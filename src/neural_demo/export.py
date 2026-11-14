"""Export the trained weights for the browser, plus reference outputs to check it against."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torchvision import datasets

from .data import DATA_DIR
from .model import DigitCNN
from .train import ARTIFACTS, CHECKPOINT

Array = npt.NDArray[np.float32]
REFERENCE_SAMPLES = 12
# Order of the operations, written into weights.js so the browser follows the same recipe.
ARCHITECTURE = [
    "conv", "relu", "pool",
    "conv", "relu", "pool",
    "flatten", "dense", "relu", "dense",
]  # fmt: skip


@dataclass(frozen=True)
class Layer:
    """A weighted layer. ``kind`` is "conv" (weight: out, in, 3, 3) or "dense" (out, in)."""

    kind: str
    weight: Array
    bias: Array


def _to_numpy(tensor: torch.Tensor | None) -> Array:
    """Detach a tensor and copy it as float32 (a bias is optional in PyTorch)."""
    if tensor is None:
        raise ValueError("expected a bias tensor")
    return tensor.detach().cpu().numpy().astype(np.float32)


def layers_from_model(model: DigitCNN) -> list[Layer]:
    """Copy the weighted layers of ``model`` as float32 NumPy arrays."""
    parts: list[tuple[str, nn.Conv2d | nn.Linear]] = [
        ("conv", model.conv1),
        ("conv", model.conv2),
        ("dense", model.fc1),
        ("dense", model.fc2),
    ]
    return [Layer(kind, _to_numpy(m.weight), _to_numpy(m.bias)) for kind, m in parts]


def round_to_fp16(layers: list[Layer]) -> list[Layer]:
    """Round every value to float16 and back, exactly what the browser will receive."""
    return [
        Layer(
            layer.kind,
            layer.weight.astype(np.float16).astype(np.float32),
            layer.bias.astype(np.float16).astype(np.float32),
        )
        for layer in layers
    ]


def encode_fp16(values: Array) -> str:
    """Encode an array as little-endian float16, then base64."""
    return base64.b64encode(values.astype("<f2").tobytes()).decode("ascii")


def decode_fp16(text: str, shape: tuple[int, ...]) -> Array:
    """Inverse of :func:`encode_fp16`."""
    raw = np.frombuffer(base64.b64decode(text), dtype="<f2")
    return raw.astype(np.float32).reshape(shape)


def _conv3x3(x: Array, weight: Array, bias: Array) -> Array:
    """3x3 convolution with zero padding of 1. ``x`` is (N, C_in, H, W)."""
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
    out = np.einsum("nchwij,ocij->nohw", windows, weight, optimize=True)
    return np.asarray(out + bias[None, :, None, None], dtype=np.float32)


def _max_pool2(x: Array) -> Array:
    """2x2 max pooling. ``x`` is (N, C, H, W) with even H and W."""
    n, c, h, w = x.shape
    return np.asarray(x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5)), dtype=np.float32)


def forward_numpy(layers: list[Layer], x: Array) -> Array:
    """Forward pass in NumPy. ``x`` has shape (N, 1, 28, 28) with values in [0, 1]."""
    conv1, conv2, fc1, fc2 = layers
    h = _max_pool2(np.maximum(_conv3x3(x, conv1.weight, conv1.bias), 0))
    h = _max_pool2(np.maximum(_conv3x3(h, conv2.weight, conv2.bias), 0))
    h = h.reshape(h.shape[0], -1)
    h = np.maximum(h @ fc1.weight.T + fc1.bias, 0)
    return np.asarray(h @ fc2.weight.T + fc2.bias, dtype=np.float32)


def weights_js(layers: list[Layer], accuracy: float) -> str:
    """Return the JavaScript file that exposes the weights as ``window.NEURAL_WEIGHTS``."""
    payload = {
        "format": "float16-base64",
        "test_accuracy": round(accuracy, 4),
        "architecture": ARCHITECTURE,
        "layers": [
            {
                "kind": layer.kind,
                "shape": list(layer.weight.shape),
                "w": encode_fp16(layer.weight),
                "b": encode_fp16(layer.bias),
            }
            for layer in layers
        ],
    }
    return "window.NEURAL_WEIGHTS = " + json.dumps(payload, separators=(",", ":")) + ";\n"


def reference_json(layers: list[Layer], images: npt.NDArray[np.uint8]) -> str:
    """Pixels (0..255) of a few test images and the logits the browser must reproduce."""
    x = (images[:, None, :, :] / 255.0).astype(np.float32)
    logits = forward_numpy(layers, x)
    samples = [
        {
            "pixels": images[i].reshape(-1).tolist(),
            "logits": [round(float(v), 5) for v in logits[i]],
            "digit": int(np.argmax(logits[i])),
        }
        for i in range(len(images))
    ]
    return json.dumps({"samples": samples}, separators=(",", ":"))


def main() -> None:
    """Load the checkpoint, export the weights and the reference outputs."""
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = DigitCNN()
    model.load_state_dict(checkpoint["state_dict"])
    exact = layers_from_model(model)
    rounded = round_to_fp16(exact)

    test = datasets.MNIST(str(DATA_DIR), train=False, download=True)
    images = test.data.numpy().astype(np.uint8)
    labels = test.targets.numpy()
    x = (images[:, None, :, :] / 255.0).astype(np.float32)

    def accuracy(layers: list[Layer]) -> float:
        hits = 0
        for start in range(0, len(x), 2000):
            batch = forward_numpy(layers, x[start : start + 2000])
            hits += int((batch.argmax(axis=1) == labels[start : start + 2000]).sum())
        return hits / len(x)

    acc32, acc16 = accuracy(exact), accuracy(rounded)
    print(f"accuracy float32={acc32:.4f}  float16={acc16:.4f}")

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "weights.js").write_text(weights_js(rounded, acc16), encoding="utf-8")
    (ARTIFACTS / "reference.json").write_text(
        reference_json(rounded, images[:REFERENCE_SAMPLES]), encoding="utf-8"
    )
    size = Path(ARTIFACTS / "weights.js").stat().st_size
    print(f"wrote {ARTIFACTS / 'weights.js'} ({size / 1000:.0f} KB) and reference.json")


if __name__ == "__main__":
    main()
