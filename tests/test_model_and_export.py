from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_demo.export import (
    ARCHITECTURE,
    decode_fp16,
    encode_fp16,
    forward_numpy,
    layers_from_model,
    reference_json,
    round_to_fp16,
    weights_js,
)
from neural_demo.model import NUM_CLASSES, DigitCNN
from neural_demo.train import evaluate


def make_model(seed: int = 0) -> DigitCNN:
    torch.manual_seed(seed)
    return DigitCNN().eval()


def test_output_shape_and_parameter_count() -> None:
    model = make_model()
    out = model(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, NUM_CLASSES)
    # conv1 80 + conv2 1168 + fc1 25120 + fc2 330
    assert sum(p.numel() for p in model.parameters()) == 26_698


def test_numpy_forward_matches_torch() -> None:
    model = make_model()
    x = torch.rand(5, 1, 28, 28)
    expected = model(x).detach().numpy()
    actual = forward_numpy(layers_from_model(model), x.numpy())
    assert np.allclose(actual, expected, atol=1e-4)


def test_fp16_roundtrip_is_close() -> None:
    values = np.random.default_rng(0).normal(size=(8, 5)).astype(np.float32)
    decoded = decode_fp16(encode_fp16(values), values.shape)
    assert np.allclose(decoded, values, atol=2e-3)


def test_weights_js_is_valid_and_complete() -> None:
    layers = round_to_fp16(layers_from_model(make_model()))
    text = weights_js(layers, 0.99)
    assert text.startswith("window.NEURAL_WEIGHTS = ") and text.endswith(";\n")
    data = json.loads(text[len("window.NEURAL_WEIGHTS = ") : -2])
    assert data["architecture"] == ARCHITECTURE
    assert [layer["kind"] for layer in data["layers"]] == ["conv", "conv", "dense", "dense"]
    assert [layer["shape"] for layer in data["layers"]] == [
        [8, 1, 3, 3],
        [16, 8, 3, 3],
        [32, 784],
        [10, 32],
    ]
    first = data["layers"][0]
    assert np.array_equal(decode_fp16(first["w"], (8, 1, 3, 3)), layers[0].weight)


def test_reference_matches_rounded_network() -> None:
    layers = round_to_fp16(layers_from_model(make_model()))
    images = np.random.default_rng(1).integers(0, 256, size=(3, 28, 28), dtype=np.uint8)
    data = json.loads(reference_json(layers, images))
    x = (images[:, None, :, :] / 255.0).astype(np.float32)
    expected = forward_numpy(layers, x)
    for index, sample in enumerate(data["samples"]):
        assert len(sample["pixels"]) == 784
        assert np.allclose(sample["logits"], expected[index], atol=1e-4)
        assert sample["digit"] == int(np.argmax(expected[index]))


def test_evaluate_counts_correct_predictions() -> None:
    model = make_model()
    images = torch.rand(40, 1, 28, 28)
    labels = model(images).argmax(dim=1)  # the model's own answers: accuracy must be 100%
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)
    assert evaluate(model, loader) == 1.0  # type: ignore[arg-type]
