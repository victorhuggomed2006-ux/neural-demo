"""Train the digit MLP on CPU and save a checkpoint."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Loader, make_loaders
from .model import DigitCNN

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
CHECKPOINT = ARTIFACTS / "mlp.pt"


def set_seed(seed: int) -> None:
    """Make runs repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader: Loader) -> float:
    """Return the accuracy (0..1) of ``model`` on ``loader``."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / total


def train(epochs: int = 14, lr: float = 3e-3, batch_size: int = 128, seed: int = 0) -> float:
    """Train, save ``artifacts/mlp.pt`` and return the test accuracy."""
    set_seed(seed)
    torch.set_num_threads(2)  # keep the machine responsive
    train_loader, test_loader = make_loaders(batch_size, seed)
    model = DigitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    accuracy = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            running += float(loss.item()) * labels.numel()
        scheduler.step()
        accuracy = evaluate(model, test_loader)
        mean_loss = running / len(train_loader.dataset)  # type: ignore[arg-type]
        print(
            f"epoch {epoch:2d}/{epochs} loss={mean_loss:.4f} "
            f"test_acc={accuracy:.4f} ({time.time() - started:.0f}s)",
            flush=True,
        )
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "test_accuracy": accuracy}, CHECKPOINT)
    print(f"saved {CHECKPOINT}")
    return accuracy


def main() -> None:
    """Command line entry point."""
    train()


if __name__ == "__main__":
    main()
