"""MNIST loaders. Pixels stay in [0, 1] (no mean/std), so the browser can do the same."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


def make_loaders(batch_size: int = 128, seed: int = 0) -> tuple[Loader, Loader]:
    """Return (train, test) loaders. Training images get small random shifts, scale and rotation."""
    train_tf = transforms.Compose(
        [
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ]
    )
    test_tf = transforms.ToTensor()
    train_set = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=test_tf)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    test = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train, test
