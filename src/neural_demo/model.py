"""The network: a small CNN for 28x28 digit images (about 26.7 thousand parameters).

input 1x28x28
-> conv 3x3 (8 channels)  + ReLU + max-pool 2   -> 8x14x14
-> conv 3x3 (16 channels) + ReLU + max-pool 2   -> 16x7x7
-> flatten (784) -> dense 32 + ReLU -> dense 10 (logits)
"""

from __future__ import annotations

import torch
from torch import nn

CONV1_CHANNELS = 8
CONV2_CHANNELS = 16
KERNEL = 3
HIDDEN = 32
NUM_CLASSES = 10
FLAT_SIZE = CONV2_CHANNELS * 7 * 7


class DigitCNN(nn.Module):
    """Two convolution blocks followed by two dense layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, CONV1_CHANNELS, KERNEL, padding=1)
        self.conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, KERNEL, padding=1)
        self.fc1 = nn.Linear(FLAT_SIZE, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape (N, 10) for images with shape (N, 1, 28, 28) in [0, 1]."""
        x = nn.functional.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = nn.functional.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.relu(self.fc1(x.reshape(x.shape[0], -1)))
        logits: torch.Tensor = self.fc2(x)
        return logits
