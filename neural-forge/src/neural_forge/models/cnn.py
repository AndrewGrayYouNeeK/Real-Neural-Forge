"""Simple convolutional neural network."""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_forge.models.base import BaseModel


class SimpleCNN(BaseModel):
    """A compact CNN suitable for small image classification tasks.

    Architecture:
        Conv2d → BN → ReLU → MaxPool (×2 blocks) → Flatten → Linear

    Args:
        in_channels:  Number of input image channels (e.g. 1 for grayscale, 3 for RGB).
        num_classes:  Number of output classes.
        base_filters: Number of filters in the first convolutional block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_filters: int = 32,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(base_filters * 2 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
