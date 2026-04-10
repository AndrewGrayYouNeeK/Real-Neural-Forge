"""Generic PyTorch dataset wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class NeuralForgeDataset(Dataset):
    """Thin wrapper around numpy arrays or pre-loaded tensors.

    Args:
        features:   Array or tensor of shape ``(N, …)``.
        labels:     Array or tensor of shape ``(N,)`` or ``(N, C)``.
        transform:  Optional callable applied to each feature sample.
        target_transform: Optional callable applied to each label.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels)
        self.transform = transform
        self.target_transform = target_transform

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        features_path: str | Path,
        labels_path: str | Path,
        **kwargs: Any,
    ) -> "NeuralForgeDataset":
        """Load dataset from ``.npy`` files."""
        features = np.load(features_path)
        labels = np.load(labels_path)
        return cls(features, labels, **kwargs)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
