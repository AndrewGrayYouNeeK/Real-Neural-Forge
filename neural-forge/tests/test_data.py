"""Unit tests for data loading utilities."""

import numpy as np
import pytest
import torch

from neural_forge.data.dataset import NeuralForgeDataset
from neural_forge.data.loader import build_dataloaders


class TestNeuralForgeDataset:
    def _make(self, n=100, feat=8, classes=3):
        X = np.random.randn(n, feat).astype(np.float32)
        y = np.random.randint(0, classes, size=(n,))
        return NeuralForgeDataset(X, y)

    def test_len(self):
        ds = self._make(50)
        assert len(ds) == 50

    def test_getitem_shapes(self):
        ds = self._make()
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_mismatched_lengths_raise(self):
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(9, dtype=np.int64)
        with pytest.raises(ValueError):
            NeuralForgeDataset(X, y)

    def test_transform_applied(self):
        X = np.ones((5, 4), dtype=np.float32)
        y = np.zeros(5, dtype=np.int64)
        ds = NeuralForgeDataset(X, y, transform=lambda t: t * 2)
        x, _ = ds[0]
        assert torch.all(x == 2.0)


class TestBuildDataloaders:
    def _dataset(self, n=200):
        X = np.random.randn(n, 8).astype(np.float32)
        y = np.random.randint(0, 3, size=(n,))
        return NeuralForgeDataset(X, y)

    def test_keys(self):
        loaders = build_dataloaders(self._dataset(), batch_size=16, num_workers=0)
        assert set(loaders) == {"train", "val", "test"}

    def test_split_sizes(self):
        ds = self._dataset(200)
        loaders = build_dataloaders(
            ds, val_split=0.1, test_split=0.1, batch_size=16, num_workers=0
        )
        total = sum(
            len(loader.dataset)  # type: ignore[arg-type]
            for loader in loaders.values()
        )
        assert total == 200

    def test_train_shuffled(self):
        ds = self._dataset(100)
        loaders = build_dataloaders(ds, batch_size=100, num_workers=0)
        # Loader should be iterable without error
        batch = next(iter(loaders["train"]))
        assert len(batch) == 2
