"""DataLoader construction helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split


def build_dataloaders(
    dataset: Dataset,
    *,
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool | None = None,
    **loader_kwargs: Any,
) -> dict[str, DataLoader]:
    """Split *dataset* and return ``train``, ``val``, and ``test`` :class:`DataLoader` objects.

    Args:
        dataset:     Full dataset to split.
        val_split:   Fraction of samples used for validation.
        test_split:  Fraction of samples used for testing.
        batch_size:  Mini-batch size.
        num_workers: Worker processes for data loading.
        seed:        Random seed for reproducible splits.
        pin_memory:  Pin memory for GPU transfer (auto-detected by default).

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``.
    """
    n = len(dataset)  # type: ignore[arg-type]
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        **loader_kwargs,
    }

    return {
        "train": DataLoader(train_ds, shuffle=True, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
    }
