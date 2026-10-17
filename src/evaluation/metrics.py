"""Evaluation metrics."""

from __future__ import annotations

import torch


def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean((preds - targets) ** 2).item()


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of predictions equal to the targets.

    ``preds`` may be class labels (``(N,)`` or ``(N, 1)``) or per-class scores
    ``(N, C)`` with ``C > 1``, which are reduced with ``argmax``. Targets are
    flattened so ``(N,)`` and ``(N, 1)`` layouts compare element-wise instead
    of broadcasting to an ``(N, N)`` matrix.
    """
    if preds.dim() > 1 and preds.size(-1) > 1:
        preds = preds.argmax(dim=-1)
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    if preds.numel() != targets.numel():
        raise ValueError(
            f"accuracy: got {preds.numel()} predictions for {targets.numel()} targets"
        )
    if preds.numel() == 0:
        raise ValueError("accuracy: empty input")
    return (preds == targets.to(preds.dtype)).float().mean().item()
