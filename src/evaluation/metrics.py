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
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)
    return (preds == targets).float().mean().item()
