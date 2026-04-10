"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from neural_forge.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
    **extra: Any,
) -> None:
    """Save model and optimiser state to *path*.

    Args:
        model:     Model whose weights are saved.
        optimizer: Optimiser whose state is saved.
        epoch:     Current epoch number (stored in the checkpoint).
        path:      Destination file path (``.pt`` recommended).
        **extra:   Additional metadata to store in the checkpoint dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **extra,
    }
    torch.save(payload, path)
    logger.debug("Saved checkpoint → %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model / optimiser state.

    Args:
        path:         Path to the checkpoint file.
        model:        If provided, its weights are restored in-place.
        optimizer:    If provided, its state is restored in-place.
        map_location: Device mapping for :func:`torch.load`.

    Returns:
        The raw checkpoint dictionary.
    """
    ckpt: dict[str, Any] = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %s)", path, ckpt.get("epoch", "?"))
    return ckpt
