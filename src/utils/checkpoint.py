"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: dict[str, Any],
    path: str | Path,
    scaler: dict[str, Any] | None = None,
    forecast: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
        "scaler": scaler,
    }
    # Omit the key on legacy saves so older readers see the same payload.
    if forecast is not None:
        payload["forecast"] = forecast
    torch.save(payload, path)
