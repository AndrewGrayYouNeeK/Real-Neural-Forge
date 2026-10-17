"""Inference predictor for time-series models."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from src.data.scaling import StandardScaler


class Predictor:
    """Wraps a trained model for batch inference."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        scaler: StandardScaler | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.scaler = scaler

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> Predictor:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        scaler_state = ckpt.get("scaler") if "model_state_dict" in ckpt else None
        scaler = StandardScaler.from_state_dict(scaler_state) if scaler_state else None
        return cls(model, device=device, scaler=scaler)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if self.scaler is not None:
            x = self.scaler.transform_inputs(x)
        out = cast(torch.Tensor, self.model(x))
        if self.scaler is not None:
            out = self.scaler.inverse_targets(out)
        return out
