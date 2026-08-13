"""Inference predictor for time-series models."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class Predictor:
    """Wraps a trained model for batch inference."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

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
        return cls(model, device=device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return self.model(x)
