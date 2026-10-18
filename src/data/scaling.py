"""Feature/target standardisation fitted on the training split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_EPS = 1e-8


@dataclass
class StandardScaler:
    """Per-feature z-score scaling for inputs ``(N, T, F)`` and targets ``(N, O)``."""

    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    @classmethod
    def fit(cls, inputs: torch.Tensor, targets: torch.Tensor) -> StandardScaler:
        if inputs.numel() == 0 or targets.numel() == 0:
            raise ValueError("Cannot fit a scaler on an empty dataset.")
        x = inputs.reshape(-1, inputs.size(-1))
        y = targets.reshape(-1, targets.size(-1))
        # Constant columns get std 1 so they pass through as zeros, not NaN.
        x_std = x.std(dim=0, unbiased=False)
        y_std = y.std(dim=0, unbiased=False)
        return cls(
            x_mean=x.mean(dim=0),
            x_std=torch.where(x_std < _EPS, torch.ones_like(x_std), x_std),
            y_mean=y.mean(dim=0),
            y_std=torch.where(y_std < _EPS, torch.ones_like(y_std), y_std),
        )

    def transform_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def transform_targets(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)

    def inverse_targets(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)

    def state_dict(self) -> dict[str, list[float]]:
        return {
            k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "y_mean", "y_std")
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> StandardScaler:
        return cls(
            **{
                k: torch.tensor(state[k], dtype=torch.float32)
                for k in ("x_mean", "x_std", "y_mean", "y_std")
            }
        )
