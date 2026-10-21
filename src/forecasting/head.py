"""Output head for multi-step and quantile forecasts.

The one-step point model keeps ``nn.Linear`` so existing checkpoints still
load. This head is used only when the horizon is longer than one step or
quantile levels are configured.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastHead(nn.Module):
    """Project a pooled encoder state to ``(horizon, quantiles, output_dim)``.

    Quantile outputs are built as a base level plus softplus increments so
    ``q10 <= q50 <= q90`` holds for every sample.
    """

    def __init__(
        self,
        in_features: int,
        output_dim: int,
        horizon: int,
        quantiles: tuple[float, ...] = (),
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        self.output_dim = output_dim
        self.horizon = horizon
        self.quantiles = quantiles
        n_quantiles = len(quantiles) if quantiles else 1
        self.projection = nn.Linear(in_features, horizon * n_quantiles * output_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.projection(hidden)
        n_quantiles = len(self.quantiles) if self.quantiles else 1
        view = raw.view(raw.size(0), self.horizon, n_quantiles, self.output_dim)
        if self.quantiles and n_quantiles > 1:
            base = view[:, :, :1, :]
            increments = F.softplus(view[:, :, 1:, :])
            view = torch.cat([base, increments], dim=2).cumsum(dim=2)
        if not self.quantiles:
            point = view[:, :, 0, :]
            if self.horizon == 1:
                return cast(torch.Tensor, point[:, 0, :])
            return cast(torch.Tensor, point)
        if self.horizon == 1:
            return cast(torch.Tensor, view[:, 0, :, :])
        return cast(torch.Tensor, view)
