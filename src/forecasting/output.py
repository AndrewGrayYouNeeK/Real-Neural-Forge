"""Shapes shared by training, metrics, and interval calibration."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.forecasting.spec import ForecastSpec


def median_index(quantiles: Sequence[float]) -> int:
    """Quantile closest to 0.5; the earlier level wins a tie."""
    return min(
        range(len(quantiles)), key=lambda index: (abs(quantiles[index] - 0.5), index)
    )


def point_forecast(
    output: torch.Tensor,
    *,
    horizon: int,
    quantiles: Sequence[float] | None,
) -> torch.Tensor:
    """Reduce a head output to a point forecast.

    One-step point models already return ``(batch, output_dim)``. Quantile
    heads return the level nearest the median.
    """
    if not quantiles:
        return output
    index = median_index(quantiles)
    if horizon <= 1:
        if output.dim() != 3:
            raise ValueError(
                f"Expected quantile output (batch, quantiles, output_dim), got {tuple(output.shape)}."
            )
        return output[:, index, :]
    if output.dim() != 4:
        raise ValueError(
            "Expected quantile output (batch, horizon, quantiles, output_dim), "
            f"got {tuple(output.shape)}."
        )
    return output[:, :, index, :]


def quantile_bounds(
    output: torch.Tensor, *, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest and highest quantile from a head output."""
    if horizon <= 1:
        if output.dim() != 3:
            raise ValueError(
                f"Expected quantile output (batch, quantiles, output_dim), got {tuple(output.shape)}."
            )
        return output[:, 0, :], output[:, -1, :]
    if output.dim() != 4:
        raise ValueError(
            "Expected quantile output (batch, horizon, quantiles, output_dim), "
            f"got {tuple(output.shape)}."
        )
    return output[:, :, 0, :], output[:, :, -1, :]


def horizon_targets(targets: torch.Tensor, horizon: int) -> torch.Tensor:
    """Return ``(samples, horizon, output_dim)``."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if targets.dim() == 2:
        if horizon == 1:
            return targets.unsqueeze(1)
        if targets.size(-1) == horizon:
            return targets.unsqueeze(-1)
        raise ValueError(
            f"Cannot interpret targets {tuple(targets.shape)} as horizon {horizon}."
        )
    if targets.dim() == 3:
        if targets.size(1) != horizon:
            raise ValueError(
                f"Expected horizon {horizon}, got targets {tuple(targets.shape)}."
            )
        return targets
    raise ValueError(f"Cannot interpret targets {tuple(targets.shape)} as a forecast.")


def training_targets(targets: torch.Tensor, spec: ForecastSpec) -> torch.Tensor:
    """Targets the network head is trained against.

    Recursive models learn the first step only. A one-step direct model keeps
    the historical ``(samples, output_dim)`` layout.
    """
    stepped = horizon_targets(targets, spec.horizon)
    if spec.strategy == "recursive" or spec.horizon == 1:
        return stepped[:, 0, :]
    return stepped
