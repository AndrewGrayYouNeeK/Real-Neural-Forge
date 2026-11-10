"""Shapes shared by training, metrics, and interval calibration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from src.forecasting.intervals import interval_confidence, interval_is_calibrated
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


def uncertainty_report(
    p50: Sequence[float],
    p10: Sequence[float] | None = None,
    p90: Sequence[float] | None = None,
    *,
    coverage_80: float | None = None,
    p10_hit_rate: float | None = None,
    p90_hit_rate: float | None = None,
    coverage_n: int = 0,
    coverage_window: int | None = None,
    coverage_horizon: int | None = None,
    nominal: float = 0.8,
) -> dict[str, Any]:
    """Additive uncertainty block for one forecast.

    ``p50`` is the point path. Missing bounds collapse to that point and the
    block stays uncalibrated. Stored coverage is quoted only when a real
    interval was available to score.
    """
    if not p50:
        raise ValueError("uncertainty_report requires at least one p50 value.")
    midpoint = [float(value) for value in p50]
    lows = (
        list(p10) if p10 is not None and len(p10) == len(midpoint) else list(midpoint)
    )
    highs = (
        list(p90) if p90 is not None and len(p90) == len(midpoint) else list(midpoint)
    )
    ordered_low: list[float] = []
    ordered_high: list[float] = []
    has_interval = False
    for center, low, high in zip(midpoint, lows, highs, strict=True):
        bottom = float(min(low, high))
        top = float(max(low, high))
        bottom = min(bottom, center)
        top = max(top, center)
        if top - bottom > 1e-12:
            has_interval = True
        ordered_low.append(bottom)
        ordered_high.append(top)
    coverage: dict[str, Any] | None = None
    if (
        has_interval
        and coverage_80 is not None
        and coverage_n > 0
        and coverage_window is not None
        and coverage_horizon is not None
    ):
        coverage = {
            "horizon": coverage_horizon,
            "window_size": coverage_window,
            "n": coverage_n,
            "coverage_80": coverage_80,
            "p10_hit_rate": p10_hit_rate,
            "p90_hit_rate": p90_hit_rate,
        }
    quoted_coverage = coverage_80 if coverage is not None else None
    quoted_n = coverage_n if coverage is not None else 0
    return {
        "p10": ordered_low,
        "p50": midpoint,
        "p90": ordered_high,
        "interval_80": {"lower": ordered_low, "upper": ordered_high},
        "confidence": interval_confidence(quoted_coverage, quoted_n, nominal=nominal),
        "calibrated": interval_is_calibrated(
            quoted_coverage, quoted_n, nominal=nominal
        ),
        "coverage": coverage,
    }


def training_targets(targets: torch.Tensor, spec: ForecastSpec) -> torch.Tensor:
    """Targets the network head is trained against.

    Recursive models learn the first step only. A one-step direct model keeps
    the historical ``(samples, output_dim)`` layout.
    """
    stepped = horizon_targets(targets, spec.horizon)
    if spec.strategy == "recursive" or spec.horizon == 1:
        return stepped[:, 0, :]
    return stepped
