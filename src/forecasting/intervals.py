"""Pinball loss, interval coverage, and split-conformal calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


def pinball_loss(
    preds: torch.Tensor, targets: torch.Tensor, quantiles: Sequence[float]
) -> torch.Tensor:
    """Average pinball (quantile) loss.

    ``preds`` is ``(batch, quantiles, output_dim)`` or
    ``(batch, horizon, quantiles, output_dim)``. Targets omit the quantile axis.
    """
    if not quantiles:
        raise ValueError("pinball_loss requires at least one quantile.")
    levels = torch.as_tensor(list(quantiles), device=preds.device, dtype=preds.dtype)
    if preds.dim() == 3:
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)
        level_view = levels.view(1, -1, 1)
    elif preds.dim() == 4:
        if targets.dim() == 3:
            targets = targets.unsqueeze(2)
        level_view = levels.view(1, 1, -1, 1)
    else:
        raise ValueError(
            "pinball_loss expected (batch, quantiles, output_dim) or "
            f"(batch, horizon, quantiles, output_dim), got {tuple(preds.shape)}."
        )
    if targets.shape != preds.shape:
        try:
            targets = targets.expand(preds.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"pinball targets {tuple(targets.shape)} do not match predictions {tuple(preds.shape)}."
            ) from exc
    error = targets - preds
    return torch.maximum(level_view * error, (level_view - 1) * error).mean()


def interval_coverage(
    lower: torch.Tensor, upper: torch.Tensor, targets: torch.Tensor
) -> float:
    """Fraction of target elements that fall inside ``[lower, upper]``."""
    if lower.shape != upper.shape or lower.shape != targets.shape:
        raise ValueError(
            "coverage shapes differ: "
            f"lower {tuple(lower.shape)}, upper {tuple(upper.shape)}, targets {tuple(targets.shape)}."
        )
    if lower.numel() == 0:
        raise ValueError("coverage received an empty interval.")
    inside = (targets >= lower) & (targets <= upper)
    return float(inside.float().mean().item())


def mean_interval_width(lower: torch.Tensor, upper: torch.Tensor) -> float:
    if lower.shape != upper.shape:
        raise ValueError(
            f"width shapes differ: lower {tuple(lower.shape)}, upper {tuple(upper.shape)}."
        )
    if lower.numel() == 0:
        raise ValueError("width received an empty interval.")
    return float((upper - lower).mean().item())


def conformal_quantile(scores: torch.Tensor, nominal: float) -> float:
    """Finite-sample empirical quantile of ``scores`` at ``nominal`` coverage.

    Rank is ``ceil((n + 1) * nominal)``, clipped to the sample, which is the
    split-conformal level for target coverage ``nominal``.
    """
    if not 0.0 < nominal < 1.0:
        raise ValueError(f"nominal coverage must lie in (0, 1), got {nominal}.")
    flat = scores.detach().reshape(-1).to(dtype=torch.float64)
    if flat.numel() == 0:
        raise ValueError("conformal_quantile received no scores.")
    if not torch.isfinite(flat).all():
        raise ValueError("conformal scores must be finite.")
    ordered = torch.sort(flat).values
    n_scores = int(ordered.numel())
    rank = math.ceil((n_scores + 1) * nominal)
    rank = min(max(rank, 1), n_scores)
    return float(ordered[rank - 1].item())


def apply_qhat(
    lower: torch.Tensor, upper: torch.Tensor, qhat: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand (or shrink) each lead time by its conformal offset.

    ``lower`` and ``upper`` are ``(samples, horizon, output_dim)``. A negative
    offset tightens quantile bounds; if that would cross, the bound collapses
    to the midpoint instead of flipping.
    """
    if lower.shape != upper.shape:
        raise ValueError(
            f"qhat shapes differ: lower {tuple(lower.shape)}, upper {tuple(upper.shape)}."
        )
    if lower.dim() != 3:
        raise ValueError(
            f"qhat expected (samples, horizon, output_dim), got {tuple(lower.shape)}."
        )
    if len(qhat) != lower.size(1):
        raise ValueError(
            f"conformal offset length {len(qhat)} does not match horizon {int(lower.size(1))}."
        )
    delta = torch.tensor(list(qhat), device=lower.device, dtype=lower.dtype).view(
        1, -1, 1
    )
    adjusted_lower = lower - delta
    adjusted_upper = upper + delta
    crossed = adjusted_lower > adjusted_upper
    if bool(crossed.any()):
        midpoint = (lower + upper) / 2
        adjusted_lower = torch.where(crossed, midpoint, adjusted_lower)
        adjusted_upper = torch.where(crossed, midpoint, adjusted_upper)
    return adjusted_lower, adjusted_upper


@dataclass(frozen=True)
class IntervalCalibration:
    """Conformal offset and the coverage numbers we are willing to quote.

    ``validation_coverage`` is measured on the later half of the rows.
    ``qhat`` is fit only on the earlier half. ``raw_coverage`` is the
    uncalibrated quantile interval on every row, and is ``None`` for residual
    intervals because those intervals do not exist before ``qhat`` is fit.
    """

    method: str
    qhat: tuple[float, ...]
    nominal_coverage: float
    validation_coverage: float | None
    raw_coverage: float | None
    interval_width: float | None
    coverage_horizon: int


def calibrate_intervals(
    *,
    point: torch.Tensor,
    targets: torch.Tensor,
    nominal: float,
    method: str,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> IntervalCalibration:
    """Fit a per-lead conformal offset. Rows must already be in time order."""
    if method not in {"cqr", "residual"}:
        raise ValueError("interval method must be 'cqr' or 'residual'.")
    if not 0.0 < nominal < 1.0:
        raise ValueError(f"nominal coverage must lie in (0, 1), got {nominal}.")

    point_steps = _as_steps(point)
    target_steps = _as_steps(targets)
    if point_steps.shape != target_steps.shape:
        raise ValueError(
            f"point {tuple(point_steps.shape)} does not match targets {tuple(target_steps.shape)}."
        )
    n_rows, horizon, _ = point_steps.shape
    if n_rows < 1:
        raise ValueError("cannot calibrate intervals on an empty set.")

    if method == "cqr":
        if lower is None or upper is None:
            raise ValueError("cqr calibration requires quantile bounds.")
        lower_steps = _as_steps(lower)
        upper_steps = _as_steps(upper)
        if (
            lower_steps.shape != point_steps.shape
            or upper_steps.shape != point_steps.shape
        ):
            raise ValueError("quantile bounds do not match the point forecast.")
        raw_coverage = interval_coverage(lower_steps, upper_steps, target_steps)
        base_lower, base_upper = lower_steps, upper_steps
        scores = torch.maximum(lower_steps - target_steps, target_steps - upper_steps)
    else:
        raw_coverage = None
        base_lower, base_upper = point_steps, point_steps
        scores = (target_steps - point_steps).abs()

    # Four rows leave two to fit and two to score. Fewer than that cannot
    # support a coverage claim that was not used to choose the offset.
    if n_rows >= 4:
        n_cal = n_rows // 2
        qhat = _qhat_per_step(scores[:n_cal], nominal)
        adjusted_lower, adjusted_upper = apply_qhat(base_lower, base_upper, qhat)
        report = slice(n_cal, None)
        validation_coverage = interval_coverage(
            adjusted_lower[report], adjusted_upper[report], target_steps[report]
        )
        width = mean_interval_width(adjusted_lower[report], adjusted_upper[report])
    else:
        qhat = _qhat_per_step(scores, nominal)
        adjusted_lower, adjusted_upper = apply_qhat(base_lower, base_upper, qhat)
        validation_coverage = None
        width = mean_interval_width(adjusted_lower, adjusted_upper)

    return IntervalCalibration(
        method=method,
        qhat=qhat,
        nominal_coverage=nominal,
        validation_coverage=validation_coverage,
        raw_coverage=raw_coverage,
        interval_width=width,
        coverage_horizon=horizon,
    )


def _as_steps(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    raise ValueError(
        f"Expected (samples, output) or (samples, horizon, output), got {tuple(tensor.shape)}."
    )


def _qhat_per_step(scores: torch.Tensor, nominal: float) -> tuple[float, ...]:
    # scores: (samples, horizon, output_dim). Pool features so the offset is marginal.
    return tuple(
        conformal_quantile(scores[:, step, :], nominal)
        for step in range(scores.size(1))
    )
