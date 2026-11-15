"""Counterfactual what-if: shock the last input and compare forecasts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from src.inference.predictor import Predictor


def apply_shock(
    window: list[list[float]],
    *,
    mode: str,
    value: float,
    feature_index: int = 0,
) -> tuple[float, float, list[list[float]]]:
    """Return (before, after, shocked_window) after mutating the last step."""
    if not window or not window[-1]:
        raise ValueError("predict window is empty.")
    if feature_index < 0 or feature_index >= len(window[-1]):
        raise ValueError("feature_index is out of range for the window.")
    shocked = deepcopy(window)
    before = float(shocked[-1][feature_index])
    if mode == "pct":
        after = before * (1.0 + float(value) / 100.0)
    elif mode == "abs":
        after = before + float(value)
    else:
        raise ValueError("shock mode must be 'pct' or 'abs'.")
    shocked[-1][feature_index] = after
    return before, after, shocked


def _path_from_batch(batch: Any) -> list[float]:
    return [float(item) for item in batch.point[0, :, 0].detach().cpu().tolist()]


def compare_window_forecasts(
    predictor: Predictor,
    window: list[list[float]],
    *,
    horizon: int = 1,
    mode: str = "pct",
    value: float = -10.0,
    feature_index: int = 0,
) -> dict[str, Any]:
    """Forecast baseline vs shocked last input; return paths and plain-English delta."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    before, after, shocked_window = apply_shock(
        window, mode=mode, value=value, feature_index=feature_index
    )
    tensor = torch.tensor(window, dtype=torch.float32)
    shocked_tensor = torch.tensor(shocked_window, dtype=torch.float32)
    baseline_batch = predictor.forecast(tensor, horizon=horizon)
    shocked_batch = predictor.forecast(shocked_tensor, horizon=horizon)
    baseline = _path_from_batch(baseline_batch)
    shocked = _path_from_batch(shocked_batch)
    deltas = [s - b for b, s in zip(baseline, shocked, strict=True)]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    if mode == "pct":
        shock_text = f"A {value:+.1f}% shock to the last input"
    else:
        shock_text = f"An absolute shock of {value:+.3g} to the last input"
    if abs(mean_delta) < 1e-9:
        delta_text = (
            f"{shock_text} leaves the {horizon}-step forecast almost unchanged."
        )
    elif mean_delta > 0:
        delta_text = (
            f"{shock_text} lifts the {horizon}-step path by about {mean_delta:.3g} "
            "on average."
        )
    else:
        delta_text = (
            f"{shock_text} lowers the {horizon}-step path by about {abs(mean_delta):.3g} "
            "on average."
        )
    return {
        "horizon": horizon,
        "shock": {
            "mode": mode,
            "value": float(value),
            "feature_index": feature_index,
            "before": before,
            "after": after,
        },
        "baseline": {"p50": baseline, "forecast": baseline},
        "shocked": {"p50": shocked, "forecast": shocked},
        "delta": {"p50": deltas, "mean": mean_delta, "text": delta_text},
        "summary": delta_text,
    }
