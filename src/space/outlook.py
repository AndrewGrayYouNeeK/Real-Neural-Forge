"""Twenty-four month outlooks with residual quantile bands.

The network stays a one-step model so the holdout comparison with
persistence, the seasonal naive, and the mean is the same comparison the
other space series use. The published path rolls that one-step model forward.
Each lead's p10 and p90 are empirical residual quantiles fit on the earlier
half of the validation origins that still have that lead inside the series.
Coverage is measured on the later half and is not a second untouched test:
the checkpoint itself was chosen on the validation loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.baselines import temporal_validation_target_rows
from src.forecasting.intervals import conformal_quantile
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.utils.paths import resolve_repo_path


def build_outlook(cfg: dict[str, Any]) -> dict[str, Any]:
    """Forecast ``space.outlook_months`` steps past the end of the CSV."""
    space_cfg = cfg.get("space") or {}
    months = int(space_cfg.get("outlook_months") or 0)
    if months < 1:
        raise ValueError("space.outlook_months must be positive")
    data_cfg = cfg["data"]
    seq_len = int(data_cfg.get("seq_len", 64))
    path = resolve_repo_path(data_cfg["path"])
    frame = pd.read_csv(path)
    columns = list(data_cfg.get("feature_columns", ["value"]))
    target_name = str(data_cfg.get("target_column", "value"))
    missing = [
        column
        for column in (*columns, target_name, "date")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Outlook CSV is missing {missing}")
    features = frame[columns].to_numpy(dtype=np.float64)
    target = frame[target_name].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Outlook series has non-finite values.")
    dates = pd.to_datetime(frame["date"])
    rows = temporal_validation_target_rows(
        len(target), seq_len, float(data_cfg.get("val_split", 0.0))
    )
    if int(rows.min()) < seq_len:
        raise ValueError("Outlook validation rows start before a full window.")
    windows = np.stack([features[int(row) - seq_len : int(row)] for row in rows])
    checkpoint = resolve_repo_path(cfg["inference"]["checkpoint_path"])
    predictor = Predictor.from_checkpoint(build_model(cfg), checkpoint, device="cpu")
    predicted = _points(predictor, windows, months)
    actual = np.full((len(rows), months), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        for lead in range(months):
            position = int(row) + lead
            if position < len(target):
                actual[index, lead] = target[position]
    low, high, coverage, extrapolated = _lead_quantiles(predicted, actual)
    tail = _points(predictor, features[-seq_len:], months)[0]
    lower_bound = space_cfg.get("outlook_min")
    upper_bound = space_cfg.get("outlook_max")
    last = pd.Timestamp(dates.iloc[-1])
    published: list[dict[str, Any]] = []
    for lead in range(months):
        point = float(tail[lead])
        band_low = point + low[lead]
        band_high = point + high[lead]
        point, band_low, band_high = _clip_ordered(
            point, band_low, band_high, lower_bound, upper_bound
        )
        month = last + pd.DateOffset(months=lead + 1)
        published.append(
            {
                "date": pd.Timestamp(month).strftime("%Y-%m-%d"),
                "p10": round(band_low, 4),
                "p50": round(point, 4),
                "p90": round(band_high, 4),
            }
        )
    finite_coverage = [value for value in coverage if value is not None]
    return {
        "method": (
            "recursive one-step forecast; p10 and p90 are split residual "
            "quantiles (nominal 0.1 and 0.9) fit on the earlier half of the "
            "validation origins"
        ),
        "nominal_coverage": 0.8,
        "validation_coverage": (
            None if not finite_coverage else round(float(np.mean(finite_coverage)), 4)
        ),
        "interval_extrapolated": extrapolated,
        "months": published,
    }


def _points(predictor: Predictor, windows: np.ndarray, months: int) -> np.ndarray:
    tensor = torch.tensor(np.asarray(windows, dtype=np.float32))
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    batch = predictor.forecast(tensor, horizon=months)
    return batch.point[:, :, 0].detach().cpu().numpy().astype(np.float64)


def _lead_quantiles(
    predicted: np.ndarray, actual: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[float | None], bool]:
    months = predicted.shape[1]
    low = np.zeros(months, dtype=np.float64)
    high = np.zeros(months, dtype=np.float64)
    coverage: list[float | None] = []
    extrapolated = False
    previous: tuple[float, float] | None = None
    for lead in range(months):
        residuals = actual[:, lead] - predicted[:, lead]
        finite = residuals[np.isfinite(residuals)]
        if finite.size == 0:
            if previous is None:
                raise ValueError("No validation residual is available for the outlook.")
            low[lead], high[lead] = previous
            coverage.append(None)
            extrapolated = True
            continue
        q10, q90, held = _split_quantile(finite)
        low[lead] = q10
        high[lead] = q90
        previous = (q10, q90)
        coverage.append(None if held is None else round(held, 4))
    return low, high, coverage, extrapolated


def _split_quantile(residuals: np.ndarray) -> tuple[float, float, float | None]:
    """Fit on the earlier half. Score coverage on the later half."""
    ordered = np.asarray(residuals, dtype=np.float64)
    if ordered.size < 4:
        return _quantile(ordered, 0.1), _quantile(ordered, 0.9), None
    split = ordered.size // 2
    low = _quantile(ordered[:split], 0.1)
    high = _quantile(ordered[:split], 0.9)
    held = ordered[split:]
    covered = float(np.mean((held >= low) & (held <= high)))
    return low, high, covered


def _quantile(values: np.ndarray, nominal: float) -> float:
    return float(conformal_quantile(torch.tensor(values, dtype=torch.float64), nominal))


def _clip_ordered(
    point: float,
    low: float,
    high: float,
    lower_bound: object,
    upper_bound: object,
) -> tuple[float, float, float]:
    point, low, high = (
        _clip(point, lower_bound, upper_bound),
        _clip(low, lower_bound, upper_bound),
        _clip(high, lower_bound, upper_bound),
    )
    low = min(low, point)
    high = max(high, point)
    return point, low, high


def _clip(value: float, lower_bound: object, upper_bound: object) -> float:
    clipped = value
    if isinstance(lower_bound, int | float) and not isinstance(lower_bound, bool):
        clipped = max(clipped, float(lower_bound))
    if isinstance(upper_bound, int | float) and not isinstance(upper_bound, bool):
        clipped = min(clipped, float(upper_bound))
    return clipped
