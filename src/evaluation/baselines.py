"""Persistence and seasonal-naive forecasts on a temporal validation split.

The row selection matches ``TimeSeriesTrainer``: sliding windows of length
``seq_len`` step forward by one day, and the last ``int(n_windows * val_split)``
windows are the validation set. The label of a window is the next row after it.

Persistence predicts that next day with the previous observation (tomorrow =
today). Seasonal naive predicts it with the same calendar day one year earlier.
29 February falls back to 28 February when the previous year has no 29 February.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.timeseries import resolve_val_count
from src.evaluation.metrics import mae, mse, rmse


def temporal_validation_target_rows(
    n_rows: int, seq_len: int, val_split: float
) -> np.ndarray:
    """Row indexes of the labels in the temporal validation split."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if n_rows <= seq_len:
        raise ValueError("series must contain more rows than seq_len")
    n_windows = n_rows - seq_len
    n_val = resolve_val_count(n_windows, val_split)
    if n_val == 0:
        raise ValueError("baseline comparison requires a non-empty validation split")
    n_fit = n_windows - n_val
    return np.arange(n_fit + seq_len, n_rows)


def persistence_forecast(values: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """Tomorrow equals today: the observation immediately before each label."""
    rows = np.asarray(target_rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("persistence forecast needs at least one target row")
    if int(rows.min()) < 1:
        raise ValueError("persistence needs the previous observation")
    return np.asarray(values, dtype=np.float64)[rows - 1]


def _same_day_previous_year(day: np.datetime64) -> np.datetime64:
    stamp = pd.Timestamp(day)
    try:
        previous = stamp.replace(year=stamp.year - 1)
    except ValueError:
        if stamp.month == 2 and stamp.day == 29:
            previous = pd.Timestamp(year=stamp.year - 1, month=2, day=28)
        else:
            raise
    return np.datetime64(previous.date(), "D")


def seasonal_naive_calendar(
    values: np.ndarray,
    dates: np.ndarray,
    target_rows: np.ndarray,
) -> np.ndarray:
    """Predict each label with the same month and day in the previous year."""
    series = np.asarray(values, dtype=np.float64)
    calendar = np.asarray(dates, dtype="datetime64[D]")
    if len(series) != len(calendar):
        raise ValueError("values and dates must have the same length")
    lookup: dict[np.datetime64, float] = {}
    for index, day in enumerate(calendar):
        if day in lookup:
            raise ValueError("dates must be unique for a seasonal naive forecast")
        lookup[day] = float(series[index])

    rows = np.asarray(target_rows, dtype=np.int64)
    preds = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        position = int(row)
        if position < 0 or position >= len(series):
            raise ValueError(f"target row {position} is outside the series")
        key = _same_day_previous_year(calendar[position])
        if key not in lookup:
            raise ValueError(
                "No observation on "
                f"{key} for the seasonal naive forecast of {calendar[position]}"
            )
        preds[index] = lookup[key]
    return preds


def regression_scores(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """MSE, RMSE, and MAE in the units of ``targets``."""
    predicted = torch.tensor(np.asarray(preds, dtype=np.float32)).reshape(-1, 1)
    actual = torch.tensor(np.asarray(targets, dtype=np.float32)).reshape(-1, 1)
    if predicted.shape != actual.shape:
        raise ValueError(
            f"got {predicted.shape[0]} predictions for {actual.shape[0]} targets"
        )
    return {
        "mse": mse(predicted, actual),
        "rmse": rmse(predicted, actual),
        "mae": mae(predicted, actual),
    }


def load_baseline_series(
    path: str | Path,
    target_column: str,
    date_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read the target column, and dates when a date column is named.

    Row order is the series order. That matches ``load_csv_timeseries``, which
    also keeps the file order.
    """
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column {target_column!r} in {path}")
    values = frame[target_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Target column {target_column!r} contains non-finite values.")
    dates = None
    if date_column:
        if date_column not in frame.columns:
            raise ValueError(f"Missing date column {date_column!r} in {path}")
        dates = pd.to_datetime(frame[date_column], format="%Y-%m-%d").to_numpy(
            dtype="datetime64[D]"
        )
    return values, dates


def compare_to_baselines(
    values: np.ndarray,
    target_rows: np.ndarray,
    model_preds: np.ndarray,
    *,
    dates: np.ndarray | None = None,
    model_targets: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Score the model, persistence, and (when dates exist) seasonal naive.

    ``model_targets``, when given, must be the validation labels in the same
    order as ``target_rows``. A mismatch means the model was scored on a
    different split than the baselines.
    """
    series = np.asarray(values, dtype=np.float64)
    rows = np.asarray(target_rows, dtype=np.int64)
    targets = series[rows]
    if model_targets is not None:
        reported = np.asarray(model_targets, dtype=np.float64).reshape(-1)
        if reported.shape != targets.shape or not np.allclose(
            reported, targets, rtol=1e-4, atol=1e-3
        ):
            raise ValueError(
                "Model targets do not match the temporal validation rows of the series."
            )
    predicted = np.asarray(model_preds, dtype=np.float64).reshape(-1)
    if predicted.shape != targets.shape:
        raise ValueError(
            f"got {predicted.shape[0]} model predictions for {targets.shape[0]} targets"
        )

    scores = {"model": regression_scores(predicted, targets)}
    scores["persistence"] = regression_scores(
        persistence_forecast(series, rows), targets
    )
    if dates is not None:
        scores["seasonal_naive"] = regression_scores(
            seasonal_naive_calendar(series, dates, rows), targets
        )
    return scores


def format_score_table(scores: dict[str, dict[str, float]]) -> str:
    """Plain-text RMSE/MAE table for the training log."""
    lines = [f"{'method':<16} {'RMSE':>10} {'MAE':>10}"]
    for name, metrics in scores.items():
        lines.append(f"{name:<16} {metrics['rmse']:>10.4f} {metrics['mae']:>10.4f}")
    return "\n".join(lines)
