"""Persistence, seasonal-naive, and mean forecasts on a chronological holdout.

The holdout matches :class:`src.training.trainer.TimeSeriesTrainer`: with
``n_windows = n_rows - seq_len`` and ``n_val = int(n_windows * val_split)``,
the validation targets are the last ``n_val`` windows. Window ``i`` predicts
the row at index ``i + seq_len``.
"""

from __future__ import annotations

import math

import numpy as np


def target_positions(
    n_rows: int, seq_len: int, val_split: float
) -> tuple[int, np.ndarray]:
    """Return ``(n_train_windows, validation_target_indices)``.

    ``validation_target_indices`` are positions in the raw series, not window
    indices. The value at each position is the one-step target.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    if n_rows <= seq_len:
        raise ValueError("series must be longer than seq_len")
    n_windows = n_rows - seq_len
    n_val = int(n_windows * val_split)
    if val_split > 0 and (n_val == 0 or n_val == n_windows):
        raise ValueError("val_split leaves an empty train or validation set")
    n_train = n_windows - n_val
    targets = np.arange(n_train + seq_len, n_rows)
    if len(targets) != n_val:
        raise AssertionError("validation target count drifted from n_val")
    return n_train, targets


def persistence(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Predict each target with the previous observation."""
    if np.any(targets < 1):
        raise ValueError("persistence needs a previous observation for every target")
    series = np.asarray(values, dtype=float)
    return np.array(series[np.asarray(targets, dtype=int) - 1], dtype=float)


def seasonal_naive(values: np.ndarray, targets: np.ndarray, period: int) -> np.ndarray:
    """Predict each target with the observation ``period`` steps earlier.

    Targets without that much history fall back to persistence.
    """
    if period < 1:
        raise ValueError(f"seasonal period must be positive, got {period}")
    series = np.asarray(values, dtype=float)
    preds = persistence(series, targets)
    idx = targets - period
    ok = idx >= 0
    preds = preds.copy()
    preds[ok] = series[idx[ok]]
    return preds


def mean_forecast(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Predict every target with the mean of observations before the holdout."""
    series = np.asarray(values, dtype=float)
    first = int(targets[0])
    if first < 1:
        raise ValueError("mean forecast needs history before the first target")
    history = series[:first]
    if history.size == 0:
        raise ValueError("mean forecast history is empty")
    return np.full(len(targets), float(history.mean()))


def regression_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, and MSE. Empty input is rejected."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    yhat = np.asarray(predicted, dtype=float).reshape(-1)
    if y.size == 0 or y.size != yhat.size:
        raise ValueError(
            f"expected equal non-empty arrays, got {y.size} and {yhat.size}"
        )
    err = yhat - y
    mse = float(np.mean(err**2))
    return {
        "n": float(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": math.sqrt(mse),
        "mse": mse,
    }


def skill_score(model_mae: float, baseline_mae: float) -> float | None:
    """``1 - model / baseline``. Positive means the model has lower MAE."""
    if baseline_mae < 0:
        raise ValueError("MAE cannot be negative")
    if baseline_mae == 0:
        return None
    return 1.0 - model_mae / baseline_mae
