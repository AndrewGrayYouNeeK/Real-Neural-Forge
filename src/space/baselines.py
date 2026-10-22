"""Lag baselines, drift, and storm-day scores for the space series.

Persistence and the one-year calendar naive already live in
``src.evaluation.baselines`` and run when a config sets
``data.report_baselines``. The lags here are the ones that matter for the
Sun: one 27-day solar rotation, and one ~11-year Schwabe cycle.
"""

from __future__ import annotations

import math

import numpy as np

# Bartels / Carrington rotation used as the seasonal lag for daily solar indices.
SOLAR_ROTATION_DAYS = 27
# 11 * 365.25 = 4017.75, rounded to a whole-day lag.
SCHWABE_CYCLE_DAYS = 4018
# 11 calendar years of monthly bins.
SCHWABE_CYCLE_MONTHS = 132


def seasonal_naive_lag(
    values: np.ndarray, target_rows: np.ndarray, period: int
) -> np.ndarray:
    """Predict each label with the observation ``period`` steps earlier."""
    if period < 1:
        raise ValueError(f"seasonal period must be positive, got {period}")
    series = np.asarray(values, dtype=np.float64)
    rows = np.asarray(target_rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("seasonal naive lag needs at least one target row")
    if int(rows.min()) < period:
        raise ValueError(
            f"seasonal lag {period} needs history before row {int(rows.min())}"
        )
    if int(rows.max()) >= len(series):
        raise ValueError("target row is outside the series")
    return series[rows - period]


def drift_forecast(values: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """One-step drift: previous value plus the mean change before validation.

    The mean step is fit only on rows before the first validation label, so
    the holdout does not leak into the baseline.
    """
    series = np.asarray(values, dtype=np.float64)
    rows = np.asarray(target_rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("drift forecast needs at least one target row")
    origin = int(rows.min())
    if origin < 2:
        raise ValueError("drift needs at least two observations before validation")
    step = float(np.mean(np.diff(series[:origin])))
    return series[rows - 1] + step


def mean_forecast(values: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """Predict every label with the mean of observations before the holdout."""
    series = np.asarray(values, dtype=np.float64)
    rows = np.asarray(target_rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("mean forecast needs at least one target row")
    origin = int(rows.min())
    if origin < 1:
        raise ValueError("mean forecast needs history before the first target")
    history = series[:origin]
    if history.size == 0 or not np.isfinite(history).all():
        raise ValueError("mean forecast history must be finite")
    return np.full(rows.shape, float(history.mean()))


def discovery_log_regression(
    values: np.ndarray, covariate: np.ndarray, target_rows: np.ndarray
) -> np.ndarray:
    """Predict counts from log1p(catalog size), fit only before the holdout.

    ``covariate`` is aligned to ``values``. For close approaches it is the
    number of NEOs whose first observation falls before that month. The fit
    does not see the validation labels or the covariate values on those rows.
    """
    series = np.asarray(values, dtype=np.float64)
    catalog = np.asarray(covariate, dtype=np.float64)
    rows = np.asarray(target_rows, dtype=np.int64)
    if catalog.shape != series.shape:
        raise ValueError("discovery covariate length does not match the series")
    if rows.size == 0:
        raise ValueError("discovery regression needs at least one target row")
    origin = int(rows.min())
    if origin < 2:
        raise ValueError("discovery regression needs history before validation")
    if (
        not np.isfinite(series[:origin]).all()
        or not np.isfinite(catalog[:origin]).all()
    ):
        raise ValueError("discovery regression history must be finite")
    features = np.log1p(np.clip(catalog[:origin], 0.0, None))
    if float(np.std(features)) == 0.0:
        raise ValueError("discovery covariate has no variance before validation")
    design = np.column_stack([np.ones(origin), features])
    coef, _, _, _ = np.linalg.lstsq(design, series[:origin], rcond=None)
    future = np.log1p(np.clip(catalog[rows], 0.0, None))
    return np.asarray(coef[0] + coef[1] * future, dtype=np.float64)


def pearson(preds: np.ndarray, targets: np.ndarray) -> float:
    """Pearson correlation, or NaN when either side has no variance."""
    predicted = np.asarray(preds, dtype=np.float64).reshape(-1)
    actual = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape:
        raise ValueError(
            f"got {predicted.shape[0]} predictions for {actual.shape[0]} targets"
        )
    if predicted.size < 2:
        return float("nan")
    if float(np.std(predicted)) == 0.0 or float(np.std(actual)) == 0.0:
        return float("nan")
    return float(np.corrcoef(predicted, actual)[0, 1])


def storm_day_scores(
    preds: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, float]:
    """Score a Kp-style storm alarm: predict storm when the forecast crosses.

    A day is a storm day when the observed value is ``>= threshold``. Precision
    is NaN when the forecast never raises an alarm, and recall is NaN when the
    validation window contains no storm days.
    """
    predicted = np.asarray(preds, dtype=np.float64).reshape(-1) >= threshold
    actual = np.asarray(targets, dtype=np.float64).reshape(-1) >= threshold
    if predicted.shape != actual.shape:
        raise ValueError(
            f"got {predicted.shape[0]} predictions for {actual.shape[0]} targets"
        )
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0.0:
        f1 = float("nan")
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    n = int(actual.size)
    return {
        "threshold": float(threshold),
        "n": float(n),
        "storm_days": float(tp + fn),
        "storm_rate": (tp + fn) / n if n else float("nan"),
        "alarms": float(tp + fp),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
