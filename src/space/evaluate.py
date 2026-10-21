"""Score a saved space-series checkpoint against the solar baselines.

The model numbers themselves come from ``TimeSeriesTrainer``, which reloads
the best checkpoint and compares persistence and the one-year calendar naive.
This module adds the 27-day and 11-year lags, the drift baseline, storm-day
detection, and the validation traces used for charts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.timeseries import load_csv_timeseries
from src.evaluation.baselines import (
    persistence_forecast,
    regression_scores,
    temporal_validation_target_rows,
)
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.space.baselines import (
    drift_forecast,
    pearson,
    seasonal_naive_lag,
    storm_day_scores,
)
from src.utils.paths import resolve_repo_path


def validation_frame(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return validation targets, dates, and best-checkpoint predictions."""
    data_cfg = cfg["data"]
    seq_len = int(data_cfg.get("seq_len", 64))
    val_split = float(data_cfg.get("val_split", 0.0))
    path = resolve_repo_path(data_cfg["path"])
    frame = pd.read_csv(path)
    target_column = str(data_cfg.get("target_column", "value"))
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column {target_column!r} in {path}")
    values = frame[target_column].to_numpy(dtype=np.float64)
    dates = None
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[D]")
    rows = temporal_validation_target_rows(len(values), seq_len, val_split)
    inputs, targets = load_csv_timeseries(
        path,
        feature_columns=list(data_cfg.get("feature_columns", ["value"])),
        target_column=target_column,
        seq_len=seq_len,
    )
    n_val = len(rows)
    val_inputs = inputs[-n_val:]
    val_targets = targets[-n_val:].reshape(-1).numpy()
    if not np.allclose(val_targets, values[rows], rtol=1e-4, atol=1e-3):
        raise ValueError("Validation labels do not match the target column.")

    checkpoint = resolve_repo_path(cfg["inference"]["checkpoint_path"])
    model = build_model(cfg)
    predictor = Predictor.from_checkpoint(model, checkpoint, device="cpu")
    predictions = predictor.predict(val_inputs).reshape(-1).detach().cpu().numpy()
    return {
        "values": values,
        "dates": dates,
        "rows": rows,
        "targets": values[rows],
        "predictions": np.asarray(predictions, dtype=np.float64),
    }


def extra_baseline_scores(cfg: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Lag, drift, and storm scores for the ``space`` block of a config."""
    space_cfg = cfg.get("space") or {}
    values = np.asarray(trace["values"], dtype=np.float64)
    rows = np.asarray(trace["rows"])
    targets = np.asarray(trace["targets"], dtype=np.float64)
    predictions = np.asarray(trace["predictions"], dtype=np.float64)
    lags: dict[str, dict[str, float] | None] = {}
    lag_traces: dict[str, np.ndarray] = {}
    for period in space_cfg.get("lags") or []:
        key = f"lag_{int(period)}"
        try:
            forecast = seasonal_naive_lag(values, rows, int(period))
        except ValueError:
            lags[key] = None
            continue
        scores = regression_scores(forecast, targets)
        scores["pearson_r"] = pearson(forecast, targets)
        lags[key] = scores
        lag_traces[key] = forecast

    drift_scores: dict[str, float] | None = None
    drift_trace: np.ndarray | None = None
    if space_cfg.get("drift"):
        drift_trace = drift_forecast(values, rows)
        drift_scores = regression_scores(drift_trace, targets)
        drift_scores["pearson_r"] = pearson(drift_trace, targets)

    persistence = persistence_forecast(values, rows)
    model_scores = regression_scores(predictions, targets)
    model_scores["pearson_r"] = pearson(predictions, targets)

    threshold = space_cfg.get("storm_threshold")
    storms: dict[str, dict[str, float]] | None = None
    if threshold is not None:
        cutoff = float(threshold)
        storms = {
            "model": storm_day_scores(predictions, targets, cutoff),
            "persistence": storm_day_scores(persistence, targets, cutoff),
        }
        for key, forecast in lag_traces.items():
            storms[key] = storm_day_scores(forecast, targets, cutoff)

    return {
        "model": model_scores,
        "lags": lags,
        "drift": drift_scores,
        "storms": storms,
        "persistence_trace": persistence,
        "lag_traces": lag_traces,
        "drift_trace": drift_trace,
    }
