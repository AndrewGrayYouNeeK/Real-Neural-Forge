"""Score a saved checkpoint against persistence, seasonal-naive, and mean.

Metrics are in the CSV's original units. The checkpoint is the one selected
on validation loss, so the model number is slightly optimistic relative to
baselines that were not tuned on that holdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.scaling import StandardScaler
from src.data.timeseries import load_csv_timeseries
from src.models.registry import build_model
from src.realworld.baselines import (
    mean_forecast,
    persistence,
    regression_scores,
    seasonal_naive,
    skill_score,
    target_positions,
)
from src.realworld.spells import (
    exponential_trend_mean,
    fit_exponential_trend,
    lag1_correlation,
)
from src.utils.config import load_config
from src.utils.paths import resolve_repo_path


def compare_checkpoint(config_path: str | Path) -> dict[str, Any]:
    """Load ``best_model.pt`` for ``config_path`` and score the temporal holdout."""
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    realworld = cfg.get("realworld") or {}
    path = resolve_repo_path(data_cfg["path"])
    seq_len = int(data_cfg.get("seq_len", 64))
    val_split = float(data_cfg.get("val_split", 0.2))
    period = int(data_cfg["seasonal_period"])
    target_column = data_cfg.get("target_column", "value")
    feature_columns = list(data_cfg.get("feature_columns", ["value"]))

    frame = pd.read_csv(path)
    values = frame[target_column].to_numpy(dtype=np.float64)
    _n_train, targets = target_positions(len(values), seq_len, val_split)
    actual = values[targets]
    baseline_scores = {
        "persistence": regression_scores(actual, persistence(values, targets)),
        "seasonal_naive": regression_scores(
            actual, seasonal_naive(values, targets, period)
        ),
        "mean": regression_scores(actual, mean_forecast(values, targets)),
    }
    extras = _extra_baselines(realworld, frame, values, targets, actual)
    baseline_scores.update(extras)

    inputs, labels = load_csv_timeseries(
        path,
        feature_columns=feature_columns,
        target_column=target_column,
        seq_len=seq_len,
    )
    label_values = labels.numpy().reshape(-1)
    if not np.allclose(label_values[-len(targets) :], actual):
        raise RuntimeError("validation targets are not the tail of the CSV series")
    val_inputs = inputs[-len(targets) :]
    predicted = _predict(cfg, val_inputs)
    model_scores = regression_scores(actual, predicted)
    best_name, best_mae = min(
        (
            (name, scores["mae"])
            for name, scores in baseline_scores.items()
            if name in {"persistence", "seasonal_naive", "mean"}
        ),
        key=lambda item: item[1],
    )
    return {
        "id": realworld.get("id", Path(config_path).stem),
        "title": realworld.get("title", Path(config_path).stem),
        "units": realworld.get("units", ""),
        "config": str(config_path),
        "n_rows": int(len(values)),
        "seq_len": seq_len,
        "seasonal_period": period,
        "n_val": int(len(targets)),
        "lag1_correlation": lag1_correlation(values),
        "model": model_scores,
        "baselines": baseline_scores,
        "best_simple_baseline": best_name,
        "skill_vs_persistence": skill_score(
            model_scores["mae"], baseline_scores["persistence"]["mae"]
        ),
        "skill_vs_seasonal_naive": skill_score(
            model_scores["mae"], baseline_scores["seasonal_naive"]["mae"]
        ),
        "skill_vs_best_simple_baseline": skill_score(model_scores["mae"], best_mae),
    }


def _predict(cfg: dict[str, Any], inputs: torch.Tensor) -> np.ndarray:
    checkpoint_path = resolve_repo_path(
        cfg["training"].get("checkpoint_dir", "checkpoints")
    )
    checkpoint_path = checkpoint_path / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler_state = checkpoint.get("scaler")
    batch = inputs
    scaler: StandardScaler | None = None
    if scaler_state:
        scaler = StandardScaler.from_state_dict(scaler_state)
        batch = scaler.transform_inputs(inputs)
    batch_size = int(cfg["training"].get("batch_size", 32))
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, batch.shape[0], batch_size):
            pieces.append(model(batch[start : start + batch_size]))
    predicted = torch.cat(pieces)
    if scaler is not None:
        predicted = scaler.inverse_targets(predicted)
    return predicted.numpy().reshape(-1)


def _extra_baselines(
    realworld: dict[str, Any],
    frame: pd.DataFrame,
    values: np.ndarray,
    targets: np.ndarray,
    actual: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Time-varying exponential mean for inter-onset gaps.

    A constant-rate exponential forecast is the training mean, already reported
    as the mean baseline. The annual Poisson forecast needs the onset list,
    which the analysis script scores separately on this same holdout.
    """
    if realworld.get("extra_baselines") != "onset_gaps":
        return {}
    if "year" not in frame.columns:
        raise ValueError("onset-gap baselines need a year column")
    years = frame["year"].to_numpy(dtype=np.float64)
    first = int(targets[0])
    trend = fit_exponential_trend(years[:first], values[:first])
    trend_pred = np.array(
        [exponential_trend_mean(trend, float(years[index])) for index in targets]
    )
    return {"exponential_trend": regression_scores(actual, trend_pred)}
