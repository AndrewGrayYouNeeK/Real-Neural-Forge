"""Held-out return scores against persistence and the training-set mean.

The split matches ``TimeSeriesTrainer`` for CSV data: windows are ordered in
time, ``n_val = int(n_windows * val_split)`` of the most recent windows are the
holdout, and the checkpoint scaler was fit on the training windows only.
Metrics are in log-return units (the model's target after inverting that
scaler). A prediction of zero is the random-walk / price-persistence baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.scaling import StandardScaler
from src.data.timeseries import load_csv_timeseries, resolve_val_count
from src.evaluation.metrics import mae, rmse
from src.inference.predictor import Predictor
from src.models.registry import build_model


def split_sizes(n_windows: int, val_split: float) -> tuple[int, int]:
    """Return ``(n_train, n_val)`` window counts, mirroring the trainer.

    ``load_csv_timeseries`` already emits one window per eligible row, and
    ``resolve_val_count`` holds out ``int(n_windows * val_split)`` of those
    windows from the end. ``n_windows`` is that count, not the CSV row count.
    """
    if n_windows <= 1:
        raise ValueError("not enough windows for a training set and a holdout")
    n_val = resolve_val_count(n_windows, val_split)
    return n_windows - n_val, n_val


def label_date_bounds(
    dates: list[str], seq_len: int, n_train: int, n_val: int
) -> dict[str, str]:
    """ISO dates of the first and last training and holdout labels.

    Window ``i`` predicts the return on row ``seq_len + i``. The holdout is the
    last ``n_val`` windows, so its last label is the last row of the file.
    """
    val_start = seq_len + n_train
    val_end = val_start + n_val - 1
    if val_end != len(dates) - 1:
        raise ValueError("holdout does not end on the last observation")
    return {
        "train_start": dates[seq_len],
        "train_end": dates[val_start - 1],
        "val_start": dates[val_start],
        "val_end": dates[val_end],
    }


def directional_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Share of nonzero outcomes whose sign matches the prediction.

    Exact zeros in the outcome are excluded: a zero move has no direction.
    A prediction of exact zero does not match a nonzero move.
    """
    pred = preds.reshape(-1)
    actual = targets.reshape(-1)
    if pred.numel() != actual.numel():
        raise ValueError(
            f"directional accuracy: {pred.numel()} predictions for {actual.numel()} targets"
        )
    mask = actual != 0
    if int(mask.sum()) == 0:
        raise ValueError("directional accuracy: no nonzero targets")
    hits = torch.sign(pred[mask]) == torch.sign(actual[mask])
    return float(hits.float().mean())


def positive_fraction(targets: torch.Tensor) -> float:
    """Share of nonzero outcomes that are positive. This is the always-long hit rate."""
    actual = targets.reshape(-1)
    mask = actual != 0
    if int(mask.sum()) == 0:
        raise ValueError("positive fraction: no nonzero targets")
    return float((actual[mask] > 0).float().mean())


def _scores(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return {"rmse": rmse(preds, targets), "mae": mae(preds, targets)}


def score_checkpoint(
    feature_csv: str | Path,
    checkpoint: str | Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Score the saved checkpoint on the most recent holdout, in return units."""
    data_cfg = cfg["data"]
    seq_len = int(data_cfg["seq_len"])
    val_split = float(data_cfg["val_split"])
    feature_columns = list(data_cfg["feature_columns"])
    target_column = str(data_cfg["target_column"])
    inputs, targets = load_csv_timeseries(
        feature_csv,
        feature_columns=feature_columns,
        target_column=target_column,
        seq_len=seq_len,
    )
    n_windows = int(inputs.shape[0])
    n_train, n_val = split_sizes(n_windows, val_split)
    dates = pd.read_csv(feature_csv)["date"].astype(str).tolist()
    if len(dates) != n_windows + seq_len:
        raise ValueError("feature file length does not match the model windows")
    bounds = label_date_bounds(dates, seq_len, n_train, n_val)
    train_y = targets[:n_train]
    val_x = inputs[n_train:]
    val_y = targets[n_train:]
    if int(val_x.shape[0]) != n_val:
        raise ValueError("holdout size does not match the temporal split")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    scaler_state = ckpt.get("scaler")
    scaler = StandardScaler.from_state_dict(scaler_state) if scaler_state else None
    predictor = Predictor(model, device="cpu", scaler=scaler)
    preds = predictor.predict(val_x).cpu()

    mean_value = float(train_y.mean())
    zero = torch.zeros_like(val_y)
    mean_pred = torch.full_like(val_y, mean_value)
    model_scores = _scores(preds, val_y)
    zero_scores = _scores(zero, val_y)
    mean_scores = _scores(mean_pred, val_y)
    scaler_mean = None
    if scaler is not None:
        scaler_mean = float(scaler.y_mean.reshape(-1)[0])

    return {
        "n_train": n_train,
        "n_val": n_val,
        **bounds,
        "model_rmse": model_scores["rmse"],
        "model_mae": model_scores["mae"],
        "directional_accuracy": directional_accuracy(preds, val_y),
        "persistence_rmse": zero_scores["rmse"],
        "persistence_mae": zero_scores["mae"],
        "mean_return": mean_value,
        "mean_rmse": mean_scores["rmse"],
        "mean_mae": mean_scores["mae"],
        "mean_directional_accuracy": directional_accuracy(mean_pred, val_y),
        "positive_fraction": positive_fraction(val_y),
        "scaler_y_mean": scaler_mean,
        "best_epoch": int(ckpt.get("epoch", 0)),
        "rmse_ratio_vs_persistence": model_scores["rmse"] / zero_scores["rmse"],
        "rmse_ratio_vs_mean": model_scores["rmse"] / mean_scores["rmse"],
        "beats_persistence_rmse": model_scores["rmse"] < zero_scores["rmse"],
        "beats_mean_rmse": model_scores["rmse"] < mean_scores["rmse"],
    }
