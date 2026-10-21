"""Score a checkpoint on the latest held-out window and forecast one step ahead."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.prepare import load_series
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.retrain.baselines import regression_metrics
from src.utils.paths import resolve_repo_path


class ScoreError(RuntimeError):
    """A checkpoint could not be scored on the requested rows."""


def predict_rows(
    checkpoint: Path,
    inputs: torch.Tensor,
    indices: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return original-unit predictions for ``indices`` using ``checkpoint``."""
    predictor = _predictor(checkpoint, device)
    rows = inputs[indices]
    outputs: list[torch.Tensor] = []
    step = max(1, int(batch_size))
    for start in range(0, int(rows.shape[0]), step):
        outputs.append(predictor.predict(rows[start : start + step]).detach().cpu())
    if not outputs:
        raise ScoreError("No rows to score.")
    return torch.cat(outputs, dim=0)


def score_checkpoint(
    checkpoint: Path,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_idx: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Return validation metrics for ``checkpoint`` in original units."""
    try:
        preds = predict_rows(
            checkpoint,
            inputs,
            val_idx,
            batch_size=batch_size,
            device=device,
        )
        return regression_metrics(preds, targets[val_idx])
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise ScoreError(str(exc)) from exc


def forward_window(cfg: dict[str, Any]) -> tuple[torch.Tensor, str]:
    """Return the context window for a forecast past the loaded series.

    CSV data uses the last ``seq_len`` rows, which is one step beyond the last
    training target. Synthetic data has no rows past the generated samples, so
    the last input window is used and labelled ``last_window``.
    """
    data = cfg.get("data")
    data = data if isinstance(data, dict) else {}
    if data.get("source") == "csv":
        window = _csv_tail(data)
        return window, "beyond_last_observation"
    inputs, _targets = load_series(cfg)
    return inputs[-1].clone(), "last_window"


def forecast_values(
    checkpoint: Path,
    window: torch.Tensor,
    *,
    device: torch.device,
) -> list[float]:
    """Predict the next output vector from a single ``(seq_len, features)`` window."""
    try:
        predictor = _predictor(checkpoint, device)
        output = predictor.predict(window).detach().cpu().reshape(-1)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise ScoreError(str(exc)) from exc
    return [float(value) for value in output.tolist()]


def _predictor(checkpoint: Path, device: torch.device) -> Predictor:
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ScoreError(f"Could not read {checkpoint}: {exc}") from exc
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ScoreError(f"{checkpoint} is not a Neural Forge checkpoint.")
    model_cfg = ckpt.get("config")
    if not isinstance(model_cfg, dict):
        raise ScoreError(f"{checkpoint} is missing the training config.")
    try:
        model = build_model(model_cfg)
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreError(f"Could not build the model in {checkpoint}: {exc}") from exc
    return Predictor.from_checkpoint(model, checkpoint, device=device)


def _csv_tail(data: dict[str, Any]) -> torch.Tensor:
    path = data.get("path")
    columns = data.get("feature_columns", ["value"])
    seq_len = int(data.get("seq_len", 64))
    if not isinstance(path, str) or not path.strip():
        raise ScoreError("data.path is required to forecast a CSV series.")
    if not isinstance(columns, list) or not all(
        isinstance(col, str) for col in columns
    ):
        raise ScoreError("data.feature_columns must be a list of column names.")
    frame = pd.read_csv(resolve_repo_path(path))
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ScoreError(f"Missing columns in CSV: {missing}")
    features = frame[columns].to_numpy(dtype="float32")
    if len(features) < seq_len:
        raise ScoreError("CSV must contain at least seq_len rows to forecast.")
    return torch.tensor(features[-seq_len:], dtype=torch.float32)
