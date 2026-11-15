"""CPU-small JEPA vs default transformer accuracy compare on a shipped starter.

The transformer baseline is the **shipped starter checkpoint** (same model the
dashboard forecast uses), evaluated on a temporal holdout in original units.
JEPA is briefly pretrained + fine-tuned on the same windows with z-score
normalization so its MAE is also reported in original units. Memoized per
starter mtime.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.scaling import StandardScaler
from src.data.timeseries import load_csv_timeseries
from src.evaluation.metrics import mae
from src.inference.predictor import Predictor
from src.models.jepa import TimeSeriesJEPA
from src.workspace.errors import WorkspaceError
from src.workspace.starter_forecast import STARTER_UNITS, predictor_for_starter
from src.workspace.starters import starter_by_id

_CACHE: dict[tuple[str, int], dict[str, Any]] = {}

_SEQ_LEN = 28
_MAX_WINDOWS = 200
_BATCH = 32
_PRETRAIN_EPOCHS = 2
_FINETUNE_EPOCHS = 3


def clear_jepa_demo_cache() -> None:
    """Drop cached demos (tests)."""
    _CACHE.clear()


def _fit_supervised(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
) -> None:
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=1e-3
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(_BATCH, len(x_train)),
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            if isinstance(model, TimeSeriesJEPA):
                model.update_ema()


def _fit_jepa(
    model: TimeSeriesJEPA,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
) -> None:
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=1e-3
    )
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(_BATCH, len(x_train)),
        shuffle=True,
    )
    model.train()
    for _ in range(_PRETRAIN_EPOCHS):
        for batch_x, _batch_y in loader:
            optimizer.zero_grad()
            loss = model.jepa_loss(batch_x)
            loss.backward()
            optimizer.step()
            model.update_ema()
    _fit_supervised(model, x_train, y_train, epochs=_FINETUNE_EPOCHS)


@torch.no_grad()
def _starter_mae(
    predictor: Predictor, x_val: torch.Tensor, y_val: torch.Tensor
) -> float:
    """MAE of the shipped starter Predictor in original series units."""
    preds: list[torch.Tensor] = []
    for index in range(len(x_val)):
        preds.append(predictor.predict(x_val[index]))
    stacked = torch.cat(preds, dim=0)
    if stacked.dim() == 1:
        stacked = stacked.unsqueeze(-1)
    return float(mae(stacked.cpu(), y_val.cpu()))


@torch.no_grad()
def _jepa_mae(
    model: TimeSeriesJEPA,
    scaler: StandardScaler,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    """MAE after inverse-scaling JEPA predictions back to original units."""
    model.eval()
    x_scaled = scaler.transform_inputs(x_val)
    pred_scaled = model(x_scaled)
    pred = scaler.inverse_targets(pred_scaled)
    return float(mae(pred.cpu(), y_val.cpu()))


def starter_jepa_compare(starter_id: str) -> dict[str, Any]:
    """Compare a tiny JEPA fine-tune against the shipped starter transformer."""
    starter = starter_by_id(starter_id)
    if starter is None:
        raise WorkspaceError(404, "Starter not found.")
    if not starter.dataset_path.is_file():
        raise WorkspaceError(503, "The sample dataset is not available right now.")
    if not starter.model_path.is_file():
        raise WorkspaceError(503, "The sample model is not available right now.")
    stamp = starter.dataset_path.stat().st_mtime_ns
    cache_key = (starter.id, stamp)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    cfg = starter.config()
    raw_data = cfg.get("data")
    data_cfg: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    feature_columns = list(data_cfg.get("feature_columns") or ["value"])
    target_column = str(data_cfg.get("target_column") or feature_columns[0])
    seq_len = int(data_cfg.get("seq_len") or _SEQ_LEN)
    try:
        inputs, targets = load_csv_timeseries(
            starter.dataset_path,
            feature_columns=feature_columns,
            target_column=target_column,
            seq_len=seq_len,
            horizon=1,
        )
    except (OSError, ValueError) as exc:
        raise WorkspaceError(503, "Could not read the sample series.") from exc
    if len(inputs) < 24:
        raise WorkspaceError(503, "The sample series is too short for this demo.")

    n = min(len(inputs), _MAX_WINDOWS)
    inputs = inputs[-n:]
    targets = targets[-n:]
    n_train = max(12, int(len(inputs) * 0.8))
    x_train, y_train = inputs[:n_train], targets[:n_train]
    x_val, y_val = inputs[n_train:], targets[n_train:]
    if len(x_val) < 4:
        raise WorkspaceError(503, "Not enough validation windows for a compare.")

    predictor = predictor_for_starter(starter, cfg)
    transformer_mae = _starter_mae(predictor, x_val, y_val)

    scaler = StandardScaler.fit(x_train, y_train)
    x_scaled = scaler.transform_inputs(x_train)
    y_scaled = scaler.transform_targets(y_train)

    torch.manual_seed(0)
    input_dim = int(inputs.size(-1))
    jepa = TimeSeriesJEPA(
        input_dim=input_dim,
        output_dim=1,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=max(seq_len, 64),
        horizon=1,
        ema_momentum=0.99,
        mask_ratio=0.25,
        mask_spans=1,
        predictor_hidden=64,
    )
    _fit_jepa(jepa, x_scaled, y_scaled)
    jepa_mae = _jepa_mae(jepa, scaler, x_val, y_val)

    winner = (
        "jepa_forecaster"
        if jepa_mae < transformer_mae
        else "time_series_transformer"
        if transformer_mae < jepa_mae
        else "tie"
    )
    unit = starter.unit or STARTER_UNITS.get(starter.id)
    payload = {
        "starter_id": starter.id,
        "name": starter.name,
        "unit": unit,
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "jepa_pretrain_epochs": _PRETRAIN_EPOCHS,
        "jepa_finetune_epochs": _FINETUNE_EPOCHS,
        "models": [
            {
                "name": "time_series_transformer",
                "label": "Default transformer (shipped)",
                "mae": transformer_mae,
            },
            {
                "name": "jepa_forecaster",
                "label": "JEPA (pretrain + fine-tune)",
                "mae": jepa_mae,
            },
        ],
        "winner": winner,
        "summary": (
            f"On the “{starter.name}” sample holdout (original units), "
            f"shipped transformer MAE={transformer_mae:.4f}, "
            f"JEPA MAE={jepa_mae:.4f}."
        ),
        "note": (
            "Transformer numbers use the shipped starter checkpoint (same as "
            "Forecast). JEPA is a short offline pretrain/fine-tune on normalized "
            "windows; MAE is inverse-scaled back to the series units."
        ),
        "demo": True,
    }
    _CACHE[cache_key] = payload
    return payload
