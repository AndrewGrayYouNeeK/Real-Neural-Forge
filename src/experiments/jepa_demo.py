"""CPU-small JEPA vs transformer accuracy compare on a shipped starter.

Brief SSL pretrain + fine-tune for ``jepa_forecaster``, then the same short
supervised fit for a tiny transformer on the same temporal split. Memoized per
starter mtime so the dashboard button stays cheap on repeat clicks.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.timeseries import load_csv_timeseries
from src.evaluation.metrics import mae
from src.models.jepa import TimeSeriesJEPA
from src.models.transformer import TimeSeriesTransformer
from src.workspace.errors import WorkspaceError
from src.workspace.starter_forecast import STARTER_UNITS
from src.workspace.starters import starter_by_id

_CACHE: dict[tuple[str, int], dict[str, Any]] = {}

_SEQ_LEN = 28
_MAX_WINDOWS = 160
_BATCH = 32
_PRETRAIN_EPOCHS = 1
_FINETUNE_EPOCHS = 1
_TRANSFORMER_EPOCHS = 2


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
def _val_mae(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    model.eval()
    pred = model(x_val)
    return float(mae(pred.cpu(), y_val.cpu()))


def starter_jepa_compare(starter_id: str) -> dict[str, Any]:
    """Compare a tiny JEPA fine-tune against a tiny transformer on one starter."""
    starter = starter_by_id(starter_id)
    if starter is None:
        raise WorkspaceError(404, "Starter not found.")
    if not starter.dataset_path.is_file():
        raise WorkspaceError(503, "The sample dataset is not available right now.")
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
    try:
        inputs, targets = load_csv_timeseries(
            starter.dataset_path,
            feature_columns=feature_columns,
            target_column=target_column,
            seq_len=_SEQ_LEN,
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

    torch.manual_seed(0)
    input_dim = int(inputs.size(-1))
    transformer = TimeSeriesTransformer(
        input_dim=input_dim,
        output_dim=1,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_seq_len=max(_SEQ_LEN, 64),
        horizon=1,
    )
    jepa = TimeSeriesJEPA(
        input_dim=input_dim,
        output_dim=1,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_seq_len=max(_SEQ_LEN, 64),
        horizon=1,
        ema_momentum=0.99,
        mask_ratio=0.25,
        mask_spans=1,
        predictor_hidden=32,
    )
    _fit_supervised(transformer, x_train, y_train, epochs=_TRANSFORMER_EPOCHS)
    _fit_jepa(jepa, x_train, y_train)
    transformer_mae = _val_mae(transformer, x_val, y_val)
    jepa_mae = _val_mae(jepa, x_val, y_val)
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
        "transformer_epochs": _TRANSFORMER_EPOCHS,
        "models": [
            {
                "name": "time_series_transformer",
                "label": "Default transformer",
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
            f"On the “{starter.name}” sample holdout, "
            f"transformer MAE={transformer_mae:.4f}, "
            f"JEPA MAE={jepa_mae:.4f} "
            f"(tiny CPU demo — not a production bake-off)."
        ),
        "note": (
            "Signed-out visitors use the shipped starter series. This runs a "
            "short offline compare on the request path; results are cached so "
            "repeat opens stay light within public sample caps."
        ),
        "demo": True,
    }
    _CACHE[cache_key] = payload
    return payload
