"""Train the lottery transformer with optional covariate channels.

Each input step is a past draw. When covariates are present, that step also
carries the covariate values known before the *next* draw, so the last step
of a window contains the information available for the draw being predicted
and does not contain that draw's balls.

The plain model is the same network with those extra channels removed. The
holdout is the temporal split used by the Powerball sanity check.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.scaling import StandardScaler
from src.evaluation.lottery import temporal_window_split
from src.models.registry import build_model

BALLS = 6


def build_windows(
    balls: np.ndarray,
    covariates: np.ndarray | None,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(inputs, targets, target_index)``.

    ``inputs`` has shape ``(n_windows, seq_len, features)``. ``target_index``
    is the draw index predicted by each window.
    """
    numbers = np.asarray(balls, dtype=np.float64)
    if numbers.ndim != 2 or numbers.shape[1] != BALLS:
        raise ValueError("balls must have shape (n, 6)")
    n_draws = int(numbers.shape[0])
    if seq_len < 1 or n_draws <= seq_len:
        raise ValueError("need more draws than seq_len")
    if covariates is None:
        features = numbers
        n_windows = n_draws - seq_len
        inputs = np.stack(
            [features[start : start + seq_len] for start in range(n_windows)]
        )
        target_index = np.arange(seq_len, n_draws)
        return inputs, numbers[target_index], target_index

    extra = np.asarray(covariates, dtype=np.float64)
    if extra.ndim != 2 or extra.shape[0] != n_draws:
        raise ValueError("covariates must have shape (n_draws, n_features)")
    # Row i describes draw i and the covariates known before draw i + 1.
    features = np.concatenate([numbers[:-1], extra[1:]], axis=1)
    n_windows = n_draws - seq_len
    if features.shape[0] < n_windows + seq_len - 1:
        raise ValueError("covariate feature rows do not cover the windows")
    inputs = np.stack([features[start : start + seq_len] for start in range(n_windows)])
    target_index = np.arange(seq_len, n_draws)
    return inputs, numbers[target_index], target_index


def finite_windows(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """True where a window and its target contain only finite numbers."""
    mask = np.isfinite(inputs).all(axis=(1, 2)) & np.isfinite(targets).all(axis=1)
    return np.asarray(mask, dtype=bool)


def train_holdout(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    n_draws: int,
    training: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the early windows and return ``(val_target_index, raw_predictions)``.

    Predictions are in the original number units, not the scaled units used
    for the loss. Windows with a non-finite value are left out of both splits.
    """
    seq_len = int(inputs.shape[1])
    split = temporal_window_split(
        n_draws, seq_len, float(training.get("val_split", 0.2))
    )
    if int(split["n_windows"]) != int(inputs.shape[0]):
        raise ValueError("window count does not match the temporal split")
    usable = finite_windows(inputs, targets)
    n_fit = int(split["n_fit"])
    train_idx = np.flatnonzero(usable[:n_fit])
    val_idx = np.flatnonzero(usable[n_fit:]) + n_fit
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("the temporal split has no finite train or validation windows")

    seed = training.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    x_all = torch.tensor(inputs, dtype=torch.float32)
    y_all = torch.tensor(targets, dtype=torch.float32)
    scaler = StandardScaler.fit(x_all[train_idx], y_all[train_idx])
    x_scaled = scaler.transform_inputs(x_all)
    y_scaled = scaler.transform_targets(y_all)

    device = _device(str(training.get("device", "cpu")))
    model = build_model(_model_config(inputs.shape[-1], training)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 1e-3))
    )
    criterion = nn.MSELoss()
    batch_size = max(1, min(int(training.get("batch_size", 64)), int(train_idx.size)))
    loader = DataLoader(
        TensorDataset(x_scaled[train_idx], y_scaled[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    epochs = int(training.get("epochs", 20))
    patience = int(training.get("patience", 0) or 0)
    max_grad_norm = float(training.get("max_grad_norm", 0) or 0)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    val_x = x_scaled[val_idx].to(device)

    for _epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(
                criterion(model(val_x), y_scaled[val_idx].to(device)).item()
            )
        if val_loss < best_loss:
            best_loss = val_loss
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if patience and stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scaled = model(val_x).detach().cpu()
    restored = scaler.inverse_targets(scaled).numpy()
    target_index = np.arange(seq_len, n_draws)[val_idx]
    return target_index, restored


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _model_config(input_dim: int, training: dict[str, Any]) -> dict[str, Any]:
    d_model = int(training.get("d_model", 64))
    nhead = int(training.get("nhead", 4))
    if d_model % nhead != 0:
        raise ValueError("d_model must be divisible by nhead")
    return {
        "model": {
            "name": "time_series_transformer",
            "input_dim": int(input_dim),
            "output_dim": BALLS,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": int(training.get("num_encoder_layers", 3)),
            "dim_feedforward": int(training.get("dim_feedforward", 256)),
            "dropout": float(training.get("dropout", 0.1)),
            "max_seq_len": int(training.get("max_seq_len", 128)),
        }
    }
