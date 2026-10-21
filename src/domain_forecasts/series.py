"""Train the existing transformer on a regular dated series and score baselines.

The CSV is one row per time step. Windows are chronological. The last
``test_fraction`` of windows is a test set that does not influence fitting,
early stopping, or the baseline coefficients. ``val_fraction`` is the
early-stopping slice immediately before that test block.
"""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.scaling import StandardScaler
from src.domain_forecasts.baselines import (
    doubling_time_years,
    halving_time_years,
    linear_fit,
    mae,
    one_step_persistence,
    r_squared,
    rmse,
    window_split_bounds,
)
from src.domain_forecasts.reporting import write_results
from src.models.registry import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.paths import resolve_repo_path


def load_series_frame(
    path: str,
    date_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(resolve_repo_path(path))
    missing = [
        col for col in [date_column, *feature_columns] if col not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    frame = frame.copy()
    # A numeric step (the QRNG symbol index) is not a calendar date. Spread it
    # one step per day so the shared year-axis code stays defined; doubling
    # times on that axis are not a physical clock.
    if pd.api.types.is_numeric_dtype(frame[date_column]):
        steps = pd.to_numeric(frame[date_column], errors="coerce")
        frame[date_column] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
            steps, unit="D"
        )
    else:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[date_column, *feature_columns])
    frame = frame.sort_values(date_column)
    if frame.empty:
        raise ValueError(f"No usable rows in {path}")
    return frame.reset_index(drop=True)


def apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform in ("none", ""):
        return values
    if transform == "log1p":
        if np.any(values < 0):
            raise ValueError("log1p transform requires non-negative values")
        return cast(np.ndarray, np.log1p(values))
    raise ValueError(f"unknown transform {transform!r}")


def invert_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform in ("none", ""):
        return values
    if transform == "log1p":
        return cast(np.ndarray, np.expm1(values))
    raise ValueError(f"unknown transform {transform!r}")


def make_windows(values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows. ``values`` is ``(T, F)``; each target is the next row."""
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if len(values) <= seq_len:
        raise ValueError("series must be longer than seq_len")
    inputs = np.stack([values[i : i + seq_len] for i in range(len(values) - seq_len)])
    targets = values[seq_len:]
    return inputs.astype(np.float32), targets.astype(np.float32)


def _years(dates: pd.Series) -> np.ndarray:
    origin = dates.iloc[0]
    seconds = (dates - origin).dt.total_seconds().to_numpy(dtype=float)
    return cast(np.ndarray, seconds / (365.25 * 24.0 * 3600.0))


def _baseline_predictions(
    raw: np.ndarray,
    years: np.ndarray,
    seq_len: int,
    train_end: int,
    val_end: int,
    *,
    already_log10: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """One-step persistence and a train-only log-linear forecast, original units.

    ``raw`` is the untransformed series, shape ``(T, F)``. Window ``i`` predicts
    raw index ``seq_len + i``.
    """
    n_windows = len(raw) - seq_len
    target_index = np.arange(seq_len, seq_len + n_windows)
    persistence = one_step_persistence(raw, target_index)[val_end:]
    test_index = target_index[val_end:]
    train_index = target_index[:train_end]
    n_test = n_windows - val_end
    train_mean = raw[train_index].mean(axis=0)
    mean_forecast = np.repeat(train_mean.reshape(1, -1), n_test, axis=0)
    log_linear = np.zeros((n_test, raw.shape[1]), dtype=float)
    fits: list[dict[str, Any]] = []
    for column in range(raw.shape[1]):
        train_y = raw[train_index, column]
        train_t = years[train_index]
        test_t = years[test_index]
        if already_log10:
            intercept, slope = linear_fit(train_t, train_y)
            predicted = intercept + slope * test_t
            fitted = intercept + slope * train_t
            scale = "log10"
            doubling = doubling_time_years(slope, "log10")
            halving = halving_time_years(slope, "log10")
            fit_r2 = r_squared(train_y, fitted)
        else:
            logged = np.log(np.clip(train_y, 1e-6, None))
            intercept, slope = linear_fit(train_t, logged)
            predicted = np.exp(intercept + slope * test_t)
            fitted = intercept + slope * train_t
            scale = "log"
            doubling = doubling_time_years(slope, "log")
            halving = halving_time_years(slope, "log")
            fit_r2 = r_squared(logged, fitted)
        log_linear[:, column] = predicted
        fits.append(
            {
                "intercept": intercept,
                "slope_per_year": slope,
                "value_scale": scale,
                "train_r2": fit_r2,
                "doubling_time_years": doubling,
                "halving_time_years": halving,
            }
        )
    return persistence, log_linear, mean_forecast, fits


def _predict(
    model: nn.Module, inputs: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    model.eval()
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size].to(device)
            pieces.append(model(batch).cpu())
    return torch.cat(pieces, dim=0)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def train_series(config_path: str) -> dict[str, Any]:
    """Fit one series config and return metrics against the baselines."""
    started = time.perf_counter()
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    columns = list(data_cfg["feature_columns"])
    frame = load_series_frame(
        data_cfg["path"], data_cfg.get("date_column", "date"), columns
    )
    raw = frame[columns].to_numpy(dtype=np.float64)
    dates = frame[data_cfg.get("date_column", "date")]
    years = _years(dates)
    transform = str(data_cfg.get("transform", "none"))
    transformed = apply_transform(raw, transform).astype(np.float32)
    seq_len = int(data_cfg["seq_len"])
    inputs_np, targets_np = make_windows(transformed, seq_len)
    train_end, val_end = window_split_bounds(
        len(inputs_np),
        float(data_cfg.get("test_fraction", 0.2)),
        float(data_cfg.get("val_fraction", 0.1)),
    )

    seed = train_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    device = _resolve_device(str(train_cfg.get("device", "cpu")))
    model = build_model(cfg).to(device)
    if int(cfg["model"]["input_dim"]) != len(columns) or int(
        cfg["model"]["output_dim"]
    ) != len(columns):
        raise ValueError(
            f"model input_dim/output_dim must both be {len(columns)} for columns {columns}"
        )

    inputs = torch.tensor(inputs_np)
    targets = torch.tensor(targets_np)
    normalise = bool(data_cfg.get("normalize", True))
    scaler: StandardScaler | None = None
    if normalise:
        scaler = StandardScaler.fit(inputs[:train_end], targets[:train_end])
        inputs = scaler.transform_inputs(inputs)
        targets = scaler.transform_targets(targets)

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(inputs[:train_end], targets[:train_end]),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(train_cfg["learning_rate"])
    )
    criterion = nn.MSELoss()
    patience = int(train_cfg.get("patience", 0) or 0)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0) or 0)
    epochs = int(train_cfg["epochs"])
    best_loss = float("inf")
    stale = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0

    def scaled_mse(start: int, end: int) -> float:
        predicted = _predict(
            model, inputs[start:end], int(train_cfg["batch_size"]), device
        )
        return float(torch.mean((predicted - targets[start:end]) ** 2).item())

    for epoch in range(1, epochs + 1):
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
        select = scaled_mse(train_end, val_end)
        if select < best_loss:
            best_loss = select
            stale = 0
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if patience and stale >= patience:
                break

    model.load_state_dict(best_state)
    checkpoint_dir = resolve_repo_path(
        train_cfg.get("checkpoint_dir", "checkpoints/domain")
    )
    save_checkpoint(
        model,
        optimizer,
        best_epoch,
        best_loss,
        cfg,
        checkpoint_dir / "best_model.pt",
        scaler=scaler.state_dict() if scaler is not None else None,
    )

    scaled_pred = _predict(
        model, inputs[val_end:], int(train_cfg["batch_size"]), device
    )
    if scaler is not None:
        model_trans = scaler.inverse_targets(scaled_pred).numpy()
    else:
        model_trans = scaled_pred.numpy()
    model_raw = invert_transform(model_trans, transform)
    actual_raw = raw[seq_len + val_end :]
    persistence, log_linear, mean_forecast, fits = _baseline_predictions(
        raw,
        years,
        seq_len,
        train_end,
        val_end,
        already_log10=bool(data_cfg.get("already_log10", False)),
    )
    per_column: dict[str, Any] = {}
    for index, name in enumerate(columns):
        per_column[name] = {
            "model_mae": mae(actual_raw[:, index], model_raw[:, index]),
            "model_rmse": rmse(actual_raw[:, index], model_raw[:, index]),
            "persistence_mae": mae(actual_raw[:, index], persistence[:, index]),
            "persistence_rmse": rmse(actual_raw[:, index], persistence[:, index]),
            "log_linear_mae": mae(actual_raw[:, index], log_linear[:, index]),
            "log_linear_rmse": rmse(actual_raw[:, index], log_linear[:, index]),
            "mean_mae": mae(actual_raw[:, index], mean_forecast[:, index]),
            "mean_rmse": rmse(actual_raw[:, index], mean_forecast[:, index]),
            "beats_constant_mean": bool(
                rmse(actual_raw[:, index], model_raw[:, index])
                < 0.95 * rmse(actual_raw[:, index], mean_forecast[:, index])
            ),
            **fits[index],
        }
    target_dates = dates.iloc[seq_len + val_end :].dt.strftime("%Y-%m-%d").tolist()
    payload: dict[str, Any] = {
        "config": str(config_path),
        "seconds": time.perf_counter() - started,
        "n_rows": int(len(frame)),
        "n_train_windows": int(train_end),
        "n_val_windows": int(val_end - train_end),
        "n_test_windows": int(len(inputs_np) - val_end),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "date_start": dates.iloc[0].strftime("%Y-%m-%d"),
        "date_end": dates.iloc[-1].strftime("%Y-%m-%d"),
        "test_start": target_dates[0],
        "test_end": target_dates[-1],
        "columns": per_column,
        "test_dates": target_dates,
        "test_actual": actual_raw.tolist(),
        "test_model": model_raw.tolist(),
        "test_persistence": persistence.tolist(),
        "test_log_linear": log_linear.tolist(),
        "test_mean": mean_forecast.tolist(),
        "param_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
    output = cfg.get("output", {}).get("results")
    if output:
        payload["results_path"] = str(write_results(output, payload))
    return payload
