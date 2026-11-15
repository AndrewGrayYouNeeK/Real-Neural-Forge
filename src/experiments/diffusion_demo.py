"""CPU-small diffusion futures demo for a shipped starter series.

Trains a tiny conditional diffusion model for a few steps on the starter CSV
(in z-score space), draws sample paths, inverse-scales to original units, and
clips to a plausible band around the historical range. Memoized per starter
mtime so repeated opens stay cheap.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.scaling import StandardScaler
from src.data.timeseries import load_csv_timeseries
from src.models.diffusion import DiffusionForecaster
from src.workspace.errors import WorkspaceError
from src.workspace.starter_forecast import STARTER_UNITS, predictor_for_starter
from src.workspace.starters import starter_by_id

_CACHE: dict[tuple[str, int, int, int], dict[str, Any]] = {}

# Keep the request path light: short windows, few denoise steps.
_SEQ_LEN = 28
_HORIZON = 7
_EPOCHS = 2
_BATCH = 32
_MAX_WINDOWS = 200
_N_PATHS = 24
_DIFFUSION_STEPS = 12
_SAMPLE_STEPS = 6
_HISTORY_POINTS = 48
# Band must stay within this multiple of the historical range beyond min/max.
_RANGE_PAD = 0.35


def clear_diffusion_demo_cache() -> None:
    """Drop cached demos (tests)."""
    _CACHE.clear()


def _plausible_bounds(series: torch.Tensor) -> tuple[float, float, float, float]:
    """Return (data_min, data_max, clip_low, clip_high) from 1-D values."""
    flat = series.detach().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        raise WorkspaceError(503, "The sample series has no finite values.")
    data_min = float(finite.min().item())
    data_max = float(finite.max().item())
    span = max(data_max - data_min, 1e-3)
    pad = _RANGE_PAD * span
    return data_min, data_max, data_min - pad, data_max + pad


def starter_diffusion_futures(
    starter_id: str,
    *,
    horizon: int = _HORIZON,
    n_paths: int = _N_PATHS,
) -> dict[str, Any]:
    """Return history + p10/p50/p90 (and a few sample paths) for a starter."""
    if horizon < 1 or horizon > 14:
        raise WorkspaceError(422, "horizon must be between 1 and 14.")
    if n_paths < 4 or n_paths > 32:
        raise WorkspaceError(422, "n_paths must be between 4 and 32.")
    starter = starter_by_id(starter_id)
    if starter is None:
        raise WorkspaceError(404, "Starter not found.")
    if not starter.dataset_path.is_file():
        raise WorkspaceError(503, "The sample dataset is not available right now.")
    stamp = starter.dataset_path.stat().st_mtime_ns
    cache_key = (starter.id, stamp, horizon, n_paths)
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
            horizon=horizon,
        )
    except (OSError, ValueError) as exc:
        raise WorkspaceError(503, "Could not read the sample series.") from exc
    if len(inputs) < 16:
        raise WorkspaceError(503, "The sample series is too short for this demo.")

    # Temporal holdout: train on earlier windows, forecast from the latest.
    n = min(len(inputs), _MAX_WINDOWS)
    inputs = inputs[-n:]
    targets = targets[-n:]
    n_train = max(8, int(len(inputs) * 0.8))
    x_train = inputs[:n_train]
    y_train = targets[:n_train]
    if y_train.dim() == 3:
        y_fit = y_train.reshape(y_train.size(0), -1)
    else:
        y_fit = y_train

    data_min, data_max, clip_low, clip_high = _plausible_bounds(
        torch.cat([x_train[..., 0].reshape(-1), y_fit.reshape(-1)], dim=0)
    )

    # Train + sample in z-score space so ε-prediction stays well-scaled.
    scaler = StandardScaler.fit(x_train, y_fit)
    x_scaled = scaler.transform_inputs(x_train)
    y_scaled = scaler.transform_targets(y_fit)

    torch.manual_seed(0)
    model = DiffusionForecaster(
        input_dim=int(inputs.size(-1)),
        output_dim=1,
        hidden_size=24,
        horizon=horizon,
        diffusion_steps=_DIFFUSION_STEPS,
        sample_steps=_SAMPLE_STEPS,
        n_paths=n_paths,
        quantiles=(0.1, 0.5, 0.9),
        max_seq_len=max(seq_len, 64),
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(x_scaled, y_scaled),
        batch_size=min(_BATCH, len(x_scaled)),
        shuffle=True,
    )
    model.train()
    for _ in range(_EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = model.diffusion_loss(batch_x, batch_y)
            loss.backward()
            optimizer.step()

    context = inputs[-1:].contiguous()
    context_scaled = scaler.transform_inputs(context)
    # Anchor the band on the shipped starter forecast when available so a
    # barely-trained denoiser cannot wander to the clip edges.
    last_value = float(context[0, -1, 0].item())
    center = torch.full((horizon,), last_value, dtype=torch.float32)
    try:
        predictor = predictor_for_starter(starter, cfg)
        batch = predictor.forecast(context[0], horizon=horizon)
        center = batch.point[0, :, 0].detach().cpu().float()
    except (WorkspaceError, OSError, RuntimeError, ValueError, TypeError, KeyError):
        center = torch.full((horizon,), last_value, dtype=torch.float32)

    history_series = context[0, :, 0].detach().cpu().float()
    if history_series.numel() >= 2:
        step_vol = float(history_series.diff().std(unbiased=False).item())
    else:
        step_vol = 1.0
    step_vol = max(step_vol, 0.5)

    model.eval()
    with torch.no_grad():
        paths = model.sample_paths(context_scaled, n_paths=n_paths, steps=_SAMPLE_STEPS)
        # paths: (1, P, H, 1) in z-space → original units.
        flat = scaler.inverse_targets(paths[0, :, :, 0])
        # Residuals around the path median, shrunk to recent day-to-day vol,
        # then added to the champion / persistence center.
        median_path = flat.median(dim=0).values
        residuals = flat - median_path.unsqueeze(0)
        typical = float(residuals.abs().mean().clamp(min=1e-3).item())
        shrink = min(1.0, (1.75 * step_vol) / typical)
        flat = center.unsqueeze(0) + residuals * shrink
        flat = flat.clamp(min=clip_low, max=clip_high)
        p10 = torch.quantile(flat, 0.1, dim=0)
        p50 = torch.quantile(flat, 0.5, dim=0)
        p90 = torch.quantile(flat, 0.9, dim=0)

    history_vals = context[0, :, 0].detach().cpu().tolist()
    if len(history_vals) > _HISTORY_POINTS:
        history_vals = history_vals[-_HISTORY_POINTS:]
    history = [
        {"index": index, "actual": float(value)}
        for index, value in enumerate(history_vals)
    ]
    steps = [
        {
            "step": offset + 1,
            "p10": float(p10[offset].item()),
            "p50": float(p50[offset].item()),
            "p90": float(p90[offset].item()),
        }
        for offset in range(horizon)
    ]
    show_paths = min(5, int(flat.size(0)))
    sample_paths = [
        [float(value) for value in flat[path_index].tolist()]
        for path_index in range(show_paths)
    ]
    unit = starter.unit or STARTER_UNITS.get(starter.id)
    payload = {
        "starter_id": starter.id,
        "name": starter.name,
        "unit": unit,
        "horizon": horizon,
        "n_paths": n_paths,
        "history": history,
        "forecast": steps,
        "sample_paths": sample_paths,
        "data_min": data_min,
        "data_max": data_max,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "summary": (
            f"Diffusion drew {n_paths} futures for the next {horizon} steps on the "
            f"“{starter.name}” sample (tiny CPU demo — not the default model)."
        ),
        "note": (
            "Signed-out visitors use the shipped starter series. Futures are "
            "sampled in normalized space, inverse-scaled, and clipped to a "
            "plausible band around the historical range."
        ),
        "demo": True,
    }
    _CACHE[cache_key] = payload
    return payload
