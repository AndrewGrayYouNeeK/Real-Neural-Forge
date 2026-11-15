"""Generate realistic synthetic time series for demos and Train-on-my-CSV."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.diffusion import DiffusionForecaster

PatternName = Literal["sine", "trend", "seasonal", "noise", "walk", "diffusion"]

_VALID_PATTERNS = frozenset({"sine", "trend", "seasonal", "noise", "walk", "diffusion"})


def _classical(
    pattern: str, length: int, noise: float, rng: np.random.Generator
) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    if pattern == "sine":
        values = np.sin(2.0 * np.pi * t / 24.0)
    elif pattern == "trend":
        values = 0.03 * t
    elif pattern == "seasonal":
        values = (
            np.sin(2.0 * np.pi * t / 7.0)
            + 0.45 * np.sin(2.0 * np.pi * t / 30.0)
            + 0.01 * t
        )
    elif pattern == "noise":
        values = np.zeros(length, dtype=np.float64)
    elif pattern == "walk":
        values = np.cumsum(rng.normal(0.0, 1.0, size=length))
        values = values - float(values[0])
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    if noise > 0:
        values = values + noise * rng.normal(0.0, 1.0, size=length)
    return values.astype(np.float64)


def _diffusion_series(length: int, noise: float, seed: int) -> np.ndarray:
    """Tiny CPU diffusion sample path conditioned on a seasonal context."""
    rng = np.random.default_rng(seed)
    seq_len = 24
    horizon = min(16, max(4, length // 8))
    base = _classical("seasonal", seq_len + horizon + 80, noise, rng)
    windows: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    for start in range(0, len(base) - seq_len - horizon, 3):
        windows.append(base[start : start + seq_len])
        futures.append(base[start + seq_len : start + seq_len + horizon])
        if len(windows) >= 48:
            break
    if len(windows) < 8:
        return _classical("seasonal", length, noise, rng)
    x = torch.tensor(np.stack(windows)[:, :, None], dtype=torch.float32)
    y = torch.tensor(np.stack(futures), dtype=torch.float32)
    torch.manual_seed(seed)
    model = DiffusionForecaster(
        input_dim=1,
        output_dim=1,
        hidden_size=16,
        horizon=horizon,
        diffusion_steps=8,
        sample_steps=3,
        n_paths=4,
        max_seq_len=max(seq_len, 64),
        dropout=0.0,
    )
    model.train()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for history, future in loader:
        optim.zero_grad()
        loss = model.diffusion_loss(history, future)
        loss.backward()
        optim.step()
    model.eval()
    context = torch.tensor(
        _classical("seasonal", seq_len, noise, rng)[None, :, None],
        dtype=torch.float32,
    )
    with torch.no_grad():
        paths = model.sample_paths(context, n_paths=1, steps=3)
    path = paths[0, 0, :, 0].detach().cpu().numpy()
    prefix_len = max(0, length - len(path))
    prefix = (
        _classical("seasonal", prefix_len, noise, rng) if prefix_len else np.asarray([])
    )
    values = np.concatenate([prefix, path])[:length]
    if len(values) < length:
        pad = _classical("seasonal", length - len(values), noise, rng)
        values = np.concatenate([values, pad])
    return np.asarray(values, dtype=np.float64)


def series_to_csv(values: np.ndarray) -> str:
    lines = ["date,value"]
    for index, value in enumerate(values):
        day = index % 28 + 1
        month = (index // 28) % 12 + 1
        year = 2024 + index // 336
        lines.append(f"{year:04d}-{month:02d}-{day:02d},{float(value):.6f}")
    return "\n".join(lines) + "\n"


def generate_synthetic_series(
    *,
    pattern: str = "sine",
    length: int = 256,
    noise: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Return JSON-friendly series plus a CSV body for download / upload."""
    if pattern not in _VALID_PATTERNS:
        raise ValueError(
            "pattern must be one of: sine, trend, seasonal, noise, walk, diffusion."
        )
    if length < 32 or length > 2048:
        raise ValueError("length must be between 32 and 2048.")
    if noise < 0.0 or noise > 1.0:
        raise ValueError("noise must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    if pattern == "diffusion":
        try:
            values = _diffusion_series(length, noise, seed)
            method = "diffusion"
        except (RuntimeError, ValueError, TypeError):
            values = _classical("seasonal", length, noise, rng)
            method = "seasonal_fallback"
    else:
        values = _classical(pattern, length, noise, rng)
        method = pattern
    rows = [
        {"index": int(index), "value": float(value)}
        for index, value in enumerate(values.tolist())
    ]
    csv_body = series_to_csv(values)
    return {
        "pattern": pattern,
        "method": method,
        "length": length,
        "noise": noise,
        "seed": seed,
        "columns": ["date", "value"],
        "series": rows,
        "csv": csv_body,
        "summary": (
            f"Generated {length} points with pattern “{pattern}” "
            f"(noise={noise:g}, seed={seed})."
        ),
    }
