"""Time-series dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.utils.paths import resolve_repo_path


def resolve_val_count(n_items: int, val_split: float) -> int:
    """Return how many items a validation split holds out.

    ``n_val = int(n_items * val_split)``. A positive split that would leave
    either side empty is rejected so callers can rely on both sides existing.
    """
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"data.val_split must be in [0, 1), got {val_split}")
    n_val = int(n_items * val_split)
    if val_split > 0 and (n_val == 0 or n_val == n_items):
        raise ValueError("data.val_split leaves an empty train or validation set.")
    return n_val


def make_sine_dataset(
    n_samples: int,
    seq_len: int,
    noise: float = 0.05,
    seed: int | None = None,
    horizon: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic sine-wave dataset for demonstration.

    Pass ``seed`` for a reproducible dataset; ``None`` uses the global RNG.
    ``horizon`` 1 returns targets of shape ``(samples, 1)``. A longer horizon
    returns ``(samples, horizon, 1)``, the next ``horizon`` values after the
    input window. The sine spans two periods, so its sample period is
    ``(seq_len + horizon - 1) / 2``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t = torch.linspace(0, 4 * torch.pi, seq_len + horizon)
    offsets = torch.rand(n_samples, generator=generator) * 2 * torch.pi
    series = torch.sin(t.unsqueeze(0) + offsets.unsqueeze(1))
    series += torch.randn(series.shape, generator=generator) * noise
    inputs = series[:, :seq_len].unsqueeze(-1)
    future = series[:, seq_len:]
    if horizon == 1:
        targets = future[:, 0].unsqueeze(-1)
    else:
        targets = future.unsqueeze(-1)
    return inputs, targets


def load_csv_timeseries(
    path: str | Path,
    feature_columns: list[str],
    target_column: str,
    seq_len: int,
    horizon: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load sliding-window sequences from a CSV file.

    ``horizon`` 1 keeps the historical target layout ``(samples, 1)``. A longer
    horizon stacks the next ``horizon`` target values as ``(samples, horizon, 1)``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    df = pd.read_csv(path)
    missing = [
        col for col in feature_columns + [target_column] if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    features = df[feature_columns].to_numpy(dtype=np.float32)
    targets = df[target_column].to_numpy(dtype=np.float32)

    n_windows = len(features) - seq_len - horizon + 1
    if n_windows <= 0:
        raise ValueError("CSV must contain more rows than seq_len + horizon - 1.")

    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for start in range(n_windows):
        end = start + seq_len
        windows.append(features[start:end])
        labels.append(np.array(targets[end : end + horizon], copy=True))

    x = torch.tensor(np.stack(windows), dtype=torch.float32)
    if horizon == 1:
        # Match the historical one-step labels exactly: one Python float per window.
        y = torch.tensor(
            [float(row[0]) for row in labels], dtype=torch.float32
        ).unsqueeze(-1)
    else:
        y = torch.tensor(np.stack(labels), dtype=torch.float32).unsqueeze(-1)
    return x, y


def load_windows_from_config(
    data_cfg: dict[str, Any], horizon: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the configured source as input windows and ``horizon``-step targets."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if data_cfg.get("source") == "csv" and data_cfg.get("path"):
        return load_csv_timeseries(
            resolve_repo_path(data_cfg["path"]),
            feature_columns=list(data_cfg.get("feature_columns", ["value"])),
            target_column=str(data_cfg.get("target_column", "value")),
            seq_len=int(data_cfg.get("seq_len", 64)),
            horizon=horizon,
        )
    seed = data_cfg.get("seed")
    return make_sine_dataset(
        n_samples=int(data_cfg.get("n_samples", 1024)),
        seq_len=int(data_cfg.get("seq_len", 64)),
        noise=float(data_cfg.get("noise", 0.05)),
        seed=int(seed) if seed is not None else None,
        horizon=horizon,
    )
