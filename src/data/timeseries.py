"""Time-series dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def make_sine_dataset(
    n_samples: int,
    seq_len: int,
    noise: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic sine-wave dataset for demonstration."""
    t = torch.linspace(0, 4 * torch.pi, seq_len + 1)
    offsets = torch.rand(n_samples) * 2 * torch.pi
    series = torch.sin(t.unsqueeze(0) + offsets.unsqueeze(1))
    series += torch.randn_like(series) * noise
    inputs = series[:, :-1].unsqueeze(-1)
    targets = series[:, -1].unsqueeze(-1)
    return inputs, targets


def load_csv_timeseries(
    path: str | Path,
    feature_columns: list[str],
    target_column: str,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load sliding-window sequences from a CSV file."""
    df = pd.read_csv(path)
    missing = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    features = df[feature_columns].to_numpy(dtype=np.float32)
    targets = df[target_column].to_numpy(dtype=np.float32)

    if len(features) <= seq_len:
        raise ValueError("CSV must contain more rows than seq_len.")

    inputs: list[np.ndarray] = []
    labels: list[float] = []
    for start in range(len(features) - seq_len):
        end = start + seq_len
        inputs.append(features[start:end])
        labels.append(float(targets[end]))

    x = torch.tensor(np.stack(inputs), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
    return x, y
