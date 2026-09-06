"""Dataset loading for time-series training."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_PREFERRED_COLUMNS = ("value", "y", "target", "close", "price", "adj_close")


def make_sine_dataset(
    n_samples: int,
    seq_len: int,
    noise: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic sine-wave dataset for demonstration.

    Each sample is a sequence of length seq_len drawn from sin(t + offset)
    with added Gaussian noise.  The target is the value one step ahead.

    Returns:
        inputs: (n_samples, seq_len, 1)
        targets: (n_samples, 1)
    """
    t = torch.linspace(0, 4 * torch.pi, seq_len + 1)
    offsets = torch.rand(n_samples) * 2 * torch.pi
    series = torch.sin(t.unsqueeze(0) + offsets.unsqueeze(1))
    series += torch.randn_like(series) * noise
    inputs = series[:, :-1].unsqueeze(-1)
    targets = series[:, -1].unsqueeze(-1)
    return inputs, targets


def _first_row_is_header(row: list[str]) -> bool:
    if not row:
        return False
    try:
        float(row[-1].replace(",", "").strip())
    except ValueError:
        return True
    return False


def _column_index(header: list[str], value_column: str | None) -> int:
    names = [cell.strip() for cell in header]
    lowered = [name.lower() for name in names]
    if value_column:
        key = value_column.strip().lower()
        if key in lowered:
            return lowered.index(key)
        raise ValueError(
            f"Column '{value_column}' not found. Available columns: {names}"
        )
    for preferred in _PREFERRED_COLUMNS:
        if preferred in lowered:
            return lowered.index(preferred)
    return len(names) - 1


def load_csv_series(
    path: str | Path,
    value_column: str | None = None,
) -> torch.Tensor:
    """
    Load a 1-D numeric series from a CSV file.

    Accepts a single-column file, or a multi-column file where the value
    column is named (value, y, target, close, price) or selected via
    ``value_column``. Falls back to the last column.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open(newline="") as handle:
        rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    has_header = _first_row_is_header(rows[0]) or value_column is not None
    if has_header:
        col_idx = _column_index(rows[0], value_column)
        data_rows = rows[1:]
    else:
        col_idx = len(rows[0]) - 1
        data_rows = rows

    values: list[float] = []
    for row in data_rows:
        if col_idx >= len(row):
            continue
        cell = row[col_idx].replace(",", "").strip()
        if not cell or cell.startswith("#"):
            continue
        try:
            values.append(float(cell))
        except ValueError:
            continue

    if len(values) < 2:
        raise ValueError(f"Need at least 2 numeric values in {csv_path}")

    logger.info("Loaded %d points from %s", len(values), csv_path)
    return torch.tensor(values, dtype=torch.float32)


def window_series(
    series: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cut a 1-D series into next-step prediction windows.

    Returns:
        inputs: (n_windows, seq_len, 1)
        targets: (n_windows, 1)
    """
    if series.ndim != 1:
        raise ValueError(f"Expected a 1-D series, got shape {tuple(series.shape)}")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if series.numel() <= seq_len:
        raise ValueError(
            f"Need more than {seq_len} points to build windows, got {series.numel()}"
        )

    n_windows = series.numel() - seq_len
    offsets = torch.arange(seq_len).unsqueeze(0) + torch.arange(n_windows).unsqueeze(1)
    inputs = series[offsets].unsqueeze(-1)
    targets = series[seq_len:].unsqueeze(-1)
    return inputs, targets


def temporal_split(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_split: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split windows in time order: earlier windows train, later windows validate.
    """
    if inputs.size(0) != targets.size(0):
        raise ValueError("inputs and targets must have the same length")
    n = inputs.size(0)
    if n < 2:
        raise ValueError("Need at least 2 windows for a train/val split")

    val_split = float(val_split)
    if val_split < 0 or val_split >= 1:
        raise ValueError("val_split must be in [0, 1)")

    n_val = int(round(n * val_split)) if val_split > 0 else 0
    n_val = min(max(n_val, 1 if val_split > 0 else 0), n - 1)
    n_train = n - n_val
    return inputs[:n_train], targets[:n_train], inputs[n_train:], targets[n_train:]


def load_train_val(
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Build train/val tensors from config.

    Uses ``data.csv_path`` when set; otherwise falls back to the sine generator.
    """
    data_cfg = cfg.get("data") or {}
    seq_len = int(data_cfg.get("seq_len", 64))
    val_split = float(data_cfg.get("val_split", 0.2))
    csv_path = data_cfg.get("csv_path")

    if csv_path:
        series = load_csv_series(csv_path, data_cfg.get("value_column"))
        inputs, targets = window_series(series, seq_len)
        source = f"csv:{csv_path}"
    else:
        inputs, targets = make_sine_dataset(
            n_samples=int(data_cfg.get("n_samples", 1024)),
            seq_len=seq_len,
            noise=float(data_cfg.get("noise", 0.05)),
        )
        source = "sine"

    train_x, train_y, val_x, val_y = temporal_split(inputs, targets, val_split)
    logger.info(
        "Dataset source=%s  train=%d  val=%d  seq_len=%d",
        source,
        train_x.size(0),
        val_x.size(0),
        seq_len,
    )
    return train_x, train_y, val_x, val_y, source
