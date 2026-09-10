"""Dataset loading for time-series training."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
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


def temporal_split_with_test(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_split: float = 0.2,
    test_split: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Peel a final test holdout, then split the remainder into train/val.
    """
    test_split = float(test_split)
    if test_split < 0 or test_split >= 1:
        raise ValueError("test_split must be in [0, 1)")

    n = inputs.size(0)
    if test_split == 0:
        n_test = 0
    else:
        if n < 3:
            raise ValueError("Need at least 3 windows for a train/val/test split")
        n_test = min(max(int(round(n * test_split)), 1), n - 2)

    test_x = inputs[n - n_test :]
    test_y = targets[n - n_test :]
    rest_x = inputs[: n - n_test]
    rest_y = targets[: n - n_test]
    train_x, train_y, val_x, val_y = temporal_split(rest_x, rest_y, val_split)
    return train_x, train_y, val_x, val_y, test_x, test_y


@dataclass
class DatasetSplits:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    source: str


def load_splits(cfg: dict) -> DatasetSplits:
    """
    Build train/val/test tensors from config.

    Uses ``data.csv_path`` when set; otherwise falls back to the sine generator.
    """
    data_cfg = cfg.get("data") or {}
    seq_len = int(data_cfg.get("seq_len", 64))
    val_split = float(data_cfg.get("val_split", 0.2))
    test_split = float(data_cfg.get("test_split", 0.0))
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

    train_x, train_y, val_x, val_y, test_x, test_y = temporal_split_with_test(
        inputs, targets, val_split=val_split, test_split=test_split
    )
    logger.info(
        "Dataset source=%s  train=%d  val=%d  test=%d  seq_len=%d",
        source,
        train_x.size(0),
        val_x.size(0),
        test_x.size(0),
        seq_len,
    )
    return DatasetSplits(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        source=source,
    )


def load_train_val(
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Build train/val tensors from config (test holdout is available via load_splits)."""
    splits = load_splits(cfg)
    return splits.train_x, splits.train_y, splits.val_x, splits.val_y, splits.source


def sequence_from_series(series: torch.Tensor, seq_len: int) -> list[list[float]]:
    """Take the last ``seq_len`` points as a model input sequence."""
    if series.ndim != 1:
        raise ValueError(f"Expected a 1-D series, got shape {tuple(series.shape)}")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if series.numel() < seq_len:
        raise ValueError(f"Need at least {seq_len} points, got {series.numel()}")
    tail = series[-seq_len:]
    return [[float(value)] for value in tail.tolist()]


def fit_scaler(train_x: torch.Tensor) -> dict[str, float]:
    """Fit a mean/std scaler on training windows only."""
    mean = float(train_x.mean().item())
    std = float(train_x.std(unbiased=False).clamp_min(1e-8).item())
    return {"mean": mean, "std": std}


def apply_scaler(
    tensor: torch.Tensor,
    scaler: dict[str, float],
    inverse: bool = False,
) -> torch.Tensor:
    """Scale or inverse-scale a tensor with a fitted mean/std scaler."""
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    if std <= 0:
        raise ValueError("scaler std must be > 0")
    if inverse:
        return tensor * std + mean
    return (tensor - mean) / std


def maybe_standardize(
    cfg: dict,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float] | None]:
    """Standardize train/val if ``data.standardize`` is enabled (default on)."""
    data_cfg = cfg.get("data") or {}
    if not data_cfg.get("standardize", True):
        return train_x, train_y, val_x, val_y, None

    scaler = fit_scaler(train_x)
    train_x = apply_scaler(train_x, scaler)
    train_y = apply_scaler(train_y, scaler)
    val_x = apply_scaler(val_x, scaler)
    val_y = apply_scaler(val_y, scaler)
    logger.info("Standardized features  mean=%.6f  std=%.6f", scaler["mean"], scaler["std"])
    return train_x, train_y, val_x, val_y, scaler
