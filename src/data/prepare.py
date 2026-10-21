"""Load a configured series and the train/validation indices shared by training.

New dataset sources belong in ``load_series``. Scheduled retraining calls it
too, so a config added under ``config/`` does not need a second loader.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.utils.data import random_split

from src.data.timeseries import load_csv_timeseries, make_sine_dataset
from src.forecasting.spec import forecast_spec_from_config
from src.utils.paths import resolve_repo_path


def load_series(cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(inputs, targets)`` in original units for ``cfg``.

    ``forecast.horizon`` 1 keeps the historical ``(samples, 1)`` targets. A
    longer horizon stacks that many future steps.
    """
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    horizon = forecast_spec_from_config(cfg).horizon
    source = data_cfg.get("source", "synthetic")
    if source == "csv":
        path = data_cfg.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("data.source is csv but data.path is missing.")
        columns = data_cfg.get("feature_columns", ["value"])
        if not isinstance(columns, list) or not all(
            isinstance(col, str) for col in columns
        ):
            raise ValueError("data.feature_columns must be a list of column names.")
        target = data_cfg.get("target_column", "value")
        if not isinstance(target, str) or not target:
            raise ValueError("data.target_column must be a column name.")
        return load_csv_timeseries(
            resolve_repo_path(path),
            feature_columns=columns,
            target_column=target,
            seq_len=int(data_cfg.get("seq_len", 64)),
            horizon=horizon,
        )
    if source != "synthetic":
        raise ValueError(
            f"Unknown data.source {source!r}. Add it to src.data.prepare.load_series, "
            "or set source: csv and a fetch step that writes data.path."
        )
    seed = data_cfg.get("seed")
    return make_sine_dataset(
        n_samples=int(data_cfg.get("n_samples", 1024)),
        seq_len=int(data_cfg.get("seq_len", 64)),
        noise=float(data_cfg.get("noise", 0.05)),
        seed=int(seed) if seed is not None else None,
        horizon=horizon,
    )


def resolve_split_mode(data_cfg: dict[str, Any]) -> str:
    """Return the split mode, including the CSV/synthetic defaults."""
    mode = data_cfg.get("split_mode")
    if mode is None:
        return "temporal" if data_cfg.get("source") == "csv" else "random"
    return str(mode)


def split_indices(
    n: int,
    val_fraction: float,
    split_mode: str,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """Return ``(train_indices, validation_indices)``.

    A zero validation fraction returns every row as training. ``temporal``
    holds out the last fraction of rows, which is the latest window for data
    that was loaded in time order. ``random`` matches ``torch.utils.data.random_split``.
    """
    if split_mode not in ("random", "temporal"):
        raise ValueError(
            f"data.split_mode must be 'random' or 'temporal', got {split_mode!r}"
        )
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"data.val_split must be in [0, 1), got {val_fraction}")
    n_val = int(n * val_fraction)
    if val_fraction > 0 and (n_val == 0 or n_val == n):
        raise ValueError("data.val_split leaves an empty train or validation set.")
    if n_val == 0:
        return list(range(n)), []
    if split_mode == "temporal":
        n_fit = n - n_val
        return list(range(n_fit)), list(range(n_fit, n))
    generator = torch.Generator().manual_seed(int(split_seed))
    # Same permutation as random_split on a sequence of this length. The range
    # is only a sized sequence; random_split does not read its values.
    train_subset, val_subset = random_split(
        cast(Any, range(n)), [n - n_val, n_val], generator=generator
    )
    return list(train_subset.indices), list(val_subset.indices)
