"""Sliding windows for CSV series, including calendar and missing values.

The default path (no calendar additions and no missing-value indicators) matches
``load_csv_timeseries``. Indicators are decided from rows strictly before the
first validation label, and missing entries are filled with that same row
block's per-column median.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.calendar import calendar_feature_names, calendar_features
from src.data.lunar import lunar_feature_frame, lunar_feature_names
from src.data.splits import fraction_window_indices, split_by_label_dates
from src.data.timeseries import load_csv_timeseries
from src.utils.paths import resolve_repo_path


@dataclass
class PreparedSeries:
    """Windows in original units, plus the chronological indexes that use them."""

    inputs: torch.Tensor
    targets: torch.Tensor
    feature_names: list[str]
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    label_dates: np.ndarray | None
    target_values: np.ndarray
    series_dates: np.ndarray | None


def prepare_csv_series(
    data_cfg: dict, path: str | Path | None = None
) -> PreparedSeries:
    """Build windows and the train/validation/test indexes for one CSV config."""
    csv_path = resolve_repo_path(path or data_cfg["path"])
    frame = pd.read_csv(csv_path)
    seq_len = int(data_cfg.get("seq_len", 64))
    feature_columns = list(data_cfg.get("feature_columns", ["value"]))
    target_column = str(data_cfg.get("target_column", "value"))
    date_column = data_cfg.get("date_column") or None
    calendar = list(data_cfg.get("calendar_features") or [])
    future_calendar = list(data_cfg.get("future_calendar") or [])
    lunar = bool(data_cfg.get("lunar_features", False))
    future_lunar = bool(data_cfg.get("future_lunar", False))
    lunar_tz = str(data_cfg.get("lunar_timezone", "UTC"))
    lunar_at = str(data_cfg.get("lunar_at", "12:00"))
    missing_indicators = bool(data_cfg.get("missing_indicators", False))

    dates = _date_array(frame, date_column) if date_column else None
    if (calendar or future_calendar or lunar or future_lunar) and dates is None:
        raise ValueError(
            "calendar_features, future_calendar, lunar_features, and "
            "future_lunar require data.date_column."
        )

    train_index, val_index, test_index = _split_indexes(
        n_rows=len(frame),
        seq_len=seq_len,
        data_cfg=data_cfg,
        dates=dates,
    )
    held_out = val_index if len(val_index) else test_index
    fit_row_end = int(held_out[0]) + seq_len if len(held_out) else len(frame)

    use_extensions = bool(
        calendar or future_calendar or missing_indicators or lunar or future_lunar
    )
    if use_extensions:
        inputs, targets, names = _extended_windows(
            frame,
            feature_columns=feature_columns,
            target_column=target_column,
            seq_len=seq_len,
            dates=dates,
            calendar=calendar,
            future_calendar=future_calendar,
            lunar=lunar,
            future_lunar=future_lunar,
            lunar_tz=lunar_tz,
            lunar_at=lunar_at,
            missing_indicators=missing_indicators,
            fit_row_end=fit_row_end,
        )
    else:
        inputs, targets = load_csv_timeseries(
            csv_path,
            feature_columns=feature_columns,
            target_column=target_column,
            seq_len=seq_len,
        )
        names = list(feature_columns)

    label_dates = None if dates is None else dates[seq_len:]
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column {target_column!r} in {csv_path}")
    target_values = pd.to_numeric(frame[target_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    return PreparedSeries(
        inputs=inputs,
        targets=targets,
        feature_names=names,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        label_dates=label_dates,
        target_values=target_values,
        series_dates=dates,
    )


def _split_indexes(
    *,
    n_rows: int,
    seq_len: int,
    data_cfg: dict,
    dates: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_rows <= seq_len:
        raise ValueError("CSV must contain more rows than seq_len.")
    val_start = data_cfg.get("val_start") or None
    test_start = data_cfg.get("test_start") or None
    n_windows = n_rows - seq_len
    if val_start or test_start:
        if dates is None:
            raise ValueError(
                "data.val_start and data.test_start require data.date_column."
            )
        return split_by_label_dates(
            dates[seq_len:],
            str(val_start) if val_start else None,
            str(test_start) if test_start else None,
        )
    return fraction_window_indices(
        n_windows,
        float(data_cfg.get("val_split", 0.0)),
        float(data_cfg.get("test_split", 0.0) or 0.0),
    )


def _date_array(frame: pd.DataFrame, date_column: str) -> np.ndarray:
    if date_column not in frame.columns:
        raise ValueError(f"Missing date column {date_column!r}.")
    parsed = pd.to_datetime(frame[date_column], format="%Y-%m-%d", errors="coerce")
    if bool(parsed.isna().any()):
        parsed = pd.to_datetime(frame[date_column], errors="raise")
    return np.asarray(parsed.to_numpy(dtype="datetime64[D]"))


def _extended_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    seq_len: int,
    dates: np.ndarray | None,
    calendar: list[str],
    future_calendar: list[str],
    lunar: bool,
    future_lunar: bool,
    lunar_tz: str,
    lunar_at: str,
    missing_indicators: bool,
    fit_row_end: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    missing = [
        column
        for column in feature_columns + [target_column]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    observed = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    targets = pd.to_numeric(frame[target_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(targets).all():
        raise ValueError(f"Target column {target_column!r} contains non-finite values.")

    names = list(feature_columns)
    columns = [
        observed[column].to_numpy(dtype=np.float64) for column in feature_columns
    ]
    if missing_indicators:
        if fit_row_end <= 0:
            raise ValueError("missing_indicators requires at least one fitting row.")
        for column in feature_columns:
            values = observed[column].to_numpy(dtype=np.float64)
            if np.isnan(values[:fit_row_end]).any():
                names.append(f"{column}_missing")
                columns.append(np.isnan(values).astype(np.float64))

    matrix = np.column_stack(columns) if columns else np.zeros((len(frame), 0))
    if calendar:
        generated = calendar_features(pd.Series(dates), calendar)
        overlap = [name for name in generated.columns if name in names]
        if overlap:
            raise ValueError(
                f"Calendar features {overlap} are already in feature_columns."
            )
        names.extend(generated.columns.tolist())
        matrix = np.column_stack([matrix, generated.to_numpy(dtype=np.float64)])

    if lunar:
        if dates is None:
            raise ValueError("lunar_features requires dates.")
        generated = lunar_feature_frame(pd.Series(dates), tz=lunar_tz, at=lunar_at)
        overlap = [name for name in generated.columns if name in names]
        if overlap:
            raise ValueError(
                f"Lunar features {overlap} are already in feature_columns."
            )
        names.extend(generated.columns.tolist())
        matrix = np.column_stack([matrix, generated.to_numpy(dtype=np.float64)])

    n_windows = len(frame) - seq_len
    inputs = np.empty((n_windows, seq_len, matrix.shape[1]), dtype=np.float64)
    for start in range(n_windows):
        inputs[start] = matrix[start : start + seq_len]

    if future_calendar:
        if dates is None:
            raise ValueError("future_calendar requires dates.")
        future_names = calendar_feature_names(future_calendar, future=True)
        overlap = [name for name in future_names if name in names]
        if overlap:
            raise ValueError(f"Future calendar features collide with {overlap}.")
        future = calendar_features(pd.Series(dates), future_calendar).to_numpy(
            dtype=np.float64
        )
        label_future = future[seq_len:]
        tiled = np.repeat(label_future[:, None, :], seq_len, axis=1)
        inputs = np.concatenate([inputs, tiled], axis=-1)
        names.extend(future_names)

    if future_lunar:
        if dates is None:
            raise ValueError("future_lunar requires dates.")
        future_names = lunar_feature_names(future=True)
        overlap = [name for name in future_names if name in names]
        if overlap:
            raise ValueError(f"Future lunar features collide with {overlap}.")
        future = lunar_feature_frame(
            pd.Series(dates), tz=lunar_tz, at=lunar_at
        ).to_numpy(dtype=np.float64)
        label_future = future[seq_len:]
        tiled = np.repeat(label_future[:, None, :], seq_len, axis=1)
        inputs = np.concatenate([inputs, tiled], axis=-1)
        names.extend(future_names)

    medians = _train_medians(matrix, fit_row_end)
    # Medians cover the observed and row-calendar columns. Future columns are
    # finite by construction and are appended after the median vector.
    observed_width = matrix.shape[1]
    filled = inputs.copy()
    if observed_width:
        block = filled[:, :, :observed_width]
        mask = np.isnan(block)
        if mask.any():
            positions = np.where(mask)
            block[positions] = medians[positions[-1]]
        filled[:, :, :observed_width] = block
    if np.isnan(filled).any():
        raise ValueError("Windows still contain NaN after training-split imputation.")

    label_values = targets[seq_len:].astype(np.float32)
    return (
        torch.tensor(filled, dtype=torch.float32),
        torch.tensor(label_values, dtype=torch.float32).unsqueeze(-1),
        names,
    )


def _train_medians(matrix: np.ndarray, fit_row_end: int) -> np.ndarray:
    """Per-column median of rows ``[:fit_row_end]``. Empty columns use 0."""
    block = matrix[:fit_row_end]
    medians = np.zeros(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        finite = block[:, column]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            medians[column] = float(np.median(finite))
    return medians
