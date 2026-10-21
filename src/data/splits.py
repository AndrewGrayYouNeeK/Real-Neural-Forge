"""Chronological window splits for time-series training.

Windows are ordered in time. A fraction split holds out the latest windows.
Optional label-date boundaries hold out the same calendar days when the
sequence length changes, which keeps model comparisons on one split.
"""

from __future__ import annotations

import numpy as np

from src.data.timeseries import resolve_val_count


def split_counts(
    n_windows: int, val_split: float, test_split: float = 0.0
) -> tuple[int, int, int]:
    """Return ``(n_train, n_val, n_test)`` window counts.

    The test block is the latest ``test_split`` fraction. Validation is the
    latest ``val_split`` fraction of what remains. ``test_split=0`` matches
    ``resolve_val_count`` on the full window list.
    """
    if n_windows <= 0:
        raise ValueError(f"n_windows must be positive, got {n_windows}")
    if not 0.0 <= test_split < 1.0:
        raise ValueError(f"data.test_split must be in [0, 1), got {test_split}")
    n_test = resolve_val_count(n_windows, test_split) if test_split else 0
    n_remaining = n_windows - n_test
    n_val = resolve_val_count(n_remaining, val_split)
    n_train = n_remaining - n_val
    if n_train <= 0:
        raise ValueError("The split leaves no training windows.")
    return n_train, n_val, n_test


def fraction_window_indices(
    n_windows: int, val_split: float, test_split: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train, validation, and test window indexes for a fraction split."""
    n_train, n_val, n_test = split_counts(n_windows, val_split, test_split)
    train = np.arange(0, n_train)
    val = np.arange(n_train, n_train + n_val)
    test = np.arange(n_train + n_val, n_train + n_val + n_test)
    return train, val, test


def split_by_label_dates(
    label_dates: np.ndarray,
    val_start: str | None = None,
    test_start: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split windows by the calendar date of the label.

    Validation labels are on or after ``val_start`` and, when ``test_start``
    is set, strictly before that date. Later labels are the test split. With
    only ``test_start``, every earlier label is training and validation is empty.
    """
    if val_start is None and test_start is None:
        raise ValueError("A date split needs data.val_start or data.test_start.")
    labels = np.asarray(label_dates, dtype="datetime64[D]")
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("label_dates must be a non-empty 1-D array of dates.")
    val_boundary = np.datetime64(val_start, "D") if val_start is not None else None
    test_boundary = np.datetime64(test_start, "D") if test_start is not None else None
    if (
        val_boundary is not None
        and test_boundary is not None
        and val_boundary >= test_boundary
    ):
        raise ValueError(
            f"data.val_start ({val_start}) must be earlier than "
            f"data.test_start ({test_start})."
        )
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for index, day in enumerate(labels):
        if test_boundary is not None and day >= test_boundary:
            test.append(index)
        elif val_boundary is not None and day >= val_boundary:
            val.append(index)
        else:
            train.append(index)
    if not train or (val_boundary is not None and not val):
        raise ValueError(
            "Date split leaves an empty train or validation set. "
            f"val_start={val_start!r} test_start={test_start!r}."
        )
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(val, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )
