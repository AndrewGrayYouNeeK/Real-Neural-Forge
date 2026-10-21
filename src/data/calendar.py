"""Calendar features derived from a date column.

The features are functions of the date alone, so they are known before that
day's observation. ``future_*`` columns use the label date: the day being
forecast, which is known when the horizon is a fixed number of steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mean tropical year, matching the weather CSV's day-of-year encoding.
YEAR_LENGTH_DAYS = 365.25

_COMPONENTS: dict[str, tuple[str, str]] = {
    "doy": ("doy_sin", "doy_cos"),
    "dow": ("dow_sin", "dow_cos"),
    "month": ("month_sin", "month_cos"),
}


def calendar_feature_names(components: list[str], *, future: bool = False) -> list[str]:
    """Column names produced for ``components`` (``doy``, ``dow``, ``month``)."""
    names: list[str] = []
    for component in components:
        if component not in _COMPONENTS:
            known = ", ".join(sorted(_COMPONENTS))
            raise ValueError(
                f"Unknown calendar component {component!r}. Expected one of: {known}."
            )
        left, right = _COMPONENTS[component]
        if future:
            left, right = f"future_{left}", f"future_{right}"
        names.extend((left, right))
    return names


def calendar_features(dates: pd.Series, components: list[str]) -> pd.DataFrame:
    """Sine/cosine columns aligned with ``dates``."""
    parsed = _parse_dates(dates)
    columns: dict[str, np.ndarray] = {}
    if "doy" in components:
        angle = 2.0 * np.pi * parsed.dt.dayofyear.to_numpy(dtype=np.float64)
        angle = angle / YEAR_LENGTH_DAYS
        columns["doy_sin"] = np.sin(angle)
        columns["doy_cos"] = np.cos(angle)
    if "dow" in components:
        angle = 2.0 * np.pi * parsed.dt.dayofweek.to_numpy(dtype=np.float64) / 7.0
        columns["dow_sin"] = np.sin(angle)
        columns["dow_cos"] = np.cos(angle)
    if "month" in components:
        angle = 2.0 * np.pi * (parsed.dt.month.to_numpy(dtype=np.float64) - 1.0) / 12.0
        columns["month_sin"] = np.sin(angle)
        columns["month_cos"] = np.cos(angle)
    unknown = [component for component in components if component not in _COMPONENTS]
    if unknown:
        known = ", ".join(sorted(_COMPONENTS))
        raise ValueError(
            f"Unknown calendar component {unknown[0]!r}. Expected one of: {known}."
        )
    return pd.DataFrame(columns, index=dates.index)


def _parse_dates(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    if bool(parsed.isna().any()):
        parsed = pd.to_datetime(dates, errors="raise")
    return parsed
