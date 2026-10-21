"""Align a series to Powerball draws without using anything known after the draw.

Powerball draws happen at 22:59 America/New_York. A row is eligible only when
its value was already known at that instant.

* A clock time is eligible when it is at or before 22:59 ET on the draw date.
  The value used is the latest eligible one.
* A date-only row has no clock. ``prior_calendar_day`` keeps the latest row
  dated strictly before the draw. That is the rule for an end-of-day summary
  such as a NOAA daily maximum, which is not a pre-draw publication.
* ``same_day_before_draw`` keeps the latest row dated on or before the draw.
  Use it only for a daily close that is published before 22:59 ET, such as
  the 16:00 ET equity close.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DRAW_HOUR = 22
DRAW_MINUTE = 59
DRAW_TIMEZONE = "America/New_York"
PRIOR_CALENDAR_DAY = "prior_calendar_day"
SAME_DAY_BEFORE_DRAW = "same_day_before_draw"


def draw_cutoff(
    draw_day: date,
    timezone: str = DRAW_TIMEZONE,
    hour: int = DRAW_HOUR,
    minute: int = DRAW_MINUTE,
) -> pd.Timestamp:
    """Return the draw instant as a timezone-aware timestamp."""
    zone = ZoneInfo(timezone)
    local = datetime(
        draw_day.year,
        draw_day.month,
        draw_day.day,
        hour,
        minute,
        tzinfo=zone,
    )
    return pd.Timestamp(local)


def looks_like_calendar_dates(raw: pd.Series) -> bool:
    """True when values are dates, including midnight timestamps of a date.

    A clock time other than 00:00 is a real timestamp and is compared with the
    22:59 ET draw. Midnight is treated as a calendar date so a date serialized
    as ``T00:00:00`` still follows the date-only availability rule.
    """
    clock = raw.astype(str).str.extract(r"(\d{2}):(\d{2})", expand=True)
    if clock.isna().all(axis=None):
        return True
    hours = clock[0]
    minutes = clock[1]
    non_midnight = hours.notna() & ~((hours == "00") & (minutes == "00"))
    return not bool(non_midnight.any())


def prepare_observations(
    observed_at: pd.Series,
    values: np.ndarray,
    *,
    timezone: str = DRAW_TIMEZONE,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sort observations, drop missing values, and collapse duplicate times.

    Returns ``(keys, values, calendar)``. Calendar keys are proleptic Gregorian
    day numbers. Timestamp keys are UTC nanoseconds.
    """
    raw_values = np.asarray(values, dtype=np.float64)
    if raw_values.ndim == 1:
        raw_values = raw_values.reshape(-1, 1)
    if raw_values.shape[0] != len(observed_at):
        raise ValueError("observed_at and values must have the same length")
    calendar = looks_like_calendar_dates(observed_at)
    keys = _day_numbers(observed_at) if calendar else _utc_ns(observed_at, timezone)
    finite = np.isfinite(raw_values).all(axis=1)
    finite = finite & (keys != np.iinfo(np.int64).min)
    keys = keys[finite]
    kept = raw_values[finite]
    if keys.size == 0:
        return keys, kept.reshape(0, raw_values.shape[1]), calendar
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    kept = kept[order]
    # Keep the last row at each timestamp. Walking from the right is enough.
    _, first_from_right = np.unique(keys[::-1], return_index=True)
    chosen = np.sort(keys.size - 1 - first_from_right)
    return keys[chosen], kept[chosen], calendar


def align_observations(
    draw_days: list[date],
    observed_at: pd.Series,
    values: np.ndarray,
    *,
    availability: str,
    timezone: str = DRAW_TIMEZONE,
    hour: int = DRAW_HOUR,
    minute: int = DRAW_MINUTE,
) -> tuple[np.ndarray, str]:
    """Return one value row per draw, with NaN where nothing was known yet.

    The second result is the rule that was actually applied.
    """
    if availability not in (PRIOR_CALENDAR_DAY, SAME_DAY_BEFORE_DRAW):
        raise ValueError(
            "availability must be 'prior_calendar_day' or 'same_day_before_draw', "
            f"got {availability!r}"
        )
    keys, kept, calendar = prepare_observations(observed_at, values, timezone=timezone)
    width = 1 if np.asarray(values).ndim == 1 else int(np.asarray(values).shape[1])
    aligned = np.full((len(draw_days), width), np.nan, dtype=np.float64)
    if keys.size == 0:
        rule = "calendar:" + availability if calendar else "timestamp"
        return aligned, rule
    if calendar:
        query = _draw_day_numbers(draw_days)
        if availability == PRIOR_CALENDAR_DAY:
            positions = np.searchsorted(keys, query, side="left") - 1
        else:
            positions = np.searchsorted(keys, query, side="right") - 1
        rule = "calendar:" + availability
    else:
        query = _cutoff_ns(draw_days, timezone, hour, minute)
        positions = np.searchsorted(keys, query, side="right") - 1
        rule = "timestamp:at_or_before_draw"
    usable = positions >= 0
    aligned[usable] = kept[positions[usable]]
    return aligned, rule


def _day_numbers(raw: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(raw, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(DRAW_TIMEZONE).dt.tz_localize(None)
    days = parsed.dt.normalize().to_numpy(dtype="datetime64[D]")
    numbers = np.full(len(raw), np.iinfo(np.int64).min, dtype=np.int64)
    valid = ~pd.isna(parsed)
    numbers[valid.to_numpy()] = days[valid.to_numpy()].astype(np.int64)
    return numbers


def _draw_day_numbers(draw_days: list[date]) -> np.ndarray:
    as_days = np.array([np.datetime64(day, "D") for day in draw_days])
    return as_days.astype(np.int64)


def _utc_ns(raw: pd.Series, timezone: str) -> np.ndarray:
    parsed = pd.to_datetime(raw, errors="coerce", utc=False)
    numbers = np.full(len(raw), np.iinfo(np.int64).min, dtype=np.int64)
    if len(parsed) == 0:
        return numbers
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(
            ZoneInfo(timezone),
            ambiguous="NaT",
            nonexistent="NaT",
        )
    utc = parsed.dt.tz_convert("UTC")
    nanos = utc.to_numpy(dtype="datetime64[ns]")
    valid = ~pd.isna(parsed)
    numbers[valid.to_numpy()] = nanos[valid.to_numpy()].astype(np.int64)
    return numbers


def _cutoff_ns(
    draw_days: list[date], timezone: str, hour: int, minute: int
) -> np.ndarray:
    cutoffs = [draw_cutoff(day, timezone, hour, minute) for day in draw_days]
    utc = pd.DatetimeIndex(cutoffs).tz_convert("UTC")
    return np.asarray(utc.to_numpy(dtype="datetime64[ns]"), dtype=np.int64)
