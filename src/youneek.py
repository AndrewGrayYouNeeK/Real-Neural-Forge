"""YouNeeK Time, calendar, and lunar clocks.

YouNeeK Time is a 100-unit day (not 10/10/10): 100 units, 100 minutes,
100 seconds. Midnight is 00:00:00 and noon is 50:00:00.

The YouNeeK year is 354 days (11 shorter than a common Gregorian year)
with 10-day weeks. The epoch is 2026-01-01 00:00:00 UTC. Mid-year is
50:00:00 on the year clock.

The lunar clock maps one mean synodic month onto the same 100:100:100
scale: new moon is 00:00:00 and full moon is 50:00:00.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import floor
from typing import Any

UNITS_PER_CYCLE = 100
MINUTES_PER_UNIT = 100
SECONDS_PER_MINUTE = 100
TICKS_PER_CYCLE = UNITS_PER_CYCLE * MINUTES_PER_UNIT * SECONDS_PER_MINUTE  # 1_000_000

GREGORIAN_SECONDS_PER_DAY = 86_400
YOUNEEK_MINUTE_EARTH_SECONDS = GREGORIAN_SECONDS_PER_DAY / (
    UNITS_PER_CYCLE * MINUTES_PER_UNIT
)

DAYS_PER_YEAR = 354
DAYS_PER_WEEK = 10
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Mean synodic month (days) and a conventional mean new-moon epoch (Meeus).
SYNODIC_MONTH_DAYS = 29.530588853
MEAN_NEW_MOON_EPOCH = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def ticks_from_fraction(fraction: float) -> int:
    """Map a unit interval [0, 1) onto 1_000_000 YouNeeK ticks."""
    wrapped = fraction % 1.0
    tick = int(floor(wrapped * TICKS_PER_CYCLE))
    if tick >= TICKS_PER_CYCLE:
        return 0
    return tick


def format_clock(ticks: int) -> str:
    ticks = ticks % TICKS_PER_CYCLE
    units = ticks // (MINUTES_PER_UNIT * SECONDS_PER_MINUTE)
    remainder = ticks % (MINUTES_PER_UNIT * SECONDS_PER_MINUTE)
    minutes = remainder // SECONDS_PER_MINUTE
    seconds = remainder % SECONDS_PER_MINUTE
    return f"{units:02d}:{minutes:02d}:{seconds:02d}"


def clock_parts(ticks: int) -> dict[str, int | str]:
    ticks = ticks % TICKS_PER_CYCLE
    units = ticks // (MINUTES_PER_UNIT * SECONDS_PER_MINUTE)
    remainder = ticks % (MINUTES_PER_UNIT * SECONDS_PER_MINUTE)
    minutes = remainder // SECONDS_PER_MINUTE
    seconds = remainder % SECONDS_PER_MINUTE
    return {
        "units": units,
        "minutes": minutes,
        "seconds": seconds,
        "display": f"{units:02d}:{minutes:02d}:{seconds:02d}",
        "ticks": ticks,
    }


def time_of_day_ticks(moment: datetime) -> int:
    utc = _ensure_utc(moment)
    midnight = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (utc - midnight).total_seconds()
    return ticks_from_fraction(elapsed / GREGORIAN_SECONDS_PER_DAY)


def calendar_state(moment: datetime) -> dict[str, Any]:
    utc = _ensure_utc(moment)
    elapsed_days = (utc - EPOCH).total_seconds() / GREGORIAN_SECONDS_PER_DAY
    # Year index can be negative before the epoch.
    year_index = int(floor(elapsed_days / DAYS_PER_YEAR))
    day_in_year_float = elapsed_days - year_index * DAYS_PER_YEAR
    day_in_year = int(floor(day_in_year_float))
    week = day_in_year // DAYS_PER_WEEK
    weekday = day_in_year % DAYS_PER_WEEK
    year_ticks = ticks_from_fraction(day_in_year_float / DAYS_PER_YEAR)
    return {
        "epoch": EPOCH.isoformat().replace("+00:00", "Z"),
        "year_index": year_index,
        "day_in_year": day_in_year,
        "days_per_year": DAYS_PER_YEAR,
        "week": week,
        "weekday": weekday,
        "days_per_week": DAYS_PER_WEEK,
        "year_clock": clock_parts(year_ticks),
    }


def lunar_state(moment: datetime) -> dict[str, Any]:
    utc = _ensure_utc(moment)
    elapsed_days = (utc - MEAN_NEW_MOON_EPOCH).total_seconds() / GREGORIAN_SECONDS_PER_DAY
    cycle_fraction = (elapsed_days / SYNODIC_MONTH_DAYS) % 1.0
    ticks = ticks_from_fraction(cycle_fraction)
    # Illumination peaks at full moon (0.5).
    illumination = 0.5 - 0.5 * abs(2.0 * cycle_fraction - 1.0)
    if cycle_fraction < 0.5:
        phase_name = "waxing"
    elif cycle_fraction == 0.5:
        phase_name = "full"
    else:
        phase_name = "waning"
    if cycle_fraction < 1e-12 or cycle_fraction > 1 - 1e-12:
        phase_name = "new"
    return {
        "synodic_month_days": SYNODIC_MONTH_DAYS,
        "cycle_fraction": cycle_fraction,
        "illumination": illumination,
        "phase": phase_name,
        "clock": clock_parts(ticks),
        "new_moon": "00:00:00",
        "full_moon": "50:00:00",
    }


def next_youneek_minute(moment: datetime) -> dict[str, Any]:
    utc = _ensure_utc(moment)
    midnight = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (utc - midnight).total_seconds()
    minutes_elapsed = elapsed / YOUNEEK_MINUTE_EARTH_SECONDS
    next_index = int(floor(minutes_elapsed)) + 1
    if next_index >= UNITS_PER_CYCLE * MINUTES_PER_UNIT:
        boundary = midnight + timedelta(days=1)
        next_index = 0
    else:
        boundary = midnight + timedelta(seconds=next_index * YOUNEEK_MINUTE_EARTH_SECONDS)
    units = next_index // MINUTES_PER_UNIT
    minutes = next_index % MINUTES_PER_UNIT
    return {
        "at": boundary.isoformat().replace("+00:00", "Z"),
        "youneek_display": f"{units:02d}:{minutes:02d}:00",
        "earth_seconds_until": (boundary - utc).total_seconds(),
        "youneek_minute_earth_seconds": YOUNEEK_MINUTE_EARTH_SECONDS,
    }


@dataclass(frozen=True)
class YouNeeKSnapshot:
    utc: datetime
    time_of_day: dict[str, int | str]
    calendar: dict[str, Any]
    lunar: dict[str, Any]
    next_minute: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": {
                "units_per_day": UNITS_PER_CYCLE,
                "minutes_per_unit": MINUTES_PER_UNIT,
                "seconds_per_minute": SECONDS_PER_MINUTE,
                "noon": "50:00:00",
                "note": "YouNeeK Time is 100/100/100, not App Store 10/10/10.",
            },
            "utc": self.utc.isoformat().replace("+00:00", "Z"),
            "time": self.time_of_day,
            "calendar": self.calendar,
            "lunar": self.lunar,
            "forecast": {"next_minute": self.next_minute},
        }


def snapshot(moment: datetime) -> YouNeeKSnapshot:
    utc = _ensure_utc(moment)
    return YouNeeKSnapshot(
        utc=utc,
        time_of_day=clock_parts(time_of_day_ticks(utc)),
        calendar=calendar_state(utc),
        lunar=lunar_state(utc),
        next_minute=next_youneek_minute(utc),
    )


def parse_moment(value: str | None) -> datetime:
    if value is None or value == "":
        return datetime.now(timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return _ensure_utc(parsed)
