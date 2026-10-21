"""Parsers for the public files the fetch scripts download.

Each function accepts text or already-decoded records and returns plain rows.
Nothing here touches the network.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta

GED_REGIONS = ("Africa", "Americas", "Asia", "Europe", "Middle East")
POLLEN_VARIABLES = (
    "alder_pollen",
    "birch_pollen",
    "grass_pollen",
    "mugwort_pollen",
    "olive_pollen",
    "ragweed_pollen",
)


def parse_fred_csv(text: str) -> list[tuple[date, float]]:
    """Parse a fredgraph CSV (``observation_date`` plus one value column).

    Missing FRED values encoded as ``.`` are dropped.
    """
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff").strip()))
    if not reader.fieldnames:
        raise ValueError("FRED CSV has no header")
    fields = [name.strip() for name in reader.fieldnames if name and name.strip()]
    date_key = "observation_date" if "observation_date" in fields else fields[0]
    value_keys = [name for name in fields if name != date_key]
    if not value_keys:
        raise ValueError("FRED CSV has no value column")
    value_key = value_keys[0]
    rows: list[tuple[date, float]] = []
    for raw in reader:
        stamp = (raw.get(date_key) or "").strip()
        value = (raw.get(value_key) or "").strip()
        if not stamp or value in {"", "."}:
            continue
        rows.append((date.fromisoformat(stamp), float(value)))
    rows.sort(key=lambda item: item[0])
    return rows


def year_over_year_percent(
    observations: list[tuple[date, float]],
) -> list[tuple[date, float]]:
    """12-month percent change, paired on calendar month."""
    by_month = {(day.year, day.month): value for day, value in observations}
    changes: list[tuple[date, float]] = []
    for day, value in observations:
        previous = by_month.get((day.year - 1, day.month))
        if previous is None or previous == 0:
            continue
        changes.append((day, 100.0 * (value / previous - 1.0)))
    return changes


def parse_silso_monthly(text: str) -> list[tuple[date, float]]:
    """Parse SILSO ``SN_m_tot_V2.0`` semicolon text.

    Columns are year, month, decimal date, monthly mean, then standard
    deviation and observation counts. A negative monthly mean is missing.
    """
    rows: list[tuple[date, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "Year")):
            continue
        parts = [part.strip() for part in stripped.split(";")]
        if len(parts) < 4 or not parts[0][:1].isdigit():
            continue
        year, month = int(parts[0]), int(parts[1])
        sunspots = float(parts[3])
        if sunspots < 0 or not 1 <= month <= 12:
            continue
        rows.append((date(year, month, 1), sunspots))
    rows.sort(key=lambda item: item[0])
    return rows


def parse_match_date(raw: str) -> date:
    """Parse football-data dates (``dd/mm/yy`` or ``dd/mm/yyyy``)."""
    text = raw.strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised match date: {raw!r}")


def parse_team_goal_difference(text: str, team: str) -> list[dict[str, object]]:
    """One row per match for ``team``: goal difference from that team's side.

    ``value`` is goals for minus goals against. League points (3/1/0) are not
    used. Seasons without a result for the team are simply absent.
    """
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames or "HomeTeam" not in reader.fieldnames:
        raise ValueError("match CSV is missing HomeTeam")
    rows: list[dict[str, object]] = []
    for raw in reader:
        home = (raw.get("HomeTeam") or "").strip()
        away = (raw.get("AwayTeam") or "").strip()
        if team not in {home, away}:
            continue
        scored = (raw.get("FTHG") or "").strip()
        conceded = (raw.get("FTAG") or "").strip()
        if scored == "" or conceded == "":
            continue
        home_goals, away_goals = int(scored), int(conceded)
        if home == team:
            goals_for, goals_against, opponent, at_home = (
                home_goals,
                away_goals,
                away,
                1,
            )
        else:
            goals_for, goals_against, opponent, at_home = (
                away_goals,
                home_goals,
                home,
                0,
            )
        rows.append(
            {
                "date": parse_match_date(raw["Date"]).isoformat(),
                "value": goals_for - goals_against,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "opponent": opponent,
                "home": at_home,
            }
        )
    rows.sort(key=lambda item: str(item["date"]))
    return rows


def eia_hour_stamp(raw: str) -> str:
    """Convert an EIA bulk hour (``20190101T00``) to an ISO-8601 UTC stamp."""
    text = raw.strip().rstrip("Z")
    if len(text) < 11 or text[8] != "T":
        raise ValueError(f"unrecognised EIA timestamp: {raw!r}")
    return f"{text[0:4]}-{text[4:6]}-{text[6:8]}T{text[9:11]}:00:00Z"


def parse_eia_series_line(
    line: bytes, series_id: str = "EBA.US48-ALL.D.H"
) -> list[tuple[str, float]] | None:
    """Parse one EIA bulk JSON line when it is ``series_id``.

    Returns ``None`` for every other line so a caller can stream the file.
    Observations are sorted ascending. Null values are dropped.
    """
    # The bulk file has no space after the colon. A spaced encoding still
    # puts the id in the first few dozen bytes, so either form is accepted.
    if series_id.encode() not in line[:80]:
        return None
    payload = json.loads(line)
    if payload.get("series_id") != series_id:
        return None
    rows: list[tuple[str, float]] = []
    for stamp, value in payload.get("data", []):
        if value is None or value == "":
            continue
        rows.append((eia_hour_stamp(str(stamp)), float(value)))
    rows.sort(key=lambda item: item[0])
    return rows


def tail_rows(rows: list[tuple[str, float]], n: int | None) -> list[tuple[str, float]]:
    """Keep the last ``n`` rows. ``None`` or a non-positive ``n`` keeps all."""
    if n is None or n <= 0 or len(rows) <= n:
        return rows
    return rows[-n:]


def parse_usgs_event_csv(text: str) -> list[tuple[datetime, float]]:
    """Parse a USGS FDSN CSV into ``(time, magnitude)`` pairs."""
    reader = csv.DictReader(io.StringIO(text))
    events: list[tuple[datetime, float]] = []
    for raw in reader:
        stamp = (raw.get("time") or "").strip()
        magnitude = (raw.get("mag") or "").strip()
        if not stamp or not magnitude:
            continue
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        events.append((when, float(magnitude)))
    return events


def daily_earthquake_rows(
    events: Iterable[tuple[datetime, float]], start: date, end: date
) -> list[dict[str, object]]:
    """Daily M4.5+ counts and a relative energy sum.

    ``energy_rel`` is ``sum(10 ** (1.5 * magnitude))``, proportional to
    Gutenberg-Richter seismic energy. Days with no events are explicit zeros
    so a seasonal period of 365 steps is 365 calendar days.
    """
    if end < start:
        raise ValueError("earthquake end date is before the start date")
    counts: dict[date, int] = defaultdict(int)
    energy: dict[date, float] = defaultdict(float)
    for when, magnitude in events:
        day = when.date()
        if day < start or day > end:
            continue
        counts[day] += 1
        energy[day] += 10 ** (1.5 * magnitude)
    rows: list[dict[str, object]] = []
    day = start
    while day <= end:
        total = energy.get(day, 0.0)
        rows.append(
            {
                "date": day.isoformat(),
                "value": counts.get(day, 0),
                "energy_rel": total,
                "log10_energy": math.log10(total) if total > 0 else "",
            }
        )
        day += timedelta(days=1)
    return rows


def daily_air_quality(payload: Mapping[str, object]) -> list[dict[str, float | str]]:
    """Aggregate an Open-Meteo hourly payload to daily means.

    Pollen is the sum of the CAMS taxa at each hour, then the daily mean of
    that sum. Hours with a null taxon contribute 0 for that taxon when at
    least one taxon is present. Days with no pollen observations are omitted
    from the pollen column (the key is absent).
    """
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("Open-Meteo payload has no hourly object")
    times = hourly.get("time")
    if not isinstance(times, list):
        raise ValueError("Open-Meteo payload has no hourly times")
    buckets: dict[str, dict[str, list[float]]] = {}
    for index, stamp in enumerate(times):
        day = str(stamp)[:10]
        bucket = buckets.setdefault(day, {"pm2_5": [], "ozone": [], "pollen": []})
        _append_number(bucket["pm2_5"], _at(hourly, "pm2_5", index))
        _append_number(bucket["ozone"], _at(hourly, "ozone", index))
        taxa = [_at(hourly, name, index) for name in POLLEN_VARIABLES if name in hourly]
        if taxa and any(value is not None for value in taxa):
            bucket["pollen"].append(_sum_numbers(taxa))
    rows: list[dict[str, float | str]] = []
    for day in sorted(buckets):
        bucket = buckets[day]
        if not bucket["pm2_5"] or not bucket["ozone"]:
            continue
        row: dict[str, float | str] = {
            "date": day,
            "pm2_5": sum(bucket["pm2_5"]) / len(bucket["pm2_5"]),
            "ozone": sum(bucket["ozone"]) / len(bucket["ozone"]),
        }
        if bucket["pollen"]:
            row["pollen"] = sum(bucket["pollen"]) / len(bucket["pollen"])
        rows.append(row)
    return rows


def aggregate_ged_monthly(
    records: Iterable[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Monthly global event counts and best-estimate fatalities from GED rows.

    Interior months with no events are filled with zeros. All violence types
    in the file are included.
    """
    counts: dict[str, int] = defaultdict(int)
    deaths: dict[str, float] = defaultdict(float)
    region_counts: dict[tuple[str, str], int] = defaultdict(int)
    region_deaths: dict[tuple[str, str], float] = defaultdict(float)
    for record in records:
        stamp = (record.get("date_start") or "").strip()
        if len(stamp) < 7:
            continue
        month = stamp[:7]
        counts[month] += 1
        fatalities = _float_or_zero(record.get("best"))
        deaths[month] += fatalities
        region = (record.get("region") or "").strip()
        if region:
            region_counts[(month, region)] += 1
            region_deaths[(month, region)] += fatalities
    if not counts:
        return []
    start = _month_date(min(counts))
    end = _month_date(max(counts))
    rows: list[dict[str, object]] = []
    cursor = start
    while cursor <= end:
        month = f"{cursor.year:04d}-{cursor.month:02d}"
        row: dict[str, object] = {
            "date": f"{month}-01",
            "events": counts.get(month, 0),
            "fatalities": deaths.get(month, 0.0),
        }
        for region in GED_REGIONS:
            slug = region.lower().replace(" ", "_")
            row[f"events_{slug}"] = region_counts.get((month, region), 0)
            row[f"fatalities_{slug}"] = region_deaths.get((month, region), 0.0)
        rows.append(row)
        year = cursor.year + (1 if cursor.month == 12 else 0)
        month_index = 1 if cursor.month == 12 else cursor.month + 1
        cursor = date(year, month_index, 1)
    return rows


def _append_number(bucket: list[float], value: object) -> None:
    number = _as_float(value)
    if number is not None:
        bucket.append(number)


def _sum_numbers(values: list[object]) -> float:
    total = 0.0
    for value in values:
        number = _as_float(value)
        if number is not None:
            total += number
    return total


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _at(hourly: Mapping[str, object], name: str, index: int) -> object:
    column = hourly.get(name)
    if not isinstance(column, list) or index >= len(column):
        return None
    return column[index]


def _float_or_zero(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if text == "":
        return 0.0
    return float(text)


def _month_date(month: str) -> date:
    year, month_index = month.split("-")
    return date(int(year), int(month_index), 1)
