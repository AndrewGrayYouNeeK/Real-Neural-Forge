"""Download the rest of the solar system and serve the summary report.

Run from the repository root:

    python -m scripts.fetch_solar_system

Raw responses and the processed CSVs live under ``data/space/`` and are not
committed. No API key is used. The weekly retrain calls the same fetch; each
page is cached, so later configs in that job do not download it again.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.space.approaches import (
    AU_PER_LD,
    LOOKAHEAD_DAYS,
    OUTLOOK_MONTHS,
    published_forecasts,
    utc_today,
    window_bounds,
)
from src.space.fetch import (
    SBDB_QUERY_URL,
    _load_json,
    _require_range,
    cached_download,
    write_series,
)
from src.space.near_earth import _collect_cad
from src.space.planets import (
    DISCOVERY_SPECS,
    MOON_SOURCE,
    PLANETS,
    SHOWER_NOTE,
    SOLAR_JPL_KEYS,
    SOLAR_SYSTEM_NOTE,
    TIME_SCALE_NOTE,
    Planet,
    alignment_events,
    approach_month_frame,
    comet_passages,
    monthly_count_frame,
    parse_horizons_table,
    parse_moon_discoveries,
    valid_discovery_dates,
)
from src.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
GEO_CENTER = "500@399"
SUN_CENTER = "500@10"
LIST_CAP = 2000
HISTORY_START = "1900-01-01"

EVENTS_PATH = "data/space/solar_approach_events.csv"
DISCOVERY_PATH = "data/space/solar_discoveries_monthly.csv"
APPROACH_MONTHLY_PATH = "data/space/solar_approaches_monthly.csv"
PERIHELIA_MONTHLY_PATH = "data/space/comet_perihelia_monthly.csv"
COMET_CATALOG_PATH = "data/space/solar_comet_catalog.csv"
HORIZONS_TRAIN_PATH = "data/space/horizons_train.csv"
HORIZONS_MONTHLY_PATH = "data/space/horizons_monthly.csv"
HORIZONS_DAILY_PATH = "data/space/horizons_daily.csv"
MOONS_PATH = "data/space/solar_moons.json"
MANIFEST_PATH = "data/space/solar_system_manifest.json"
FORECAST_PATH = "results/solar_system_forecast.json"

_SUM_BANDS: dict[str, tuple[float, float]] = {
    "mercury": (500, 100_000),
    "venus": (500, 100_000),
    "mars": (500, 100_000),
    "jupiter": (100, 100_000),
    "neo": (10_000, 200_000),
    "main_belt": (100_000, 3_000_000),
    "trojan": (1_000, 100_000),
    "centaur": (100, 20_000),
    "tno": (1_000, 50_000),
    "comet": (2_000, 20_000),
    "perihelia": (100, 500_000),
}
_EARTH_BANDS: dict[str, tuple[float, float]] = {
    "mercury": (0.3, 1.5),
    "venus": (0.2, 1.8),
    "mars": (0.3, 2.8),
    "jupiter": (3.8, 6.6),
    "saturn": (7.5, 11.5),
    "uranus": (17.0, 21.5),
    "neptune": (28.0, 31.5),
    "pluto": (28.0, 51.0),
}
_SUN_BANDS: dict[str, tuple[float, float]] = {
    "mercury": (0.25, 0.55),
    "venus": (0.65, 0.80),
    "mars": (1.30, 1.75),
    "jupiter": (4.8, 5.6),
    "saturn": (8.8, 10.6),
    "uranus": (17.5, 20.8),
    "neptune": (29.0, 31.0),
    "pluto": (29.0, 50.5),
}
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def fetch_solar_system(
    *, refresh: bool = False, output_dir: str | Path = "data/space"
) -> dict[str, object]:
    """Download planetary approaches, discoveries, ephemerides, and moons."""
    root = resolve_repo_path(output_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = utc_today()
    through = today.replace(day=1)
    date_max = (today + pd.Timedelta(days=LOOKAHEAD_DAYS)).strftime("%Y-%m-%d")

    events = _approaches(raw_dir, start=HISTORY_START, end=date_max, refresh=refresh)
    discoveries, dropped, catalog = _discoveries(
        raw_dir, through=through, refresh=refresh
    )
    perihelia = _perihelia_monthly(catalog, discoveries, through=through)
    horizons = _horizons(raw_dir, today=today, date_max=date_max, refresh=refresh)
    train = horizons.loc[pd.to_datetime(horizons["date"]) < through].copy()
    moons = _moons(raw_dir, refresh=refresh)

    _check_sums(events, discoveries, perihelia)
    _check_distances(horizons)
    write_series(discoveries, DISCOVERY_PATH)
    write_series(
        approach_month_frame(
            events,
            through=through,
            known_neos_prev=discoveries.set_index("date")["neo_prev"],
        ),
        APPROACH_MONTHLY_PATH,
    )
    write_series(perihelia, PERIHELIA_MONTHLY_PATH)
    write_series(train, HORIZONS_TRAIN_PATH)
    write_series(horizons, HORIZONS_MONTHLY_PATH)
    _write_table(events, EVENTS_PATH)
    _write_table(catalog, COMET_CATALOG_PATH)
    moon_path = resolve_repo_path(MOONS_PATH)
    moon_path.write_text(json.dumps(moons, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, object] = {
        "fetched_on": today.strftime("%Y-%m-%d"),
        "date_max": date_max,
        "through": through.strftime("%Y-%m-%d"),
        "approaches": {
            planet.name: int((events["body"] == planet.name).sum())
            for planet in PLANETS
        },
        "discoveries": {
            spec["column"]: int(discoveries[str(spec["column"])].sum())
            for spec in DISCOVERY_SPECS
        },
        "discovery_dates_dropped": dropped,
        "perihelia": int(perihelia["perihelia"].sum()),
        "horizons_months": int(len(horizons)),
        "moons": int(sum(row["count"] for row in moons["by_planet"])),
        "lookahead_days": LOOKAHEAD_DAYS,
    }
    manifest = resolve_repo_path(MANIFEST_PATH)
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Solar-system fetch wrote %s", summary)
    return summary


def solar_system_report(
    *, days: int = LOOKAHEAD_DAYS, today: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Notable events in the day window, plus the saved 24-month forecasts."""
    stamp = utc_today() if today is None else pd.Timestamp(today).normalize()
    start, end = window_bounds(stamp, days)
    events = _read_csv(EVENTS_PATH)
    catalog = _read_csv(COMET_CATALOG_PATH)
    daily = _read_csv(HORIZONS_DAILY_PATH)
    horizons = _read_csv(HORIZONS_MONTHLY_PATH)
    moons = _read_json_object(MOONS_PATH)
    forecasts = _enrich_forecasts(published_forecasts(FORECAST_PATH), horizons=horizons)
    approaches, truncated = _approaches_in_window(events, start, end)
    passages, passages_truncated = _perihelia_in_window(catalog, start, end)
    alignments = _alignments_in_window(daily, start, end)
    monthly = _monthly_in_window(approaches, passages, start, end)
    for forecast in forecasts:
        key = SOLAR_JPL_KEYS.get(str(forecast.get("id")))
        forecast["jpl_monthly_key"] = key
    moon_block = _moon_block(moons, year=int(stamp.year))
    available = bool(
        approaches
        or passages
        or alignments
        or forecasts
        or moon_block["by_planet"]
        or not events.empty
    )
    return {
        "available": available,
        "days": int(days),
        "window_start": start.strftime("%Y-%m-%d"),
        "window_end": end.strftime("%Y-%m-%d"),
        "time_scale": TIME_SCALE_NOTE,
        "survey_growth_note": SOLAR_SYSTEM_NOTE,
        "notes": [SOLAR_SYSTEM_NOTE, TIME_SCALE_NOTE, SHOWER_NOTE],
        "truncated": truncated or passages_truncated,
        "approaches": approaches,
        "perihelia": passages,
        "alignments": alignments,
        "monthly": monthly,
        "bodies": _body_rows(events, start, end),
        "moons": moon_block,
        "showers": {"available": False, "note": SHOWER_NOTE},
        "forecasts": forecasts,
        "outlook_months": OUTLOOK_MONTHS,
    }


def _approaches(raw_dir: Path, *, start: str, end: str, refresh: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for planet in PLANETS:
        logger.info("CAD %s within %s AU", planet.name, planet.dist_max_au)
        frame = _planet_cad(
            raw_dir,
            planet,
            start=start,
            end=end,
            refresh=refresh,
        )
        frame["body"] = planet.name
        frame["body_code"] = planet.code
        frame["dist_max_au"] = planet.dist_max_au
        frames.append(frame)
    populated = [frame for frame in frames if not frame.empty]
    if not populated:
        raise ValueError("CAD returned no planetary approaches at any cutoff.")
    events = pd.concat(populated, ignore_index=True)
    events = events.drop_duplicates(["body", "designation", "cd"], keep="first")
    if events.empty:
        raise ValueError("CAD returned no planetary approaches at any cutoff.")
    events["cd"] = pd.to_datetime(events["cd"])
    return events.sort_values(["cd", "body", "dist_au"]).reset_index(drop=True)


def _planet_cad(
    raw_dir: Path,
    planet: Planet,
    *,
    start: str,
    end: str,
    refresh: bool,
) -> pd.DataFrame:
    """One CAD body. A zero-total response omits ``fields`` and ``data``."""
    distance = _distance_token(planet.dist_max_au)
    try:
        return _collect_cad(
            raw_dir,
            body=planet.code,
            start=start,
            end=end,
            dist_max=distance,
            refresh=refresh,
        )
    except ValueError as exc:
        cached = raw_dir / f"cad_{planet.code}_{start}_{end}_{distance}.json"
        if not cached.is_file():
            raise
        payload = json.loads(cached.read_text(encoding="utf-8"))
        total = int(payload.get("total", payload.get("count", 0)) or 0)
        data = payload.get("data")
        if total == 0 and not data:
            return pd.DataFrame(
                columns=["body", "designation", "name", "cd", "dist_au", "h"]
            )
        raise exc


def _discoveries(
    raw_dir: Path, *, through: pd.Timestamp, refresh: bool
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    columns: dict[str, pd.Series] = {}
    dropped: dict[str, int] = {}
    catalog = pd.DataFrame()
    for spec in DISCOVERY_SPECS:
        column = str(spec["column"])
        kind = spec.get("kind")
        if kind:
            catalog = _sbdb_pages(
                raw_dir,
                cache_stem="solar_comets",
                refresh=refresh,
                sb_kind=str(kind),
                fields="pdes,full_name,first_obs,tp,per",
            )
            source = catalog
        else:
            pages = [
                _sbdb_pages(
                    raw_dir,
                    cache_stem=f"solar_sbdb_{code}",
                    refresh=refresh,
                    sb_class=str(code),
                    fields="pdes,first_obs",
                )
                for code in spec["classes"]
            ]
            source = pd.concat(pages, ignore_index=True).drop_duplicates(
                "pdes", keep="first"
            )
        dated, missed = valid_discovery_dates(source["first_obs"], through=through)
        columns[column] = dated
        dropped[column] = missed
        logger.info(
            "%s discoveries: %s dated, %s incomplete dates dropped",
            column,
            len(dated),
            missed,
        )
    return monthly_count_frame(columns, through=through), dropped, catalog


def _perihelia_monthly(
    catalog: pd.DataFrame, discoveries: pd.DataFrame, *, through: pd.Timestamp
) -> pd.DataFrame:
    passages = comet_passages(
        catalog,
        start=pd.Timestamp("1700-01-01"),
        end=through,
        known_only=True,
    )
    index = pd.to_datetime(discoveries["date"])
    if passages.empty:
        counts = pd.Series(0, index=index, dtype=int)
    else:
        months = pd.to_datetime(passages["date"]).dt.to_period("M").dt.to_timestamp()
        counts = months.value_counts().reindex(index, fill_value=0).astype(int)
    return pd.DataFrame(
        {
            "date": index,
            "perihelia": counts.to_numpy(),
            "comet_prev": discoveries["comet_prev"].to_numpy(),
        }
    )


def _horizons(
    raw_dir: Path, *, today: pd.Timestamp, date_max: str, refresh: bool
) -> pd.DataFrame:
    daily_start = (today - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    daily_stop = (today + pd.Timedelta(days=LOOKAHEAD_DAYS + 2)).strftime("%Y-%m-%d")
    monthly_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    for planet in PLANETS:
        logger.info("Horizons %s", planet.name)
        earth = _horizons_frame(
            raw_dir,
            command=planet.horizons_command,
            center=GEO_CENTER,
            start=HISTORY_START,
            stop=date_max,
            step="1 MO",
            quantities="20,23",
            refresh=refresh,
        )
        sun = _horizons_frame(
            raw_dir,
            command=planet.horizons_command,
            center=SUN_CENTER,
            start=HISTORY_START,
            stop=date_max,
            step="1 MO",
            quantities="20",
            refresh=refresh,
        )
        daily = _horizons_frame(
            raw_dir,
            command=planet.horizons_command,
            center=GEO_CENTER,
            start=daily_start,
            stop=daily_stop,
            step="1 d",
            quantities="20,23",
            refresh=refresh,
        )
        monthly_parts.append(_distance_columns(planet, earth, sun))
        block = daily.copy()
        block["body"] = planet.name
        block["earth_au"] = block["delta_au"]
        daily_parts.append(
            block.loc[
                :,
                ["date", "body", "earth_au", "elongation_deg", "elongation_side"],
            ]
        )
    monthly = monthly_parts[0]
    for part in monthly_parts[1:]:
        monthly = monthly.merge(part, on="date", how="inner")
    monthly = monthly.sort_values("date").reset_index(drop=True)
    _require_month_starts(monthly, "Horizons monthly distances")
    daily_frame = pd.concat(daily_parts, ignore_index=True)
    elongation = pd.to_numeric(daily_frame["elongation_deg"], errors="coerce")
    if not bool(elongation.between(0, 180).all()):
        raise RuntimeError("Horizons elongation left the 0..180 degree range.")
    _write_table(daily_frame, HORIZONS_DAILY_PATH)
    return monthly


def _distance_columns(
    planet: Planet, earth: pd.DataFrame, sun: pd.DataFrame
) -> pd.DataFrame:
    left = pd.DataFrame(
        {
            "date": pd.to_datetime(earth["date"]).dt.normalize(),
            f"{planet.column}_earth_au": pd.to_numeric(
                earth["delta_au"], errors="coerce"
            ),
        }
    )
    right = pd.DataFrame(
        {
            "date": pd.to_datetime(sun["date"]).dt.normalize(),
            f"{planet.column}_sun_au": pd.to_numeric(sun["delta_au"], errors="coerce"),
        }
    )
    return left.merge(right, on="date", how="inner")


def _horizons_frame(
    raw_dir: Path,
    *,
    command: str,
    center: str,
    start: str,
    stop: str,
    step: str,
    quantities: str,
    refresh: bool,
) -> pd.DataFrame:
    params = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": f"'{center}'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "QUANTITIES": f"'{quantities}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
    }
    url = f"{HORIZONS_API}?{urllib.parse.urlencode(params)}"
    slug = _SLUG.sub(
        "_", f"horizons_{command}_{center}_{start}_{stop}_{step}_{quantities}"
    )
    payload = _load_json(
        cached_download(url, raw_dir / f"{slug}.json", refresh=refresh)
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Horizons {command} response was not an object.")
    if payload.get("error"):
        raise RuntimeError(f"Horizons {command} {center}: {payload['error']}")
    text = str(payload.get("result") or "")
    if "$$SOE" not in text:
        span = pd.Timestamp(stop) - pd.Timestamp(start)
        if span.days < 40:
            raise RuntimeError(f"Horizons {command} returned no table: {text[:400]}")
        mid = pd.Timestamp(start) + span / 2
        left_end = (mid - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        right_start = mid.strftime("%Y-%m-%d")
        left = _horizons_frame(
            raw_dir,
            command=command,
            center=center,
            start=start,
            stop=left_end,
            step=step,
            quantities=quantities,
            refresh=refresh,
        )
        right = _horizons_frame(
            raw_dir,
            command=command,
            center=center,
            start=right_start,
            stop=stop,
            step=step,
            quantities=quantities,
            refresh=refresh,
        )
        combined = pd.concat([left, right], ignore_index=True)
        return combined.drop_duplicates("date", keep="first")
    return parse_horizons_table(text)


def _moons(raw_dir: Path, *, refresh: bool) -> dict[str, Any]:
    payload = cached_download(
        MOON_SOURCE, raw_dir / "satellite_discoveries.html", refresh=refresh
    )
    parsed = parse_moon_discoveries(payload.decode("utf-8", "replace"))
    total = sum(int(row["count"]) for row in parsed["by_planet"])
    if total < 400:
        raise RuntimeError(
            f"JPL moon table parsed {total} satellites; expected about 460."
        )
    for row in parsed["by_planet"]:
        if int(row["count"]) != int(row["listed"]):
            raise RuntimeError(
                f"{row['planet']} header lists {row['listed']} satellites "
                f"but {row['count']} rows were parsed."
            )
    parsed["source"] = MOON_SOURCE
    return parsed


def _sbdb_pages(
    raw_dir: Path,
    *,
    cache_stem: str,
    refresh: bool,
    fields: str,
    sb_class: str | None = None,
    sb_kind: str | None = None,
) -> pd.DataFrame:
    offset = 0
    frames: list[pd.DataFrame] = []
    reported = 0
    while True:
        params: dict[str, str] = {
            "fields": fields,
            "limit": "100000",
            "limit-from": str(offset),
        }
        if sb_class:
            params["sb-class"] = sb_class
            params["sort"] = "pdes"
        if sb_kind:
            params["sb-kind"] = sb_kind
        url = f"{SBDB_QUERY_URL}?{urllib.parse.urlencode(params)}"
        payload = _load_json(
            cached_download(
                url, raw_dir / f"{cache_stem}_{offset}.json", refresh=refresh
            )
        )
        frame = _payload_frame(payload)
        count = int(payload.get("count", 0) or 0)
        reported = max(reported, count)
        frames.append(frame)
        offset += len(frame)
        logger.info(
            "%s page offset %s rows %s of %s", cache_stem, offset, len(frame), count
        )
        if frame.empty or offset >= count or len(frame) < 100_000:
            break
    if not frames:
        raise RuntimeError(f"SBDB {cache_stem} returned no pages.")
    raw_rows = sum(len(frame) for frame in frames)
    if reported and raw_rows < reported:
        raise RuntimeError(f"SBDB {cache_stem} returned {raw_rows} of {reported} rows.")
    combined = pd.concat(frames, ignore_index=True)
    if "pdes" not in combined.columns:
        raise ValueError(f"SBDB {cache_stem} has no pdes column.")
    deduped = combined.drop_duplicates("pdes", keep="first").reset_index(drop=True)
    # Pages can overlap if the catalog changes mid-download. A collapse down to
    # one repeated page means limit-from was ignored.
    if reported and len(deduped) < int(reported * 0.98):
        raise RuntimeError(
            f"SBDB {cache_stem} kept {len(deduped)} of {reported} designations."
        )
    return deduped


def _payload_frame(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("SBDB response was not an object.")
    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError("SBDB response is missing fields or data.")
    names = [str(field) for field in fields]
    if not data:
        return pd.DataFrame(columns=names)
    if isinstance(data[0], dict):
        return pd.DataFrame(data)
    return pd.DataFrame(data, columns=names)


def _check_sums(
    events: pd.DataFrame, discoveries: pd.DataFrame, perihelia: pd.DataFrame
) -> None:
    for planet in PLANETS:
        if not planet.train:
            continue
        total = float((events["body"] == planet.name).sum())
        low, high = _SUM_BANDS[planet.column]
        _require_range(f"{planet.name} approaches", total, low, high)
    for spec in DISCOVERY_SPECS:
        column = str(spec["column"])
        low, high = _SUM_BANDS[column]
        _require_range(
            f"{column} discoveries", float(discoveries[column].sum()), low, high
        )
    low, high = _SUM_BANDS["perihelia"]
    _require_range("comet perihelia", float(perihelia["perihelia"].sum()), low, high)


def _check_distances(frame: pd.DataFrame) -> None:
    for planet in PLANETS:
        for suffix, bands in (("earth", _EARTH_BANDS), ("sun", _SUN_BANDS)):
            column = f"{planet.column}_{suffix}_au"
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            if not np.isfinite(values).all() or bool((values <= 0).any()):
                raise RuntimeError(
                    f"{column} has a non-positive or non-finite distance."
                )
            low, high = bands[planet.column]
            _require_range(column, float(np.median(values)), low, high)


def _require_month_starts(frame: pd.DataFrame, name: str) -> None:
    dates = pd.to_datetime(frame["date"])
    expected = pd.date_range(dates.min(), dates.max(), freq="MS")
    if len(expected) != len(dates) or not bool(dates.is_monotonic_increasing):
        raise RuntimeError(f"{name} is not a continuous month-start series.")


def _approaches_in_window(
    events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[list[dict[str, Any]], bool]:
    if events.empty or "cd" not in events.columns:
        return [], False
    frame = events.copy()
    frame["cd"] = pd.to_datetime(frame["cd"], errors="coerce")
    frame = frame.dropna(subset=["cd"])
    stop = end + pd.Timedelta(days=1)
    chosen = frame.loc[(frame["cd"] >= start) & (frame["cd"] < stop)]
    truncated = len(chosen) > LIST_CAP
    if truncated:
        chosen = chosen.nsmallest(LIST_CAP, "dist_au")
    rows: list[dict[str, Any]] = []
    for record in chosen.sort_values(["cd", "dist_au"]).to_dict(orient="records"):
        magnitude = pd.to_numeric(record.get("h"), errors="coerce")
        distance = float(record["dist_au"])
        name = str(record.get("name") or "").strip()
        designation = str(record.get("designation") or "").strip()
        rows.append(
            {
                "body": str(record.get("body") or ""),
                "body_code": str(record.get("body_code") or ""),
                "designation": designation,
                "name": name or designation,
                "date": pd.Timestamp(record["cd"]).strftime("%Y-%m-%dT%H:%M:%S"),
                "dist_au": round(distance, 6),
                "dist_ld": round(distance / AU_PER_LD, 3),
                "h": None if pd.isna(magnitude) else round(float(magnitude), 2),
            }
        )
    return rows, truncated


def _perihelia_in_window(
    catalog: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[list[dict[str, Any]], bool]:
    if catalog.empty or "tp" not in catalog.columns:
        return [], False
    passages = comet_passages(
        catalog, start=start, end=end + pd.Timedelta(days=1), known_only=False
    )
    truncated = len(passages) > LIST_CAP
    if truncated:
        passages = passages.head(LIST_CAP)
    rows: list[dict[str, Any]] = []
    for record in passages.to_dict(orient="records"):
        period = record.get("period_days")
        period_value = pd.to_numeric(period, errors="coerce")
        rows.append(
            {
                "date": pd.Timestamp(record["date"]).strftime("%Y-%m-%d"),
                "designation": str(record.get("designation") or ""),
                "name": str(record.get("name") or ""),
                "period_days": None
                if pd.isna(period_value)
                else round(float(period_value), 1),
            }
        )
    return rows, truncated


def _alignments_in_window(
    daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> list[dict[str, Any]]:
    if daily.empty or "body" not in daily.columns:
        return []
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for planet in PLANETS:
        chosen = frame.loc[frame["body"] == planet.name]
        for event in alignment_events(chosen, inner=planet.inner):
            stamp = pd.Timestamp(event["date"])
            if start <= stamp <= end:
                event = dict(event)
                event["body"] = planet.name
                rows.append(event)
    rows.sort(key=lambda row: (str(row["date"]), str(row["body"])))
    return rows


def _monthly_in_window(
    approaches: list[dict[str, Any]],
    perihelia: list[dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    index = pd.date_range(
        start.to_period("M").to_timestamp(),
        end.to_period("M").to_timestamp(),
        freq="MS",
    )
    counts = {
        planet.column: dict.fromkeys(index, 0)
        for planet in PLANETS
        if planet.train
    }
    perihelion_counts = dict.fromkeys(index, 0)
    for row in approaches:
        planet = next(
            (item for item in PLANETS if item.name == row["body"] and item.train),
            None,
        )
        if planet is None:
            continue
        month = pd.Timestamp(str(row["date"])).to_period("M").to_timestamp()
        if month in counts[planet.column]:
            counts[planet.column][month] += 1
    for row in perihelia:
        month = pd.Timestamp(str(row["date"])).to_period("M").to_timestamp()
        if month in perihelion_counts:
            perihelion_counts[month] += 1
    payload: list[dict[str, Any]] = []
    for stamp in index:
        record: dict[str, Any] = {"month": stamp.strftime("%Y-%m-%d")}
        for column, values in counts.items():
            record[column] = values[stamp]
        record["perihelia"] = perihelion_counts[stamp]
        payload.append(record)
    return payload


def _body_rows(
    events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_events = not events.empty and "cd" in events.columns
    frame = events.copy()
    if has_events:
        frame["cd"] = pd.to_datetime(frame["cd"], errors="coerce")
    for planet in PLANETS:
        chosen = (
            frame.loc[frame["body"] == planet.name] if has_events else frame.iloc[0:0]
        )
        dates = (
            chosen["cd"]
            if has_events and not chosen.empty
            else pd.Series(dtype="datetime64[ns]")
        )
        historical = int((dates < start).sum()) if len(dates) else 0
        upcoming = (
            int(((dates >= start) & (dates < end + pd.Timedelta(days=1))).sum())
            if len(dates)
            else 0
        )
        rows.append(
            {
                "code": planet.code,
                "name": planet.name,
                "dist_max_au": planet.dist_max_au,
                "trained": planet.train,
                "historical_count": historical,
                "upcoming_count": upcoming,
            }
        )
    return rows


def _enrich_forecasts(
    forecasts: list[dict[str, Any]], *, horizons: pd.DataFrame
) -> list[dict[str, Any]]:
    lookup: dict[str, pd.Series] = {}
    if not horizons.empty and "date" in horizons.columns:
        dated = horizons.copy()
        dated["date"] = pd.to_datetime(dated["date"]).dt.strftime("%Y-%m-%d")
        for record in dated.to_dict(orient="records"):
            lookup[str(record["date"])] = pd.Series(record)
    enriched: list[dict[str, Any]] = []
    for forecast in forecasts:
        copied = dict(forecast)
        column = copied.get("physics_column")
        raw_outlook = copied.get("outlook")
        outlook: dict[str, Any] = raw_outlook if isinstance(raw_outlook, dict) else {}
        months_in = outlook.get("months")
        months: list[dict[str, Any]] = []
        errors: list[float] = []
        for month in months_in if isinstance(months_in, list) else []:
            if not isinstance(month, dict):
                continue
            point = dict(month)
            date = str(point.get("date") or "")[:10]
            physics = _physics_value(lookup.get(date), column)
            point["physics"] = physics
            p50 = point.get("p50")
            if (
                physics is not None
                and isinstance(p50, int | float)
                and not isinstance(p50, bool)
            ):
                errors.append(abs(float(p50) - physics))
            months.append(point)
        copied["outlook"] = dict(outlook)
        copied["outlook"]["months"] = months
        copied["physics_outlook_mae"] = (
            round(sum(errors) / len(errors), 4) if errors else None
        )
        enriched.append(copied)
    return enriched


def _physics_value(row: pd.Series | None, column: object) -> float | None:
    if row is None or not isinstance(column, str) or column not in row.index:
        return None
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def _moon_block(document: dict[str, Any], *, year: int) -> dict[str, Any]:
    discoveries = document.get("discoveries")
    rows = discoveries if isinstance(discoveries, list) else []
    recent = [
        row
        for row in rows
        if isinstance(row, dict) and int(row.get("year") or 0) >= year - 10
    ]
    recent.sort(
        key=lambda row: (-int(row.get("year") or 0), str(row.get("name") or ""))
    )
    by_planet = document.get("by_planet")
    return {
        "available": bool(rows),
        "source": document.get("source") or MOON_SOURCE,
        "by_planet": by_planet if isinstance(by_planet, list) else [],
        "recent": recent[:40],
        "note": (
            "Counts are discovery years from the JPL planetary satellite table. "
            "Only a year is published, and the finds arrive in survey bursts, "
            "so this table is not a trained monthly series."
        ),
    }


def _read_csv(path: str) -> pd.DataFrame:
    destination = resolve_repo_path(path)
    if not destination.is_file():
        return pd.DataFrame()
    return pd.read_csv(destination)


def _read_json_object(path: str) -> dict[str, Any]:
    destination = resolve_repo_path(path)
    if not destination.is_file():
        return {}
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_table(frame: pd.DataFrame, path: str) -> None:
    destination = resolve_repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = pd.to_datetime(output[column]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    output.to_csv(destination, index=False)


def _distance_token(distance: float) -> str:
    if math.isclose(distance, round(distance)):
        return str(int(round(distance)))
    return f"{distance:.2f}".rstrip("0").rstrip(".")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. See ``python -m src.space.solar_system --help``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description="Download solar-system series.")
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached responses."
    )
    parser.add_argument("--output-dir", default="data/space")
    args = parser.parse_args(argv)
    summary = fetch_solar_system(refresh=args.refresh, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
