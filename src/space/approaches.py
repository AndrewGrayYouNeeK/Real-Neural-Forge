"""Close approaches to Earth and the Moon, fireballs, and reentries.

The CAD distances are astronomical units. One lunar distance (LD) is
384,400 km, the same conventional length JPL prints next to an AU miss.
``H < 22`` is the usual stand-in for a body about 140 m across or larger.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import resolve_repo_path

# 384,400 km / IAU astronomical unit.
AU_PER_LD = 384_400 / 149_597_870.7
CLOSE_AU = 0.05
H_LARGE = 22.0
LOOKAHEAD_DAYS = 730
OUTLOOK_MONTHS = 24
LIST_CAP = 8000

EVENTS_PATH = "data/space/near_earth_events.csv"
FORECAST_PATH = "results/near_earth_forecast.json"
REENTRY_EVENTS_PATH = "data/space/reentry_events.csv"
FIREBALL_EVENTS_PATH = "data/space/fireball_events.csv"
SENTRY_PATH = "data/space/sentry_summary.json"
INTERSTELLAR_PATH = "data/space/interstellar_objects.csv"
MANIFEST_PATH = "data/space/near_earth_manifest.json"

SURVEY_GROWTH_NOTE = (
    "Monthly close-approach counts from the JPL catalog include approaches "
    "that happened before an object was discovered. As surveys improve, those "
    "past approaches are added, so the historical counts rise with the catalog. "
    "The known-at-the-time series keeps only objects whose first observation is "
    "on or before the approach, and the per-1,000-known-NEO rate divides by the "
    "NEO catalog size at the start of the month. Both controls still depend on "
    "which objects the surveys find. H < 22 (about 140 m and larger) is less "
    "incomplete in recent years and is not a complete record back to 1900. "
    "Future months in the CAD feed are predicted approaches of objects known "
    "today; they do not include objects nobody has found yet."
)
REENTRY_NOTE = (
    "Reentries are decay dates in the public CelesTrak SATCAT. That catalog "
    "records decays after they happen. It is not a predicted-reentry feed, and "
    "Space-Track TIP messages are not used because they require an account. "
    "A decay dated in the future is shown when the catalog carries one."
)
FIREBALL_NOTE = (
    "Fireballs are observed atmospheric impacts from JPL's fireball catalog, "
    "not predictions. The monthly forecast is a statistical extrapolation. "
    "Detection depends on the sensor network, so the counts are not a complete "
    "flux. Radiated energy is the JPL value in units of 10^10 joules. Impact "
    "energy is the JPL estimate in kilotons of TNT."
)

_COMET = re.compile(r"\d+\s*[PDC]\b|\b[PDC]/", re.IGNORECASE)
_INTERSTELLAR_TOKEN = re.compile(r"\b\d+I\b", re.IGNORECASE)
_BODIES = (("Earth", "earth"), ("Moon", "moon"))

# Forecast id -> key in the upcoming-month table. Future CAD rows are objects
# already known, so the known-at-the-time model is compared with that schedule.
JPL_MONTHLY_KEYS = {
    "earth_within_1ld": "earth_within_1ld",
    "earth_within_0_05au": "earth_within_0_05au",
    "earth_closest_au": "earth_closest_au",
    "earth_large_h22": "earth_large_h22",
    "earth_known_0_05au": "earth_within_0_05au",
    "moon_within_1ld": "moon_within_1ld",
    "moon_within_0_05au": "moon_within_0_05au",
    "moon_closest_au": "moon_closest_au",
    "moon_large_h22": "moon_large_h22",
    "reentries": "reentries",
}


def utc_today() -> pd.Timestamp:
    """Return today's date in UTC, with no timezone attached."""
    now = pd.Timestamp.now(tz="UTC")
    return pd.Timestamp(year=int(now.year), month=int(now.month), day=int(now.day))


def normalize_designation(value: object) -> str:
    """Uppercase a designation and drop punctuation."""
    text = str(value or "").upper().replace("'", "")
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", text).split())


def classify_kind(designation: object, name: object, interstellar: set[str]) -> str:
    """Label an approach as interstellar, comet, or asteroid."""
    des = str(designation or "")
    title = str(name or "")
    if (
        normalize_designation(des) in interstellar
        or normalize_designation(title) in interstellar
    ):
        return "interstellar"
    if _INTERSTELLAR_TOKEN.search(f"{des} {title}"):
        return "interstellar"
    if _COMET.search(f"{des} {title}"):
        return "comet"
    return "asteroid"


def cad_rows_frame(payload: dict[str, Any], *, body: str) -> pd.DataFrame:
    """Turn one CAD JSON object into approach rows. ``body`` is Earth or Moon."""
    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError("CAD payload is missing fields or data.")
    required = ("des", "cd", "dist")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"CAD payload is missing {missing}.")
    index = {str(name): pos for pos, name in enumerate(fields)}
    rows: list[dict[str, object]] = []
    for raw in data:
        if not isinstance(raw, list):
            continue
        if max(index[name] for name in required) >= len(raw):
            continue
        fullname = ""
        if (
            "fullname" in index
            and index["fullname"] < len(raw)
            and raw[index["fullname"]]
        ):
            fullname = str(raw[index["fullname"]]).strip()
        magnitude = np.nan
        if "h" in index and index["h"] < len(raw):
            magnitude = pd.to_numeric(raw[index["h"]], errors="coerce")
        rows.append(
            {
                "body": body,
                "designation": str(raw[index["des"]]).strip(),
                "name": fullname,
                "cd": pd.to_datetime(str(raw[index["cd"]]), format="%Y-%b-%d %H:%M"),
                "dist_au": float(raw[index["dist"]]),
                "h": float(magnitude) if pd.notna(magnitude) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["body", "designation", "name", "cd", "dist_au", "h"]
        )
    return frame


def attach_kinds(
    events: pd.DataFrame, interstellar: set[str], discoveries: dict[str, pd.Timestamp]
) -> pd.DataFrame:
    """Add kind, lunar distance, and whether the object was already known."""
    if events.empty:
        empty = events.copy()
        empty["dist_ld"] = []
        empty["kind"] = []
        empty["known"] = []
        return empty
    frame = events.copy()
    frame["dist_ld"] = frame["dist_au"] / AU_PER_LD
    frame["kind"] = [
        classify_kind(des, name, interstellar)
        for des, name in zip(frame["designation"], frame["name"], strict=True)
    ]
    first = frame["designation"].map(
        lambda value: discoveries.get(normalize_designation(value))
    )
    first_obs = pd.to_datetime(first)
    frame["known"] = first_obs.notna() & (first_obs <= frame["cd"])
    return frame


def monthly_approaches(
    events: pd.DataFrame, discoveries: pd.DataFrame, *, through: pd.Timestamp
) -> pd.DataFrame:
    """Monthly Earth and Moon series for months strictly before ``through``.

    ``through`` is the first instant of the open month. Later CAD rows are
    predictions and stay out of the training series. A month with no approach
    inside 0.05 AU has closest distance set to that search limit and
    ``*_closest_censored`` of 1.
    """
    if "cd" not in events.columns or events.empty:
        raise ValueError("close-approach events are empty")
    cutoff = pd.Timestamp(through)
    historical = events.loc[events["cd"] < cutoff].copy()
    if historical.empty:
        raise ValueError("no historical close approaches before the open month")
    historical["month"] = historical["cd"].dt.to_period("M").dt.to_timestamp()
    end = (cutoff - pd.Timedelta(days=1)).to_period("M").to_timestamp()
    start = pd.Timestamp(historical["month"].min())
    index = pd.date_range(start, end, freq="MS")
    if len(index) == 0:
        raise ValueError("close-approach month index is empty")
    search = historical.loc[historical["dist_au"] <= CLOSE_AU + 1e-12].copy()
    frame = pd.DataFrame({"date": index})
    for body, prefix in _BODIES:
        sub = search.loc[search["body"] == body]
        if sub.empty:
            raise ValueError(f"no historical {body} approaches within {CLOSE_AU} AU")
        within_ld = sub.loc[sub["dist_au"] <= AU_PER_LD]
        large = sub.loc[sub["h"].notna() & (sub["h"] < H_LARGE)]
        known = sub.loc[sub["known"].astype(bool)]
        frame[f"{prefix}_within_0_05au"] = _month_counts(sub, index)
        frame[f"{prefix}_within_1ld"] = _month_counts(within_ld, index)
        frame[f"{prefix}_large_h22"] = _month_counts(large, index)
        frame[f"{prefix}_known_0_05au"] = _month_counts(known, index)
        frame[f"{prefix}_known_1ld"] = _month_counts(
            known.loc[known["dist_au"] <= AU_PER_LD], index
        )
        frame[f"{prefix}_comets_0_05au"] = _month_counts(
            sub.loc[sub["kind"] == "comet"], index
        )
        frame[f"{prefix}_asteroids_0_05au"] = _month_counts(
            sub.loc[sub["kind"] == "asteroid"], index
        )
        closest, censored = _closest(sub, index)
        frame[f"{prefix}_closest_au"] = closest
        frame[f"{prefix}_closest_censored"] = censored
    frame["known_neos_prev"] = _catalog_before(discoveries, index)
    positive = frame["known_neos_prev"] > 0
    if not bool(positive.any()):
        raise ValueError("NEO catalog size is zero for every approach month.")
    frame = frame.loc[positive].reset_index(drop=True)
    if frame["date"].diff().iloc[1:].gt(pd.Timedelta(days=32)).any():
        raise ValueError("dropping zero catalog months left a gap in the series")
    known_prev = frame["known_neos_prev"].to_numpy(dtype=float)
    frame["earth_0_05au_per_1000"] = (
        frame["earth_within_0_05au"].to_numpy(dtype=float) / known_prev * 1000.0
    )
    return frame


def monthly_sum_count(
    dates: pd.Series,
    values: pd.Series,
    column: str,
    *,
    through: pd.Timestamp,
    count_column: str = "count",
) -> pd.DataFrame:
    """Monthly event counts and sums for instants strictly before ``through``."""
    stamps = pd.to_datetime(dates)
    keep = stamps < pd.Timestamp(through)
    stamps = stamps.loc[keep]
    amounts = pd.to_numeric(values, errors="coerce").loc[keep]
    if stamps.empty:
        raise ValueError(f"{column} has no rows before the open month")
    months = stamps.dt.to_period("M").dt.to_timestamp()
    end = (pd.Timestamp(through) - pd.Timedelta(days=1)).to_period("M").to_timestamp()
    index = pd.date_range(months.min(), end, freq="MS")
    grouped = pd.DataFrame({"month": months, "amount": amounts.fillna(0.0)})
    counts = grouped.groupby("month").size().reindex(index, fill_value=0).astype(int)
    totals = grouped.groupby("month")["amount"].sum().reindex(index, fill_value=0.0)
    return pd.DataFrame(
        {"date": index, count_column: counts.to_numpy(), column: totals.to_numpy()}
    )


def fireball_events(payload: dict[str, Any]) -> pd.DataFrame:
    """One row per fireball. Energy columns follow the JPL field units."""
    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError("Fireball payload is missing fields or data.")
    if "date" not in fields:
        raise ValueError("Fireball payload has no date field.")
    index = {str(name): pos for pos, name in enumerate(fields)}
    rows: list[dict[str, object]] = []
    for raw in data:
        if not isinstance(raw, list) or index["date"] >= len(raw):
            continue
        rows.append(
            {
                "date": pd.to_datetime(str(raw[index["date"]]), errors="coerce"),
                "radiated_1e10_j": _optional_number(raw, index, "energy"),
                "energy_kt": _optional_number(raw, index, "impact-e"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["date"].isna().all():
        raise ValueError("Fireball payload has no dated events.")
    return frame.dropna(subset=["date"]).reset_index(drop=True)


def sentry_summary(payload: dict[str, Any], *, limit: int = 12) -> dict[str, Any]:
    """Keep the current impact-risk list, highest Palermo scale first."""
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("Sentry payload has no data list.")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        palermo = pd.to_numeric(row.get("ps_cum"), errors="coerce")
        parsed.append(
            {
                "designation": row.get("des"),
                "name": str(row.get("fullname") or "").strip() or row.get("des"),
                "h": _finite_or_none(row.get("h")),
                "diameter_km": _finite_or_none(row.get("diameter")),
                "impact_probability": _finite_or_none(row.get("ip")),
                "palermo": None if pd.isna(palermo) else float(palermo),
                "range": row.get("range"),
                "last_obs": row.get("last_obs"),
            }
        )
    parsed.sort(
        key=lambda item: (
            item["palermo"] is None,
            -(item["palermo"] if item["palermo"] is not None else 0.0),
        )
    )
    return {
        "count": len(parsed),
        "objects": parsed[:limit],
        "note": (
            "Sentry is JPL's current impact-risk list, a snapshot rather than "
            "a monthly history. Palermo is cumulative; a value closer to zero "
            "is a higher risk than a more negative one."
        ),
    }


def satcat_decays(frame: pd.DataFrame) -> pd.DataFrame:
    """Decay rows from a CelesTrak SATCAT table."""
    required = {"OBJECT_NAME", "NORAD_CAT_ID", "DECAY_DATE"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"SATCAT is missing columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["DECAY_DATE"], errors="coerce")
    kept = frame.loc[dates.notna(), ["OBJECT_NAME", "NORAD_CAT_ID"]].copy()
    kept["date"] = dates.loc[dates.notna()].to_numpy()
    kept = kept.rename(columns={"OBJECT_NAME": "name", "NORAD_CAT_ID": "norad_id"})
    if kept.empty:
        raise ValueError("SATCAT has no decay dates.")
    return kept.reset_index(drop=True)


def window_bounds(today: pd.Timestamp, days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive UTC date window ``[today, today + days]``."""
    if (
        isinstance(days, bool)
        or not isinstance(days, int)
        or not 1 <= days <= LOOKAHEAD_DAYS
    ):
        raise ValueError(f"days must be an integer from 1 to {LOOKAHEAD_DAYS}.")
    start = pd.Timestamp(today).normalize()
    end = start + pd.Timedelta(days=int(days))
    return start, end


def near_earth_report(
    *, days: int = LOOKAHEAD_DAYS, today: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Upcoming approaches, reentries, and the saved monthly forecasts."""
    stamp = utc_today() if today is None else pd.Timestamp(today).normalize()
    start, end = window_bounds(stamp, days)
    events = _read_csv(EVENTS_PATH)
    approaches, monthly, truncated = _approaches_in_window(events, start, end)
    reentries, recent_reentries = _reentries_in_window(start, end)
    _attach_reentry_counts(monthly, reentries)
    forecasts = _forecast_payload()
    fireballs = _recent_fireballs(stamp)
    sentry = _read_json(SENTRY_PATH)
    interstellar = _read_csv(INTERSTELLAR_PATH)
    manifest = _read_json(MANIFEST_PATH)
    covered = manifest.get("date_max") if isinstance(manifest, dict) else None
    available = bool(approaches) or bool(forecasts) or bool(reentries)
    return {
        "available": available,
        "days": int(days),
        "window_start": start.strftime("%Y-%m-%d"),
        "window_end": end.strftime("%Y-%m-%d"),
        "time_scale": "TDB calendar dates as recorded by JPL CAD",
        "covered_through": covered,
        "truncated": truncated,
        "survey_growth_note": SURVEY_GROWTH_NOTE,
        "reentries_note": REENTRY_NOTE,
        "fireball_note": FIREBALL_NOTE,
        "approaches": approaches,
        "closest": _closest_rows(approaches, limit=8),
        "monthly": monthly,
        "reentries": reentries,
        "recent_reentries": recent_reentries,
        "fireballs_recent": fireballs,
        "sentry": sentry if isinstance(sentry, dict) else {"count": 0, "objects": []},
        "interstellar": _interstellar_block(interstellar, approaches),
        "forecasts": forecasts,
        "outlook_months": OUTLOOK_MONTHS,
    }


def _month_counts(events: pd.DataFrame, index: pd.DatetimeIndex) -> list[int]:
    if events.empty:
        return [0] * len(index)
    counts = events.groupby("month").size().reindex(index, fill_value=0)
    return [int(value) for value in counts.to_numpy()]


def _closest(
    events: pd.DataFrame, index: pd.DatetimeIndex
) -> tuple[list[float], list[int]]:
    if events.empty:
        mins = pd.Series(np.nan, index=index)
    else:
        mins = events.groupby("month")["dist_au"].min().reindex(index)
    censored = mins.isna()
    filled = mins.fillna(CLOSE_AU)
    return (
        [float(value) for value in filled.to_numpy()],
        [int(value) for value in censored.astype(int).to_numpy()],
    )


def _catalog_before(discoveries: pd.DataFrame, index: pd.DatetimeIndex) -> list[int]:
    """NEOs with ``first_obs`` before each month, carried forward after the last."""
    if discoveries.empty or "date" not in discoveries.columns:
        raise ValueError("NEO discovery series is empty.")
    column = "discoveries" if "discoveries" in discoveries.columns else None
    if column is None:
        raise ValueError("NEO discovery series has no discoveries column.")
    frame = discoveries.loc[:, ["date", column]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = frame.groupby("date")[column].sum().sort_index().astype(float)
    full = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full, fill_value=0.0)
    before = monthly.cumsum().shift(1).fillna(0.0)
    aligned = before.reindex(before.index.union(index)).ffill().reindex(index)
    return [int(value) for value in aligned.fillna(0.0).to_numpy()]


def _optional_number(raw: list[object], index: dict[str, int], name: str) -> float:
    if name not in index or index[name] >= len(raw):
        return 0.0
    value = pd.to_numeric(raw[index[name]], errors="coerce")
    if pd.isna(value):
        return 0.0
    return float(value)


def _finite_or_none(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def _read_csv(path: str) -> pd.DataFrame:
    destination = resolve_repo_path(path)
    if not destination.is_file():
        return pd.DataFrame()
    return pd.read_csv(destination)


def _read_json(path: str) -> Any:
    destination = resolve_repo_path(path)
    if not destination.is_file():
        return None
    return json.loads(destination.read_text(encoding="utf-8"))


def _approaches_in_window(
    events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if events.empty or "date" not in events.columns:
        return [], [], False
    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    stop = end + pd.Timedelta(days=1)
    mask = (frame["date"] >= start) & (frame["date"] < stop)
    if "dist_au" in frame.columns:
        search = frame["dist_au"] <= CLOSE_AU + 1e-9
        interstellar = frame.get("kind", pd.Series("", index=frame.index)).eq(
            "interstellar"
        )
        mask = mask & (search | interstellar)
    chosen = frame.loc[mask].sort_values(["date", "dist_au"], na_position="last")
    monthly = _window_months(chosen, start, end)
    truncated = len(chosen) > LIST_CAP
    if truncated:
        chosen = chosen.nsmallest(LIST_CAP, "dist_au")
        chosen = chosen.sort_values(["date", "dist_au"])
    return [_approach_row(row) for _, row in chosen.iterrows()], monthly, truncated


def _approach_row(row: pd.Series) -> dict[str, Any]:
    magnitude = row.get("h")
    magnitude_value = _finite_or_none(magnitude)
    dist = float(row["dist_au"])
    return {
        "body": str(row.get("body") or ""),
        "designation": str(row.get("designation") or ""),
        "name": str(row.get("name") or row.get("designation") or ""),
        "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%dT%H:%M"),
        "dist_au": dist,
        "dist_ld": float(row["dist_ld"])
        if pd.notna(row.get("dist_ld"))
        else dist / AU_PER_LD,
        "h": magnitude_value,
        "kind": str(row.get("kind") or "asteroid"),
        "large": magnitude_value is not None and magnitude_value < H_LARGE,
        "interstellar": str(row.get("kind") or "") == "interstellar",
    }


def _window_months(
    events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> list[dict[str, Any]]:
    """JPL counts and closest misses for each calendar month the window touches."""
    months = pd.date_range(
        start.to_period("M").to_timestamp(),
        end.to_period("M").to_timestamp(),
        freq="MS",
    )
    if events.empty:
        dated = events
    else:
        dated = events.copy()
        dated["month"] = dated["date"].dt.to_period("M").dt.to_timestamp()
    rows: list[dict[str, Any]] = []
    for month in months:
        month_end = month + pd.offsets.MonthEnd(0)
        partial = start > month or end < month_end.normalize()
        if dated.empty:
            sub = dated
        else:
            sub = dated.loc[dated["month"] == month]
        row: dict[str, Any] = {
            "month": pd.Timestamp(month).strftime("%Y-%m-%d"),
            "partial": bool(partial),
        }
        for body, prefix in _BODIES:
            body_rows = (
                sub.loc[sub["body"] == body]
                if not sub.empty and "body" in sub.columns
                else sub.iloc[0:0]
            )
            search = body_rows
            if not body_rows.empty and "dist_au" in body_rows.columns:
                search = body_rows.loc[body_rows["dist_au"] <= CLOSE_AU + 1e-9]
            within_ld = (
                search.loc[search["dist_au"] <= AU_PER_LD]
                if not search.empty
                else search
            )
            large = (
                search.loc[pd.to_numeric(search["h"], errors="coerce") < H_LARGE]
                if not search.empty and "h" in search.columns
                else search.iloc[0:0]
            )
            row[f"{prefix}_within_0_05au"] = int(len(search))
            row[f"{prefix}_within_1ld"] = int(len(within_ld))
            row[f"{prefix}_large_h22"] = int(len(large))
            if search.empty:
                row[f"{prefix}_closest_au"] = None
            else:
                row[f"{prefix}_closest_au"] = float(search["dist_au"].min())
        if not sub.empty and "kind" in sub.columns and sub["kind"].eq("reentry").any():
            row["reentries"] = int(sub["kind"].eq("reentry").sum())
        rows.append(row)
    return rows


def _reentries_in_window(
    start: pd.Timestamp, end: pd.Timestamp
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = _read_csv(REENTRY_EVENTS_PATH)
    if frame.empty or "date" not in frame.columns:
        return [], []
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    stop = end + pd.Timedelta(days=1)
    upcoming = frame.loc[(frame["date"] >= start) & (frame["date"] < stop)]
    recent_start = start - pd.Timedelta(days=120)
    recent = frame.loc[(frame["date"] >= recent_start) & (frame["date"] < start)]
    recent = recent.sort_values("date").tail(12)
    return _reentry_rows(upcoming), _reentry_rows(recent)


def _reentry_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values("date").iterrows():
        rows.append(
            {
                "name": str(row.get("name") or ""),
                "norad_id": None
                if pd.isna(row.get("norad_id"))
                else str(row.get("norad_id")),
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            }
        )
    return rows


def _recent_fireballs(today: pd.Timestamp) -> list[dict[str, Any]]:
    frame = _read_csv(FIREBALL_EVENTS_PATH)
    if frame.empty or "date" not in frame.columns:
        return []
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    start = today - pd.Timedelta(days=120)
    chosen = frame.loc[
        (frame["date"] >= start) & (frame["date"] < today + pd.Timedelta(days=1))
    ]
    chosen = chosen.sort_values("date").tail(8)
    rows: list[dict[str, Any]] = []
    for _, row in chosen.iterrows():
        rows.append(
            {
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%dT%H:%M:%S"),
                "energy_kt": _finite_or_none(row.get("energy_kt")),
                "radiated_1e10_j": _finite_or_none(row.get("radiated_1e10_j")),
            }
        )
    return rows


def published_forecasts(path: str) -> list[dict[str, Any]]:
    """Slim saved outlook rows from a forecast JSON file."""
    document = _read_json(path)
    if isinstance(document, dict):
        series = document.get("series")
    elif isinstance(document, list):
        series = document
    else:
        return []
    if not isinstance(series, list):
        return []
    payloads: list[dict[str, Any]] = []
    for row in series:
        if isinstance(row, dict) and isinstance(row.get("outlook"), dict):
            payloads.append(_slim_forecast(row))
    return payloads


def _forecast_payload() -> list[dict[str, Any]]:
    return published_forecasts(FORECAST_PATH)


def _slim_forecast(row: dict[str, Any]) -> dict[str, Any]:
    outlook = row.get("outlook") if isinstance(row.get("outlook"), dict) else {}
    months = outlook.get("months") if isinstance(outlook, dict) else None
    model = _mae(row.get("model"))
    naive = {
        "persistence": _mae(_nested_mae(row, "trainer_baselines", "persistence")),
        "seasonal_naive": _mae(_nested_mae(row, "trainer_baselines", "seasonal_naive")),
        "mean": _mae(row.get("mean")),
    }
    discovery = _mae(row.get("discovery"))
    naive_values = [value for value in naive.values() if value is not None]
    best_naive = min(naive_values) if naive_values else None
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "units": row.get("units"),
        "survey_growth": bool(row.get("survey_growth")),
        "control": row.get("control"),
        "physics_column": row.get("physics_column"),
        "model_mae": model,
        "model_rmse": _metric(row.get("model"), "rmse"),
        "baselines": naive,
        "discovery_regression_mae": discovery,
        "beats_naive_baselines": (
            None if model is None or best_naive is None else bool(model < best_naive)
        ),
        "jpl_monthly_key": JPL_MONTHLY_KEYS.get(str(row.get("id"))),
        "outlook": {
            "method": outlook.get("method") if isinstance(outlook, dict) else None,
            "nominal_coverage": outlook.get("nominal_coverage")
            if isinstance(outlook, dict)
            else None,
            "validation_coverage": outlook.get("validation_coverage")
            if isinstance(outlook, dict)
            else None,
            "months": months if isinstance(months, list) else [],
        },
    }


def _nested_mae(row: dict[str, Any], block: str, name: str) -> Any:
    parent = row.get(block)
    if isinstance(parent, dict):
        return parent.get(name)
    return None


def _mae(block: Any) -> float | None:
    return _metric(block, "mae")


def _metric(block: Any, name: str) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _attach_reentry_counts(
    monthly: list[dict[str, Any]], reentries: list[dict[str, Any]]
) -> None:
    counts: dict[str, int] = {}
    for row in reentries:
        month = (
            pd.Timestamp(row["date"]).to_period("M").to_timestamp().strftime("%Y-%m-%d")
        )
        counts[month] = counts.get(month, 0) + 1
    for row in monthly:
        row["reentries"] = counts.get(str(row["month"]), 0)


def _closest_rows(
    approaches: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    earth = [
        row
        for row in approaches
        if row.get("body") == "Earth" and not row.get("interstellar")
    ]
    earth.sort(key=lambda row: float(row["dist_ld"]))
    return earth[:limit]


def _interstellar_block(
    catalog: pd.DataFrame, approaches: list[dict[str, Any]]
) -> dict[str, Any]:
    known: list[dict[str, Any]] = []
    if not catalog.empty:
        for _, row in catalog.iterrows():
            known.append(
                {
                    "id": None
                    if pd.isna(row.get("interstellar_id"))
                    else str(row.get("interstellar_id")),
                    "designation": None
                    if pd.isna(row.get("designation"))
                    else str(row.get("designation")),
                    "name": None if pd.isna(row.get("name")) else str(row.get("name")),
                    "orbit_class": None
                    if pd.isna(row.get("orbit_class"))
                    else str(row.get("orbit_class")),
                    "first_obs": None
                    if pd.isna(row.get("first_obs"))
                    else str(row.get("first_obs")),
                }
            )
    upcoming = [row for row in approaches if row.get("interstellar")]
    return {
        "note": (
            "Interstellar objects are the IAU nI designations resolved from "
            "JPL SBDB (1I, 2I, 3I, and any later number the service returns). "
            "Hyperbolic asteroids that were never given an I designation are "
            "not labeled interstellar."
        ),
        "known": known,
        "upcoming": upcoming,
    }
