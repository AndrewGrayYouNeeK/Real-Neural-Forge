"""Download Earth and Moon close approaches, fireballs, Sentry, and reentries.

Run from the repository root:

    python -m scripts.fetch_near_earth

Raw responses and the processed CSVs live under ``data/space/`` and are not
committed. The event files the API serves are also committed in
``space_snapshot/`` so the image can answer before a refresh finishes. No API
key is used.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.space.approaches import (
    CLOSE_AU,
    FIREBALL_EVENTS_PATH,
    INTERSTELLAR_PATH,
    LOOKAHEAD_DAYS,
    MANIFEST_PATH,
    REENTRY_EVENTS_PATH,
    SENTRY_PATH,
    attach_kinds,
    cad_rows_frame,
    fireball_events,
    monthly_approaches,
    monthly_sum_count,
    normalize_designation,
    satcat_decays,
    sentry_summary,
    utc_today,
)
from src.space.fetch import (
    CAD_API_URL,
    NEO_ORBIT_CLASSES,
    SBDB_QUERY_URL,
    _decode,
    _load_json,
    _require_range,
    _summary,
    cached_download,
    sbdb_url,
    write_series,
)
from src.utils.paths import resolve_repo_path

FIREBALL_API_URL = "https://ssd-api.jpl.nasa.gov/fireball.api"
SENTRY_API_URL = "https://ssd-api.jpl.nasa.gov/sentry.api"
SBDB_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
SATCAT_URL = "https://celestrak.org/pub/satcat.csv"
USER_AGENT = "real-neural-forge space fetch"

NEAR_EARTH_SOURCES: tuple[dict[str, str], ...] = (
    {
        "id": "cad",
        "name": "JPL SBDB close approaches to Earth and the Moon",
        "url": CAD_API_URL,
        "licence": (
            "NASA/JPL SBDB Close-Approach Data API, public domain U.S. "
            f"government work. {CAD_API_URL}. Distances are AU. The training "
            f"counts use approaches within {CLOSE_AU} AU. The lookahead file "
            "also keeps interstellar approaches out to 1 AU."
        ),
    },
    {
        "id": "fireballs",
        "name": "JPL fireball and bolide reports",
        "url": FIREBALL_API_URL,
        "licence": (
            "NASA/JPL Fireball Data API, public domain U.S. government work. "
            f"{FIREBALL_API_URL}. Radiated energy is in units of 10^10 joules. "
            "Impact energy is an estimate in kilotons of TNT."
        ),
    },
    {
        "id": "sentry",
        "name": "JPL Sentry impact-risk list",
        "url": SENTRY_API_URL,
        "licence": (
            "NASA/JPL Sentry API, public domain U.S. government work. "
            f"{SENTRY_API_URL}. A current snapshot, not a historical series."
        ),
    },
    {
        "id": "sbdb",
        "name": "JPL SBDB designations and first observations",
        "url": SBDB_QUERY_URL,
        "licence": (
            "NASA/JPL Small-Body Database Query API, public domain U.S. "
            f"government work. {SBDB_QUERY_URL}. first_obs is the discovery "
            "control for close-approach counts."
        ),
    },
    {
        "id": "satcat",
        "name": "CelesTrak SATCAT decay dates",
        "url": SATCAT_URL,
        "licence": (
            "CelesTrak SATCAT redistribution of the public satellite catalog. "
            f"{SATCAT_URL}. Cite CelesTrak (T.S. Kelso). Decay dates record "
            "reentries that have already happened."
        ),
    },
)


def cad_query_url(
    *,
    start: str,
    end: str,
    body: str,
    dist_max: str,
    kind: str | None = None,
    designation: str | None = None,
) -> str:
    """One CAD request. ``dist_max`` is an AU string or an LD string such as ``1LD``."""
    params: dict[str, str] = {
        "date-min": start,
        "date-max": end,
        "dist-max": dist_max,
        "body": body,
        "fullname": "true",
        "limit": "100000",
    }
    if kind:
        params["kind"] = kind
    if designation:
        params["des"] = designation
    return f"{CAD_API_URL}?{urllib.parse.urlencode(params)}"


def sbdb_kind_url(kind: str) -> str:
    """Designation and first observation for one SBDB object kind."""
    query = urllib.parse.urlencode(
        {"fields": "pdes,first_obs", "sb-kind": kind, "limit": "100000"}
    )
    return f"{SBDB_QUERY_URL}?{query}"


def download_json_optional(url: str) -> dict[str, Any] | None:
    """GET a JSON object. A 400 or 404 response means the object is absent."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 404}:
            return None
        raise
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object from {url}.")
    return payload


def fetch_near_earth(
    *, refresh: bool = False, output_dir: str | Path = "data/space"
) -> dict[str, object]:
    """Download the near-Earth files and write the training CSVs."""
    root = resolve_repo_path(output_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = utc_today()
    through = today.replace(day=1)
    date_max = (today + pd.Timedelta(days=LOOKAHEAD_DAYS)).strftime("%Y-%m-%d")
    start = "1900-01-01"

    interstellar = _interstellar_objects(raw_dir, refresh=refresh)
    keys = _interstellar_keys(interstellar)
    discoveries, index = _discovery_index(raw_dir, refresh=refresh, through=through)

    frames = [
        _collect_cad(
            raw_dir,
            body=body,
            start=start,
            end=date_max,
            dist_max=str(CLOSE_AU),
            refresh=refresh,
        )
        for body in ("Earth", "Moon")
    ]
    frames.extend(
        _interstellar_approaches(
            raw_dir, interstellar, start=start, end=date_max, refresh=refresh
        )
    )
    events = attach_kinds(pd.concat(frames, ignore_index=True), keys, index)
    events = events.drop_duplicates(["body", "designation", "cd"], keep="first")
    monthly = monthly_approaches(events, discoveries, through=through)
    _require_range(
        "Earth close approaches",
        float(monthly["earth_within_0_05au"].sum()),
        1000.0,
        500_000.0,
    )
    _require_range(
        "Moon close approaches",
        float(monthly["moon_within_0_05au"].sum()),
        100.0,
        2_000_000.0,
    )
    write_series(monthly, root / "near_earth_monthly.csv")
    _write_upcoming(
        events, root / "near_earth_events.csv", today=today, date_max=date_max
    )

    fireballs = _fireballs(raw_dir, refresh=refresh)
    fireball_monthly = monthly_sum_count(
        fireballs["date"],
        fireballs["energy_kt"],
        "fireball_energy_kt",
        through=through,
        count_column="fireball_count",
    )
    radiated = monthly_sum_count(
        fireballs["date"],
        fireballs["radiated_1e10_j"],
        "fireball_radiated_1e10j",
        through=through,
        count_column="fireball_count",
    )
    fireball_monthly["fireball_radiated_1e10j"] = radiated["fireball_radiated_1e10j"]
    _require_range(
        "fireballs", float(fireball_monthly["fireball_count"].sum()), 100.0, 100_000.0
    )
    write_series(fireball_monthly, root / "fireballs_monthly.csv")
    _write_fireball_events(fireballs, resolve_repo_path(FIREBALL_EVENTS_PATH))

    decays = _satcat(raw_dir, refresh=refresh)
    reentry_monthly = monthly_sum_count(
        decays["date"],
        pd.Series(1.0, index=decays.index),
        "reentries",
        through=through,
        count_column="reentry_events",
    )
    reentry_monthly = reentry_monthly.loc[:, ["date", "reentries"]]
    _require_range(
        "reentries", float(reentry_monthly["reentries"].sum()), 1000.0, 100_000.0
    )
    write_series(reentry_monthly, root / "reentries_monthly.csv")
    _write_reentry_events(decays, resolve_repo_path(REENTRY_EVENTS_PATH), today=today)

    sentry = _sentry(raw_dir, refresh=refresh)
    summary: dict[str, object] = {
        "sources": list(NEAR_EARTH_SOURCES),
        "fetched_on": today.strftime("%Y-%m-%d"),
        "date_min": start,
        "date_max": date_max,
        "dist_max_au": CLOSE_AU,
        "near_earth": _summary(
            monthly,
            earth_approaches=int(monthly["earth_within_0_05au"].sum()),
            moon_approaches=int(monthly["moon_within_0_05au"].sum()),
        ),
        "fireballs": _summary(
            fireball_monthly,
            events=int(fireball_monthly["fireball_count"].sum()),
            energy_kt=float(fireball_monthly["fireball_energy_kt"].sum()),
        ),
        "reentries": _summary(
            reentry_monthly, events=int(reentry_monthly["reentries"].sum())
        ),
        "sentry_objects": int(sentry["count"]),
        "interstellar_objects": int(len(interstellar)),
    }
    manifest_path = resolve_repo_path(MANIFEST_PATH)
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _collect_cad(
    raw_dir: Path,
    *,
    body: str,
    start: str,
    end: str,
    dist_max: str,
    refresh: bool,
    kind: str | None = None,
    designation: str | None = None,
) -> pd.DataFrame:
    """Download one CAD window, splitting it when the service truncates."""
    rows = _cad_pages(
        raw_dir,
        body=body,
        start=start,
        end=end,
        dist_max=dist_max,
        refresh=refresh,
        kind=kind,
        designation=designation,
    )
    return cad_rows_frame({"fields": rows["fields"], "data": rows["data"]}, body=body)


def _cad_pages(
    raw_dir: Path,
    *,
    body: str,
    start: str,
    end: str,
    dist_max: str,
    refresh: bool,
    kind: str | None,
    designation: str | None,
) -> dict[str, Any]:
    field_names: list[str] = []

    def collect(window_start: str, window_end: str) -> list[list[object]]:
        url = cad_query_url(
            start=window_start,
            end=window_end,
            body=body,
            dist_max=dist_max,
            kind=kind,
            designation=designation,
        )
        slug = f"{body}_{window_start}_{window_end}_{dist_max}"
        if kind:
            slug += f"_{kind}"
        if designation:
            slug += "_" + normalize_designation(designation).replace(" ", "_")
        payload = _load_json(
            cached_download(url, raw_dir / f"cad_{slug}.json", refresh=refresh)
        )
        fields = payload.get("fields")
        data = payload.get("data")
        total = int(payload.get("total", payload.get("count", 0)) or 0)
        if not isinstance(fields, list) or not isinstance(data, list):
            raise ValueError(f"CAD {body} response is missing fields or data.")
        if not field_names:
            field_names.extend(str(field) for field in fields)
        if len(data) >= total or total == 0:
            return data
        span = pd.Timestamp(window_end) - pd.Timestamp(window_start)
        if span.days <= 1:
            raise RuntimeError(
                f"CAD API truncated {body} {window_start}..{window_end} "
                f"({len(data)} of {total})."
            )
        mid = pd.Timestamp(window_start) + span / 2
        left_end = (mid - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        right_start = mid.strftime("%Y-%m-%d")
        return collect(window_start, left_end) + collect(right_start, window_end)

    data = collect(start, end)
    if not field_names:
        raise ValueError(f"CAD {body} returned no field names.")
    return {"fields": field_names, "data": data}


def _discovery_index(
    raw_dir: Path, *, refresh: bool, through: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    """NEO monthly discoveries, plus a designation map that also includes comets."""
    frames: list[pd.DataFrame] = []
    for orbit_class in NEO_ORBIT_CLASSES:
        frames.append(
            _sbdb_frame(
                sbdb_url(orbit_class),
                raw_dir / f"sbdb_{orbit_class}.json",
                refresh=refresh,
            )
        )
    neo = pd.concat(frames, ignore_index=True)
    discoveries = _monthly_discoveries(neo, through=through)
    comet = _sbdb_frame(
        sbdb_kind_url("c"), raw_dir / "sbdb_comets.json", refresh=refresh
    )
    index = _first_obs_map(pd.concat([neo, comet], ignore_index=True))
    return discoveries, index


def _monthly_discoveries(frame: pd.DataFrame, *, through: pd.Timestamp) -> pd.DataFrame:
    """Count distinct designations by first_obs, including the last complete month."""
    combined = frame.drop_duplicates("pdes", keep="first")
    dates = pd.to_datetime(combined["first_obs"], errors="coerce").dropna()
    monthly = monthly_sum_count(
        dates,
        pd.Series(0.0, index=dates.index),
        "ignored",
        through=through,
        count_column="discoveries",
    )
    return monthly.loc[:, ["date", "discoveries"]]


def _sbdb_frame(url: str, destination: Path, *, refresh: bool) -> pd.DataFrame:
    payload = _load_json(cached_download(url, destination, refresh=refresh))
    fields = payload.get("fields")
    data = payload.get("data")
    count = int(payload.get("count", 0) or 0)
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError(f"SBDB response {destination.name} is missing rows.")
    if count and len(data) < count:
        raise RuntimeError(
            f"SBDB {destination.name} returned {len(data)} of {count} rows."
        )
    return pd.DataFrame(data, columns=[str(field) for field in fields])


def _first_obs_map(frame: pd.DataFrame) -> dict[str, pd.Timestamp]:
    if "pdes" not in frame.columns or "first_obs" not in frame.columns:
        raise ValueError("SBDB frames need pdes and first_obs columns.")
    dates = pd.to_datetime(frame["first_obs"], errors="coerce")
    index: dict[str, pd.Timestamp] = {}
    for designation, stamp in zip(frame["pdes"], dates, strict=True):
        if pd.isna(stamp):
            continue
        key = normalize_designation(designation)
        previous = index.get(key)
        if previous is None or stamp < previous:
            index[key] = pd.Timestamp(stamp)
    if not index:
        raise ValueError("SBDB frames produced no designation dates.")
    return index


def _interstellar_objects(raw_dir: Path, *, refresh: bool) -> pd.DataFrame:
    """Resolve 1I, 2I, 3I, ... until SBDB has no further object."""
    rows: list[dict[str, object]] = []
    for number in range(1, 16):
        label = f"{number}I"
        cache = raw_dir / f"sbdb_{label}.json"
        payload: dict[str, Any] | None
        if cache.is_file() and cache.stat().st_size > 0 and not refresh:
            payload = _load_json(cache.read_bytes())
        else:
            payload = download_json_optional(f"{SBDB_API_URL}?sstr={label}")
            if payload is None or "object" not in payload:
                break
            cache.write_bytes(json.dumps(payload).encode("utf-8"))
        if payload is None or "object" not in payload:
            break
        obj = payload.get("object")
        if not isinstance(obj, dict):
            break
        orbit_raw = payload.get("orbit")
        orbit = orbit_raw if isinstance(orbit_raw, dict) else {}
        class_raw = obj.get("orbit_class")
        orbit_class = class_raw if isinstance(class_raw, dict) else {}
        rows.append(
            {
                "interstellar_id": label,
                "designation": obj.get("des"),
                "name": str(obj.get("fullname") or "").strip(),
                "orbit_class": orbit_class.get("code"),
                "first_obs": orbit.get("first_obs"),
            }
        )
    frame = pd.DataFrame(rows)
    destination = resolve_repo_path(INTERSTELLAR_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def _interstellar_keys(frame: pd.DataFrame) -> set[str]:
    keys: set[str] = set()
    if frame.empty:
        return keys
    for column in ("interstellar_id", "designation", "name"):
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            key = normalize_designation(value)
            if key:
                keys.add(key)
    return keys


def _interstellar_approaches(
    raw_dir: Path,
    catalog: pd.DataFrame,
    *,
    start: str,
    end: str,
    refresh: bool,
) -> list[pd.DataFrame]:
    """Approaches of each nI object out to 1 AU, tried under every alias."""
    frames: list[pd.DataFrame] = []
    if catalog.empty:
        return frames
    for _, row in catalog.iterrows():
        aliases = {
            str(row.get("interstellar_id") or ""),
            str(row.get("designation") or ""),
        }
        for alias in aliases:
            if not alias or alias == "None":
                continue
            try:
                frame = _collect_cad(
                    raw_dir,
                    body="Earth",
                    start=start,
                    end=end,
                    dist_max="1",
                    refresh=refresh,
                    designation=alias,
                )
            except (RuntimeError, ValueError, urllib.error.URLError):
                continue
            if not frame.empty:
                frames.append(frame)
    return frames


def _fireballs(raw_dir: Path, *, refresh: bool) -> pd.DataFrame:
    url = f"{FIREBALL_API_URL}?{urllib.parse.urlencode({'limit': '100000'})}"
    payload = _load_json(
        cached_download(url, raw_dir / "fireball.json", refresh=refresh)
    )
    return fireball_events(payload)


def _sentry(raw_dir: Path, *, refresh: bool) -> dict[str, Any]:
    payload = _load_json(
        cached_download(SENTRY_API_URL, raw_dir / "sentry.json", refresh=refresh)
    )
    summary = sentry_summary(payload)
    destination = resolve_repo_path(SENTRY_PATH)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _satcat(raw_dir: Path, *, refresh: bool) -> pd.DataFrame:
    text = _decode(cached_download(SATCAT_URL, raw_dir / "satcat.csv", refresh=refresh))
    table = pd.read_csv(io.StringIO(text))
    return satcat_decays(table)


def _write_upcoming(
    events: pd.DataFrame, path: Path, *, today: pd.Timestamp, date_max: str
) -> None:
    stop = pd.Timestamp(date_max) + pd.Timedelta(days=1)
    chosen = events.loc[events["cd"] >= today].copy()
    chosen = chosen.loc[chosen["cd"] < stop]
    search = chosen["dist_au"] <= CLOSE_AU + 1e-9
    extra = chosen["kind"].eq("interstellar")
    chosen = chosen.loc[search | extra]
    output = pd.DataFrame(
        {
            "body": chosen["body"],
            "designation": chosen["designation"],
            "name": chosen["name"],
            "date": chosen["cd"].dt.strftime("%Y-%m-%dT%H:%M"),
            "dist_au": chosen["dist_au"],
            "dist_ld": chosen["dist_ld"],
            "h": chosen["h"],
            "kind": chosen["kind"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def _write_fireball_events(events: pd.DataFrame, path: Path) -> None:
    output = events.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def _write_reentry_events(
    events: pd.DataFrame, path: Path, *, today: pd.Timestamp
) -> None:
    recent = events.loc[events["date"] >= today - pd.Timedelta(days=120)].copy()
    recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    recent.loc[:, ["name", "norad_id", "date"]].to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. See ``python -m src.space.near_earth --help``."""
    parser = argparse.ArgumentParser(
        description=(
            "Download Earth and Moon close approaches, fireballs, Sentry, "
            "and SATCAT decay dates into data/space/."
        )
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-dir", default="data/space")
    args = parser.parse_args(argv)
    summary = fetch_near_earth(refresh=args.refresh, output_dir=args.output_dir)
    for key, value in summary.items():
        if key == "sources":
            continue
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
