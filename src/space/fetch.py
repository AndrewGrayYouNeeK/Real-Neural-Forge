"""Download the outer-space series into ``data/space/``.

Run from the repository root:

    python -m scripts.fetch_space

The same command is ``python -m src.space.fetch``. Raw responses and the
processed CSVs live under ``data/`` and are not committed.

The geomagnetic file also carries the SILSO international sunspot number.
That column is ignored here; another dataset batch forecasts the SILSO
monthly sunspot series itself.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.space.parse import (
    align_on_dates,
    parse_close_approaches,
    parse_exoplanet_years,
    parse_gcat_launches,
    parse_goes_xrs_reports,
    parse_neo_discoveries,
    parse_omni_daily,
    parse_space_weather_csv,
    regularize_daily,
    trim_open_period,
)
from src.utils.paths import resolve_repo_path

USER_AGENT = "real-neural-forge space fetch"
CELESTRAK_URL = "https://celestrak.org/SpaceData/SW-All.csv"
CELESTRAK_FORMAT_URL = "https://celestrak.org/SpaceData/SpaceWx-format.php"
GFZ_KP_URL = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
OMNI_DAILY_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni_01_av.dat"
OMNI_FORMAT_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text"
XRS_BASE_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
    "solar-features/solar-flares/x-rays/goes/xrs/"
)
CAD_API_URL = "https://ssd-api.jpl.nasa.gov/cad.api"
SBDB_QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
EXOPLANET_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
GCAT_LAUNCH_URL = "https://planet4589.org/space/gcat/tsv/launch/launch.tsv"
GCAT_COLUMNS_URL = "https://planet4589.org/space/gcat/web/launch/lcols.html"
GCAT_HOME_URL = "https://planet4589.org/space/gcat"

# Apollo, Amor, Aten, and Atira (IEO). These four classes are the NEOs.
NEO_ORBIT_CLASSES = ("APO", "AMO", "ATE", "IEO")
CLOSE_APPROACH_AU = "0.05"
MIN_PLASMA_HOURS = 4

LICENCE_KP = (
    "Kp, ap, and Ap: GFZ Helmholtz Centre for Geosciences, CC BY 4.0 "
    "(Matzka et al., 2021, doi:10.5880/Kp.0001). The GFZ host did not answer "
    f"from this environment ({GFZ_KP_URL}); the bytes are the CelesTrak "
    f"SW-All.csv redistribution ({CELESTRAK_URL}), which documents that GFZ "
    f"file as its geomagnetic source ({CELESTRAK_FORMAT_URL}). "
    "Cite CelesTrak as well as GFZ. The ISN column in the same file is the "
    "SILSO sunspot number and is not used."
)
LICENCE_F107 = (
    "Observed 10.7 cm solar radio flux (F10.7_OBS, sfu) from the Penticton "
    "observatory, as carried in the same CelesTrak SW-All.csv file. "
    "Measurements: Natural Resources Canada, Dominion Radio Astrophysical "
    "Observatory. Redistributed by NOAA SWPC (U.S. government work, public "
    "domain) and by CelesTrak. Days flagged INT are the file's own short "
    "interpolations; PRD and PRM predictions are excluded."
)
LICENCE_OMNI = (
    "NASA OMNI2 daily averages (omni_01_av.dat), public domain as a U.S. "
    "government work. Cite King and Papitashvili, NASA/GSFC OMNI, "
    f"{OMNI_DAILY_URL}. Dst in this file is the daily mean of the Kyoto WDC "
    "index (World Data Center for Geomagnetism, Kyoto), not the daily minimum."
)
LICENCE_FLARES = (
    "NOAA NCEI GOES XRS flare event reports, 1975 through the 2017 "
    "year-to-date file. Public domain (U.S. government work). "
    f"Directory: {XRS_BASE_URL}"
)
LICENCE_CAD = (
    "NASA/JPL SBDB Close-Approach Data API, public domain U.S. government "
    f"work. {CAD_API_URL}. Counts are approaches within {CLOSE_APPROACH_AU} AU "
    "of NEOs currently in the database, including approaches before discovery."
)
LICENCE_SBDB = (
    "NASA/JPL Small-Body Database Query API, public domain U.S. government "
    f"work. {SBDB_QUERY_URL}. Monthly counts use first_obs for NEO classes "
    "APO, AMO, ATE, and IEO. That date mostly reflects survey effort."
)
LICENCE_GCAT = (
    "Jonathan C. McDowell, General Catalog of Artificial Space Objects "
    f"(GCAT), Creative Commons CC-BY. {GCAT_HOME_URL}. "
    "Short citation requested by the catalog: "
    "'data from GCAT (J. McDowell, planet4589.org/space/gcat)'."
)
LICENCE_EXOPLANETS = (
    "NASA Exoplanet Archive TAP (Caltech/IPAC), composite planetary systems "
    f"table pscomppars. {EXOPLANET_TAP_URL}. Free to use; cite the NASA "
    "Exoplanet Archive. Counts are planets per discovery year, so a catalog "
    "release can put hundreds of Kepler planets in a single year."
)

SOURCES: tuple[dict[str, str], ...] = (
    {
        "id": "kp",
        "name": "GFZ Kp and Ap via CelesTrak SW-All.csv",
        "url": CELESTRAK_URL,
        "licence": LICENCE_KP,
    },
    {
        "id": "f107",
        "name": "Penticton/NOAA F10.7 via CelesTrak SW-All.csv",
        "url": CELESTRAK_URL,
        "licence": LICENCE_F107,
    },
    {
        "id": "omni",
        "name": "NASA OMNI2 daily solar wind and Dst",
        "url": OMNI_DAILY_URL,
        "licence": LICENCE_OMNI,
    },
    {
        "id": "flares",
        "name": "NOAA NCEI GOES XRS flare reports",
        "url": XRS_BASE_URL,
        "licence": LICENCE_FLARES,
    },
    {
        "id": "approaches",
        "name": "JPL SBDB close-approach API",
        "url": CAD_API_URL,
        "licence": LICENCE_CAD,
    },
    {
        "id": "discoveries",
        "name": "JPL SBDB NEO first-observation dates",
        "url": SBDB_QUERY_URL,
        "licence": LICENCE_SBDB,
    },
    {
        "id": "launches",
        "name": "McDowell GCAT orbital launch attempts",
        "url": GCAT_LAUNCH_URL,
        "licence": LICENCE_GCAT,
    },
    {
        "id": "exoplanets",
        "name": "NASA Exoplanet Archive discoveries per year",
        "url": EXOPLANET_TAP_URL,
        "licence": LICENCE_EXOPLANETS,
    },
)


def xrs_report_names() -> list[str]:
    """Yearly GOES XRS report filenames, including the partial 2017 file."""
    names = [f"goes-xrs-report_{year}.txt" for year in range(1975, 2017)]
    names.append("goes-xrs-report_2017-ytd.txt")
    return names


def cad_url(start: str, end: str) -> str:
    """Close approaches between two dates, within 0.05 AU."""
    query = urllib.parse.urlencode(
        {
            "date-min": start,
            "date-max": end,
            "dist-max": CLOSE_APPROACH_AU,
            "limit": "100000",
            "fullname": "false",
        }
    )
    return f"{CAD_API_URL}?{query}"


def sbdb_url(orbit_class: str) -> str:
    """One NEO orbit class, with designation and first observation."""
    query = urllib.parse.urlencode(
        {
            "fields": "pdes,first_obs",
            "sb-class": orbit_class,
            "limit": "100000",
        }
    )
    return f"{SBDB_QUERY_URL}?{query}"


def exoplanet_url() -> str:
    """Aggregated discovery-year counts from the composite planet table."""
    query = urllib.parse.urlencode(
        {
            "query": (
                "select disc_year, count(*) as n from pscomppars "
                "where disc_year is not null "
                "group by disc_year order by disc_year"
            ),
            "format": "csv",
        }
    )
    return f"{EXOPLANET_TAP_URL}?{query}"


def download_bytes(url: str, *, timeout: float = 180.0, attempts: int = 4) -> bytes:
    """GET ``url``, retrying transient network errors."""
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return bytes(response.read())
        except urllib.error.URLError as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}: {last}") from last


def cached_download(url: str, destination: Path, *, refresh: bool) -> bytes:
    """Return ``destination``'s bytes, downloading when missing or refreshed."""
    if destination.is_file() and destination.stat().st_size > 0 and not refresh:
        return destination.read_bytes()
    payload = download_bytes(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return payload


def write_series(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a series CSV with ``YYYY-MM-DD`` dates."""
    destination = resolve_repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output.to_csv(destination, index=False)
    return destination


def _require_range(name: str, value: float, low: float, high: float) -> None:
    """Fail the fetch when a parsed series is not in a physically plausible band."""
    if not low <= value <= high:
        raise RuntimeError(
            f"{name} median/total {value} is outside {low}..{high}; the parser "
            "likely read the wrong column."
        )


def _summary(frame: pd.DataFrame, **extra: object) -> dict[str, object]:
    dates = pd.to_datetime(frame["date"])
    record: dict[str, object] = {
        "rows": int(len(frame)),
        "start": dates.min().strftime("%Y-%m-%d"),
        "end": dates.max().strftime("%Y-%m-%d"),
    }
    record.update(extra)
    return record


def _decode(payload: bytes) -> str:
    return payload.decode("utf-8-sig", errors="replace")


def _load_json(payload: bytes) -> dict[str, Any]:
    parsed = json.loads(_decode(payload))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _collect_approaches(raw_dir: Path, *, refresh: bool) -> dict[str, Any]:
    """Download close approaches, splitting the window if the API truncates."""

    field_names: list[str] = []

    def collect(start: str, end: str) -> list[list[object]]:
        url = cad_url(start, end)
        cache = raw_dir / f"cad_{start}_{end}.json"
        payload = _load_json(cached_download(url, cache, refresh=refresh))
        fields = payload.get("fields")
        data = payload.get("data")
        total = int(payload.get("total", payload.get("count", 0)))
        if not isinstance(fields, list) or not isinstance(data, list):
            raise ValueError("CAD response is missing fields or data.")
        if not field_names:
            field_names.extend(str(field) for field in fields)
        if len(data) >= total:
            return data
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if (end_ts - start_ts).days <= 1:
            raise RuntimeError(
                f"CAD API truncated {start}..{end} ({len(data)} of {total})."
            )
        mid = start_ts + (end_ts - start_ts) / 2
        left_end = (mid - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        right_start = mid.strftime("%Y-%m-%d")
        return collect(start, left_end) + collect(right_start, end)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    rows = collect("1900-01-01", today)
    if "cd" not in field_names:
        raise ValueError("CAD response has no cd column.")
    cd_index = field_names.index("cd")
    return {"fields": ["cd"], "data": [[row[cd_index]] for row in rows]}


def fetch_space(
    *, refresh: bool = False, output_dir: str | Path = "data/space"
) -> dict[str, object]:
    """Download every series and write the processed CSVs plus a manifest."""
    root = resolve_repo_path(output_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"sources": list(SOURCES)}

    weather_bytes = cached_download(
        CELESTRAK_URL, raw_dir / "SW-All.csv", refresh=refresh
    )
    kp_raw, f107_raw = parse_space_weather_csv(_decode(weather_bytes))
    kp, kp_filled = regularize_daily(kp_raw, ["kp_max", "ap"], max_gap_days=3)
    f107, f107_filled = regularize_daily(f107_raw, ["f107"], max_gap_days=3)
    kp = trim_open_period(kp, "D")
    f107 = trim_open_period(f107, "D")
    _require_range("kp_max", float(kp["kp_max"].median()), 0.5, 6.0)
    _require_range("f107", float(f107["f107"].median()), 60.0, 250.0)
    write_series(kp, root / "kp_daily.csv")
    write_series(f107, root / "f107_daily.csv")
    manifest["kp"] = _summary(kp, interpolated_days=kp_filled)
    manifest["f107"] = _summary(f107, interpolated_days=f107_filled)

    omni_bytes = cached_download(
        OMNI_DAILY_URL, raw_dir / "omni_01_av.dat", refresh=refresh
    )
    wind_raw, dst_raw = parse_omni_daily(
        _decode(omni_bytes), min_plasma_hours=MIN_PLASMA_HOURS
    )
    wind, wind_filled = regularize_daily(wind_raw, ["speed", "density"], max_gap_days=5)
    dst, dst_filled = regularize_daily(dst_raw, ["dst"], max_gap_days=3)
    wind = trim_open_period(wind, "D")
    dst = trim_open_period(dst, "D")
    _require_range("solar-wind speed", float(wind["speed"].median()), 250.0, 800.0)
    _require_range("solar-wind density", float(wind["density"].median()), 0.5, 30.0)
    _require_range("dst", float(dst["dst"].median()), -50.0, 20.0)
    write_series(wind, root / "solar_wind_daily.csv")
    write_series(dst, root / "dst_daily.csv")
    manifest["solar_wind"] = _summary(
        wind, interpolated_days=wind_filled, min_plasma_hours=MIN_PLASMA_HOURS
    )
    manifest["dst"] = _summary(dst, interpolated_days=dst_filled)

    joined = align_on_dates(
        kp.loc[:, ["date", "kp_max"]],
        wind.loc[:, ["date", "speed", "density"]],
    )
    joined = trim_open_period(joined, "D")
    write_series(joined, root / "kp_solar_wind.csv")
    manifest["kp_solar_wind"] = _summary(joined)

    flare_texts: list[str] = []
    for name in xrs_report_names():
        payload = cached_download(
            XRS_BASE_URL + name, raw_dir / "xrs" / name, refresh=refresh
        )
        flare_texts.append(_decode(payload))
    flares = parse_goes_xrs_reports(flare_texts)
    flares = trim_open_period(flares, "MS")
    flare_total = int(flares[["c_class", "m_class", "x_class"]].to_numpy().sum())
    _require_range("GOES C/M/X flares", float(flare_total), 1000.0, 500_000.0)
    write_series(flares, root / "flares_monthly.csv")
    manifest["flares"] = _summary(flares)

    approaches = parse_close_approaches(_collect_approaches(raw_dir, refresh=refresh))
    approaches = trim_open_period(approaches, "MS")
    _require_range(
        "close approaches", float(approaches["approaches"].sum()), 1000.0, 200_000.0
    )
    write_series(approaches, root / "neo_approaches_monthly.csv")
    manifest["neo_approaches"] = _summary(
        approaches, dist_max_au=float(CLOSE_APPROACH_AU)
    )

    discovery_frames = []
    for orbit_class in NEO_ORBIT_CLASSES:
        payload = cached_download(
            sbdb_url(orbit_class),
            raw_dir / f"sbdb_{orbit_class}.json",
            refresh=refresh,
        )
        body = _load_json(payload)
        fields = body.get("fields")
        data = body.get("data")
        count = int(body.get("count", 0))
        if not isinstance(data, list) or not isinstance(fields, list):
            raise ValueError(f"SBDB {orbit_class} response is missing rows.")
        if len(data) < count:
            raise RuntimeError(
                f"SBDB {orbit_class} returned {len(data)} of {count} rows."
            )
        discovery_frames.append(
            pd.DataFrame(data, columns=[str(field) for field in fields])
        )
    discoveries = parse_neo_discoveries(discovery_frames)
    discoveries = trim_open_period(discoveries, "MS")
    _require_range(
        "NEO first observations",
        float(discoveries["discoveries"].sum()),
        1000.0,
        200_000.0,
    )
    write_series(discoveries, root / "neo_discoveries_monthly.csv")
    manifest["neo_discoveries"] = _summary(
        discoveries, orbit_classes=list(NEO_ORBIT_CLASSES)
    )

    launch_bytes = cached_download(
        GCAT_LAUNCH_URL, raw_dir / "launch.tsv", refresh=refresh
    )
    launches = parse_gcat_launches(_decode(launch_bytes))
    launches = trim_open_period(launches, "MS")
    _require_range(
        "orbital launch attempts", float(launches["attempts"].sum()), 5000.0, 20000.0
    )
    write_series(launches, root / "launches_monthly.csv")
    manifest["launches"] = _summary(launches, column_notes=GCAT_COLUMNS_URL)

    exoplanet_bytes = cached_download(
        exoplanet_url(), raw_dir / "exoplanets.csv", refresh=refresh
    )
    exoplanets = parse_exoplanet_years(_decode(exoplanet_bytes))
    exoplanets = trim_open_period(exoplanets, "YS")
    _require_range(
        "exoplanet discoveries", float(exoplanets["discoveries"].sum()), 1000.0, 20000.0
    )
    write_series(exoplanets, root / "exoplanets_yearly.csv")
    manifest["exoplanets"] = _summary(exoplanets)

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. See ``python -m src.space.fetch --help``."""
    parser = argparse.ArgumentParser(
        description=(
            "Download outer-space time series into data/space/. "
            "Sources are keyless. Raw files are not committed."
        )
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download again even when a raw file is already cached.",
    )
    parser.add_argument("--output-dir", default="data/space")
    args = parser.parse_args(argv)
    manifest = fetch_space(refresh=args.refresh, output_dir=args.output_dir)
    for key, value in manifest.items():
        if key == "sources":
            continue
        print(f"{key}: {value}")
    print("Licences:")
    for source in SOURCES:
        print(f"- {source['name']}: {source['licence']}")


if __name__ == "__main__":
    main()
