"""Download the Central Park daily maximum temperature series.

The file is a U.S. government work, so it is in the public domain (17 U.S.C.
§ 105) and may be copied and redistributed. NOAA National Centers for
Environmental Information (NCEI) distribute it through the Daily Summaries
access service with no use fee.

Station ``USW00094728`` (New York Central Park), variable ``TMAX``, metric
units (degrees Celsius), 1981-01-01 through 2025-12-31. Scientific background
for the underlying GHCNd archive: Menne et al., 2012, *Journal of Atmospheric
and Oceanic Technology*.

The CSV is written under ``data/`` and is not committed (that directory is
gitignored).
"""

from __future__ import annotations

import argparse
import io
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import resolve_repo_path

STATION_ID = "USW00094728"
STATION_NAME = "NEW YORK CENTRAL PARK, NY US"
START_DATE = "1981-01-01"
END_DATE = "2025-12-31"
SOURCE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
LICENCE = (
    "Public domain (U.S. government work, 17 U.S.C. § 105). "
    "NOAA NCEI Daily Summaries for GHCND station USW00094728, "
    "variable TMAX, degrees Celsius. Redistribution is permitted."
)
DEFAULT_OUTPUT = Path("data/weather_central_park.csv")
# Mean tropical year, so the calendar features stay aligned across leap years.
_YEAR_LENGTH_DAYS = 365.25
_NOAA_COLUMNS = ("STATION", "DATE", "TMAX")


def build_download_url(
    *,
    station: str = STATION_ID,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    data_type: str = "TMAX",
) -> str:
    """Return the NCEI Daily Summaries URL for one station and variable."""
    query = urllib.parse.urlencode(
        {
            "dataset": "daily-summaries",
            "stations": station,
            "startDate": start_date,
            "endDate": end_date,
            "dataTypes": data_type,
            "units": "metric",
            "format": "csv",
            "includeStationName": "false",
        }
    )
    return f"{SOURCE_URL}?{query}"


def download_noaa_csv(url: str, timeout: float = 120.0) -> str:
    """Download a Daily Summaries response as text."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "real-neural-forge weather fetch"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download NOAA daily summaries: {exc}") from exc
    return bytes(raw).decode("utf-8")


def assert_regular_daily(dates: pd.Series) -> None:
    """Reject empty, duplicate, or gapped daily indexes.

    Gaps are not interpolated: a filled day would be an invented observation.
    """
    if dates.empty:
        raise ValueError("Weather series is empty.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Weather dates must be sorted ascending.")
    if bool(dates.duplicated().any()):
        raise ValueError("Weather series contains duplicate dates.")
    deltas = dates.diff().iloc[1:]
    gap_count = int((deltas != pd.Timedelta(days=1)).sum())
    if gap_count:
        span = int((dates.iloc[-1] - dates.iloc[0]).days) + 1
        missing = span - int(len(dates))
        raise ValueError(
            f"Weather series is not continuous "
            f"({missing} missing day(s) across {gap_count} gap(s)). "
            "Refusing to interpolate."
        )


def parse_noaa_daily_summaries(
    csv_text: str, *, station: str = STATION_ID
) -> pd.DataFrame:
    """Parse an NCEI Daily Summaries CSV into ``date`` and ``temperature_c``."""
    frame = pd.read_csv(io.StringIO(csv_text))
    missing = [column for column in _NOAA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"NOAA daily summaries response is missing columns: {missing}")
    if frame.empty:
        raise ValueError("NOAA daily summaries response contained no observations.")
    stations = set(frame["STATION"].astype(str))
    if stations != {station}:
        raise ValueError(f"Expected only station {station}, found {sorted(stations)}")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["DATE"], format="%Y-%m-%d"),
            "temperature_c": pd.to_numeric(frame["TMAX"], errors="coerce"),
        }
    )
    if bool(parsed["temperature_c"].isna().any()):
        raise ValueError("NOAA daily summaries response contains missing TMAX values.")
    parsed = parsed.sort_values("date").reset_index(drop=True)
    assert_regular_daily(parsed["date"])
    return parsed


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine of day-of-year.

    ``angle = 2π * day_of_year / 365.25``. These are functions of the date, so
    they are known before that day's temperature is observed.
    """
    day_of_year = frame["date"].dt.dayofyear.to_numpy(dtype=np.float64)
    angle = 2.0 * np.pi * day_of_year / _YEAR_LENGTH_DAYS
    enriched = frame.copy()
    enriched["doy_sin"] = np.sin(angle)
    enriched["doy_cos"] = np.cos(angle)
    return enriched


def write_weather_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the training CSV. Relative paths resolve from the repo root."""
    destination = resolve_repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    to_write = frame.copy()
    to_write["date"] = pd.to_datetime(to_write["date"]).dt.strftime("%Y-%m-%d")
    to_write.to_csv(destination, index=False)
    return destination


_METRIC_COLUMN_NAMES = {
    "TMAX": "tmax_c",
    "TMIN": "tmin_c",
    "PRCP": "prcp_mm",
}


def build_station_url(
    *,
    station: str,
    start_date: str,
    end_date: str,
    data_types: tuple[str, ...] = ("TMAX",),
) -> str:
    """Daily Summaries URL for one station and one or more GHCNd variables.

    ``data_types`` use the NCEI names (``TMAX``, ``TMIN``, ``PRCP``). Metric
    units are degrees Celsius and millimetres.
    """
    if not data_types:
        raise ValueError("data_types must name at least one GHCNd variable.")
    query = urllib.parse.urlencode(
        {
            "dataset": "daily-summaries",
            "stations": station,
            "startDate": start_date,
            "endDate": end_date,
            "dataTypes": ",".join(data_types),
            "units": "metric",
            "format": "csv",
            "includeStationName": "false",
        },
        safe=",",
    )
    return f"{SOURCE_URL}?{query}"


def parse_station_observations(
    csv_text: str,
    *,
    station: str,
    data_types: tuple[str, ...] = ("TMAX",),
) -> pd.DataFrame:
    """Parse a multi-variable Daily Summaries CSV.

    Dates may contain gaps. Missing measurements stay as NaN; callers decide
    whether a gap can be carried forward. Column names are ``date`` plus
    ``tmax_c``, ``tmin_c``, and ``prcp_mm`` for the requested variables.
    """
    if not data_types:
        raise ValueError("data_types must name at least one GHCNd variable.")
    unknown = [name for name in data_types if name not in _METRIC_COLUMN_NAMES]
    if unknown:
        raise ValueError(f"Unsupported GHCNd variables: {unknown}")
    frame = pd.read_csv(io.StringIO(csv_text))
    required = ["STATION", "DATE", *data_types]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"NOAA daily summaries response is missing columns: {missing}")
    if frame.empty:
        raise ValueError("NOAA daily summaries response contained no observations.")
    stations = set(frame["STATION"].astype(str))
    if stations != {station}:
        raise ValueError(f"Expected only station {station}, found {sorted(stations)}")

    parsed = pd.DataFrame({"date": pd.to_datetime(frame["DATE"], format="%Y-%m-%d")})
    for data_type in data_types:
        parsed[_METRIC_COLUMN_NAMES[data_type]] = pd.to_numeric(
            frame[data_type], errors="coerce"
        )
    parsed = parsed.sort_values("date").reset_index(drop=True)
    if bool(parsed["date"].duplicated().any()):
        raise ValueError("Weather series contains duplicate dates.")
    if not bool(parsed["date"].is_monotonic_increasing):
        raise ValueError("Weather dates must be sorted ascending.")
    return parsed


def fetch_station_observations(
    *,
    station: str,
    data_types: tuple[str, ...],
    start_date: str,
    end_date: str,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Download one station's daily observations from NOAA NCEI."""
    url = build_station_url(
        station=station,
        start_date=start_date,
        end_date=end_date,
        data_types=data_types,
    )
    return parse_station_observations(
        download_noaa_csv(url, timeout=timeout),
        station=station,
        data_types=data_types,
    )


def fetch_weather(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    station: str = STATION_ID,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    timeout: float = 120.0,
) -> Path:
    """Download the series and write ``date,temperature_c,doy_sin,doy_cos``."""
    url = build_download_url(station=station, start_date=start_date, end_date=end_date)
    frame = add_calendar_features(
        parse_noaa_daily_summaries(
            download_noaa_csv(url, timeout=timeout), station=station
        )
    )
    return write_weather_csv(frame, output)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. See ``python -m src.data.weather --help``."""
    parser = argparse.ArgumentParser(
        description=(
            "Download NOAA NCEI daily maximum temperature for Central Park "
            f"({STATION_ID}) into a CSV. {LICENCE}"
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Destination CSV (default: data/weather_central_park.csv)",
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--station", default=STATION_ID)
    args = parser.parse_args(argv)
    path = fetch_weather(
        args.output,
        station=args.station,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    frame = pd.read_csv(path)
    print(f"Wrote {len(frame)} rows to {path}")
    print(f"Station: {args.station} ({STATION_NAME})")
    print(f"Source: {SOURCE_URL}")
    print(f"Licence: {LICENCE}")


if __name__ == "__main__":
    main()
