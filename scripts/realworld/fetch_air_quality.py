"""Daily PM2.5, ozone, and pollen for Berlin from the Open-Meteo air-quality API.

No API key. Pollen coverage is the Copernicus European domain, so the city
is Berlin rather than a US city. Values are daily means of the hourly series
in Europe/Berlin local time.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from urllib.parse import urlencode

from scripts.realworld.download import fetch_bytes
from src.realworld.parsers import POLLEN_VARIABLES, daily_air_quality
from src.realworld.tabular import write_csv
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

LATITUDE = 52.52
LONGITUDE = 13.41
START = date(2022, 1, 1)


def fetch_hourly(start: date, end: date) -> dict[str, object]:
    """One Open-Meteo request. Raises if the body is not the hourly payload."""
    variables = ["pm2_5", "ozone", *POLLEN_VARIABLES]
    query = urlencode(
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "timezone": "Europe/Berlin",
            "hourly": ",".join(variables),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
    )
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?{query}"
    payload = json.loads(fetch_bytes(url))
    if "hourly" not in payload:
        reason = payload.get("reason", payload)
        raise RuntimeError(f"Open-Meteo refused {start} to {end}: {reason}")
    return payload


def fetch_daily(start: date, end: date) -> list[dict[str, float | str]]:
    """Fetch in calendar-year chunks and concatenate daily means."""
    rows: list[dict[str, float | str]] = []
    year = start.year
    while year <= end.year:
        chunk_start = date(year, 1, 1) if year > start.year else start
        chunk_end = date(year, 12, 31) if year < end.year else end
        logger.info("air quality %s → %s", chunk_start, chunk_end)
        rows.extend(daily_air_quality(fetch_hourly(chunk_start, chunk_end)))
        year += 1
    deduped: dict[str, dict[str, float | str]] = {}
    for row in rows:
        deduped[str(row["date"])] = row
    return [deduped[key] for key in sorted(deduped)]


def main() -> None:
    """Write combined and univariate daily files through yesterday's month."""
    end = date.today() - timedelta(days=1)
    rows = fetch_daily(START, end)
    out_dir = REPO_ROOT / "data" / "realworld"
    write_csv(
        out_dir / "air_berlin_daily.csv", ["date", "pm2_5", "ozone", "pollen"], rows
    )
    write_csv(
        out_dir / "air_pm25_berlin.csv",
        ["date", "value"],
        [{"date": row["date"], "value": row["pm2_5"]} for row in rows],
    )
    write_csv(
        out_dir / "air_ozone_berlin.csv",
        ["date", "value"],
        [{"date": row["date"], "value": row["ozone"]} for row in rows],
    )
    pollen = [
        {"date": row["date"], "value": row["pollen"]} for row in rows if "pollen" in row
    ]
    write_csv(out_dir / "air_pollen_berlin.csv", ["date", "value"], pollen)
    logger.info(
        "Berlin air quality: %d days, pollen days %d, %s → %s",
        len(rows),
        len(pollen),
        rows[0]["date"] if rows else "none",
        rows[-1]["date"] if rows else "none",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
