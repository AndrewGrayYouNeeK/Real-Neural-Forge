"""Daily global counts of USGS earthquakes at magnitude 4.5 and above.

The FDSN event service is public and needs no key. Each year is requested
separately so a single response stays under the service's row cap. The
training target is the daily count. ``energy_rel`` is stored alongside it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from urllib.parse import urlencode

from scripts.realworld.download import fetch_bytes
from src.realworld.parsers import daily_earthquake_rows, parse_usgs_event_csv
from src.realworld.tabular import write_csv
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

START = date(1990, 1, 1)
MIN_MAGNITUDE = 4.5
ROW_CAP = 20000


def catalog_url(start: date, end: date) -> str:
    """Half-open ``[start, end)`` query. ``end`` is exclusive via the day before."""
    query = urlencode(
        {
            "format": "csv",
            "starttime": start.isoformat(),
            "endtime": end.isoformat(),
            "minmagnitude": MIN_MAGNITUDE,
            "orderby": "time-asc",
        }
    )
    return f"https://earthquake.usgs.gov/fdsnws/event/1/query?{query}"


def fetch_events(start: date, end: date) -> list[tuple[datetime, float]]:
    """Download one window, splitting it if the service hits its row cap."""
    text = fetch_bytes(catalog_url(start, end), timeout=300).decode("utf-8")
    events = [
        event for event in parse_usgs_event_csv(text) if start <= event[0].date() < end
    ]
    if len(events) >= ROW_CAP and (end - start).days > 31:
        mid = start + timedelta(days=(end - start).days // 2)
        logger.info("splitting %s → %s (%d rows)", start, end, len(events))
        return fetch_events(start, mid) + fetch_events(mid, end)
    logger.info("USGS %s → %s: %d events", start, end, len(events))
    return events


def year_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar-year slices covering ``[start, end]`` inclusive."""
    windows: list[tuple[date, date]] = []
    year = start.year
    while year <= end.year:
        window_start = date(year, 1, 1) if year > start.year else start
        window_end = date(year + 1, 1, 1)
        if year == end.year:
            window_end = end + timedelta(days=1)
        windows.append((window_start, window_end))
        year += 1
    return windows


def main() -> None:
    """Write one row per UTC day from 1990 through yesterday."""
    end = date.today() - timedelta(days=1)
    windows = year_windows(START, end)
    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(
            pool.map(lambda window: fetch_events(window[0], window[1]), windows)
        )
    events = [event for batch in batches for event in batch]
    rows = daily_earthquake_rows(events, START, end)
    path = REPO_ROOT / "data" / "realworld" / "earthquakes_m45_daily.csv"
    write_csv(path, ["date", "value", "energy_rel", "log10_energy"], rows)
    total = sum(int(row["value"]) for row in rows)
    logger.info(
        "earthquakes: %d days, %d events, %s → %s", len(rows), total, START, end
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
