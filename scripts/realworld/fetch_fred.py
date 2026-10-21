"""Keyless FRED graph CSV downloads for unemployment, CPI inflation, and gas.

fredgraph.csv does not need an API key. CPI inflation is the 12-month percent
change of CPIAUCNS (CPI-U, all items, not seasonally adjusted), which is the
usual headline year-over-year rate.
"""

from __future__ import annotations

import logging

from scripts.realworld.download import fetch_bytes
from src.realworld.parsers import parse_fred_csv, year_over_year_percent
from src.realworld.tabular import write_csv
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

SERIES = {
    "UNRATE": "fred_unemployment.csv",
    "CPIAUCNS": "fred_cpi_index.csv",
    "GASREGW": "fred_gas.csv",
}


def fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def main() -> None:
    """Write unemployment, headline CPI inflation, and weekly regular gas price."""
    out_dir = REPO_ROOT / "data" / "realworld"
    frames = {
        series_id: parse_fred_csv(fetch_bytes(fred_url(series_id)).decode("utf-8"))
        for series_id in SERIES
    }
    unemployment = frames["UNRATE"]
    gas = frames["GASREGW"]
    inflation = year_over_year_percent(frames["CPIAUCNS"])
    write_csv(
        out_dir / "fred_unemployment.csv",
        ["date", "value"],
        [{"date": day.isoformat(), "value": value} for day, value in unemployment],
    )
    write_csv(
        out_dir / "fred_cpi_inflation.csv",
        ["date", "value"],
        [{"date": day.isoformat(), "value": value} for day, value in inflation],
    )
    write_csv(
        out_dir / "fred_gas.csv",
        ["date", "value"],
        [{"date": day.isoformat(), "value": value} for day, value in gas],
    )
    logger.info(
        "FRED: unemployment %d, CPI inflation %d, gas %d",
        len(unemployment),
        len(inflation),
        len(gas),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
