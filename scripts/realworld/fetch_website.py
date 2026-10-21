"""Import daily visit counts from a CSV, or from Cloudflare when configured.

Training and CI use ``examples/realworld/website_visits.csv``, a synthetic
series with a weekly pattern. That file is not a real site. A Cloudflare
fetch runs only when ``CLOUDFLARE_API_TOKEN`` is set together with either
``CLOUDFLARE_ACCOUNT_ID`` (Web Analytics page loads, optional
``CLOUDFLARE_SITE_TAG``) or ``CLOUDFLARE_ZONE_ID`` (GraphQL zone analytics:
unique visitors, or requests if uniques are absent). The live download is
written to ``data/realworld/website_visits.csv`` and is not required.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.realworld.tabular import write_csv
from src.realworld.website import (
    read_visit_csv,
    synthetic_daily_visits,
    visits_from_env,
)
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

EXAMPLE = REPO_ROOT / "examples" / "realworld" / "website_visits.csv"


def write_example(path: Path = EXAMPLE) -> None:
    """Rewrite the committed example from the fixed synthetic generator."""
    rows = [
        {"date": day.isoformat(), "value": value}
        for day, value in synthetic_daily_visits()
    ]
    write_csv(path, ["date", "value"], rows)


def main() -> None:
    """Fetch Cloudflare visits when credentials exist. Always check the example."""
    if not EXAMPLE.is_file():
        write_example()
    example_rows = read_visit_csv(EXAMPLE)
    logger.info(
        "example visits: %d days (synthetic, not a real site)", len(example_rows)
    )
    live = visits_from_env()
    if live is None:
        logger.info(
            "Cloudflare credentials not set; leaving the example as the training file"
        )
        return
    path = REPO_ROOT / "data" / "realworld" / "website_visits.csv"
    write_csv(
        path,
        ["date", "value"],
        [{"date": day.isoformat(), "value": value} for day, value in live],
    )
    logger.info("wrote %d Cloudflare visit rows to %s", len(live), path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
