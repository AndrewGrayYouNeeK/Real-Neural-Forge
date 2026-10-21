"""Download every real-world series this package trains on."""

from __future__ import annotations

import logging

from scripts.realworld.fetch_air_quality import main as fetch_air_quality
from scripts.realworld.fetch_earthquakes import main as fetch_earthquakes
from scripts.realworld.fetch_electricity import main as fetch_electricity
from scripts.realworld.fetch_fred import main as fetch_fred
from scripts.realworld.fetch_sports import main as fetch_sports
from scripts.realworld.fetch_sunspots import main as fetch_sunspots
from scripts.realworld.fetch_ucdp import main as fetch_ucdp
from scripts.realworld.fetch_website import main as fetch_website

logger = logging.getLogger(__name__)

STEPS = (
    ("fred", fetch_fred),
    ("sunspots", fetch_sunspots),
    ("air_quality", fetch_air_quality),
    ("sports", fetch_sports),
    ("website", fetch_website),
    ("earthquakes", fetch_earthquakes),
    ("ucdp", fetch_ucdp),
    ("electricity", fetch_electricity),
)


def main() -> None:
    """Run each fetcher. A failure stops the rest so a partial cache is obvious."""
    for name, step in STEPS:
        logger.info("fetching %s", name)
        step()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
