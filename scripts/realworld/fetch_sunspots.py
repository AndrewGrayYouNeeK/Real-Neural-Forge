"""Monthly mean total sunspot number from SILSO (version 2.0 file).

The file is the public ``SN_m_tot_V2.0.csv`` release. SILSO distributes it
under CC BY-NC 4.0.
"""

from __future__ import annotations

import logging

from scripts.realworld.download import fetch_bytes
from src.realworld.parsers import parse_silso_monthly
from src.realworld.tabular import write_csv
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

SILSO_URL = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"


def main() -> None:
    """Write ``date,value`` monthly sunspot numbers."""
    text = fetch_bytes(SILSO_URL).decode("utf-8")
    rows = [
        {"date": day.isoformat(), "value": value}
        for day, value in parse_silso_monthly(text)
    ]
    path = REPO_ROOT / "data" / "realworld" / "sunspots_monthly.csv"
    write_csv(path, ["date", "value"], rows)
    logger.info(
        "sunspots: %d months %s → %s", len(rows), rows[0]["date"], rows[-1]["date"]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
