"""Download UCDP GED, the UCDP/PRIO conflict dataset, and the termination file.

GED is aggregated to global monthly event counts and best-estimate fatalities.
The conflict-year and termination files are cached for the interval analysis.
Raw files stay under ``data/realworld/cache`` and are not committed. All three
are CC BY 4.0; cite UCDP when using them.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path

from scripts.realworld.download import download_file
from src.realworld.parsers import GED_REGIONS, aggregate_ged_monthly
from src.realworld.tabular import write_csv
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

GED_URL = "https://ucdp.uu.se/downloads/ged/ged261-csv.zip"
ACD_URL = "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-261-csv.zip"
TERMINATION_CONFLICT_URL = "https://ucdp.uu.se/downloads/monadterm/UCDPConflictTerminationDataset_v4_2024_Conflict.csv"
TERMINATION_DYAD_URL = "https://ucdp.uu.se/downloads/monadterm/UCDPConflictTerminationDataset_v4_2024_Dyad.csv"

GED_VERSION = "26.1"
ACD_VERSION = "26.1"
TERMINATION_VERSION = "4-2024"


def cache_dir() -> Path:
    path = REPO_ROOT / "data" / "realworld" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def monthly_from_ged(zip_path: Path) -> list[dict[str, object]]:
    """Stream the GED CSV inside the zip. The uncompressed file is about 270 MB."""
    with zipfile.ZipFile(zip_path) as archive:
        name = next(item for item in archive.namelist() if item.endswith(".csv"))
        with archive.open(name) as handle:
            reader = csv.DictReader(
                io.TextIOWrapper(handle, encoding="utf-8", newline="")
            )
            return aggregate_ged_monthly(reader)


def main() -> None:
    """Cache the three UCDP files and write monthly event and fatality series."""
    cache = cache_dir()
    ged_zip = download_file(GED_URL, cache / "ged261-csv.zip")
    download_file(ACD_URL, cache / "ucdp-prio-acd-261-csv.zip")
    download_file(
        TERMINATION_CONFLICT_URL, cache / "ucdp-termination-v4-2024-conflict.csv"
    )
    download_file(TERMINATION_DYAD_URL, cache / "ucdp-termination-v4-2024-dyad.csv")
    monthly = monthly_from_ged(ged_zip)
    out_dir = REPO_ROOT / "data" / "realworld"
    region_fields = []
    for region in GED_REGIONS:
        slug = region.lower().replace(" ", "_")
        region_fields.extend([f"events_{slug}", f"fatalities_{slug}"])
    write_csv(
        out_dir / "ucdp_monthly.csv",
        ["date", "events", "fatalities", *region_fields],
        monthly,
    )
    write_csv(
        out_dir / "ucdp_monthly_events.csv",
        [
            "date",
            "value",
            *[field for field in region_fields if field.startswith("events_")],
        ],
        [{"date": row["date"], "value": row["events"], **row} for row in monthly],
    )
    write_csv(
        out_dir / "ucdp_monthly_fatalities.csv",
        ["date", "value"],
        [{"date": row["date"], "value": row["fatalities"]} for row in monthly],
    )
    logger.info(
        "UCDP GED %s: %d months %s → %s",
        GED_VERSION,
        len(monthly),
        monthly[0]["date"],
        monthly[-1]["date"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
