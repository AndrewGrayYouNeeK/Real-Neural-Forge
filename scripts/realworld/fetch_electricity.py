"""Download EIA-930 US48 hourly demand from the keyless bulk file.

The bulk zip is about 700 MB and is cached under ``data/realworld/cache``.
Only the extracted hourly series is kept for training, and the training file
is the last 17,520 hours (about two years). A 168-step model on the full
history is too slow on CPU; the full series is still written beside it.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from scripts.realworld.download import download_file
from src.realworld.parsers import parse_eia_series_line, tail_rows
from src.realworld.tabular import write_csv
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

EIA_BULK_URL = "https://www.eia.gov/opendata/bulk/EBA.zip"
SERIES_ID = "EBA.US48-ALL.D.H"
TRAIN_TAIL_HOURS = 17520


def extract_series(zip_path: Path) -> list[tuple[str, float]]:
    """Stream ``EBA.txt`` and return the US48 demand series, oldest hour first."""
    with zipfile.ZipFile(zip_path) as archive, archive.open("EBA.txt") as handle:
        for line in handle:
            parsed = parse_eia_series_line(line, SERIES_ID)
            if parsed is not None:
                return parsed
    raise RuntimeError(f"{SERIES_ID} was not in {zip_path}")


def main() -> None:
    """Download the bulk file and write the full series plus the training slice."""
    cache = REPO_ROOT / "data" / "realworld" / "cache" / "EBA.zip"
    logger.info("EIA bulk cache: %s", cache)
    download_file(EIA_BULK_URL, cache)
    rows = extract_series(cache)
    full = [{"date": stamp, "value": value} for stamp, value in rows]
    train = [
        {"date": stamp, "value": value}
        for stamp, value in tail_rows(rows, TRAIN_TAIL_HOURS)
    ]
    full_path = REPO_ROOT / "data" / "realworld" / "electricity_us48_hourly_full.csv"
    train_path = REPO_ROOT / "data" / "realworld" / "electricity_us48_hourly.csv"
    write_csv(full_path, ["date", "value"], full)
    write_csv(train_path, ["date", "value"], train)
    logger.info(
        "US48 demand: %d hours %s → %s; training slice %d hours ending %s",
        len(full),
        full[0]["date"],
        full[-1]["date"],
        len(train),
        train[-1]["date"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
