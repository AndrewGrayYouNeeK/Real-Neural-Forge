"""Download the public HKJC/Singapore performances table for local training.

The upstream repository has no licence file. This script does not copy the
table into git. It writes the gzip under ``data/raw``, which is gitignored,
and prints a short summary of the Hong Kong rows the trainer keeps.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

from src.domain_forecasts.racing import tidy_hk_performances
from src.utils.paths import resolve_repo_path

SOURCE_URL = (
    "https://github.com/eprochasson/horserace_data/raw/master/data/performances.csv.gz"
)
SOURCE_REPO = "https://github.com/eprochasson/horserace_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--out", default="data/raw/horse_racing/performances.csv.gz")
    args = parser.parse_args()
    destination = resolve_repo_path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(args.url, destination)
    frame = pd.read_csv(destination)
    tidy = tidy_hk_performances(frame)
    summary = {
        "source": SOURCE_URL,
        "repository": SOURCE_REPO,
        "licence": (
            "No licence file in eprochasson/horserace_data (GitHub licence API "
            "404 on 2026-10-21). The README says the tables were scraped from "
            "public HKJC and Singapore Turf Club pages and are provided without "
            "a warranty. This project does not redistribute the raw table."
        ),
        "upstream_rows": int(len(frame)),
        "hk_runners_kept": int(len(tidy)),
        "hk_races": int(tidy["race_key"].nunique()),
        "date_start": str(tidy["race_date"].min().date()),
        "date_end": str(tidy["race_date"].max().date()),
        "local_path": str(destination),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
