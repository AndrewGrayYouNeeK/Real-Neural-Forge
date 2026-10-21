"""Command line for the Powerball dataset comparison.

Run from the repository root:

    python -m src.lottery_study --config config/lottery_dataset_study.yaml

The Powerball CSV and any dataset CSV that is missing are fetched only when
the config says so. Fetchers that are not on main yet stay in the report as
TODO rows; they are included automatically once their module or CSV appears.
"""

from __future__ import annotations

import argparse

from src.lottery_study.run import run_study
from src.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/lottery_dataset_study.yaml")
    args = parser.parse_args(argv)
    report = run_study(load_config(args.config))
    print()
    print(report["headline_table"])
    print()
    print(report["conclusion"])


if __name__ == "__main__":
    main()
