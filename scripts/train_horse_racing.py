"""Train the horse-racing form model and write its results JSON."""

from __future__ import annotations

import argparse
import json
import logging

from src.domain_forecasts.racing import train_racing
from src.domain_forecasts.reporting import to_builtin


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/horse_racing.yaml")
    args = parser.parse_args()
    payload = train_racing(args.config)
    skip = {"test_races"}
    brief = {key: value for key, value in payload.items() if key not in skip}
    print(json.dumps(to_builtin(brief), indent=2))


if __name__ == "__main__":
    main()
