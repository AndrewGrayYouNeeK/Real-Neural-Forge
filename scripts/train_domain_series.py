"""Train one dated-series config and write its results JSON."""

from __future__ import annotations

import argparse
import json
import logging

from src.domain_forecasts.reporting import to_builtin
from src.domain_forecasts.series import train_series


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = train_series(args.config)
    brief = {
        key: payload[key]
        for key in (
            "seconds",
            "n_rows",
            "n_train_windows",
            "n_val_windows",
            "n_test_windows",
            "best_epoch",
            "date_start",
            "date_end",
            "test_start",
            "test_end",
            "columns",
            "results_path",
        )
        if key in payload
    }
    print(json.dumps(to_builtin(brief), indent=2))


if __name__ == "__main__":
    main()
