"""Compare forecast models for a config and store the table.

Run ``python -m src.compare --config config/config.yaml``.
"""

from __future__ import annotations

import argparse

from src.evaluation.compare import compare_models
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare forecast models on one temporal split.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Neural-model epochs for this comparison (default: training.epochs)",
    )
    parser.add_argument(
        "--season-length",
        type=int,
        default=None,
        help="Season length for the seasonal-naive baseline",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=None,
        help="Hidden size of the LSTM baseline",
    )
    args = parser.parse_args()
    report = compare_models(
        load_config(args.config),
        epochs=args.epochs,
        season_length=args.season_length,
        lstm_hidden_size=args.lstm_hidden_size,
    )
    print(report.table)
    print(f"Stored experiment {report.experiment_id} in {report.db_path}")


if __name__ == "__main__":
    main()
