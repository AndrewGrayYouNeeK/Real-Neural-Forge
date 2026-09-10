"""CLI inference against a local checkpoint."""

from __future__ import annotations

import argparse
import json

from src.api import load_model, run_predict
from src.common import load_config
from src.data import load_csv_series, sequence_from_series


def parse_sequence(raw: str) -> list[list[float]]:
    """Accept comma-separated values or a JSON list."""
    text = raw.strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if parsed and isinstance(parsed[0], list):
            return [[float(v) for v in step] for step in parsed]
        return [[float(v)] for v in parsed]
    return [[float(part)] for part in text.split(",") if part.strip()]


def sequence_from_csv(path: str, seq_len: int, value_column: str | None = None) -> list[list[float]]:
    """Use the last ``seq_len`` values of a CSV as the model input."""
    return sequence_from_series(load_csv_series(path, value_column), seq_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neural Forge inference")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--sequence",
        help="Comma-separated values or a JSON list",
    )
    parser.add_argument(
        "--csv",
        help="CSV file; uses the last data.seq_len points",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of future steps to roll out",
    )
    args = parser.parse_args()
    if bool(args.sequence) == bool(args.csv):
        parser.error("Provide exactly one of --sequence or --csv")

    cfg = load_config(args.config)
    if args.csv:
        seq_len = int((cfg.get("data") or {}).get("seq_len", 64))
        value_column = (cfg.get("data") or {}).get("value_column")
        sequence = sequence_from_csv(args.csv, seq_len, value_column)
    else:
        sequence = parse_sequence(args.sequence)

    load_model(args.config)
    prediction = run_predict(sequence, args.horizon)
    print(json.dumps({"prediction": prediction, "horizon": args.horizon}))


if __name__ == "__main__":
    main()
