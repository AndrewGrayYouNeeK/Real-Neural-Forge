"""CLI inference against a local checkpoint."""

from __future__ import annotations

import argparse
import json

from src.api import load_model, run_predict


def parse_sequence(raw: str) -> list[list[float]]:
    """Accept comma-separated values or a JSON list."""
    text = raw.strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if parsed and isinstance(parsed[0], list):
            return [[float(v) for v in step] for step in parsed]
        return [[float(v)] for v in parsed]
    return [[float(part)] for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neural Forge inference")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Comma-separated values or a JSON list",
    )
    args = parser.parse_args()
    load_model(args.config)
    prediction = run_predict(parse_sequence(args.sequence))
    print(json.dumps({"prediction": prediction}))


if __name__ == "__main__":
    main()
