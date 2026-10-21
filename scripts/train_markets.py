"""Train per-series return models and write results/market_forecasts.json."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.markets.run import main

if __name__ == "__main__":
    main()
