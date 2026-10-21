"""Download market history into data/markets/. Prices are not committed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.markets.fetch import main

if __name__ == "__main__":
    main()
