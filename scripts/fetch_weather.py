"""Download the Central Park daily temperature series into ``data/``.

Run from the repository root:

    python -m scripts.fetch_weather

The same command is ``python -m src.data.weather``. The CSV is gitignored.
"""

from __future__ import annotations

from src.data.weather import main

if __name__ == "__main__":
    main()
