"""Download outer-space time series into ``data/space/``.

Run from the repository root:

    python -m scripts.fetch_space

The same command is ``python -m src.space.fetch``. The CSVs are gitignored.
"""

from __future__ import annotations

from src.space.fetch import main

if __name__ == "__main__":
    main()
