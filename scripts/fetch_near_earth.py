"""Download near-Earth approaches, fireballs, and reentries.

    python -m scripts.fetch_near_earth

The processed CSVs are written under ``data/space/`` and are not committed.
"""

from src.space.near_earth import main

if __name__ == "__main__":
    main()
