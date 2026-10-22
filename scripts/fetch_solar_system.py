"""Download planetary approaches, comet perihelia, and Horizons ephemerides.

    python -m scripts.fetch_solar_system

The processed CSVs are written under ``data/space/`` and are not committed.
"""

from src.space.solar_system import main

if __name__ == "__main__":
    main()
