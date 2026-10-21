#!/usr/bin/env python3
"""Download current-matrix Powerball draws from NY Open Data.

Run from the repository root:

    python scripts/fetch_powerball.py

Writes ``data/powerball.csv`` (gitignored). Source and licence are printed
and also stored in :mod:`src.data.lottery`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.lottery import (  # noqa: E402
    SOURCE_API,
    fetch_and_write,
    source_metadata,
)
from src.utils.paths import resolve_repo_path  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/powerball.csv",
        help="Destination CSV (default: data/powerball.csv)",
    )
    parser.add_argument(
        "--api",
        default=SOURCE_API,
        help="SODA endpoint (default: the NY Open Data Powerball resource)",
    )
    args = parser.parse_args(argv)
    destination = resolve_repo_path(args.output)
    parsed = fetch_and_write(destination, api_url=args.api)
    meta = source_metadata()
    first = parsed.draws[0].draw_date.isoformat()
    last = parsed.draws[-1].draw_date.isoformat()
    print(f"Wrote {len(parsed.draws)} draws ({first} to {last}) to {destination}")
    print(f"Skipped {parsed.skipped_before_matrix} draws before {meta['matrix_start']}")
    print(f"Source: {meta['page']}")
    print(f"API: {meta['api']}")
    print(f"Publisher: {meta['publisher']}")
    print(f"Licence: {meta['licence']} ({meta['licence_url']})")
    print(meta["licence_summary"])


if __name__ == "__main__":
    main()
