"""Small CSV helpers shared by the dataset fetch scripts."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path


def write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]
) -> int:
    """Write ``rows`` to ``path``, creating parent directories. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            count += 1
    return count
