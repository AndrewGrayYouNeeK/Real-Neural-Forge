"""Fetch and parse New York Open Data Powerball draws.

The CSV this module writes is reproducible from the public API and is not
committed (``data/`` is gitignored). Draws before 2015-10-07 used a different
matrix, so they are dropped before a uniform 1–26 baseline is meaningful.
"""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

DATASET_NAME = "Lottery Powerball Winning Numbers: Beginning 2010"
DATASET_ID = "d6yy-54nr"
SOURCE_PAGE = (
    "https://data.ny.gov/Government-Finance/"
    "Lottery-Powerball-Winning-Numbers-Beginning-2010/d6yy-54nr"
)
SOURCE_API = "https://data.ny.gov/resource/d6yy-54nr.json"
PUBLISHER = "New York State Gaming Commission"
LICENCE_NAME = "OPEN-NY Terms of Use"
LICENCE_URL = (
    "https://data.ny.gov/api/views/77gx-ii52/files/"
    "ef0c1840-ad54-4240-92fd-6397c49fde46"
    "?filename=OPEN-NY_20Terms_20of_20Use.pdf"
)
LICENCE_SUMMARY = (
    "OPEN-NY Terms of Use (last modified March 8, 2013) grant a non-exclusive, "
    "revocable licence to use content published on data.ny.gov. Those terms do "
    "not require attribution, share-alike, or pre-approval. The dataset view "
    "does not attach a separate SPDX licence."
)

# First draw under the current 5/69 white-ball + 1/26 Powerball matrix.
MATRIX_START = date(2015, 10, 7)
WHITE_LOW = 1
WHITE_HIGH = 69
POWERBALL_LOW = 1
POWERBALL_HIGH = 26
WHITE_BALL_COUNT = 5

CSV_COLUMNS = (
    "draw_date",
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
    "powerball",
)

WhiteBalls = tuple[int, int, int, int, int]
Opener = Callable[..., Any]


@dataclass(frozen=True)
class PowerballDraw:
    """One Powerball draw. White balls are sorted; the Powerball is separate."""

    draw_date: date
    white_balls: WhiteBalls
    powerball: int


@dataclass(frozen=True)
class ParsedDraws:
    """Draws kept for the current matrix, plus how many older draws were skipped."""

    draws: list[PowerballDraw]
    skipped_before_matrix: int


def source_metadata() -> dict[str, str]:
    """Source and licence recorded with the sanity-check results."""
    return {
        "dataset": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "page": SOURCE_PAGE,
        "api": SOURCE_API,
        "publisher": PUBLISHER,
        "licence": LICENCE_NAME,
        "licence_url": LICENCE_URL,
        "licence_summary": LICENCE_SUMMARY,
        "matrix_start": MATRIX_START.isoformat(),
        "white_ball_range": f"{WHITE_LOW}-{WHITE_HIGH}",
        "powerball_range": f"{POWERBALL_LOW}-{POWERBALL_HIGH}",
    }


def parse_draw_date(value: str) -> date:
    """Parse an ISO date or a Socrata calendar timestamp."""
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"invalid draw_date {value!r}") from exc


def parse_winning_numbers(text: str) -> tuple[WhiteBalls, int]:
    """Parse ``winning_numbers`` into sorted white balls and the Powerball.

    NY Open Data stores six space-separated integers. The last field is the
    Powerball. The first five are white balls and are returned in ascending
    order; the Powerball is not sorted in with them.
    """
    parts = text.split()
    if len(parts) != WHITE_BALL_COUNT + 1:
        raise ValueError(
            f"winning_numbers must contain {WHITE_BALL_COUNT + 1} integers, "
            f"got {text!r}"
        )
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"winning_numbers must be integers, got {text!r}") from exc

    whites = tuple(sorted(numbers[:WHITE_BALL_COUNT]))
    if len(set(whites)) != WHITE_BALL_COUNT:
        raise ValueError(f"white balls must be distinct, got {text!r}")
    white_balls = (
        whites[0],
        whites[1],
        whites[2],
        whites[3],
        whites[4],
    )
    return white_balls, numbers[WHITE_BALL_COUNT]


def draw_from_api_row(row: dict[str, Any]) -> PowerballDraw:
    """Build a draw from one SODA row."""
    raw_date = row.get("draw_date")
    raw_numbers = row.get("winning_numbers")
    if not isinstance(raw_date, str) or not isinstance(raw_numbers, str):
        raise ValueError(f"API row is missing draw_date or winning_numbers: {row!r}")
    white_balls, powerball = parse_winning_numbers(raw_numbers)
    return PowerballDraw(
        draw_date=parse_draw_date(raw_date),
        white_balls=white_balls,
        powerball=powerball,
    )


def _in_current_matrix(draw: PowerballDraw) -> bool:
    whites_ok = all(WHITE_LOW <= ball <= WHITE_HIGH for ball in draw.white_balls)
    powerball_ok = POWERBALL_LOW <= draw.powerball <= POWERBALL_HIGH
    return whites_ok and powerball_ok


def select_current_matrix(rows: list[dict[str, Any]]) -> ParsedDraws:
    """Keep draws from the current matrix and reject out-of-range rows inside it.

    Draws before ``MATRIX_START`` are skipped. A later draw outside 5/69 + 1/26
    is an error: the uniform baseline would no longer have one range.
    """
    draws: list[PowerballDraw] = []
    skipped = 0
    for row in rows:
        draw = draw_from_api_row(row)
        if draw.draw_date < MATRIX_START:
            skipped += 1
            continue
        if not _in_current_matrix(draw):
            raise ValueError(
                f"{draw.draw_date.isoformat()} is on or after "
                f"{MATRIX_START.isoformat()} but outside the 5/69 + 1/26 matrix: "
                f"whites={draw.white_balls} powerball={draw.powerball}"
            )
        draws.append(draw)

    draws.sort(key=lambda draw: draw.draw_date)
    dates = [draw.draw_date for draw in draws]
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate draw_date in current-matrix draws")
    if not draws:
        raise ValueError("no Powerball draws left after the matrix filter")
    return ParsedDraws(draws=draws, skipped_before_matrix=skipped)


def fetch_powerball_rows(
    api_url: str = SOURCE_API,
    *,
    page_size: int = 1000,
    timeout: float = 60,
    opener: Opener = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Download every SODA row, paging until a short page.

    ``opener`` matches ``urllib.request.urlopen`` and is injectable in tests.
    """
    if page_size < 1:
        raise ValueError("page_size must be positive")
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "$limit": page_size,
                "$offset": offset,
                "$order": "draw_date ASC",
            }
        )
        url = f"{api_url}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "real-neural-forge-lottery-sanity-check",
            },
        )
        try:
            with opener(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"failed to fetch Powerball draws from {url}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON list from {url}")
        page = cast(list[Any], payload)
        parsed_page: list[dict[str, Any]] = []
        for item in page:
            if not isinstance(item, dict):
                raise ValueError(f"expected JSON objects from {url}")
            parsed_page.append(cast(dict[str, Any], item))
        rows.extend(parsed_page)
        if len(parsed_page) < page_size:
            return rows
        offset += page_size
        if offset > 100_000:
            raise RuntimeError("Powerball pagination did not terminate")


def write_draws_csv(path: str | Path, draws: list[PowerballDraw]) -> Path:
    """Write draws to CSV. Parent directories are created."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for draw in draws:
            writer.writerow(
                {
                    "draw_date": draw.draw_date.isoformat(),
                    "white_1": draw.white_balls[0],
                    "white_2": draw.white_balls[1],
                    "white_3": draw.white_balls[2],
                    "white_4": draw.white_balls[3],
                    "white_5": draw.white_balls[4],
                    "powerball": draw.powerball,
                }
            )
    return destination


def fetch_and_write(
    path: str | Path,
    *,
    api_url: str = SOURCE_API,
    opener: Opener = urllib.request.urlopen,
) -> ParsedDraws:
    """Fetch the public dataset and write the current-matrix CSV."""
    rows = fetch_powerball_rows(api_url, opener=opener)
    parsed = select_current_matrix(rows)
    write_draws_csv(path, parsed.draws)
    return parsed


def read_draw_table(path: str | Path) -> pd.DataFrame:
    """Load a draws CSV, sorted and checked against the current matrix."""
    frame = pd.read_csv(path)
    missing = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Powerball CSV is missing columns: {missing}")
    frame = frame.loc[:, list(CSV_COLUMNS)].copy()
    frame["draw_date"] = pd.to_datetime(frame["draw_date"])
    if frame["draw_date"].duplicated().any():
        raise ValueError("duplicate draw_date in Powerball CSV")
    if not bool(frame["draw_date"].is_monotonic_increasing):
        raise ValueError("draw_date must be sorted ascending")

    number_columns = [column for column in CSV_COLUMNS if column != "draw_date"]
    for column in number_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.allclose(numeric, np.rint(numeric)):
            raise ValueError(f"{column} must be integers")
        frame[column] = numeric.astype(int)

    whites = frame.loc[:, ["white_1", "white_2", "white_3", "white_4", "white_5"]]
    if whites.min().min() < WHITE_LOW or whites.max().max() > WHITE_HIGH:
        raise ValueError("white balls must be inside 1..69")
    if whites.nunique(axis=1).min() != WHITE_BALL_COUNT:
        raise ValueError("white balls must be distinct")
    powerball = frame["powerball"]
    if powerball.min() < POWERBALL_LOW or powerball.max() > POWERBALL_HIGH:
        raise ValueError("powerball must be inside 1..26")
    return frame


def powerball_series(path: str | Path) -> np.ndarray:
    """Return the Powerball column as a chronological ``float64`` array."""
    frame = read_draw_table(path)
    return cast(np.ndarray, np.asarray(frame["powerball"], dtype=np.float64))


def draw_date_span(path: str | Path) -> tuple[str, str]:
    """Return the first and last draw dates as ISO strings."""
    frame = read_draw_table(path)
    start = pd.Timestamp(frame["draw_date"].iloc[0]).date().isoformat()
    end = pd.Timestamp(frame["draw_date"].iloc[-1]).date().isoformat()
    return start, end
