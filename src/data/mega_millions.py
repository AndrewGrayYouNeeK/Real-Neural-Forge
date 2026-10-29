"""Fetch and parse New York Open Data Mega Millions draws.

The current matrix is five white balls from 1–70 and a Mega Ball from 1–25.
It began with the drawing on 2017-10-31. Older drawings are skipped. The CSV
is reproducible from the public API and is not committed.
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

DATASET_NAME = "Lottery Mega Millions Winning Numbers: Beginning 2002"
DATASET_ID = "5xaw-6ayf"
SOURCE_PAGE = (
    "https://data.ny.gov/Government-Finance/"
    "Lottery-Mega-Millions-Winning-Numbers-Beginning-2002/5xaw-6ayf"
)
SOURCE_API = "https://data.ny.gov/resource/5xaw-6ayf.json"
PUBLISHER = "New York State Gaming Commission"
LICENCE_NAME = "OPEN-NY Terms of Use"
LICENCE_URL = (
    "https://data.ny.gov/api/views/77gx-ii52/files/"
    "ef0c1840-ad54-4240-92fd-6397c49fde46"
    "?filename=OPEN-NY_20Terms_20of_20Use.pdf"
)

# First draw under the current 5/70 + 1/25 matrix.
MATRIX_START = date(2017, 10, 31)
WHITE_LOW = 1
WHITE_HIGH = 70
MEGA_BALL_LOW = 1
MEGA_BALL_HIGH = 25
WHITE_BALL_COUNT = 5

CSV_COLUMNS = (
    "draw_date",
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
    "mega_ball",
)

WhiteBalls = tuple[int, int, int, int, int]
Opener = Callable[..., Any]


@dataclass(frozen=True)
class MegaMillionsDraw:
    """One Mega Millions draw. White balls are sorted; the Mega Ball is separate."""

    draw_date: date
    white_balls: WhiteBalls
    mega_ball: int


@dataclass(frozen=True)
class ParsedDraws:
    """Draws kept for the current matrix, plus how many older draws were skipped."""

    draws: list[MegaMillionsDraw]
    skipped_before_matrix: int


def source_metadata() -> dict[str, str]:
    """Source and licence recorded with the lunar comparison."""
    return {
        "dataset": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "page": SOURCE_PAGE,
        "api": SOURCE_API,
        "publisher": PUBLISHER,
        "licence": LICENCE_NAME,
        "licence_url": LICENCE_URL,
        "matrix_start": MATRIX_START.isoformat(),
        "white_ball_range": f"{WHITE_LOW}-{WHITE_HIGH}",
        "mega_ball_range": f"{MEGA_BALL_LOW}-{MEGA_BALL_HIGH}",
    }


def parse_draw_date(value: str) -> date:
    """Parse an ISO date or a Socrata calendar timestamp."""
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"invalid draw_date {value!r}") from exc


def parse_winning_numbers(text: str, mega_ball: str | int) -> tuple[WhiteBalls, int]:
    """Parse five white balls and a separate Mega Ball."""
    parts = text.split()
    if len(parts) != WHITE_BALL_COUNT:
        raise ValueError(
            f"winning_numbers must contain {WHITE_BALL_COUNT} integers, got {text!r}"
        )
    try:
        whites = tuple(sorted(int(part) for part in parts))
        bonus = int(mega_ball)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"winning numbers must be integers, got {text!r} and {mega_ball!r}"
        ) from exc
    if len(set(whites)) != WHITE_BALL_COUNT:
        raise ValueError(f"white balls must be distinct, got {text!r}")
    return (
        whites[0],
        whites[1],
        whites[2],
        whites[3],
        whites[4],
    ), bonus


def draw_from_api_row(row: dict[str, Any]) -> MegaMillionsDraw:
    """Build a draw from one SODA row."""
    raw_date = row.get("draw_date")
    raw_numbers = row.get("winning_numbers")
    raw_mega = row.get("mega_ball")
    if (
        not isinstance(raw_date, str)
        or not isinstance(raw_numbers, str)
        or raw_mega is None
    ):
        raise ValueError(f"API row is missing Mega Millions fields: {row!r}")
    if not isinstance(raw_mega, str | int):
        raise ValueError(f"mega_ball must be a string or int, got {raw_mega!r}")
    white_balls, mega_ball = parse_winning_numbers(raw_numbers, raw_mega)
    return MegaMillionsDraw(
        draw_date=parse_draw_date(raw_date),
        white_balls=white_balls,
        mega_ball=mega_ball,
    )


def _in_current_matrix(draw: MegaMillionsDraw) -> bool:
    whites_ok = all(WHITE_LOW <= ball <= WHITE_HIGH for ball in draw.white_balls)
    mega_ok = MEGA_BALL_LOW <= draw.mega_ball <= MEGA_BALL_HIGH
    return whites_ok and mega_ok


def select_current_matrix(rows: list[dict[str, Any]]) -> ParsedDraws:
    """Keep draws from the current matrix and reject out-of-range rows inside it."""
    draws: list[MegaMillionsDraw] = []
    skipped = 0
    for row in rows:
        draw = draw_from_api_row(row)
        if draw.draw_date < MATRIX_START:
            skipped += 1
            continue
        if not _in_current_matrix(draw):
            raise ValueError(
                f"{draw.draw_date.isoformat()} is on or after "
                f"{MATRIX_START.isoformat()} but outside the 5/70 + 1/25 matrix: "
                f"whites={draw.white_balls} mega_ball={draw.mega_ball}"
            )
        draws.append(draw)
    draws.sort(key=lambda item: item.draw_date)
    dates = [item.draw_date for item in draws]
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate draw_date in current-matrix draws")
    if not draws:
        raise ValueError("no Mega Millions draws left after the matrix filter")
    return ParsedDraws(draws=draws, skipped_before_matrix=skipped)


def fetch_mega_millions_rows(
    api_url: str = SOURCE_API,
    *,
    page_size: int = 1000,
    timeout: float = 60,
    opener: Opener = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Download every SODA row, paging until a short page."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"$limit": page_size, "$offset": offset, "$order": "draw_date ASC"}
        )
        url = f"{api_url}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "real-neural-forge-lunar-lottery",
            },
        )
        try:
            with opener(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"failed to fetch Mega Millions draws from {url}"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON list from {url}")
        page = cast(list[Any], payload)
        parsed: list[dict[str, Any]] = []
        for item in page:
            if not isinstance(item, dict):
                raise ValueError(f"expected JSON objects from {url}")
            parsed.append(cast(dict[str, Any], item))
        rows.extend(parsed)
        if len(parsed) < page_size:
            return rows
        offset += page_size
        if offset > 100_000:
            raise RuntimeError("Mega Millions pagination did not terminate")


def write_draws_csv(path: str | Path, draws: list[MegaMillionsDraw]) -> Path:
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
                    "mega_ball": draw.mega_ball,
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
    rows = fetch_mega_millions_rows(api_url, opener=opener)
    parsed = select_current_matrix(rows)
    write_draws_csv(path, parsed.draws)
    return parsed


def read_draw_table(path: str | Path) -> pd.DataFrame:
    """Load a draws CSV, sorted and checked against the current matrix."""
    frame = pd.read_csv(path)
    missing = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Mega Millions CSV is missing columns: {missing}")
    frame = frame.loc[:, list(CSV_COLUMNS)].copy()
    frame["draw_date"] = pd.to_datetime(frame["draw_date"])
    if frame["draw_date"].duplicated().any():
        raise ValueError("duplicate draw_date in Mega Millions CSV")
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
        raise ValueError("white balls must be inside 1..70")
    if int(whites.nunique(axis=1).min()) != WHITE_BALL_COUNT:
        raise ValueError("white balls must be distinct")
    mega = frame["mega_ball"]
    if int(mega.min()) < MEGA_BALL_LOW or int(mega.max()) > MEGA_BALL_HIGH:
        raise ValueError("mega_ball must be inside 1..25")
    return frame
