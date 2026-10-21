"""Arsenal match goal difference from football-data.co.uk Premier League files.

``value`` is goals for minus goals against in that match. It is a margin, not
league points. The series is close to unpredictable from one match to the next;
the forecast report says so rather than dressing the error up.
"""

from __future__ import annotations

import logging

from scripts.realworld.download import fetch_bytes
from src.realworld.parsers import parse_team_goal_difference
from src.realworld.tabular import write_csv
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

TEAM = "Arsenal"
FIRST_SEASON = 1992
LAST_SEASON = 2026


def season_code(start_year: int) -> str:
    """``1993`` → ``9394``, ``1999`` → ``9900``, ``2000`` → ``0001``."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_url(start_year: int) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season_code(start_year)}/E0.csv"


def main() -> None:
    """Download every available EPL season file and keep Arsenal's matches."""
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        url = season_url(year)
        try:
            text = fetch_bytes(url, timeout=60).decode("latin-1")
        except RuntimeError:
            missing.append(season_code(year))
            continue
        try:
            rows.extend(parse_team_goal_difference(text, TEAM))
        except ValueError as exc:
            logger.warning("skipping %s: %s", season_code(year), exc)
            missing.append(season_code(year))
    rows.sort(key=lambda row: str(row["date"]))
    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, object]] = set()
    for row in rows:
        key = (row["date"], row["opponent"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    path = REPO_ROOT / "data" / "realworld" / "sports_arsenal_goal_diff.csv"
    write_csv(
        path,
        ["date", "value", "goals_for", "goals_against", "opponent", "home"],
        deduped,
    )
    logger.info(
        "Arsenal goal difference: %d matches %s → %s (missing seasons: %s)",
        len(deduped),
        deduped[0]["date"] if deduped else "none",
        deduped[-1]["date"] if deduped else "none",
        ", ".join(missing) or "none",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
