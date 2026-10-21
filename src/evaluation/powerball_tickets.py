"""Powerball tickets, prize tiers, and the public monthly scorecard.

A ticket is five distinct white balls from 1–69 and one Powerball from 1–26.
Nothing here buys a ticket or talks to a lottery retailer. Prize amounts are
the published base schedule for the current matrix, used only as tier labels.
The jackpot and Power Play are not scored in dollars.

Draw nights since 2021-08-23 are Monday, Wednesday, and Saturday. The random
control ticket is a uniform draw whose seed is the public string
``powerball-control-{month}-{YYYY-MM-DD}``, so the logged ticket does not
change when the month file is read again.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.lottery import (
    POWERBALL_HIGH,
    POWERBALL_LOW,
    WHITE_BALL_COUNT,
    WHITE_HIGH,
    WHITE_LOW,
)

# Monday, Wednesday, Saturday.
DRAW_WEEKDAYS = (0, 2, 5)
DRAW_SCHEDULE_SINCE = date(2021, 8, 23)
WHITE_COMBINATIONS = comb(WHITE_HIGH, WHITE_BALL_COUNT)
TICKET_COMBINATIONS = WHITE_COMBINATIONS * POWERBALL_HIGH
CHANCE_WHITE_MATCHES = (WHITE_BALL_COUNT * WHITE_BALL_COUNT) / WHITE_HIGH
CHANCE_POWERBALL_HIT = 1.0 / POWERBALL_HIGH
# score_ticket records near misses. The lottery study detects this flag.
SCORES_CLOSE_CALLS = True

STRATEGIES = (
    "model",
    "model_weather",
    "most_hit",
    "hot_pairs",
    "neighbors",
    "overdue",
    "random",
)
PATTERN_STRATEGIES = ("most_hit", "hot_pairs", "neighbors", "overdue", "random")
RANDOM_SEED_PREFIX = "powerball-control"

# (white matches, powerball hit) -> (tier name, published base prize USD).
# None means the jackpot, which is not a fixed dollar amount.
_PRIZE_SCHEDULE: dict[tuple[int, bool], tuple[str, int | None]] = {
    (5, True): ("jackpot", None),
    (5, False): ("match_5", 1_000_000),
    (4, True): ("match_4_pb", 50_000),
    (4, False): ("match_4", 100),
    (3, True): ("match_3_pb", 100),
    (3, False): ("match_3", 7),
    (2, True): ("match_2_pb", 7),
    (1, True): ("match_1_pb", 4),
    (0, True): ("match_pb", 4),
}


@dataclass(frozen=True)
class HistoryDraw:
    """One past draw used to build pattern tickets."""

    draw_date: date
    whites: tuple[int, int, int, int, int]
    powerball: int


def white_match_probability(matches: int) -> float:
    """Probability a ticket matches exactly ``matches`` white balls."""
    if not 0 <= matches <= WHITE_BALL_COUNT:
        raise ValueError(f"white matches must be 0..{WHITE_BALL_COUNT}")
    ways = comb(WHITE_BALL_COUNT, matches) * comb(
        WHITE_HIGH - WHITE_BALL_COUNT, WHITE_BALL_COUNT - matches
    )
    return ways / WHITE_COMBINATIONS


def tier_probability(white_matches: int, powerball_hit: bool) -> float:
    """Probability of one prize tier for a uniformly random ticket."""
    powerball_ways = 1 if powerball_hit else POWERBALL_HIGH - 1
    return white_match_probability(white_matches) * powerball_ways / POWERBALL_HIGH


def expected_white_matches() -> float:
    """Expected white-ball matches. Equals ``5 * 5 / 69``."""
    return float(
        sum(
            matches * white_match_probability(matches)
            for matches in range(WHITE_BALL_COUNT + 1)
        )
    )


def prize_tier(white_matches: int, powerball_hit: bool) -> dict[str, Any]:
    """Name the published base tier. ``base_prize_usd`` is null for the jackpot."""
    found = _PRIZE_SCHEDULE.get((white_matches, powerball_hit))
    if found is None:
        return {
            "tier": "none",
            "base_prize_usd": 0,
            "white_matches": white_matches,
            "powerball_hit": powerball_hit,
        }
    name, prize = found
    return {
        "tier": name,
        "base_prize_usd": prize,
        "white_matches": white_matches,
        "powerball_hit": powerball_hit,
    }


def register_close_call_metrics(_scorecard_fn: object) -> None:
    """Close calls are already fields on ``score_ticket``. The hook acknowledges that."""
    return None


def score_ticket(ticket: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Score one logged ticket against one actual draw.

    ``within_1`` and ``within_2`` count predicted white balls whose nearest
    drawn white is at most that far away, including exact hits. ``near_miss_1``
    and ``near_miss_2`` count distance exactly 1 and exactly 2.
    """
    predicted = validate_ticket(ticket)
    observed = validate_ticket(actual)
    matches = len(set(predicted["whites"]) & set(observed["whites"]))
    powerball_hit = predicted["powerball"] == observed["powerball"]
    tier = prize_tier(matches, powerball_hit)
    pred_whites = np.asarray(predicted["whites"], dtype=np.int64)
    act_whites = np.asarray(observed["whites"], dtype=np.int64)
    distance = np.abs(pred_whites.reshape(-1, 1) - act_whites.reshape(1, -1)).min(
        axis=1
    )
    return {
        "white_matches": matches,
        "powerball_hit": powerball_hit,
        "tier": tier["tier"],
        "base_prize_usd": tier["base_prize_usd"],
        "within_1": int(np.sum(distance <= 1)),
        "within_2": int(np.sum(distance <= 2)),
        "near_miss_1": int(np.sum(distance == 1)),
        "near_miss_2": int(np.sum(distance == 2)),
        "white_sum_abs_error": int(abs(int(pred_whites.sum()) - int(act_whites.sum()))),
        "powerball_abs_error": abs(predicted["powerball"] - observed["powerball"]),
    }


def validate_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    """Return a sorted ticket or raise if it is not a legal Powerball play."""
    whites = ticket.get("whites")
    powerball = ticket.get("powerball")
    if not isinstance(whites, (list, tuple)) or len(whites) != WHITE_BALL_COUNT:
        raise ValueError("a ticket needs five white balls")
    if powerball is None:
        raise ValueError("ticket numbers must be integers")
    try:
        numbers = tuple(int(ball) for ball in whites)
        powerball_number = int(powerball)
    except (TypeError, ValueError) as exc:
        raise ValueError("ticket numbers must be integers") from exc
    if numbers != tuple(sorted(numbers)):
        raise ValueError("white balls must be sorted ascending")
    if len(set(numbers)) != WHITE_BALL_COUNT:
        raise ValueError("white balls must be unique")
    if any(ball < WHITE_LOW or ball > WHITE_HIGH for ball in numbers):
        raise ValueError("white balls must be inside 1..69")
    if not POWERBALL_LOW <= powerball_number <= POWERBALL_HIGH:
        raise ValueError("powerball must be inside 1..26")
    return {"whites": list(numbers), "powerball": powerball_number}


def repair_number_vector(values: np.ndarray) -> dict[str, Any]:
    """Turn a 6-long forecast into a legal ticket.

    The first five entries are white-ball predictions and the last is the
    Powerball. Rounding uses half-to-even. Colliding white balls step outward
    until they are unique. The result is sorted.
    """
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != WHITE_BALL_COUNT + 1:
        raise ValueError("a Powerball forecast has 6 numbers")
    if not np.isfinite(vector).all():
        raise ValueError("forecast contains a non-finite number")
    used: set[int] = set()
    chosen: list[int] = []
    for raw in np.rint(vector[:WHITE_BALL_COUNT]):
        start = int(np.clip(raw, WHITE_LOW, WHITE_HIGH))
        if start not in used:
            chosen.append(start)
            used.add(start)
            continue
        placed = False
        for delta in range(1, WHITE_HIGH):
            for sign in (1, -1):
                candidate = start + sign * delta
                if WHITE_LOW <= candidate <= WHITE_HIGH and candidate not in used:
                    chosen.append(candidate)
                    used.add(candidate)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            raise ValueError("could not repair white balls into five unique numbers")
    powerball = int(np.clip(np.rint(vector[-1]), POWERBALL_LOW, POWERBALL_HIGH))
    return validate_ticket({"whites": sorted(chosen), "powerball": powerball})


def draw_dates_for_month(month: str) -> list[date]:
    """Monday, Wednesday, and Saturday dates in ``YYYY-MM``.

    Months before the Monday draw was added are rejected so a Wed/Sat month
    is not filled with Mondays that did not exist.
    """
    year_text, month_text = month.split("-")
    year, month_number = int(year_text), int(month_text)
    if not 1 <= month_number <= 12:
        raise ValueError(f"invalid month {month}")
    first = date(year, month_number, 1)
    if first < date(DRAW_SCHEDULE_SINCE.year, DRAW_SCHEDULE_SINCE.month, 1):
        raise ValueError(
            f"draw calendar is Monday/Wednesday/Saturday only from {DRAW_SCHEDULE_SINCE.isoformat()}"
        )
    if month_number == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month_number + 1, 1)
    dates: list[date] = []
    cursor = first
    while cursor < next_month:
        if cursor.weekday() in DRAW_WEEKDAYS and cursor >= DRAW_SCHEDULE_SINCE:
            dates.append(cursor)
        cursor += timedelta(days=1)
    return dates


def month_before(month: str) -> str:
    """Return the previous ``YYYY-MM``."""
    year_text, month_text = month.split("-")
    year, month_number = int(year_text), int(month_text)
    if month_number == 1:
        return f"{year - 1}-12"
    return f"{year}-{month_number - 1:02d}"


def default_forecast_month(today: date) -> str:
    """On the 1st, forecast that month. Later in the month, forecast the next one."""
    if today.day == 1:
        return f"{today.year}-{today.month:02d}"
    if today.month == 12:
        return f"{today.year + 1}-01"
    return f"{today.year}-{today.month + 1:02d}"


def as_of_for_month(month: str) -> date:
    """Last calendar day the model may use: the day before the month starts."""
    year_text, month_text = month.split("-")
    return date(int(year_text), int(month_text), 1) - timedelta(days=1)


def _ranked_counts(counts: Counter[int], limit: int) -> list[int]:
    return [
        number
        for number, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ][:limit]


def _powerball_mode(history: list[HistoryDraw]) -> int:
    counts: Counter[int] = Counter(draw.powerball for draw in history)
    if not counts:
        raise ValueError("pattern tickets need at least one past draw")
    return _ranked_counts(counts, 1)[0]


def _fill_whites(chosen: list[int], history: list[HistoryDraw]) -> list[int]:
    filled = list(chosen)
    if len(filled) > WHITE_BALL_COUNT:
        raise ValueError("too many white balls to fill")
    counts: Counter[int] = Counter(ball for draw in history for ball in draw.whites)
    for number in _ranked_counts(counts, WHITE_HIGH):
        if len(filled) == WHITE_BALL_COUNT:
            break
        if number not in filled:
            filled.append(number)
    for number in range(WHITE_LOW, WHITE_HIGH + 1):
        if len(filled) == WHITE_BALL_COUNT:
            break
        if number not in filled:
            filled.append(number)
    if len(filled) != WHITE_BALL_COUNT:
        raise ValueError("could not fill five white balls")
    return sorted(filled)


def most_hit_ticket(history: list[HistoryDraw]) -> dict[str, Any]:
    """Five white balls and the Powerball with the highest historical counts."""
    counts: Counter[int] = Counter(ball for draw in history for ball in draw.whites)
    if not counts:
        raise ValueError("most-hit tickets need at least one past draw")
    whites = _fill_whites(_ranked_counts(counts, WHITE_BALL_COUNT), history)
    return validate_ticket({"whites": whites, "powerball": _powerball_mode(history)})


def hot_pairs_ticket(history: list[HistoryDraw]) -> dict[str, Any]:
    """Greedy white balls from the most frequent co-occurring pairs."""
    if not history:
        raise ValueError("hot-pair tickets need at least one past draw")
    pairs: Counter[tuple[int, int]] = Counter()
    for draw in history:
        balls = draw.whites
        for left in range(WHITE_BALL_COUNT):
            for right in range(left + 1, WHITE_BALL_COUNT):
                pairs[(balls[left], balls[right])] += 1
    chosen: list[int] = []
    for (left, right), _count in sorted(
        pairs.items(), key=lambda item: (-item[1], item[0])
    ):
        for number in (left, right):
            if number not in chosen:
                chosen.append(number)
            if len(chosen) == WHITE_BALL_COUNT:
                break
        if len(chosen) == WHITE_BALL_COUNT:
            break
    return validate_ticket(
        {"whites": _fill_whites(chosen, history), "powerball": _powerball_mode(history)}
    )


def neighbor_ticket(history: list[HistoryDraw]) -> dict[str, Any]:
    """White balls one step from the latest draw. The Powerball stays as drawn.

    Each white ball moves up by one, or down by one when it is already 69.
    A collision is filled from the most-hit numbers.
    """
    if not history:
        raise ValueError("neighbor tickets need at least one past draw")
    latest = history[-1]
    chosen: list[int] = []
    for ball in latest.whites:
        candidate = ball + 1 if ball < WHITE_HIGH else ball - 1
        if candidate not in chosen:
            chosen.append(candidate)
    return validate_ticket(
        {"whites": _fill_whites(chosen, history), "powerball": latest.powerball}
    )


def overdue_ticket(history: list[HistoryDraw]) -> dict[str, Any]:
    """Numbers with the longest gap since they last appeared."""
    if not history:
        raise ValueError("overdue tickets need at least one past draw")
    last_white: dict[int, int | None] = dict.fromkeys(range(WHITE_LOW, WHITE_HIGH + 1))
    last_powerball: dict[int, int | None] = dict.fromkeys(
        range(POWERBALL_LOW, POWERBALL_HIGH + 1)
    )
    for index, draw in enumerate(history):
        for ball in draw.whites:
            last_white[ball] = index
        last_powerball[draw.powerball] = index
    total = len(history)

    def gap(last_index: int | None) -> int:
        if last_index is None:
            return total + 1
        return total - 1 - last_index

    whites = sorted(
        sorted(
            range(WHITE_LOW, WHITE_HIGH + 1),
            key=lambda number: (-gap(last_white[number]), number),
        )[:WHITE_BALL_COUNT]
    )
    powerball = sorted(
        range(POWERBALL_LOW, POWERBALL_HIGH + 1),
        key=lambda number: (-gap(last_powerball[number]), number),
    )[0]
    return validate_ticket({"whites": whites, "powerball": powerball})


def random_ticket(draw_date: date, month: str) -> dict[str, Any]:
    """Uniform control ticket. The seed string is part of the public log."""
    seed_text = f"{RANDOM_SEED_PREFIX}-{month}-{draw_date.isoformat()}"
    seed = zlib.crc32(seed_text.encode("utf-8")) & 0xFFFFFFFF
    generator = np.random.default_rng(seed)
    whites = sorted(
        int(number)
        for number in generator.choice(
            np.arange(WHITE_LOW, WHITE_HIGH + 1), size=WHITE_BALL_COUNT, replace=False
        )
    )
    powerball = int(generator.integers(POWERBALL_LOW, POWERBALL_HIGH + 1))
    return validate_ticket({"whites": whites, "powerball": powerball})


def pattern_tickets(
    history: list[HistoryDraw], draw_date: date, month: str
) -> dict[str, dict[str, Any]]:
    """Pattern tickets and the random control for one upcoming draw date."""
    return {
        "most_hit": most_hit_ticket(history),
        "hot_pairs": hot_pairs_ticket(history),
        "neighbors": neighbor_ticket(history),
        "overdue": overdue_ticket(history),
        "random": random_ticket(draw_date, month),
    }


def history_from_frame(frame: pd.DataFrame, *, through: date) -> list[HistoryDraw]:
    """Draws on or before ``through``, in date order."""
    dated = frame.copy()
    dated["draw_date"] = pd.to_datetime(dated["draw_date"]).dt.date
    dated = dated.loc[dated["draw_date"] <= through].sort_values("draw_date")
    history: list[HistoryDraw] = []
    for row in dated.itertuples(index=False):
        whites = tuple(
            int(getattr(row, f"white_{index}"))
            for index in range(1, WHITE_BALL_COUNT + 1)
        )
        history.append(
            HistoryDraw(
                draw_date=row.draw_date,
                whites=(whites[0], whites[1], whites[2], whites[3], whites[4]),
                powerball=int(row.powerball),
            )
        )
    return history


def actuals_by_date(frame: pd.DataFrame) -> dict[date, dict[str, Any]]:
    """Map draw dates to validated tickets."""
    actuals: dict[date, dict[str, Any]] = {}
    for draw in history_from_frame(frame, through=date.max):
        actuals[draw.draw_date] = validate_ticket(
            {"whites": list(draw.whites), "powerball": draw.powerball}
        )
    return actuals


def _empty_totals() -> dict[str, Any]:
    return {
        "tickets": 0,
        "scored": 0,
        "white_matches": 0,
        "powerball_hits": 0,
        "prize_tickets": 0,
        "within_1": 0,
        "within_2": 0,
        "near_miss_1": 0,
        "near_miss_2": 0,
        "white_sum_abs_error": 0,
        "powerball_abs_error": 0,
        "tiers": {},
    }


def _add_score(totals: dict[str, Any], score: dict[str, Any]) -> None:
    totals["scored"] += 1
    totals["white_matches"] += int(score["white_matches"])
    totals["powerball_hits"] += int(score["powerball_hit"])
    for key in (
        "within_1",
        "within_2",
        "near_miss_1",
        "near_miss_2",
        "white_sum_abs_error",
        "powerball_abs_error",
    ):
        totals[key] += int(score[key])
    tier = str(score["tier"])
    tiers = totals["tiers"]
    tiers[tier] = int(tiers.get(tier, 0)) + 1
    if tier != "none":
        totals["prize_tickets"] += 1


def _finalize(totals: dict[str, Any]) -> dict[str, Any]:
    scored = int(totals["scored"])
    chance_white = CHANCE_WHITE_MATCHES * scored
    chance_powerball = CHANCE_POWERBALL_HIT * scored
    return {
        **totals,
        "mean_white_matches": (totals["white_matches"] / scored) if scored else None,
        "expected_white_matches": chance_white,
        "expected_powerball_hits": chance_powerball,
        "white_matches_minus_chance": totals["white_matches"] - chance_white,
        "powerball_hits_minus_chance": totals["powerball_hits"] - chance_powerball,
        "mean_within_1": (totals["within_1"] / scored) if scored else None,
        "mean_within_2": (totals["within_2"] / scored) if scored else None,
        "mean_near_miss_1": (totals["near_miss_1"] / scored) if scored else None,
        "mean_near_miss_2": (totals["near_miss_2"] / scored) if scored else None,
        "mean_white_sum_abs_error": (
            (totals["white_sum_abs_error"] / scored) if scored else None
        ),
    }


def score_logged_months(
    logs: list[dict[str, Any]], actuals: dict[date, dict[str, Any]]
) -> dict[str, Any]:
    """Score every logged ticket whose draw date is in ``actuals``.

    Logged tickets are not modified. Dates with no actual draw stay unscored,
    which covers a draw that was scheduled and then absent from the results.
    """
    months: list[dict[str, Any]] = []
    cumulative: dict[str, dict[str, Any]] = {}
    for log in logs:
        month_totals: dict[str, dict[str, Any]] = {}
        scored_draws: list[dict[str, Any]] = []
        unscored_dates: list[str] = []
        for entry in log.get("draws", []):
            draw_date = date.fromisoformat(str(entry["draw_date"]))
            tickets = entry.get("tickets", {})
            for strategy in tickets:
                month_totals.setdefault(strategy, _empty_totals())
                cumulative.setdefault(strategy, _empty_totals())
                month_totals[strategy]["tickets"] += 1
                cumulative[strategy]["tickets"] += 1
            actual = actuals.get(draw_date)
            if actual is None:
                unscored_dates.append(draw_date.isoformat())
                continue
            strategy_scores = {
                strategy: score_ticket(ticket, actual)
                for strategy, ticket in tickets.items()
            }
            for strategy, score in strategy_scores.items():
                _add_score(month_totals[strategy], score)
                _add_score(cumulative[strategy], score)
            scored_draws.append(
                {
                    "draw_date": draw_date.isoformat(),
                    "actual": actual,
                    "scores": strategy_scores,
                }
            )
        months.append(
            {
                "month": log.get("month"),
                "training_draws_through": log.get("training_draws_through"),
                "scored_draws": scored_draws,
                "unscored_dates": unscored_dates,
                "strategies": {
                    strategy: _finalize(totals)
                    for strategy, totals in month_totals.items()
                },
            }
        )
    finished = {strategy: _finalize(totals) for strategy, totals in cumulative.items()}
    return {
        "schema_version": 1,
        "note": (
            "Scorecard for logged tickets only. No tickets are purchased. "
            "Chance expectations are 25/69 white-ball matches and 1/26 Powerball "
            "hits per ticket. Base prize tiers label matches; they are not payouts."
        ),
        "chance_white_matches_per_ticket": CHANCE_WHITE_MATCHES,
        "chance_powerball_hit_per_ticket": CHANCE_POWERBALL_HIT,
        "months": months,
        "cumulative": finished,
        "summary": _summary(finished),
    }


def _summary(cumulative: dict[str, dict[str, Any]]) -> str:
    if not cumulative or not any(item["scored"] for item in cumulative.values()):
        return "No logged draw has an actual result yet."

    def piece(strategy: str) -> str:
        item = cumulative.get(strategy)
        if item is None or not item["scored"]:
            return f"{strategy}: not scored"
        return (
            f"{strategy}: white matches {item['white_matches']}"
            f" (chance {item['expected_white_matches']:.2f}), "
            f"Powerball hits {item['powerball_hits']}"
            f" (chance {item['expected_powerball_hits']:.2f}), "
            f"prize tiers {item['prize_tickets']}, "
            f"within ±1 {item['mean_within_1']:.2f}"
        )

    scored = next(iter(cumulative.values()))["scored"]
    parts = [piece(strategy) for strategy in STRATEGIES if strategy in cumulative]
    return f"Scored {scored} draws. " + "; ".join(parts) + "."


def assemble_month_log(
    *,
    month: str,
    history: list[HistoryDraw],
    model_ticket: dict[str, Any],
    weather_ticket: dict[str, Any],
    training_draws_through: date,
    generated_on: date,
    weather_note: str,
    feature_columns: list[str],
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Build one month log. Model and pattern tickets repeat; random does not.

    Both model tickets are the one-step forecast from ``training_draws_through``.
    They are copied onto every scheduled draw because no later draw has been
    observed. The random control is redrawn from its public per-date seed.
    """
    model = validate_ticket(model_ticket)
    weather = validate_ticket(weather_ticket)
    draws = []
    for draw_date in draw_dates_for_month(month):
        tickets = pattern_tickets(history, draw_date, month)
        tickets = {"model": model, "model_weather": weather, **tickets}
        draws.append({"draw_date": draw_date.isoformat(), "tickets": tickets})
    return {
        "schema_version": 1,
        "month": month,
        "generated_on": generated_on.isoformat(),
        "training_draws_through": training_draws_through.isoformat(),
        "draw_schedule": "Monday, Wednesday, Saturday",
        "purchases": False,
        "weather": weather_note,
        "feature_columns": feature_columns,
        "hyperparameters": hyperparameters,
        "random_seed_formula": (
            f"crc32('{RANDOM_SEED_PREFIX}-{{month}}-{{YYYY-MM-DD}}') unsigned"
        ),
        "note": (
            "Model and pattern tickets are the one-step forecast from "
            "training_draws_through and are repeated for each scheduled draw. "
            "The random ticket changes with the draw date. "
            "training_draws_through is on or before the day before this month, "
            "including when the file is written later as a backfill. "
            "Weather features, when present, come from earlier draws only. "
            "No ticket is purchased."
        ),
        "draws": draws,
    }


def load_month_logs(directory: str | Path) -> list[dict[str, Any]]:
    """Read ``YYYY-MM.json`` logs from a directory, oldest first."""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    logs: list[dict[str, Any]] = []
    for path in sorted(folder.glob("????-??.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and payload.get("month"):
            logs.append(payload)
    return logs


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON with a trailing newline. Existing files are replaced."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload
