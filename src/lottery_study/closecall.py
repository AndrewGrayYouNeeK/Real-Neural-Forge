"""Close-call scoring for Powerball tickets.

A ticket is five distinct white balls in 1..69 plus a Powerball in 1..26.
Partial-match rates use the hypergeometric distribution of a uniform random
ticket against a uniform draw. Near-miss counts use the same uniform ticket:
a predicted white ball is within ±d of the draw when its nearest drawn white
is at most d away. Exact matches are included in the "within" counts and
reported separately as near-misses (distance exactly 1 or exactly 2).

The monthly scorecard module is not part of this package. When one of the
modules listed in ``SCORECARD_MODULES`` appears and exposes a registration
function, :func:`attach_close_call_scoring` registers these metrics with it.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.lottery import (
    POWERBALL_HIGH,
    POWERBALL_LOW,
    WHITE_BALL_COUNT,
    WHITE_HIGH,
    WHITE_LOW,
)

SCALAR_TARGETS: tuple[str, ...] = (
    "white_sum",
    "powerball",
    "odd_count",
    "near_repeat_3",
    "near_repeat_4",
    "neighbor_of_previous",
    "consecutive_in_draw",
)

SCORECARD_MODULES: tuple[str, ...] = (
    "src.evaluation.monthly_scorecard",
    "src.evaluation.scorecard",
    "src.reporting.monthly_scorecard",
    "src.scorecard",
)

SCORECARD_KEYS: tuple[str, ...] = (
    "mean_whites_matched",
    "powerball_match_rate",
    "rate_at_least_2",
    "rate_at_least_3",
    "rate_at_least_4",
    "mean_within_1",
    "mean_within_2",
    "mean_near_miss_1",
    "mean_near_miss_2",
    "white_sum_mae",
    "full_ticket_exact_rate",
    "mean_powerball_abs_error",
    "powerball_within_1_rate",
    "powerball_within_2_rate",
)

_POPULATION = WHITE_HIGH - WHITE_LOW + 1


def white_match_probability(matches: int) -> float:
    """Probability a random ticket shares exactly ``matches`` white balls."""
    if matches < 0 or matches > WHITE_BALL_COUNT:
        return 0.0
    rest = WHITE_BALL_COUNT - matches
    undrawn = _POPULATION - WHITE_BALL_COUNT
    if rest > undrawn:
        return 0.0
    total = math.comb(_POPULATION, WHITE_BALL_COUNT)
    ways = math.comb(WHITE_BALL_COUNT, matches) * math.comb(undrawn, rest)
    return ways / total


def white_match_at_least(matches: int) -> float:
    """Probability a random ticket shares at least ``matches`` white balls."""
    return float(
        sum(white_match_probability(k) for k in range(matches, WHITE_BALL_COUNT + 1))
    )


def chance_powerball_rate() -> float:
    """Probability a uniform Powerball guess is exact."""
    return 1.0 / (POWERBALL_HIGH - POWERBALL_LOW + 1)


def binomial_two_sided_p(successes: int, n: int, p: float) -> float:
    """Two-sided binomial test, summing outcomes at least as unlikely as observed."""
    if n < 1:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    if p <= 0.0:
        return 1.0 if successes == 0 else 0.0
    if p >= 1.0:
        return 1.0 if successes == n else 0.0
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    logs = np.empty(n + 1, dtype=np.float64)
    for k in range(n + 1):
        logs[k] = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_p
            + (n - k) * log_q
        )
    shifted = np.exp(logs - logs.max())
    probs = shifted / shifted.sum()
    threshold = float(probs[successes])
    return float(probs[probs <= threshold + 1e-15].sum())


def repair_whites(raw: np.ndarray) -> np.ndarray:
    """Round, clip, and replace duplicates with the nearest unused white ball."""
    rounded = np.clip(np.rint(np.asarray(raw, dtype=np.float64)), WHITE_LOW, WHITE_HIGH)
    used: set[int] = set()
    chosen: list[int] = []
    for value in rounded.astype(int).tolist():
        if value not in used:
            used.add(value)
            chosen.append(value)
            continue
        replacement = _nearest_free(value, used)
        used.add(replacement)
        chosen.append(replacement)
    if len(chosen) != WHITE_BALL_COUNT:
        raise ValueError("white repair did not return 5 balls")
    return np.array(sorted(chosen), dtype=np.int64)


def decode_ticket(raw: np.ndarray) -> np.ndarray:
    """Turn six continuous outputs into a legal ticket."""
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != WHITE_BALL_COUNT + 1:
        raise ValueError("a ticket prediction has 6 numbers")
    whites = repair_whites(values[:WHITE_BALL_COUNT])
    powerball = int(np.clip(np.rint(values[-1]), POWERBALL_LOW, POWERBALL_HIGH))
    return np.concatenate([whites, np.array([powerball], dtype=np.int64)])


def decode_tickets(raw: np.ndarray) -> np.ndarray:
    """Decode a batch of continuous rows shaped ``(n, 6)``."""
    rows = np.asarray(raw, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != WHITE_BALL_COUNT + 1:
        raise ValueError("expected shape (n, 6)")
    return np.vstack([decode_ticket(row) for row in rows])


def score_tickets(predicted: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Per-draw close-call counts for integer tickets shaped ``(n, 6)``."""
    pred = _as_tickets(predicted)
    act = _as_tickets(actual)
    if pred.shape != act.shape:
        raise ValueError("predicted and actual tickets must have the same shape")
    pred_w = pred[:, :WHITE_BALL_COUNT]
    act_w = act[:, :WHITE_BALL_COUNT]
    matched = _white_matches(pred_w, act_w)
    distance = _min_distance(pred_w, act_w)
    powerball_abs = np.abs(pred[:, WHITE_BALL_COUNT] - act[:, WHITE_BALL_COUNT])
    return {
        "whites_matched": matched,
        "powerball_matched": (powerball_abs == 0).astype(np.int64),
        "within_1": (distance <= 1).sum(axis=1).astype(np.int64),
        "within_2": (distance <= 2).sum(axis=1).astype(np.int64),
        "near_miss_1": (distance == 1).sum(axis=1).astype(np.int64),
        "near_miss_2": (distance == 2).sum(axis=1).astype(np.int64),
        "powerball_abs_error": powerball_abs.astype(np.float64),
        "powerball_within_1": (powerball_abs <= 1).astype(np.int64),
        "powerball_within_2": (powerball_abs <= 2).astype(np.int64),
        "white_sum_abs_error": np.abs(pred_w.sum(axis=1) - act_w.sum(axis=1)).astype(
            np.float64
        ),
        "full_ticket_exact": (
            (matched == WHITE_BALL_COUNT) & (powerball_abs == 0)
        ).astype(np.int64),
    }


def close_call_scorecard(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Mean close-call rates for a scorecard row.

    ``predicted`` and ``actual`` are integer tickets shaped ``(n, 6)``.
    """
    parts = score_tickets(predicted, actual)
    matched = parts["whites_matched"]
    n = int(matched.size)
    return {
        "n": float(n),
        "mean_whites_matched": float(matched.mean()),
        "powerball_match_rate": float(parts["powerball_matched"].mean()),
        "powerball_exact_matches": float(parts["powerball_matched"].sum()),
        "rate_at_least_2": float(np.mean(matched >= 2)),
        "rate_at_least_3": float(np.mean(matched >= 3)),
        "rate_at_least_4": float(np.mean(matched >= 4)),
        "n_at_least_2": float(np.sum(matched >= 2)),
        "n_at_least_3": float(np.sum(matched >= 3)),
        "n_at_least_4": float(np.sum(matched >= 4)),
        "mean_within_1": float(parts["within_1"].mean()),
        "mean_within_2": float(parts["within_2"].mean()),
        "mean_near_miss_1": float(parts["near_miss_1"].mean()),
        "mean_near_miss_2": float(parts["near_miss_2"].mean()),
        "white_sum_mae": float(parts["white_sum_abs_error"].mean()),
        "full_ticket_exact_rate": float(parts["full_ticket_exact"].mean()),
        "mean_powerball_abs_error": float(parts["powerball_abs_error"].mean()),
        "powerball_within_1_rate": float(parts["powerball_within_1"].mean()),
        "powerball_within_2_rate": float(parts["powerball_within_2"].mean()),
    }


def lottery_target_frame(
    whites: np.ndarray, powerball: np.ndarray
) -> dict[str, np.ndarray]:
    """Draw-level targets, including near-repeat and neighbor events.

    ``near_repeat_3`` / ``near_repeat_4`` are 1 when the white balls share at
    least that many numbers with the previous draw. ``neighbor_of_previous``
    is 1 when any current white ball is exactly one away from any previous
    white ball (an exact repeat does not count). ``consecutive_in_draw`` is 1
    when the draw itself contains two adjacent whites. The first draw has no
    previous draw, so the previous-draw targets are NaN there. ``odd_count``
    counts odd white balls and does not include the Powerball.
    """
    whites = np.asarray(whites, dtype=np.int64)
    powerball = np.asarray(powerball, dtype=np.int64).reshape(-1)
    if whites.ndim != 2 or whites.shape[1] != WHITE_BALL_COUNT:
        raise ValueError("whites must have shape (n, 5)")
    if powerball.shape[0] != whites.shape[0]:
        raise ValueError("powerball length must match the draws")
    n = int(whites.shape[0])
    near3 = np.full(n, np.nan)
    near4 = np.full(n, np.nan)
    neighbor = np.full(n, np.nan)
    if n >= 2:
        shared = _white_matches(whites[1:], whites[:-1]).astype(np.float64)
        near3[1:] = (shared >= 3).astype(np.float64)
        near4[1:] = (shared >= 4).astype(np.float64)
        distance = np.abs(whites[1:, :, None] - whites[:-1, None, :]).min(axis=2)
        neighbor[1:] = np.any(distance == 1, axis=1).astype(np.float64)
    ordered = np.sort(whites, axis=1)
    consecutive = np.any(np.diff(ordered, axis=1) == 1, axis=1).astype(np.float64)
    return {
        "white_sum": whites.sum(axis=1).astype(np.float64),
        "powerball": powerball.astype(np.float64),
        "odd_count": np.sum(whites % 2 == 1, axis=1).astype(np.float64),
        "near_repeat_3": near3,
        "near_repeat_4": near4,
        "neighbor_of_previous": neighbor,
        "consecutive_in_draw": consecutive,
    }


def ball_indicators(
    whites: np.ndarray, powerball: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray]:
    """Per-number indicators: 1 when that white ball or Powerball was drawn."""
    whites = np.asarray(whites, dtype=np.int64)
    powerball = np.asarray(powerball, dtype=np.int64).reshape(-1)
    n = int(whites.shape[0])
    names = tuple(
        f"white_{ball:02d}" for ball in range(WHITE_LOW, WHITE_HIGH + 1)
    ) + tuple(
        f"powerball_{ball:02d}" for ball in range(POWERBALL_LOW, POWERBALL_HIGH + 1)
    )
    matrix = np.zeros((n, len(names)), dtype=np.float64)
    rows = np.arange(n)
    for column in range(WHITE_BALL_COUNT):
        matrix[rows, whites[:, column] - WHITE_LOW] = 1.0
    matrix[rows, _POPULATION + powerball - POWERBALL_LOW] = 1.0
    return names, matrix


def frequency_ticket(
    history_whites: np.ndarray, history_powerball: np.ndarray, *, cold: bool
) -> np.ndarray:
    """The five most (or least) common historical whites, plus the modal Powerball.

    Ties break toward the smaller number so the ticket does not depend on row order.
    """
    white_counts = np.bincount(
        np.asarray(history_whites, dtype=np.int64).ravel(), minlength=WHITE_HIGH + 1
    )[WHITE_LOW:]
    powerball_counts = np.bincount(
        np.asarray(history_powerball, dtype=np.int64), minlength=POWERBALL_HIGH + 1
    )[POWERBALL_LOW:]
    white_key = white_counts if cold else -white_counts
    powerball_key = powerball_counts if cold else -powerball_counts
    white_order = np.lexsort((np.arange(white_counts.size), white_key))
    powerball_order = np.lexsort((np.arange(powerball_counts.size), powerball_key))
    chosen = np.sort(white_order[:WHITE_BALL_COUNT] + WHITE_LOW)
    powerball = int(powerball_order[0] + POWERBALL_LOW)
    return np.concatenate([chosen, np.array([powerball], dtype=np.int64)])


def shift_ticket(previous: np.ndarray) -> np.ndarray:
    """A pattern ticket that moves every number in the previous draw up by one."""
    raw = np.asarray(previous, dtype=np.float64).reshape(-1).copy()
    raw[:WHITE_BALL_COUNT] += 1.0
    raw[-1] += 1.0
    return decode_ticket(raw)


def random_tickets(n: int, rng: np.random.Generator) -> np.ndarray:
    """Seeded uniform legal tickets."""
    if n < 1:
        raise ValueError("n must be positive")
    population = np.arange(WHITE_LOW, WHITE_HIGH + 1)
    out = np.empty((n, WHITE_BALL_COUNT + 1), dtype=np.int64)
    for index in range(n):
        out[index, :WHITE_BALL_COUNT] = np.sort(
            rng.choice(population, size=WHITE_BALL_COUNT, replace=False)
        )
        out[index, WHITE_BALL_COUNT] = int(
            rng.integers(POWERBALL_LOW, POWERBALL_HIGH + 1)
        )
    return out


def raw_sum_abs_error(raw: np.ndarray, actual_whites: np.ndarray) -> np.ndarray:
    """Absolute error of the sum of the five continuous white-ball outputs."""
    predicted = np.asarray(raw, dtype=np.float64)[:, :WHITE_BALL_COUNT].sum(axis=1)
    actual = np.asarray(actual_whites, dtype=np.float64).sum(axis=1)
    return np.asarray(np.abs(predicted - actual), dtype=np.float64)


@dataclass(frozen=True)
class ChanceReference:
    """What a uniform random ticket is expected to do on one fixed holdout."""

    rate_at_least_2: float
    rate_at_least_3: float
    rate_at_least_4: float
    rate_exact_whites: float
    powerball_exact_rate: float
    full_ticket_exact_rate: float
    mean_within_1: float
    mean_within_2: float
    mean_near_miss_1: float
    mean_near_miss_2: float
    within_1_se: float
    within_2_se: float
    powerball_within_1_rate: float
    powerball_within_2_rate: float
    white_sum_mae: float
    white_sum_mae_se: float
    white_pmf: dict[int, float]


def chance_reference(actual: np.ndarray, *, n_sim: int, seed: int) -> ChanceReference:
    """Analytical match rates, plus simulated white-sum error, on ``actual``."""
    tickets = _as_tickets(actual)
    whites = tickets[:, :WHITE_BALL_COUNT]
    powerball = tickets[:, WHITE_BALL_COUNT]
    within_1, within_1_se = _expected_within(whites, tol=1, strict=False)
    within_2, within_2_se = _expected_within(whites, tol=2, strict=False)
    near_1, _ = _expected_within(whites, tol=1, strict=True)
    near_2, _ = _expected_within(whites, tol=2, strict=True)
    sum_mae, sum_se = _expected_sum_mae(whites, n_sim=n_sim, seed=seed)
    pmf = {k: white_match_probability(k) for k in range(WHITE_BALL_COUNT + 1)}
    exact_whites = pmf[WHITE_BALL_COUNT]
    powerball_rate = chance_powerball_rate()
    return ChanceReference(
        rate_at_least_2=white_match_at_least(2),
        rate_at_least_3=white_match_at_least(3),
        rate_at_least_4=white_match_at_least(4),
        rate_exact_whites=exact_whites,
        powerball_exact_rate=powerball_rate,
        full_ticket_exact_rate=exact_whites * powerball_rate,
        mean_within_1=within_1,
        mean_within_2=within_2,
        mean_near_miss_1=near_1,
        mean_near_miss_2=near_2,
        within_1_se=within_1_se,
        within_2_se=within_2_se,
        powerball_within_1_rate=_powerball_within_rate(powerball, 1),
        powerball_within_2_rate=_powerball_within_rate(powerball, 2),
        white_sum_mae=sum_mae,
        white_sum_mae_se=sum_se,
        white_pmf=pmf,
    )


def metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Scorecard callbacks keyed by metric name."""

    def _make(key: str) -> Callable[[np.ndarray, np.ndarray], float]:
        def _metric(predicted: np.ndarray, actual: np.ndarray) -> float:
            return float(close_call_scorecard(predicted, actual)[key])

        return _metric

    return {key: _make(key) for key in SCORECARD_KEYS}


def attach_to_scorecard_module(module: Any) -> bool:
    """Register close-call metrics on a scorecard-like module.

    Supported hooks, in order: ``register_close_call_metrics(scorecard_fn)``,
    then ``register_metric(name, fn)`` or ``add_metric(name, fn)``.
    """
    register_all = getattr(module, "register_close_call_metrics", None)
    if callable(register_all):
        register_all(close_call_scorecard)
        return True
    register = getattr(module, "register_metric", None)
    if not callable(register):
        register = getattr(module, "add_metric", None)
    if not callable(register):
        return False
    for name, fn in metric_functions().items():
        register(name, fn)
    return True


def attach_close_call_scoring() -> str | None:
    """Attach to a monthly scorecard module if one is importable. Return its name."""
    for name in SCORECARD_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if attach_to_scorecard_module(module):
            return name
    return None


def _nearest_free(value: int, used: set[int]) -> int:
    for delta in range(1, _POPULATION):
        for candidate in (value - delta, value + delta):
            if WHITE_LOW <= candidate <= WHITE_HIGH and candidate not in used:
                return candidate
    raise RuntimeError("no unused white ball left")


def _as_tickets(tickets: np.ndarray) -> np.ndarray:
    values = np.asarray(tickets, dtype=np.int64)
    if (
        values.ndim != 2
        or values.shape[1] != WHITE_BALL_COUNT + 1
        or values.shape[0] < 1
    ):
        raise ValueError("tickets must have shape (n, 6) with n >= 1")
    return values


def _white_matches(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    equal = pred[:, :, None] == actual[:, None, :]
    counts = equal.any(axis=2).sum(axis=1)
    return np.asarray(counts, dtype=np.int64)


def _min_distance(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    distance = np.abs(pred[:, :, None] - actual[:, None, :]).min(axis=2)
    return np.asarray(distance, dtype=np.int64)


def _marked_count(whites: np.ndarray, tol: int, *, strict: bool) -> int:
    exact = {int(value) for value in whites}
    marked: set[int] = set()
    for ball in exact:
        low = max(WHITE_LOW, ball - tol)
        high = min(WHITE_HIGH, ball + tol)
        marked.update(range(low, high + 1))
    if strict:
        marked -= exact
    return len(marked)


def _expected_within(
    whites: np.ndarray, *, tol: int, strict: bool
) -> tuple[float, float]:
    means: list[float] = []
    variances: list[float] = []
    for row in whites:
        marked = _marked_count(row, tol, strict=strict)
        mean = WHITE_BALL_COUNT * marked / _POPULATION
        variance = (
            mean
            * (1.0 - marked / _POPULATION)
            * (_POPULATION - WHITE_BALL_COUNT)
            / (_POPULATION - 1)
        )
        means.append(mean)
        variances.append(variance)
    expected = float(np.mean(means))
    se = math.sqrt(float(np.mean(variances)) / len(means))
    return expected, se


def _powerball_within_rate(powerball: np.ndarray, tol: int) -> float:
    guesses = np.arange(POWERBALL_LOW, POWERBALL_HIGH + 1)
    distance = np.abs(guesses[None, :] - powerball[:, None])
    return float(np.mean(distance <= tol))


def _expected_sum_mae(
    whites: np.ndarray, *, n_sim: int, seed: int
) -> tuple[float, float]:
    if n_sim < 1:
        raise ValueError("n_sim must be positive")
    rng = np.random.default_rng(seed)
    population = np.arange(WHITE_LOW, WHITE_HIGH + 1)
    sums = np.empty(n_sim, dtype=np.float64)
    for index in range(n_sim):
        sums[index] = float(
            rng.choice(population, size=WHITE_BALL_COUNT, replace=False).sum()
        )
    actual_sums = whites.sum(axis=1).astype(np.float64)
    errors = np.abs(sums[:, None] - actual_sums[None, :])
    per_draw_mean = errors.mean(axis=0)
    per_draw_var = errors.var(axis=0)
    se = math.sqrt(float(per_draw_var.mean()) / actual_sums.size)
    return float(per_draw_mean.mean()), se
