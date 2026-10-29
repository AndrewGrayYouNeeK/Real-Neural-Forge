"""Do lunar features tell you anything about lottery draws?

Each draw is one observation. Permutation tests shuffle the phase label of
the draw and leave its balls alone, so the five white balls are not treated
as five independent trials. The chi-square p-value is reported next to that
permutation p-value and is only a rough check. Benjamini–Hochberg and
Bonferroni are applied to the permutation p-values.

A second, separate family trains the existing lottery transformer with and
without the lunar columns and scores the held-out draws against random
tickets. Moon phase on the draw date is known before the draw, so those
columns are not a leak of the winning numbers.

Lottery draws are built to be unpredictable. A small p-value here is a
statement about this sample, not a reason to buy a ticket.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.data.lottery import fetch_and_write as fetch_powerball
from src.data.lottery import read_draw_table as read_powerball
from src.data.lunar import (
    LUNAR_MODEL_COLUMNS,
    PHASE_NAMES,
    lunar_feature_frame,
)
from src.data.mega_millions import fetch_and_write as fetch_mega
from src.data.mega_millions import read_draw_table as read_mega
from src.evaluation.lottery import expected_uniform_score
from src.lottery_study.closecall import binomial_two_sided_p
from src.lottery_study.modeling import build_windows, train_holdout
from src.lottery_study.stats import (
    benjamini_hochberg,
    bonferroni_adjust,
    expected_luck_hits,
)
from src.utils.paths import resolve_repo_path

PUBLISHED_REPORT = "results/lunar_lottery_analysis.json"
DRAW_TIMEZONE = "America/New_York"
DRAW_CLOCK = "22:59"
N_PHASES = len(PHASE_NAMES)
ALPHA = 0.05

DEFAULT_TRAINING: dict[str, Any] = {
    "epochs": 8,
    "patience": 3,
    "batch_size": 64,
    "learning_rate": 0.001,
    "seed": 0,
    "device": "cpu",
    "val_split": 0.2,
    "d_model": 32,
    "nhead": 4,
    "num_encoder_layers": 2,
    "dim_feedforward": 64,
    "dropout": 0.0,
    "max_seq_len": 64,
    "max_grad_norm": 1.0,
}


@dataclass(frozen=True)
class LotteryGame:
    """White balls plus one bonus number, for any similar draw."""

    game_id: str
    name: str
    bonus_name: str
    white_count: int
    white_low: int
    white_high: int
    bonus_low: int
    bonus_high: int
    white_columns: tuple[str, ...]
    bonus_column: str

    def expected_white_matches(self) -> float:
        """Expected overlap of two uniform tickets. Equals ``k * k / N``."""
        return (self.white_count * self.white_count) / self.white_high

    def bonus_span(self) -> int:
        return self.bonus_high - self.bonus_low + 1


POWERBALL = LotteryGame(
    game_id="powerball",
    name="Powerball",
    bonus_name="Powerball",
    white_count=5,
    white_low=1,
    white_high=69,
    bonus_low=1,
    bonus_high=26,
    white_columns=("white_1", "white_2", "white_3", "white_4", "white_5"),
    bonus_column="powerball",
)
MEGA_MILLIONS = LotteryGame(
    game_id="mega_millions",
    name="Mega Millions",
    bonus_name="Mega Ball",
    white_count=5,
    white_low=1,
    white_high=70,
    bonus_low=1,
    bonus_high=25,
    white_columns=("white_1", "white_2", "white_3", "white_4", "white_5"),
    bonus_column="mega_ball",
)


def phase_codes(phase: np.ndarray) -> np.ndarray:
    """Map phase fractions onto the eight named phases, centered on each one."""
    shifted = np.mod(np.asarray(phase, dtype=np.float64) + (1.0 / 16.0), 1.0)
    codes = (np.floor(shifted * N_PHASES).astype(np.int64)) % N_PHASES
    return np.asarray(codes, dtype=np.int64)


def high_ball_count(whites: np.ndarray, game: LotteryGame) -> np.ndarray:
    """How many white balls sit in the upper half of the legal range.

    The midpoint is included in the high half (``2 * ball >= low + high``).
    For Powerball that is 35–69.
    """
    balls = np.asarray(whites)
    counts = np.sum(balls * 2 >= (game.white_low + game.white_high), axis=1)
    return np.asarray(counts, dtype=np.float64)


def odd_ball_count(whites: np.ndarray) -> np.ndarray:
    """How many white balls are odd."""
    counts = np.sum(np.asarray(whites) % 2 == 1, axis=1)
    return np.asarray(counts, dtype=np.float64)


def _reduced_table(counts: np.ndarray) -> np.ndarray:
    row_keep = counts.sum(axis=1) > 0
    col_keep = counts.sum(axis=0) > 0
    return counts[np.ix_(row_keep, col_keep)]


def chi_square_table(counts: np.ndarray) -> dict[str, float]:
    """Chi-square, degrees of freedom, asymptotic p, and Cramer's V.

    Empty rows and columns are dropped. A table smaller than 2×2 has no
    association to measure and returns a p-value of 1.
    """
    table = _reduced_table(np.asarray(counts, dtype=np.float64))
    total = float(table.sum())
    if table.shape[0] < 2 or table.shape[1] < 2 or total <= 0.0:
        return {
            "statistic": 0.0,
            "dof": 0.0,
            "p_chi2": 1.0,
            "cramers_v": 0.0,
            "min_expected": 0.0,
        }
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True)
    expected = expected / total
    statistic = float(np.sum((table - expected) ** 2 / expected))
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    p_value = float(chi2.sf(statistic, dof))
    cramers = math.sqrt(
        statistic / (total * min(table.shape[0] - 1, table.shape[1] - 1))
    )
    return {
        "statistic": statistic,
        "dof": float(dof),
        "p_chi2": p_value,
        "cramers_v": float(cramers),
        "min_expected": float(expected.min()),
    }


def _number_counts(
    codes: np.ndarray, numbers: np.ndarray, low: int, high: int
) -> np.ndarray:
    width = high - low + 1
    counts = np.zeros((N_PHASES, width), dtype=np.float64)
    flat_codes = np.repeat(codes, numbers.shape[1]) if numbers.ndim == 2 else codes
    flat_numbers = np.asarray(numbers, dtype=np.int64).reshape(-1)
    if flat_codes.shape[0] != flat_numbers.shape[0]:
        raise ValueError("phase codes and numbers have different lengths")
    np.add.at(counts, (flat_codes, flat_numbers - low), 1.0)
    return counts


def _permutation_p(observed: float, samples: np.ndarray) -> float:
    exceedances = int(np.sum(samples >= observed - 1e-12))
    return (exceedances + 1) / (samples.size + 1)


def _eta_squared(values: np.ndarray, codes: np.ndarray) -> float:
    outcome = np.asarray(values, dtype=np.float64)
    labels = np.asarray(codes, dtype=np.int64)
    center = float(outcome.mean())
    total = float(np.sum((outcome - center) ** 2))
    if total == 0.0:
        return 0.0
    between = 0.0
    for code in np.unique(labels):
        block = outcome[labels == code]
        between += float(block.size) * (float(block.mean()) - center) ** 2
    return between / total


def _phase_means(values: np.ndarray, codes: np.ndarray) -> dict[str, float]:
    outcome = np.asarray(values, dtype=np.float64)
    labels = np.asarray(codes, dtype=np.int64)
    means: dict[str, float] = {}
    for code in range(N_PHASES):
        block = outcome[labels == code]
        if block.size:
            means[PHASE_NAMES[code]] = float(block.mean())
    return means


def _shuffled_stats(
    codes: np.ndarray,
    statistic,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    samples = np.empty(n_perm, dtype=np.float64)
    for index in range(n_perm):
        samples[index] = float(statistic(rng.permutation(codes)))
    return samples


def association_tests(
    phase: np.ndarray,
    whites: np.ndarray,
    bonus: np.ndarray,
    game: LotteryGame,
    *,
    n_perm: int = 1999,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Phase tests for one game. The permutation p-value is the one to cite."""
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    codes = phase_codes(phase)
    whites = np.asarray(whites, dtype=np.int64)
    bonus = np.asarray(bonus, dtype=np.int64).reshape(-1)
    if whites.shape[0] != codes.shape[0] or bonus.shape[0] != codes.shape[0]:
        raise ValueError("phase, white balls, and bonus must have one row per draw")
    rng = np.random.default_rng(seed)
    tests: list[dict[str, Any]] = []

    def frequency_test(
        test_id: str,
        label: str,
        numbers: np.ndarray,
        low: int,
        high: int,
    ) -> None:
        counts = _number_counts(codes, numbers, low, high)
        fitted = chi_square_table(counts)
        observed = fitted["statistic"]

        def statistic(shuffled: np.ndarray) -> float:
            return chi_square_table(_number_counts(shuffled, numbers, low, high))[
                "statistic"
            ]

        samples = _shuffled_stats(codes, statistic, n_perm, rng)
        tests.append(
            {
                "id": test_id,
                "label": label,
                "effect_name": "cramers_v",
                "effect": fitted["cramers_v"],
                "effect_note": (
                    "Cramer's V near 0 is a weak association. "
                    "0.1 is small for a table this wide."
                ),
                "statistic_name": "chi_square",
                "statistic": fitted["statistic"],
                "dof": fitted["dof"],
                "min_expected": fitted["min_expected"],
                "p_chi2": fitted["p_chi2"],
                "p_permutation": _permutation_p(observed, samples),
                "phase_means": {},
                "mean_span": None,
                "chi2_note": (
                    "The chi-square p-value treats each ball as independent. "
                    "Balls from the same draw are not, so use the permutation p-value."
                ),
            }
        )

    frequency_test(
        "white_numbers_by_phase",
        "White-ball numbers by moon phase",
        whites,
        game.white_low,
        game.white_high,
    )
    frequency_test(
        "bonus_numbers_by_phase",
        f"{game.bonus_name} by moon phase",
        bonus.reshape(-1, 1),
        game.bonus_low,
        game.bonus_high,
    )

    def numeric_test(test_id: str, label: str, values: np.ndarray, unit: str) -> None:
        observed = _eta_squared(values, codes)
        samples = _shuffled_stats(
            codes,
            lambda shuffled: _eta_squared(values, shuffled),
            n_perm,
            rng,
        )
        means = _phase_means(values, codes)
        span = max(means.values()) - min(means.values()) if means else 0.0
        tests.append(
            {
                "id": test_id,
                "label": label,
                "effect_name": "eta_squared",
                "effect": observed,
                "effect_note": (
                    "Eta-squared is the share of the draw-to-draw variance "
                    f"that lines up with phase. The phase means differ by "
                    f"{span:.3f} {unit}."
                ),
                "statistic_name": "eta_squared",
                "statistic": observed,
                "dof": None,
                "min_expected": None,
                "p_chi2": None,
                "p_permutation": _permutation_p(observed, samples),
                "phase_means": means,
                "mean_span": span,
                "chi2_note": None,
            }
        )

    numeric_test(
        "white_sum_by_phase",
        "White-ball sum by moon phase",
        whites.sum(axis=1).astype(np.float64),
        "points of sum",
    )
    numeric_test(
        "odd_count_by_phase",
        "Odd white balls by moon phase",
        odd_ball_count(whites),
        "odd balls",
    )
    numeric_test(
        "high_count_by_phase",
        "High white balls by moon phase",
        high_ball_count(whites, game),
        "high balls",
    )
    return tests


def _nearest_free(value: int, used: set[int], low: int, high: int) -> int:
    for offset in range(0, high - low + 1):
        for candidate in (value - offset, value + offset):
            if low <= candidate <= high and candidate not in used:
                return candidate
    raise ValueError("no unused ball remains in range")


def repair_whites(raw: np.ndarray, game: LotteryGame) -> np.ndarray:
    """Round, clip, and replace duplicates with the nearest unused white ball."""
    rounded = np.clip(
        np.rint(np.asarray(raw, dtype=np.float64)),
        game.white_low,
        game.white_high,
    )
    used: set[int] = set()
    chosen: list[int] = []
    for value in rounded.astype(int).tolist():
        if value not in used:
            used.add(value)
            chosen.append(value)
            continue
        replacement = _nearest_free(value, used, game.white_low, game.white_high)
        used.add(replacement)
        chosen.append(replacement)
    return np.array(sorted(chosen), dtype=np.int64)


def decode_tickets(raw: np.ndarray, game: LotteryGame) -> np.ndarray:
    """Turn continuous rows into legal tickets shaped ``(n, white_count + 1)``."""
    rows = np.asarray(raw, dtype=np.float64)
    width = game.white_count + 1
    if rows.ndim != 2 or rows.shape[1] != width:
        raise ValueError(f"expected shape (n, {width})")
    decoded = np.empty_like(rows, dtype=np.int64)
    for index, row in enumerate(rows):
        decoded[index, : game.white_count] = repair_whites(
            row[: game.white_count], game
        )
        bonus = int(np.clip(np.rint(row[-1]), game.bonus_low, game.bonus_high))
        decoded[index, -1] = bonus
    return decoded


def white_match_counts(
    predicted: np.ndarray, actual: np.ndarray, game: LotteryGame
) -> np.ndarray:
    """How many white balls each predicted ticket shares with the draw."""
    pred = np.asarray(predicted)[:, : game.white_count]
    act = np.asarray(actual)[:, : game.white_count]
    shared = pred[:, :, None] == act[:, None, :]
    counts = shared.any(axis=2).sum(axis=1)
    return np.asarray(counts, dtype=np.int64)


def random_tickets(n: int, game: LotteryGame, seed: int) -> np.ndarray:
    """``n`` uniform legal tickets."""
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    tickets = np.empty((n, game.white_count + 1), dtype=np.int64)
    population = np.arange(game.white_low, game.white_high + 1)
    for index in range(n):
        pick = rng.choice(population, size=game.white_count, replace=False)
        tickets[index, : game.white_count] = np.sort(pick)
        tickets[index, -1] = int(rng.integers(game.bonus_low, game.bonus_high + 1))
    return tickets


def _ticket_summary(
    tickets: np.ndarray,
    actual: np.ndarray,
    game: LotteryGame,
) -> dict[str, float]:
    matches = white_match_counts(tickets, actual, game)
    bonus_hits = int(np.sum(tickets[:, -1] == actual[:, -1]))
    n = int(actual.shape[0])
    return {
        "n": float(n),
        "mean_white_matches": float(matches.mean()),
        "total_white_matches": float(matches.sum()),
        "bonus_hits": float(bonus_hits),
        "bonus_hit_rate": bonus_hits / n,
    }


def _chance_ticket_summary(n: int, game: LotteryGame) -> dict[str, float]:
    white = game.expected_white_matches()
    bonus = 1.0 / game.bonus_span()
    return {
        "n": float(n),
        "mean_white_matches": white,
        "total_white_matches": white * n,
        "bonus_hits": bonus * n,
        "bonus_hit_rate": bonus,
    }


def _monte_carlo_white_p(
    actual: np.ndarray,
    observed_total: float,
    game: LotteryGame,
    *,
    n_sim: int,
    seed: int,
) -> float:
    n = int(actual.shape[0])
    expected = game.expected_white_matches() * n
    gap = abs(observed_total - expected)
    exceedances = 0
    for index in range(n_sim):
        tickets = random_tickets(n, game, seed + 1 + index)
        total = float(white_match_counts(tickets, actual, game).sum())
        if abs(total - expected) >= gap - 1e-9:
            exceedances += 1
    return (exceedances + 1) / (n_sim + 1)


def _paired_p(
    lunar_matches: np.ndarray, plain_matches: np.ndarray, n_perm: int, seed: int
) -> dict[str, float]:
    diff = lunar_matches.astype(np.float64) - plain_matches.astype(np.float64)
    observed = float(diff.mean())
    scale = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    cohen_d = 0.0 if scale == 0.0 else observed / scale
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size))
    permuted = (signs * diff).mean(axis=1)
    exceedances = int(np.sum(np.abs(permuted) >= abs(observed) - 1e-12))
    return {
        "mean_white_match_difference": observed,
        "cohens_d": float(cohen_d),
        "p_permutation": (exceedances + 1) / (n_perm + 1),
    }


def compare_models(
    whites: np.ndarray,
    bonus: np.ndarray,
    dates: list[Any],
    game: LotteryGame,
    *,
    training: dict[str, Any] | None = None,
    seq_len: int = 16,
    n_perm: int = 1999,
    n_sim: int = 999,
    seed: int = 0,
) -> dict[str, Any]:
    """Hold out the latest draws and score plain vs lunar vs a random ticket.

    Both models see the same temporal split. Lunar columns are the moon on
    the draw date at 22:59 America/New_York, which is known before the draw.
    """
    settings = dict(DEFAULT_TRAINING)
    if training:
        settings.update(training)
    settings["seed"] = seed
    balls = np.column_stack(
        [np.asarray(whites, dtype=np.float64), np.asarray(bonus, dtype=np.float64)]
    )
    lunar = lunar_feature_frame(dates, tz=DRAW_TIMEZONE, at=DRAW_CLOCK)
    covariates = lunar.loc[:, list(LUNAR_MODEL_COLUMNS)].to_numpy(dtype=np.float64)
    plain_x, plain_y, _plain_index = build_windows(balls, None, seq_len)
    lunar_x, lunar_y, _lunar_index = build_windows(balls, covariates, seq_len)
    if plain_y.shape != lunar_y.shape:
        raise ValueError("the plain and lunar windows do not cover the same draws")
    plain_index, plain_pred = train_holdout(
        plain_x, plain_y, n_draws=balls.shape[0], training=settings
    )
    lunar_index, lunar_pred = train_holdout(
        lunar_x, lunar_y, n_draws=balls.shape[0], training=settings
    )
    if not np.array_equal(plain_index, lunar_index):
        raise ValueError("the plain and lunar holdouts are not the same draws")
    actual = balls[plain_index].astype(np.int64)
    plain_tickets = decode_tickets(plain_pred, game)
    lunar_tickets = decode_tickets(lunar_pred, game)
    random = random_tickets(actual.shape[0], game, seed)
    plain_matches = white_match_counts(plain_tickets, actual, game)
    lunar_matches = white_match_counts(lunar_tickets, actual, game)
    n = int(actual.shape[0])
    bonus_targets = actual[:, -1].astype(np.float64)
    plain_bonus = plain_pred[:, -1]
    lunar_bonus = lunar_pred[:, -1]
    chance_bonus = expected_uniform_score(
        bonus_targets, game.bonus_low, game.bonus_high
    )
    paired = _paired_p(lunar_matches, plain_matches, n_perm, seed + 17)
    lunar_summary = _ticket_summary(lunar_tickets, actual, game)
    plain_summary = _ticket_summary(plain_tickets, actual, game)
    return {
        "seq_len": seq_len,
        "holdout_draws": n,
        "first_holdout_index": int(plain_index[0]),
        "training": {
            key: settings[key]
            for key in (
                "epochs",
                "patience",
                "batch_size",
                "learning_rate",
                "seed",
                "d_model",
                "nhead",
                "num_encoder_layers",
                "dim_feedforward",
                "dropout",
                "val_split",
            )
            if key in settings
        },
        "lunar_columns": list(LUNAR_MODEL_COLUMNS),
        "plain": plain_summary,
        "lunar": lunar_summary,
        "random_ticket": _ticket_summary(random, actual, game),
        "chance": _chance_ticket_summary(n, game),
        "plain_vs_chance": {
            "white_match_p": _monte_carlo_white_p(
                actual,
                plain_summary["total_white_matches"],
                game,
                n_sim=n_sim,
                seed=seed + 100,
            ),
            "bonus_hit_p": binomial_two_sided_p(
                int(plain_summary["bonus_hits"]), n, 1.0 / game.bonus_span()
            ),
            "bonus_mae": _mae(plain_bonus, bonus_targets),
            "chance_bonus_mae": float(chance_bonus["mae"]),
        },
        "lunar_vs_chance": {
            "white_match_p": _monte_carlo_white_p(
                actual,
                lunar_summary["total_white_matches"],
                game,
                n_sim=n_sim,
                seed=seed + 200,
            ),
            "bonus_hit_p": binomial_two_sided_p(
                int(lunar_summary["bonus_hits"]), n, 1.0 / game.bonus_span()
            ),
            "bonus_mae": _mae(lunar_bonus, bonus_targets),
            "chance_bonus_mae": float(chance_bonus["mae"]),
        },
        "lunar_minus_plain": paired,
    }


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


def _correction_summary(
    tests: list[dict[str, Any]],
    *,
    n_perm: int,
    bh_key: str,
    bonferroni_key: str,
    survive_bh_key: str,
    survive_bonferroni_key: str,
) -> dict[str, Any]:
    """Attach adjusted p-values. Keys differ for the within-game and pooled families."""
    p_values = np.array(
        [float(test["p_permutation"]) for test in tests], dtype=np.float64
    )
    bh = benjamini_hochberg(p_values)
    bonferroni = bonferroni_adjust(p_values)
    for test, bh_p, bon_p in zip(tests, bh, bonferroni, strict=True):
        test[bh_key] = float(bh_p)
        test[bonferroni_key] = float(bon_p)
        test[survive_bh_key] = bool(float(bh_p) < ALPHA)
        test[survive_bonferroni_key] = bool(float(bon_p) < ALPHA)
    return {
        "family": "lunar association, permutation p-values",
        "n_tests": int(p_values.size),
        "alpha": ALPHA,
        "expected_luck_hits": expected_luck_hits(int(p_values.size), ALPHA),
        "n_raw_below_alpha": int(np.sum(p_values < ALPHA)),
        "n_bh_rejects": int(np.sum(bh < ALPHA)),
        "n_bonferroni_rejects": int(np.sum(bonferroni < ALPHA)),
        "permutation_floor": 1.0 / (n_perm + 1),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer | np.bool_):
        return int(value) if isinstance(value, np.integer) else bool(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _dates_from_frame(frame: pd.DataFrame) -> list[Any]:
    parsed = pd.to_datetime(frame["draw_date"])
    return [value.date() for value in parsed.tolist()]


def _balls(frame: pd.DataFrame, game: LotteryGame) -> tuple[np.ndarray, np.ndarray]:
    whites = frame.loc[:, list(game.white_columns)].to_numpy(dtype=np.int64)
    bonus = frame[game.bonus_column].to_numpy(dtype=np.int64)
    return whites, bonus


def analyze_game(
    frame: pd.DataFrame,
    game: LotteryGame,
    *,
    n_perm: int = 1999,
    n_sim: int = 999,
    training: dict[str, Any] | None = None,
    seq_len: int = 16,
    seed: int = 0,
    fit_models: bool = True,
) -> dict[str, Any]:
    """Association tests, and optionally the with/without model comparison."""
    dates = _dates_from_frame(frame)
    whites, bonus = _balls(frame, game)
    lunar = lunar_feature_frame(dates, tz=DRAW_TIMEZONE, at=DRAW_CLOCK)
    tests = association_tests(
        lunar["moon_phase"].to_numpy(dtype=np.float64),
        whites,
        bonus,
        game,
        n_perm=n_perm,
        seed=seed,
    )
    correction = _correction_summary(
        tests,
        n_perm=n_perm,
        bh_key="p_bh_within_game",
        bonferroni_key="p_bonferroni_within_game",
        survive_bh_key="survives_bh_within_game",
        survive_bonferroni_key="survives_bonferroni_within_game",
    )
    payload: dict[str, Any] = {
        "game_id": game.game_id,
        "game": game.name,
        "bonus_name": game.bonus_name,
        "n_draws": int(len(frame)),
        "date_start": dates[0].isoformat(),
        "date_end": dates[-1].isoformat(),
        "matrix": {
            "white": f"{game.white_count}/{game.white_high}",
            "bonus": f"1/{game.bonus_span()}",
        },
        "expected_white_matches": game.expected_white_matches(),
        "tests": tests,
        "correction": correction,
    }
    if fit_models:
        payload["models"] = compare_models(
            whites,
            bonus,
            dates,
            game,
            training=training,
            seq_len=seq_len,
            n_perm=n_perm,
            n_sim=n_sim,
            seed=seed,
        )
    return payload


def _strongest(tests: list[dict[str, Any]]) -> dict[str, Any]:
    return min(tests, key=lambda test: float(test["p_permutation"]))


def _fmt_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _game_sentence(game_report: dict[str, Any]) -> str:
    tests = game_report["tests"]
    strongest = _strongest(tests)
    effect = float(strongest["effect"])
    sentence = (
        f"{game_report['game']}: {game_report['n_draws']} draws from "
        f"{game_report['date_start']} to {game_report['date_end']}. "
        f"The smallest permutation p-value is {_fmt_p(float(strongest['p_permutation']))} "
        f"for {strongest['label']} "
        f"({strongest['effect_name']} {effect:.4f}"
    )
    span = strongest.get("mean_span")
    if span is not None:
        sentence += f", phase means differ by {float(span):.3f}"
    sentence += "). "
    correction = game_report["correction"]
    sentence += (
        f"Benjamini–Hochberg keeps {correction['n_bh_rejects']} of "
        f"{correction['n_tests']} tests in this game."
    )
    models = game_report.get("models")
    if not models:
        return sentence
    chance = models["chance"]
    plain = models["plain"]
    lunar = models["lunar"]
    paired = models["lunar_minus_plain"]
    lunar_vs = models["lunar_vs_chance"]
    direction = "fewer" if paired["mean_white_match_difference"] < 0 else "more"
    sentence += (
        f" On {models['holdout_draws']} held-out draws a random ticket expects "
        f"{chance['mean_white_matches']:.3f} white-ball matches and "
        f"{chance['bonus_hits']:.2f} {game_report['bonus_name']} hits. "
        f"The plain model averaged {plain['mean_white_matches']:.3f} white matches "
        f"and hit the {game_report['bonus_name']} {int(plain['bonus_hits'])} times. "
        f"The lunar model averaged {lunar['mean_white_matches']:.3f} "
        f"(versus chance, p {_fmt_p(float(lunar_vs['white_match_p']))}) and hit the "
        f"{game_report['bonus_name']} {int(lunar['bonus_hits'])} times "
        f"(p {_fmt_p(float(lunar_vs['bonus_hit_p']))}). "
        f"One seeded random ticket on the same draws averaged "
        f"{models['random_ticket']['mean_white_matches']:.3f} white matches and "
        f"hit the {game_report['bonus_name']} {int(models['random_ticket']['bonus_hits'])} times. "
        f"Adding the moon gave the lunar model {direction} white matches than the "
        f"plain model ({paired['mean_white_match_difference']:+.3f} per draw, "
        f"p {_fmt_p(float(paired['p_permutation']))}, Cohen's d {paired['cohens_d']:+.2f}). "
        f"Bonus MAE is {lunar_vs['bonus_mae']:.2f} with the moon and "
        f"{models['plain_vs_chance']['bonus_mae']:.2f} without it, against "
        f"{lunar_vs['chance_bonus_mae']:.2f} for a uniform guess. A guess near the "
        f"middle of the range beats a uniform draw on absolute error even when it "
        f"does not hit the number."
    )
    return sentence


def _model_chance_family(games: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Benjamini–Hochberg on lunar-model versus random-ticket scores."""
    p_values = []
    for game in games:
        models = game.get("models")
        if not isinstance(models, dict):
            continue
        lunar_vs = models.get("lunar_vs_chance")
        if not isinstance(lunar_vs, dict):
            continue
        p_values.extend(
            [float(lunar_vs["white_match_p"]), float(lunar_vs["bonus_hit_p"])]
        )
    if not p_values:
        return None
    raw = np.asarray(p_values, dtype=np.float64)
    adjusted = benjamini_hochberg(raw)
    return {
        "family": "lunar model versus a random ticket",
        "n_tests": int(raw.size),
        "alpha": ALPHA,
        "expected_luck_hits": expected_luck_hits(int(raw.size), ALPHA),
        "n_raw_below_alpha": int(np.sum(raw < ALPHA)),
        "n_bh_rejects": int(np.sum(adjusted < ALPHA)),
        "p_bh": [float(value) for value in adjusted],
    }


def _summary(games: list[dict[str, Any]], pooled: dict[str, Any]) -> str:
    pieces = [
        (
            f"{pooled['n_tests']} tests asked whether moon phase lines up with "
            f"the numbers. If every null were true, about "
            f"{pooled['expected_luck_hits']:.2f} of them would fall under "
            f"p = {pooled['alpha']} by luck. {pooled['n_raw_below_alpha']} did. "
            f"Benjamini–Hochberg kept {pooled['n_bh_rejects']}; "
            f"Bonferroni kept {pooled['n_bonferroni_rejects']}. "
            f"Permutation p-values cannot go below {pooled['permutation_floor']:.4f} "
            f"with this shuffle count."
        )
    ]
    pieces.extend(_game_sentence(game) for game in games)
    model_family = _model_chance_family(games)
    if model_family is not None:
        pieces.append(
            "The four lunar-model scores against a random ticket "
            f"(white matches and bonus hits, two games) had "
            f"{model_family['n_raw_below_alpha']} raw p-values under "
            f"{model_family['alpha']}. Benjamini–Hochberg kept "
            f"{model_family['n_bh_rejects']} of them. "
            "The paired Powerball and Mega Millions differences point in "
            "opposite directions, so they are not one lunar effect."
        )
    if pooled["n_bh_rejects"] == 0:
        pieces.append(
            "Nothing in this sample survives a multiple-testing correction. "
            "That is what random draws look like. The moon is not a reason to "
            "expect a number."
        )
    else:
        pieces.append(
            "A row that survives correction is still a small association in a "
            "lottery. There is no credible way for the moon to set the next "
            "draw. It would have to show up again, at a similar size, in later "
            "draws before it meant anything."
        )
    return " ".join(pieces)


def build_report(
    frames: dict[str, pd.DataFrame] | None = None,
    *,
    n_perm: int = 1999,
    n_sim: int = 999,
    training: dict[str, Any] | None = None,
    seq_len: int = 16,
    seed: int = 0,
    fit_models: bool = True,
    fetch_missing: bool = False,
) -> dict[str, Any]:
    """Run the lunar check. ``frames`` overrides the on-disk draw tables."""
    loaded = frames if frames is not None else _load_games(fetch_missing=fetch_missing)
    order = [POWERBALL, MEGA_MILLIONS]
    games: list[dict[str, Any]] = []
    for game in order:
        frame = loaded.get(game.game_id)
        if frame is None:
            continue
        games.append(
            analyze_game(
                frame,
                game,
                n_perm=n_perm,
                n_sim=n_sim,
                training=training,
                seq_len=seq_len,
                seed=seed,
                fit_models=fit_models,
            )
        )
    if not games:
        raise ValueError("no lottery draws were available")
    pooled_tests = [test for game in games for test in game["tests"]]
    pooled = _correction_summary(
        pooled_tests,
        n_perm=n_perm,
        bh_key="p_bh",
        bonferroni_key="p_bonferroni",
        survive_bh_key="survives_bh",
        survive_bonferroni_key="survives_bonferroni",
    )
    report = {
        "available": True,
        "question": (
            "Do moon phase, Chinese lunar date, perigee, or a geometric eclipse "
            "line up with Powerball or Mega Millions draws?"
        ),
        "answer_in_brief": "",
        "summary": "",
        "caveat": (
            "These are association tests and a small neural net, not a betting "
            "system. Draws are designed to be random. Features are the moon at "
            f"{DRAW_CLOCK} {DRAW_TIMEZONE} on the draw date, which is known "
            "before the drawing."
        ),
        "method": {
            "phase_bins": list(PHASE_NAMES),
            "phase_rule": (
                "Eight named phases, each 1/8 of the cycle and centered on "
                "new, quarters, and full."
            ),
            "high_ball": (
                "A white ball is high when 2 * ball >= low + high "
                "(Powerball 35–69, Mega Millions 36–70)."
            ),
            "permutation": (
                "Shuffle phase labels across draws. Balls from one draw stay "
                "together. The p-value is (exceedances + 1) / (shuffles + 1)."
            ),
            "multiple_testing": (
                "Benjamini–Hochberg and Bonferroni on the permutation p-values. "
                "The pooled family is every association test across games. "
                "Model-versus-chance p-values are reported separately and are "
                "not in that family."
            ),
            "model": (
                "The lottery transformer trains on earlier draws and is scored "
                "on the latest temporal holdout. Lunar columns are optional "
                "covariates for the draw being predicted. Tickets are rounded "
                "into a legal set of white balls. Chance is the expected overlap "
                "of a uniform ticket, plus one seeded random ticket."
            ),
            "n_perm": n_perm,
            "n_sim": n_sim,
            "seed": seed,
            "clock": f"{DRAW_CLOCK} {DRAW_TIMEZONE}",
        },
        "correction": pooled,
        "model_vs_chance": _model_chance_family(games),
        "games": games,
    }
    report["summary"] = _summary(games, pooled)
    model_family = report["model_vs_chance"]
    model_kept = 0 if not isinstance(model_family, dict) else int(model_family["n_bh_rejects"])
    model_raw = 0 if not isinstance(model_family, dict) else int(model_family["n_raw_below_alpha"])
    if pooled["n_bh_rejects"] == 0 and model_kept == 0:
        if model_raw == 0:
            raw_clause = "None of the lunar-model scores beat a random ticket at p = 0.05."
        elif model_raw == 1:
            raw_clause = (
                "One lunar-model score beat a random ticket before correction, "
                "and Benjamini–Hochberg does not keep it."
            )
        else:
            count = "Two" if model_raw == 2 else str(model_raw)
            raw_clause = (
                f"{count} lunar-model scores beat a random ticket before "
                "correction, and Benjamini–Hochberg keeps none of them."
            )
        report["answer_in_brief"] = (
            "No. Moon phase does not line up with these draws: none of the "
            f"association tests fell under p = 0.05. {raw_clause}"
        )
    elif pooled["n_bh_rejects"] == 0:
        report["answer_in_brief"] = (
            "The phase tests do not survive correction. A lunar-model score "
            "against a random ticket does, and it is reported below. That is "
            "not a system for buying tickets."
        )
    else:
        report["answer_in_brief"] = (
            "A corrected test crossed the line in this sample. The effect is "
            "reported below. It is not evidence you can pick the next draw."
        )
    return cast(dict[str, Any], _jsonable(report))


def _load_games(*, fetch_missing: bool) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    powerball_path = resolve_repo_path("data/powerball.csv")
    mega_path = resolve_repo_path("data/mega_millions.csv")
    if not powerball_path.is_file() and fetch_missing:
        fetch_powerball(powerball_path)
    if powerball_path.is_file():
        loaded[POWERBALL.game_id] = read_powerball(powerball_path)
    if not mega_path.is_file() and fetch_missing:
        fetch_mega(mega_path)
    if mega_path.is_file():
        loaded[MEGA_MILLIONS.game_id] = read_mega(mega_path)
    return loaded


def write_report(
    path: str | Path | None = None,
    *,
    fetch_missing: bool = True,
    n_perm: int = 1999,
    n_sim: int = 999,
    training: dict[str, Any] | None = None,
    seq_len: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute the check and write the JSON the API serves."""
    report = build_report(
        n_perm=n_perm,
        n_sim=n_sim,
        training=training,
        seq_len=seq_len,
        seed=seed,
        fetch_missing=fetch_missing,
    )
    destination = resolve_repo_path(PUBLISHED_REPORT if path is None else path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def read_lunar_analysis(path: str | Path | None = None) -> dict[str, Any]:
    """Return the published report, or an empty one when the file is absent."""
    destination = resolve_repo_path(PUBLISHED_REPORT if path is None else path)
    if not destination.is_file():
        return {
            "available": False,
            "summary": "No lunar lottery analysis has been published.",
        }
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("lunar lottery analysis is not a JSON object")
    loaded["available"] = True
    return loaded


def main() -> None:
    report = write_report(fetch_missing=True)
    print(report["answer_in_brief"])
    print(report["summary"])


if __name__ == "__main__":
    main()
