"""Scoring rules and the baselines the domain models are compared with."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np

_LOG10_2 = math.log10(2.0)
_LN_2 = math.log(2.0)


def iter_months(start: str, end: str) -> list[str]:
    """Inclusive list of ``YYYY-MM`` months from ``start`` through ``end``."""
    start_year, start_month = _parse_month(start)
    end_year, end_month = _parse_month(end)
    if (start_year, start_month) > (end_year, end_month):
        raise ValueError(f"start {start} is after end {end}")
    months: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def _parse_month(value: str) -> tuple[int, int]:
    pieces = value.split("-")
    if len(pieces) != 2:
        raise ValueError(f"expected YYYY-MM, got {value!r}")
    year, month = int(pieces[0]), int(pieces[1])
    if month < 1 or month > 12:
        raise ValueError(f"month out of range in {value!r}")
    return year, month


def window_split_bounds(
    n_windows: int, test_fraction: float, val_fraction: float
) -> tuple[int, int]:
    """Return ``(train_end, val_end)`` for a chronological window split.

    Windows are ordered in time. The layout is
    ``[0, train_end)`` train, ``[train_end, val_end)`` validation,
    ``[val_end, n_windows)`` test. Validation is used for early stopping
    only; the test block is never used to fit the model or the baselines.
    """
    if n_windows < 3:
        raise ValueError(f"need at least 3 windows, got {n_windows}")
    if not 0.0 < test_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("test_fraction and val_fraction must be in (0, 1)")
    n_test = max(1, int(round(n_windows * test_fraction)))
    n_val = max(1, int(round(n_windows * val_fraction)))
    if n_test + n_val >= n_windows:
        raise ValueError(
            f"train split is empty for n={n_windows}, "
            f"test_fraction={test_fraction}, val_fraction={val_fraction}"
        )
    val_end = n_windows - n_test
    train_end = val_end - n_val
    return train_end, val_end


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    b = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - b)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    b = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    b = np.asarray(predicted, dtype=float).reshape(-1)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def linear_fit(t_years: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares ``values = intercept + slope * t_years``."""
    t = np.asarray(t_years, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("t_years and values must have the same length")
    if t.size < 2:
        raise ValueError("linear fit needs at least two points")
    slope, intercept = np.polyfit(t, y, 1)
    return float(intercept), float(slope)


def doubling_time_years(slope_per_year: float, value_scale: str) -> float | None:
    """Years for the underlying quantity to double, or ``None`` if it does not.

    ``log10`` means ``slope_per_year`` is the slope of ``log10(quantity)``.
    ``log`` means the slope of ``ln(quantity)``. A non-positive slope has no
    doubling time (the quantity is flat or shrinking).
    """
    if slope_per_year <= 0.0 or not math.isfinite(slope_per_year):
        return None
    if value_scale == "log10":
        return _LOG10_2 / slope_per_year
    if value_scale == "log":
        return _LN_2 / slope_per_year
    raise ValueError(f"value_scale must be 'log10' or 'log', got {value_scale!r}")


def halving_time_years(slope_per_year: float, value_scale: str) -> float | None:
    """Years for a shrinking positive quantity to halve.

    Used for two-qubit error rates, where the fitted slope of ``log10(error)``
    is negative when the hardware is improving.
    """
    return doubling_time_years(-slope_per_year, value_scale)


def persistence_predict(last_value: np.ndarray, n_steps: int) -> np.ndarray:
    """Repeat the last observed row for ``n_steps`` forecast steps.

    ``last_value`` has shape ``(F,)``. The result has shape ``(n_steps, F)``.
    On a regular series this is the one-step persistence forecast when each
    test step is scored against the true previous observation instead. For
    one-step-ahead evaluation, pass the true lag at each step rather than
    this multi-step hold. See ``one_step_persistence``.
    """
    last = np.asarray(last_value, dtype=float).reshape(1, -1)
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    return np.repeat(last, n_steps, axis=0)


def one_step_persistence(series: np.ndarray, target_index: np.ndarray) -> np.ndarray:
    """One-step persistence: the value at ``t - 1`` for each target index."""
    values = np.asarray(series, dtype=float)
    index = np.asarray(target_index, dtype=int)
    if np.any(index < 1):
        raise ValueError("persistence needs a previous observation")
    return cast(np.ndarray, values[index - 1])


def market_implied_probabilities(decimal_odds: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise ``1/odds`` so they sum to 1, and return the overround.

    Decimal odds are the stake-inclusive dividend. The overround is
    ``sum(1/odds) - 1``. It is the track take already inside those dividends:
    a win-pool deduction ``t`` produces an overround near ``t / (1 - t)``.
    """
    odds = np.asarray(decimal_odds, dtype=float).reshape(-1)
    if odds.size == 0:
        raise ValueError("odds must be non-empty")
    if np.any(odds <= 0.0) or not np.all(np.isfinite(odds)):
        raise ValueError("decimal odds must be finite and positive")
    implied = 1.0 / odds
    total = float(implied.sum())
    return implied / total, total - 1.0


def multiclass_log_loss(probabilities: np.ndarray, winner_index: int) -> float:
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if winner_index < 0 or winner_index >= probs.size:
        raise ValueError("winner_index is outside the field")
    p = float(np.clip(probs[winner_index], 1e-15, 1.0))
    return float(-math.log(p))


def multiclass_brier(probabilities: np.ndarray, winner_index: int) -> float:
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if winner_index < 0 or winner_index >= probs.size:
        raise ValueError("winner_index is outside the field")
    one_hot = np.zeros(probs.size, dtype=float)
    one_hot[winner_index] = 1.0
    return float(np.sum((probs - one_hot) ** 2))


def flat_stake_profit(decimal_odds: float, won: bool) -> float:
    """Profit on a 1-unit win bet at stake-inclusive decimal odds.

    The published dividend already has the pool deduction removed, so this
    profit accounts for the track take. Applying the deduction again would
    double-count it. A winner's profit is ``odds - 1``; a loser loses 1.
    """
    odds = float(decimal_odds)
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError("decimal odds must be finite and greater than 1")
    return odds - 1.0 if won else -1.0


def rank_best_first(
    primary: np.ndarray,
    tie_breaks: Sequence[np.ndarray] = (),
    *,
    lower_is_better: bool = True,
) -> np.ndarray:
    """Indices of ``primary`` from best to worst.

    ``tie_breaks`` are used in order when ``primary`` ties. Each tie break
    is ranked lower-better, which matches barrier draw and horse id.
    """
    score = np.asarray(primary, dtype=float).reshape(-1)
    if not lower_is_better:
        score = -score
    keys: list[np.ndarray] = []
    for tie in reversed(tuple(tie_breaks)):
        keys.append(np.asarray(tie))
    keys.append(score)
    return cast(np.ndarray, np.lexsort(tuple(keys)))


def uniform_race_expectations(field_size: int) -> dict[str, float]:
    """Expected scores of a uniform random pick in a field of this size."""
    if field_size < 2:
        raise ValueError("field_size must be at least 2")
    p = 1.0 / field_size
    return {
        "win_accuracy": p,
        "top3_hit_rate": min(3, field_size) / field_size,
        "log_loss": math.log(field_size),
        "brier": 1.0 - p,
    }


def scores_to_probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax of ``-score / temperature`` so a lower score is more likely."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = -np.asarray(scores, dtype=float).reshape(-1) / float(temperature)
    logits -= np.max(logits)
    weights = np.exp(logits)
    total = float(weights.sum())
    return weights / total
