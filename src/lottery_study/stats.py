"""Correlation, mutual information, permutation tests, and error-rate control.

Pearson p-values use the usual t approximation. That approximation is the
right null here: under "no relationship" the lottery side is exchangeable, and
with more than a thousand draws the sampling distribution of r is close to
the t reference even though a draw is a small integer. Permutation p-values
are reported beside it. Mutual information has no closed null, so its p-value
is a permutation test. The permutation floor is ``1 / (n_perm + 1)``; the
report says so, because that floor can sit above a Bonferroni line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_BETACF_ITERS = 200
_BETACF_EPS = 3.0e-14


def expected_luck_hits(n_tests: int, alpha: float) -> float:
    """How many tests would fall under ``alpha`` if every null were true.

    This is ``alpha * n_tests``. It assumes independent tests. Dependence
    among targets makes the typical count smaller, not larger, so the figure
    is a ceiling on the average.
    """
    if n_tests < 0:
        raise ValueError("n_tests must be non-negative")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return float(alpha) * float(n_tests)


def bonferroni_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Bonferroni-adjusted p-values, clipped to 1."""
    raw = np.asarray(p_values, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if raw.size == 0:
        return raw.copy()
    if np.any((raw < 0.0) | (raw > 1.0) | ~np.isfinite(raw)):
        raise ValueError("p_values must be finite and inside [0, 1]")
    return np.minimum(raw * float(raw.size), 1.0)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini–Hochberg adjusted p-values, clipped to 1.

    The adjusted value at rank ``k`` (1-based, smallest p first) is
    ``min_{j >= k} p_(j) * m / j``.
    """
    raw = np.asarray(p_values, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if raw.size == 0:
        return raw.copy()
    if np.any((raw < 0.0) | (raw > 1.0) | ~np.isfinite(raw)):
        raise ValueError("p_values must be finite and inside [0, 1]")
    m = int(raw.size)
    order = np.argsort(raw, kind="mergesort")
    ranked = raw[order]
    adjusted_ranked = np.empty(m, dtype=np.float64)
    running = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        running = min(running, float(ranked[index]) * m / rank)
        adjusted_ranked[index] = running
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return ``I_x(a, b)``, the regularized incomplete beta function."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0 or x == 1.0:
        return float(x)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_stat: float, df: int) -> float:
    """Two-sided tail probability of a Student t random variable."""
    if df < 1:
        raise ValueError("df must be positive")
    if math.isnan(t_stat):
        raise ValueError("t_stat must be finite")
    if math.isinf(t_stat):
        return 0.0
    x = df / (df + t_stat * t_stat)
    probability = regularized_incomplete_beta(df / 2.0, 0.5, x)
    return min(1.0, max(0.0, probability))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation, or ``None`` when either side has no variance."""
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("x and y must have the same length")
    if left.size < 3:
        raise ValueError("need at least 3 paired observations")
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denom == 0.0:
        return None
    return float(np.dot(left, right) / denom)


def pearson_p_value(r: float, n: int) -> float:
    """Two-sided t-test p-value for a Pearson correlation."""
    if n <= 2:
        return 1.0
    if abs(r) >= 1.0 - 1e-15:
        return 0.0
    t_stat = r * math.sqrt((n - 2) / (1.0 - r * r))
    return student_t_two_sided_p(t_stat, n - 2)


def paired_difference_p(diff: np.ndarray) -> float:
    """Two-sided t p-value for a mean paired difference.

    A zero difference, or fewer than three finite pairs, returns 1. A non-zero
    difference with no spread (every pair moved by the same amount) returns 0.
    """
    values = np.asarray(diff, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 3:
        return 1.0
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std == 0.0:
        return 0.0 if mean != 0.0 else 1.0
    t_stat = mean / (std / math.sqrt(n))
    return student_t_two_sided_p(t_stat, n - 1)


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin ``values`` into at most ``n_bins`` quantile bins, coded from 0."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size == 0:
        raise ValueError("values must be non-empty")
    unique = np.unique(series)
    if unique.size <= n_bins:
        codes = np.empty(series.size, dtype=np.int64)
        mapping = {float(value): index for index, value in enumerate(unique)}
        for index, value in enumerate(series):
            codes[index] = mapping[float(value)]
        return codes
    edges = np.quantile(series, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(series.size, dtype=np.int64)
    return np.digitize(series, edges[1:-1], right=False).astype(np.int64)


def mutual_info_bits(x: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
    """Histogram mutual information, in bits.

    A binary or other low-cardinality target keeps its own levels. A
    continuous target is quantile-binned. The estimate is biased upward; use
    :func:`mutual_info_permutation` before treating a number as evidence.
    """
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 3:
        raise ValueError("x and y must be paired and have length at least 3")
    x_bins = quantile_bins(left, n_bins)
    y_bins = quantile_bins(right, n_bins)
    return _mi_from_codes(x_bins, y_bins)


def mutual_info_permutation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Return ``(mi_bits, permutation_p, n_perm_at_least_as_large)``.

    ``y`` is shuffled. The lottery target is the exchangeable side.
    """
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    observed = mutual_info_bits(x, y, n_bins)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    exceedances = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(right)
        if mutual_info_bits(x, shuffled, n_bins) >= observed - 1e-12:
            exceedances += 1
    p_value = (exceedances + 1) / (n_perm + 1)
    return observed, float(p_value), exceedances


def pearson_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Permute ``y`` and return ``(p_value, exceedances)`` for ``|r|``."""
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    observed = pearson_r(x, y)
    if observed is None:
        return 1.0, n_perm
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    order = np.argsort(rng.random((n_perm, right.size)), axis=1)
    shuffled = right[order]
    centered_x = left - left.mean()
    centered_y = shuffled - shuffled.mean(axis=1, keepdims=True)
    numerator = centered_y @ centered_x
    denom = np.sqrt(
        float(np.dot(centered_x, centered_x)) * np.sum(centered_y * centered_y, axis=1)
    )
    correlations = np.divide(
        numerator,
        denom,
        out=np.zeros(n_perm, dtype=np.float64),
        where=denom > 0.0,
    )
    exceedances = int(np.sum(np.abs(correlations) >= abs(observed) - 1e-12))
    return (exceedances + 1) / (n_perm + 1), exceedances


@dataclass(frozen=True)
class CorrectionSummary:
    """How a family of p-values looks after two standard corrections."""

    family: str
    n_tests: int
    alpha: float
    expected_luck_hits: float
    n_raw_below_alpha: int
    n_bh_rejects: int
    n_bonferroni_rejects: int
    bonferroni_cutoff: float
    permutation_floor: float | None


def summarize_corrections(
    p_values: np.ndarray,
    bh_adjusted: np.ndarray,
    bonferroni_adjusted: np.ndarray,
    *,
    family: str,
    alpha: float,
    permutation_floor: float | None = None,
) -> CorrectionSummary:
    """Count raw, BH, and Bonferroni hits."""
    raw = np.asarray(p_values, dtype=np.float64)
    return CorrectionSummary(
        family=family,
        n_tests=int(raw.size),
        alpha=float(alpha),
        expected_luck_hits=expected_luck_hits(int(raw.size), alpha),
        n_raw_below_alpha=int(np.sum(raw <= alpha)),
        n_bh_rejects=int(np.sum(np.asarray(bh_adjusted) <= alpha)),
        n_bonferroni_rejects=int(np.sum(np.asarray(bonferroni_adjusted) <= alpha)),
        bonferroni_cutoff=(alpha / raw.size) if raw.size else 1.0,
        permutation_floor=permutation_floor,
    )


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta function."""
    am = 1.0
    bm = 1.0
    az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap
    for m in range(1, _BETACF_ITERS + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        previous = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if abs(az - previous) < _BETACF_EPS * abs(az):
            return az
    raise RuntimeError("incomplete beta continued fraction did not converge")


def _mi_from_codes(x_bins: np.ndarray, y_bins: np.ndarray) -> float:
    n_x = int(x_bins.max()) + 1
    n_y = int(y_bins.max()) + 1
    joint = np.zeros((n_x, n_y), dtype=np.float64)
    np.add.at(joint, (x_bins, y_bins), 1.0)
    total = float(joint.sum())
    if total <= 0.0:
        return 0.0
    joint /= total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px * py
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / expected[mask])))
