"""Block bootstrap for the difference of two RMSE scores.

Daily forecast errors are serially correlated, so days are resampled in
contiguous blocks. A positive improvement means ``candidate`` has the lower
RMSE. The 5% quantile is the bar used to keep a change: the gain has to show
up in the lower tail, not only in the point estimate.
"""

from __future__ import annotations

import numpy as np


def rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    error = np.asarray(predictions, dtype=np.float64) - np.asarray(
        targets, dtype=np.float64
    )
    return float(np.sqrt(np.mean(error**2)))


def block_bootstrap_improvement(
    targets: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    block: int = 14,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Return the RMSE drop of ``candidate`` versus ``reference`` and its tails.

    ``improvement`` is ``rmse(reference) - rmse(candidate)`` on the original
    rows. ``p05`` / ``p95`` are quantiles of a circular block bootstrap.
    """
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if not (len(y) == len(left) == len(right)):
        raise ValueError("Bootstrap inputs must have the same length.")
    if len(y) == 0:
        raise ValueError("Bootstrap inputs must be non-empty.")
    if block < 1:
        raise ValueError(f"block must be positive, got {block}")

    point = rmse(y, left) - rmse(y, right)
    rng = np.random.default_rng(seed)
    n = len(y)
    width = min(block, n)
    n_blocks = int(np.ceil(n / width))
    deltas = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = np.arange(width)
        indexes = (starts[:, None] + offsets[None, :]).reshape(-1)[:n] % n
        deltas[draw] = rmse(y[indexes], left[indexes]) - rmse(
            y[indexes], right[indexes]
        )
    return {
        "improvement": float(point),
        "p05": float(np.quantile(deltas, 0.05)),
        "p50": float(np.quantile(deltas, 0.50)),
        "p95": float(np.quantile(deltas, 0.95)),
    }
