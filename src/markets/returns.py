"""Log-return features for a single price series.

Returns are taken between consecutive *published* observations, and only when
both prices are strictly positive. A nonpositive print (WTI in April 2020, for
example) breaks the chain instead of producing a log of a negative number or a
return that jumps across the bad print. Volatility is the population standard
deviation of the previous ``VOL_WINDOW`` one-step returns, using only rows that
are already known at that observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

VOL_WINDOW = 10


def one_step_log_returns(close: np.ndarray) -> np.ndarray:
    """Log change from the previous observation, or NaN when it is undefined."""
    prices = np.asarray(close, dtype=np.float64)
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    if prices.size < 2:
        return out
    prev = prices[:-1]
    curr = prices[1:]
    ok = np.isfinite(prev) & np.isfinite(curr) & (prev > 0) & (curr > 0)
    out[1:][ok] = np.log(curr[ok]) - np.log(prev[ok])
    return out


def rolling_population_std(values: np.ndarray, window: int) -> np.ndarray:
    """Causal population standard deviation. A NaN inside the window stays NaN."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    series = np.asarray(values, dtype=np.float64)
    out = np.full(series.shape, np.nan, dtype=np.float64)
    if series.size < window:
        return out
    for end in range(window - 1, series.size):
        chunk = series[end - window + 1 : end + 1]
        if not np.isfinite(chunk).all():
            continue
        out[end] = float(chunk.std(ddof=0))
    return out


def nonpositive_count(close: np.ndarray) -> int:
    prices = np.asarray(close, dtype=np.float64)
    return int(np.sum(~(np.isfinite(prices) & (prices > 0))))


def price_frame_from_pairs(
    dates: Sequence[Any],
    closes: Sequence[Any] | np.ndarray,
) -> pd.DataFrame:
    """Sort by date, drop missing closes, and keep the last print of a duplicate day."""
    frame = pd.DataFrame({"date": pd.to_datetime(list(dates)), "close": closes})
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    finite = np.isfinite(frame["close"].to_numpy(dtype=np.float64))
    frame = frame.loc[finite]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def build_return_frame(
    prices: pd.DataFrame, vol_window: int = VOL_WINDOW
) -> pd.DataFrame:
    """Return ``date, log_return, vol_10, close`` with warmup and gaps removed.

    ``vol_10`` is always the column name the market configs train on. ``vol_window``
    still controls the lookback, which is 10 observations unless a test overrides it.
    """
    if "date" not in prices.columns or "close" not in prices.columns:
        raise ValueError("price frame must have date and close columns")
    frame = prices.sort_values("date").drop_duplicates("date", keep="last")
    close = frame["close"].to_numpy(dtype=np.float64)
    returns = one_step_log_returns(close)
    vol = rolling_population_std(returns, vol_window)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d"),
            "log_return": returns,
            "vol_10": vol,
            "close": close,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return", "vol_10"])
    return out.reset_index(drop=True)


def coverage_span_days(prices: pd.DataFrame) -> int:
    if prices.empty:
        return 0
    start = pd.Timestamp(prices["date"].iloc[0])
    end = pd.Timestamp(prices["date"].iloc[-1])
    return int((end - start).days)
