"""Parsers for the public files the fetch script downloads.

Each parser returns a price frame (``date``, ``close``) or, for the ECB file, one
price frame per currency already expressed as units of that currency per 1 USD.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime, timezone

import pandas as pd

from src.markets.returns import price_frame_from_pairs

_ISO = re.compile(r"^[A-Z]{3}$")


def as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected an integer, got {value!r}")
    return int(value)


def as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def parse_fred_csv(text: str) -> pd.DataFrame:
    """Parse a FRED graph CSV. Missing observations are ``.``."""
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] < 2:
        raise ValueError("FRED CSV must have a date column and a value column")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return price_frame_from_pairs(dates.tolist(), values.tolist())


def parse_stooq_csv(text: str) -> pd.DataFrame:
    """Parse a Stooq daily OHLC CSV. The close is the training price."""
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError("Stooq CSV must have Date and Close columns")
    return price_frame_from_pairs(frame["date"].tolist(), frame["close"].tolist())


def parse_yahoo_chart(payload: dict[str, object]) -> tuple[pd.DataFrame, str]:
    """Parse a Yahoo chart payload. Prefer adjusted close when it is present."""
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError("Yahoo payload has no chart object")
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo chart error: {error}")
    result = chart.get("result")
    if not isinstance(result, list) or not result:
        raise ValueError("Yahoo payload has no chart result")
    block = result[0]
    if not isinstance(block, dict):
        raise ValueError("Yahoo chart result is not an object")
    raw_timestamps = block.get("timestamp") or []
    if not isinstance(raw_timestamps, list):
        raise ValueError("Yahoo timestamps are not a list")
    timestamps = [as_int(stamp) for stamp in raw_timestamps]
    indicators = block.get("indicators")
    if not isinstance(indicators, dict):
        raise ValueError("Yahoo payload has no indicators")
    quote = indicators.get("quote")
    if not isinstance(quote, list) or not quote or not isinstance(quote[0], dict):
        raise ValueError("Yahoo payload has no quote closes")
    closes = quote[0].get("close") or []
    field = "close"
    values: list[object] = list(closes) if isinstance(closes, list) else []
    adjusted = indicators.get("adjclose")
    if isinstance(adjusted, list) and adjusted and isinstance(adjusted[0], dict):
        adj_values = adjusted[0].get("adjclose")
        if isinstance(adj_values, list) and any(
            item is not None for item in adj_values
        ):
            values = list(adj_values)
            field = "adjclose"
    if len(values) != len(timestamps):
        raise ValueError("Yahoo timestamps and closes differ in length")
    dates = [datetime.fromtimestamp(stamp, timezone.utc).date() for stamp in timestamps]
    numeric = [float("nan") if item is None else as_float(item) for item in values]
    return price_frame_from_pairs(dates, numeric), field


def parse_binance_klines(rows: list[list[object]]) -> pd.DataFrame:
    """Parse Binance kline rows. Close is field 4; open time is field 0 in ms."""
    dates: list[date] = []
    closes: list[float] = []
    for row in rows:
        if len(row) < 5:
            raise ValueError("Binance kline row is shorter than 5 fields")
        opened = datetime.fromtimestamp(as_int(row[0]) / 1000, timezone.utc).date()
        dates.append(opened)
        closes.append(as_float(row[4]))
    return price_frame_from_pairs(dates, closes)


def parse_coinbase_candles(rows: list[list[object]]) -> pd.DataFrame:
    """Parse Coinbase candle rows: time, low, high, open, close, volume."""
    dates: list[date] = []
    closes: list[float] = []
    for row in rows:
        if len(row) < 5:
            raise ValueError("Coinbase candle row is shorter than 5 fields")
        dates.append(datetime.fromtimestamp(as_int(row[0]), timezone.utc).date())
        closes.append(as_float(row[4]))
    return price_frame_from_pairs(dates, closes)


def parse_ecb_hist_csv(text: str) -> pd.DataFrame:
    """Parse ``eurofxref-hist.csv`` (newest first on the ECB site)."""
    frame = pd.read_csv(io.StringIO(text), na_values=["N/A", "n/a", ""])
    frame.columns = [str(col).strip() for col in frame.columns]
    if "Date" not in frame.columns or "USD" not in frame.columns:
        raise ValueError("ECB history must have Date and USD columns")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    keep = ["Date"]
    for col in frame.columns:
        if col != "Date" and _ISO.fullmatch(col):
            keep.append(col)
    return frame[keep].reset_index(drop=True)


def per_usd_frames(ecb: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert ECB foreign-per-euro rates into foreign currency per 1 USD.

    The euro itself is stored as EUR per 1 USD, the reciprocal of the ECB USD
    rate (USD per 1 EUR). Days with a missing or nonpositive USD rate are
    dropped for every currency, because the cross cannot be formed.
    """
    if "USD" not in ecb.columns or "Date" not in ecb.columns:
        raise ValueError("ECB frame must have Date and USD columns")
    usd = pd.to_numeric(ecb["USD"], errors="coerce")
    usable = ecb.loc[usd.notna() & (usd > 0)].copy()
    usd = pd.to_numeric(usable["USD"], errors="coerce")
    frames: dict[str, pd.DataFrame] = {
        "EUR": price_frame_from_pairs(usable["Date"].tolist(), (1.0 / usd).tolist())
    }
    for col in usable.columns:
        if col in {"Date", "USD"}:
            continue
        foreign = pd.to_numeric(usable[col], errors="coerce")
        mask = foreign.notna() & (foreign > 0)
        if not bool(mask.any()):
            continue
        per_usd = foreign[mask] / usd[mask]
        frames[col] = price_frame_from_pairs(
            usable.loc[mask, "Date"].tolist(), per_usd.tolist()
        )
    return frames


def parse_frankfurter_rates(payload: dict[str, object]) -> tuple[str, dict[str, float]]:
    """Parse one Frankfurter response quoted as currency units per 1 USD."""
    if payload.get("base") != "USD":
        raise ValueError("Frankfurter check expects base=USD")
    raw_date = payload.get("date")
    if not isinstance(raw_date, str):
        raise ValueError("Frankfurter response has no date")
    rates = payload.get("rates")
    if not isinstance(rates, dict):
        raise ValueError("Frankfurter response has no rates")
    parsed = {str(code): float(value) for code, value in rates.items()}
    return raw_date, parsed


def max_relative_diff(
    left: dict[str, float], right: dict[str, float]
) -> tuple[float, int]:
    """Largest absolute percent gap across codes present in both maps."""
    diffs: list[float] = []
    for code, value in left.items():
        other = right.get(code)
        if other is None or other == 0 or not (value > 0):
            continue
        diffs.append(abs(value - other) / other)
    if not diffs:
        return float("nan"), 0
    return float(max(diffs)), len(diffs)


def parse_coingecko_markets(
    payload: list[dict[str, object]],
) -> list[dict[str, object]]:
    coins: list[dict[str, object]] = []
    for index, item in enumerate(payload):
        symbol = str(item.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        rank = item.get("market_cap_rank")
        coins.append(
            {
                "rank": int(rank) if isinstance(rank, int) else index + 1,
                "symbol": symbol,
                "name": str(item.get("name") or symbol),
                "coin_id": str(item.get("id") or symbol.lower()),
                "source": "coingecko",
            }
        )
    return coins


def parse_coinpaprika_tickers(
    payload: list[dict[str, object]],
) -> list[dict[str, object]]:
    coins: list[dict[str, object]] = []
    for index, item in enumerate(payload):
        symbol = str(item.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        rank = item.get("rank")
        coins.append(
            {
                "rank": int(rank) if isinstance(rank, int) else index + 1,
                "symbol": symbol,
                "name": str(item.get("name") or symbol),
                "coin_id": str(item.get("id") or symbol.lower()),
                "source": "coinpaprika",
            }
        )
    return coins
