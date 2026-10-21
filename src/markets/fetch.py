"""Download public market history and write gitignored return features.

Run ``python scripts/fetch_markets.py`` from the repository root. The script
records where each series came from. It does not commit prices.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.markets.parse import (
    as_int,
    max_relative_diff,
    parse_binance_klines,
    parse_coinbase_candles,
    parse_coingecko_markets,
    parse_coinpaprika_tickers,
    parse_ecb_hist_csv,
    parse_frankfurter_rates,
    parse_fred_csv,
    parse_stooq_csv,
    parse_yahoo_chart,
    per_usd_frames,
)
from src.markets.returns import (
    build_return_frame,
    coverage_span_days,
    nonpositive_count,
)
from src.markets.universe import (
    MIN_CRYPTO_BARS,
    MIN_OBSERVATIONS,
    MIN_SPAN_DAYS,
    STALE_DAYS,
    PriceSpec,
    SeriesCoverage,
    binance_candidates,
    classify_history,
    currency_name,
    dedupe_ranked,
    is_stablecoin,
    is_wrapped,
    price_specs,
)
from src.utils.config import load_config
from src.utils.paths import resolve_repo_path

ECB_ZIP_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
BINANCE_EXCHANGE_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
COINGECKO_MARKETS_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=250&page=1&sparkline=false"
)
COINPAPRIKA_URL = "https://api.coinpaprika.com/v1/tickers?limit=250"
FRED_PROBE_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS"
    "&cosd=2024-01-02&coed=2024-01-10"
)
STOOQ_PROBE_URL = "https://stooq.com/q/d/l/?s=aapl.us&i=d"
STOOQ_SYMBOLS = {
    "stock-aapl": "aapl.us",
    "stock-msft": "msft.us",
    "stock-nvda": "nvda.us",
}
HEADERS = {
    "User-Agent": "RealNeuralForge/1.0 (research; market-history fetch)",
    "Accept": "application/json,text/csv,*/*",
}
BINANCE_START_MS = 1_483_228_800_000  # 2017-01-01 UTC


class FetchError(RuntimeError):
    """A download failed and the caller should record or fall back."""


def http_get(url: str, timeout: float = 60, attempts: int = 3) -> bytes:
    delay = 1.0
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in {404, 451}:
                raise FetchError(f"HTTP {exc.code} for {url}") from exc
            if exc.code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(delay)
                delay *= 2
                continue
            raise FetchError(f"HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay *= 2
                continue
            raise FetchError(f"request failed for {url}: {exc}") from exc
    raise FetchError(f"request failed for {url}: {last}")


def http_json(url: str, timeout: float = 60, attempts: int = 3) -> Any:
    body = http_get(url, timeout=timeout, attempts=attempts)
    return json.loads(body.decode("utf-8"))


def _probe_fred() -> bool:
    try:
        body = http_get(FRED_PROBE_URL, timeout=20, attempts=1)
    except FetchError:
        return False
    return b"observation_date" in body or b"DATE" in body


def _probe_stooq() -> bool:
    try:
        body = http_get(STOOQ_PROBE_URL, timeout=15, attempts=1)
    except FetchError:
        return False
    return body.lstrip()[:4].lower() == b"date"


def _yahoo_url(symbol: str, start: str = "2010-01-01") -> str:
    period1 = int(
        datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp()
    )
    period2 = int(datetime.now(timezone.utc).timestamp())
    quoted = urllib.parse.quote(symbol, safe="")
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quoted}?interval=1d&period1={period1}&period2={period2}"
    )


def _fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _stooq_url(symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={urllib.parse.quote(symbol)}&i=d"


def _load_yahoo(symbol: str) -> tuple[pd.DataFrame, str, str]:
    url = _yahoo_url(symbol)
    payload = http_json(url, timeout=60)
    if not isinstance(payload, dict):
        raise FetchError(f"Yahoo payload for {symbol} was not an object")
    frame, field = parse_yahoo_chart(payload)
    return frame, field, url


def _load_fred(series_id: str) -> tuple[pd.DataFrame, str]:
    url = _fred_url(series_id)
    text = http_get(url, timeout=40, attempts=1).decode("utf-8")
    return parse_fred_csv(text), url


def _load_stooq(symbol: str) -> tuple[pd.DataFrame, str]:
    url = _stooq_url(symbol)
    body = http_get(url, timeout=60)
    if body.lstrip()[:4].lower() != b"date":
        raise FetchError(f"Stooq did not return a CSV for {symbol}")
    return parse_stooq_csv(body.decode("utf-8")), url


def _group_protocol(group: str) -> tuple[int, float]:
    cfg = load_config(f"config/{group}.yaml")
    data = cfg["data"]
    return int(data["seq_len"]), float(data["val_split"])


def _iso(value: object) -> str:
    return str(pd.Timestamp(value).strftime("%Y-%m-%d"))


def _finish(
    *,
    series_id: str,
    group: str,
    name: str,
    symbol: str,
    source: str,
    source_url: str,
    kind: str,
    quote: str,
    prices: pd.DataFrame | None,
    data_dir: Path,
    seq_len: int,
    val_split: float,
    min_observations: int,
    min_span_days: int,
    active: bool | None,
    block_if_stale: bool,
    rank: int | None = None,
    price_field: str | None = None,
    frequency: str = "daily",
) -> SeriesCoverage:
    """Write features when possible and decide whether the series is trainable."""
    today = datetime.now(timezone.utc).date()
    if prices is None or prices.empty:
        return SeriesCoverage(
            id=series_id,
            group=group,
            name=name,
            symbol=symbol,
            source=source,
            source_url=source_url,
            kind=kind,
            quote=quote,
            frequency=frequency,
            start=None,
            end=None,
            observations=0,
            feature_rows=0,
            feature_path=None,
            train=False,
            skip_reason="no prices downloaded",
            active=False,
            rank=rank,
            price_field=price_field,
        )
    start = prices["date"].iloc[0]
    end = prices["date"].iloc[-1]
    end_date = pd.Timestamp(end).date()
    if active is None:
        active = (today - end_date).days <= STALE_DAYS
    bad_prints = nonpositive_count(prices["close"].to_numpy())
    features = build_return_frame(prices)
    relative = f"data/markets/features/{series_id}.csv"
    if not features.empty:
        path = data_dir / "features" / f"{series_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(path, index=False)
        feature_path: str | None = relative
    else:
        feature_path = None
    reason = classify_history(
        observations=int(len(prices)),
        span_days=coverage_span_days(prices),
        feature_rows=int(len(features)),
        seq_len=seq_len,
        val_split=val_split,
        min_observations=min_observations,
        min_span_days=min_span_days,
    )
    if reason is None and block_if_stale and not active:
        reason = f"last observation {_iso(end)} is more than {STALE_DAYS} days old"
    return SeriesCoverage(
        id=series_id,
        group=group,
        name=name,
        symbol=symbol,
        source=source,
        source_url=source_url,
        kind=kind,
        quote=quote,
        frequency=frequency,
        start=_iso(start),
        end=_iso(end),
        observations=int(len(prices)),
        feature_rows=int(len(features)),
        feature_path=feature_path,
        train=reason is None,
        skip_reason=reason,
        active=bool(active),
        nonpositive_prices=bad_prints,
        price_field=price_field,
        rank=rank,
    )


def _fetch_spec(
    spec: PriceSpec,
    data_dir: Path,
    *,
    fred_ok: bool,
    stooq_ok: bool,
) -> SeriesCoverage:
    seq_len, val_split = _group_protocol(spec.group if spec.group != "fx" else "fx")
    prices: pd.DataFrame | None = None
    source = ""
    source_url = ""
    price_field: str | None = None
    errors: list[str] = []
    stooq_symbol = STOOQ_SYMBOLS.get(spec.id)
    if stooq_ok and stooq_symbol:
        try:
            prices, source_url = _load_stooq(stooq_symbol)
            source = f"Stooq {stooq_symbol}"
            price_field = "close"
        except FetchError as exc:
            errors.append(str(exc))
    if prices is None and fred_ok and spec.fred:
        try:
            prices, source_url = _load_fred(spec.fred)
            source = f"FRED {spec.fred}"
            price_field = "value"
        except FetchError as exc:
            errors.append(str(exc))
    if prices is None and spec.yahoo:
        try:
            prices, price_field, source_url = _load_yahoo(spec.yahoo)
            source = f"Yahoo {spec.yahoo}"
        except FetchError as exc:
            errors.append(str(exc))
    name, kind, quote = spec.label_for("fred" if source.startswith("FRED") else "yahoo")
    if source.startswith("Stooq"):
        name, kind, quote = spec.name, spec.kind, spec.quote
    if prices is None:
        return SeriesCoverage(
            id=spec.id,
            group=spec.group,
            name=name,
            symbol=spec.fred or spec.yahoo or spec.id,
            source=source or "unavailable",
            source_url=source_url,
            kind=kind,
            quote=quote,
            frequency="daily",
            start=None,
            end=None,
            observations=0,
            feature_rows=0,
            feature_path=None,
            train=False,
            skip_reason="; ".join(errors) or "no source available",
            active=False,
        )
    print(f"fetched {spec.id} from {source} ({len(prices)} rows)", flush=True)
    return _finish(
        series_id=spec.id,
        group=spec.group,
        name=name,
        symbol=spec.fred or spec.yahoo or spec.id,
        source=source,
        source_url=source_url,
        kind=kind,
        quote=quote,
        prices=prices,
        data_dir=data_dir,
        seq_len=seq_len,
        val_split=val_split,
        min_observations=MIN_OBSERVATIONS,
        min_span_days=MIN_SPAN_DAYS,
        active=None,
        block_if_stale=True,
        price_field=price_field,
    )


def _fetch_ecb(data_dir: Path, notes: list[str]) -> list[SeriesCoverage]:
    print("fetching ECB reference rates", flush=True)
    raw = http_get(ECB_ZIP_URL, timeout=90)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("utf-8-sig")
    wide = parse_ecb_hist_csv(text)
    frames = per_usd_frames(wide)
    ecb_max = max(pd.Timestamp(frame["date"].max()) for frame in frames.values())
    seq_len, val_split = _group_protocol("fx")
    records: list[SeriesCoverage] = []
    for code in sorted(frames):
        prices = frames[code]
        end = pd.Timestamp(prices["date"].max())
        active = (ecb_max - end).days <= 5
        record = _finish(
            series_id=f"fx-{code.lower()}",
            group="fx",
            name=currency_name(code),
            symbol=code,
            source="ECB eurofxref-hist",
            source_url=ECB_ZIP_URL,
            kind="reference rate",
            quote=f"{code} per 1 USD",
            prices=prices,
            data_dir=data_dir,
            seq_len=seq_len,
            val_split=val_split,
            min_observations=MIN_OBSERVATIONS,
            min_span_days=MIN_SPAN_DAYS,
            active=active,
            block_if_stale=False,
        )
        records.append(record)
    _frankfurter_note(frames, ecb_max, notes)
    trained = sum(record.train for record in records)
    notes.append(
        f"ECB file {_iso(wide['Date'].min())} to {_iso(wide['Date'].max())}: "
        f"{len(records)} currencies converted to per-USD, {trained} long enough to train."
    )
    return records


def _frankfurter_note(
    frames: dict[str, pd.DataFrame], ecb_max: pd.Timestamp, notes: list[str]
) -> None:
    day = ecb_max.strftime("%Y-%m-%d")
    url = f"https://api.frankfurter.app/{day}?from=USD"
    try:
        payload = http_json(url, timeout=30)
    except (FetchError, json.JSONDecodeError) as exc:
        notes.append(f"Frankfurter cross-check failed: {exc}")
        return
    if not isinstance(payload, dict):
        notes.append("Frankfurter cross-check returned an unexpected payload.")
        return
    try:
        rate_date, rates = parse_frankfurter_rates(payload)
    except (TypeError, ValueError) as exc:
        notes.append(f"Frankfurter cross-check could not be parsed: {exc}")
        return
    ours: dict[str, float] = {}
    for code, prices in frames.items():
        on_day = prices.loc[pd.to_datetime(prices["date"]) == ecb_max, "close"]
        if not on_day.empty:
            ours[code] = float(on_day.iloc[-1])
    gap, compared = max_relative_diff(ours, rates)
    if compared == 0:
        notes.append(f"Frankfurter had no overlapping USD rates on {rate_date}.")
        return
    notes.append(
        "Frankfurter USD rates on "
        f"{rate_date} differed from the ECB cross by at most {gap:.4%} "
        f"across {compared} currencies."
    )


def _ranking(notes: list[str]) -> list[dict[str, object]]:
    try:
        payload = http_json(COINGECKO_MARKETS_URL, timeout=30)
        if isinstance(payload, list):
            coins = dedupe_ranked(parse_coingecko_markets(payload))
            notes.append(
                f"CoinGecko market-cap ranking: {len(coins)} tickers on page 1."
            )
            return coins
        notes.append("CoinGecko markets endpoint did not return a list.")
    except (FetchError, json.JSONDecodeError) as exc:
        notes.append(f"CoinGecko ranking failed ({exc}).")
    payload = http_json(COINPAPRIKA_URL, timeout=30)
    if not isinstance(payload, list):
        raise FetchError("CoinPaprika tickers were not a list")
    coins = dedupe_ranked(parse_coinpaprika_tickers(payload))
    notes.append(f"CoinPaprika market-cap ranking: {len(coins)} tickers.")
    return coins


def _binance_usdt_symbols() -> set[str]:
    payload = http_json(BINANCE_EXCHANGE_URL, timeout=90)
    if not isinstance(payload, dict):
        raise FetchError("Binance exchangeInfo was not an object")
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise FetchError("Binance exchangeInfo has no symbols")
    listed: set[str] = set()
    for item in symbols:
        if not isinstance(item, dict):
            continue
        if item.get("quoteAsset") == "USDT" and item.get("status") == "TRADING":
            listed.add(str(item["symbol"]))
    return listed


def _fetch_binance_klines(symbol: str) -> pd.DataFrame:
    rows: list[list[object]] = []
    cursor = BINANCE_START_MS
    end = int(time.time() * 1000)
    while cursor < end:
        url = (
            f"{BINANCE_KLINES_URL}?symbol={symbol}&interval=1d"
            f"&startTime={cursor}&endTime={end}&limit=1000"
        )
        batch = http_json(url, timeout=60)
        if isinstance(batch, dict):
            raise FetchError(f"Binance klines error for {symbol}: {batch}")
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + 86_400_000
        if nxt <= cursor or len(batch) < 1000:
            break
        cursor = nxt
        time.sleep(0.05)
    if not rows:
        raise FetchError(f"no Binance klines for {symbol}")
    return parse_binance_klines(rows)


def _fetch_coinbase(symbol: str) -> pd.DataFrame | None:
    product = f"{symbol.upper()}-USD"
    end = datetime.now(timezone.utc)
    stop = end - timedelta(days=365 * 8)
    rows: list[list[object]] = []
    cursor_end = end
    while cursor_end > stop:
        cursor_start = max(stop, cursor_end - timedelta(days=290))
        url = (
            "https://api.exchange.coinbase.com/products/"
            f"{product}/candles?granularity=86400"
            f"&start={urllib.parse.quote(cursor_start.isoformat())}"
            f"&end={urllib.parse.quote(cursor_end.isoformat())}"
        )
        try:
            batch = http_json(url, timeout=30, attempts=2)
        except FetchError:
            return None
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        cursor_end = cursor_start
        time.sleep(0.15)
    if not rows:
        return None
    return parse_coinbase_candles(rows)


def _skip_coin(
    coin: dict[str, object], reason: str, source: str = "not downloaded"
) -> SeriesCoverage:
    symbol = str(coin["symbol"])
    rank = as_int(coin["rank"])
    return SeriesCoverage(
        id=f"crypto-{symbol.lower()}",
        group="crypto",
        name=str(coin["name"]),
        symbol=symbol,
        source=source,
        source_url="",
        kind="crypto",
        quote="USDT or USD close",
        frequency="daily",
        start=None,
        end=None,
        observations=0,
        feature_rows=0,
        feature_path=None,
        train=False,
        skip_reason=reason,
        active=False,
        rank=rank,
    )


def _try_crypto_prices(
    coin: dict[str, object], listed: set[str]
) -> tuple[pd.DataFrame, str, str, str] | str:
    """Return prices and source labels, or a skip reason."""
    symbol = str(coin["symbol"])
    market = next(
        (candidate for candidate in binance_candidates(symbol) if candidate in listed),
        None,
    )
    if market is not None:
        try:
            prices = _fetch_binance_klines(market)
        except FetchError as exc:
            return f"Binance klines failed for {market}: {exc}"
        quote = (
            "USDT close per 1000 tokens" if market.startswith("1000") else "USDT close"
        )
        url = f"{BINANCE_KLINES_URL}?symbol={market}&interval=1d"
        return prices, f"Binance {market}", url, quote
    prices = _fetch_coinbase(symbol)
    if prices is None:
        return f"no Binance USDT market and no Coinbase {symbol}-USD candles"
    url = f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles?granularity=86400"
    return prices, f"Coinbase {symbol}-USD", url, "USD close"


def _fetch_crypto(data_dir: Path, notes: list[str], limit: int) -> list[SeriesCoverage]:
    ranked = _ranking(notes)
    listed = _binance_usdt_symbols()
    notes.append(f"Binance lists {len(listed)} trading USDT symbols.")
    seq_len, val_split = _group_protocol("crypto")
    selected: list[SeriesCoverage] = []
    skipped: list[SeriesCoverage] = []
    deferred: list[dict[str, object]] = []

    def consider(coin: dict[str, object], allow_wrapped: bool) -> None:
        symbol = str(coin["symbol"])
        name = str(coin["name"])
        if is_stablecoin(symbol, name):
            skipped.append(_skip_coin(coin, "stablecoin"))
            return
        if is_wrapped(symbol, name) and not allow_wrapped:
            deferred.append(coin)
            return
        outcome = _try_crypto_prices(coin, listed)
        if isinstance(outcome, str):
            skipped.append(_skip_coin(coin, outcome))
            return
        prices, source, url, quote = outcome
        record = _finish(
            series_id=f"crypto-{symbol.lower()}",
            group="crypto",
            name=name,
            symbol=symbol,
            source=source,
            source_url=url,
            kind="crypto",
            quote=quote,
            prices=prices,
            data_dir=data_dir,
            seq_len=seq_len,
            val_split=val_split,
            min_observations=MIN_CRYPTO_BARS,
            min_span_days=700,
            active=None,
            block_if_stale=True,
            rank=as_int(coin["rank"]),
            price_field="close",
        )
        if record.train:
            selected.append(record)
            print(
                f"crypto {len(selected)}/{limit} {symbol} {record.start}→{record.end}",
                flush=True,
            )
        else:
            skipped.append(record)
            print(f"skip {symbol}: {record.skip_reason}", flush=True)

    for coin in ranked:
        if len(selected) >= limit:
            break
        consider(coin, allow_wrapped=False)
    if len(selected) < limit and deferred:
        notes.append(
            "Wrapped or staked coins were considered because non-wrapped names "
            f"produced only {len(selected)} series with two years of daily candles."
        )
    still_short = len(selected) < limit
    for coin in deferred:
        if still_short and len(selected) < limit:
            consider(coin, allow_wrapped=True)
        else:
            skipped.append(_skip_coin(coin, "wrapped or staked duplicate"))
    notes.append(
        f"Crypto sample: {len(selected)} coins trained "
        f"(cap {limit}); {len(skipped)} higher-ranked or ineligible names skipped."
    )
    return selected + skipped


def fetch_all(
    data_dir: Path | None = None,
    *,
    refresh: bool = False,
    crypto_limit: int = 50,
) -> dict[str, Any]:
    """Download every configured group. Reuses ``coverage.json`` unless refreshing."""
    root = resolve_repo_path(data_dir or "data/markets")
    coverage_path = root / "coverage.json"
    if coverage_path.exists() and not refresh:
        loaded = json.loads(coverage_path.read_text())
        if not isinstance(loaded, dict):
            raise FetchError("coverage.json must be an object")
        return cast(dict[str, Any], loaded)
    root.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    fred_ok = _probe_fred()
    stooq_ok = _probe_stooq()
    notes.append(
        "FRED graph CSV responded."
        if fred_ok
        else "FRED graph CSV timed out or was unreachable from this environment. "
        "S&P 500, WTI, Brent, and Henry Hub fall back to Yahoo where a symbol exists. "
        "The New Taiwan dollar (H.10 DEXTWUS) is omitted when FRED is down. "
        "IMF monthly aluminum is not fetched; daily aluminum futures are."
    )
    notes.append(
        "Stooq returned a daily CSV."
        if stooq_ok
        else "Stooq returned an HTML browser check rather than a CSV, so AAPL, MSFT, "
        "and NVDA come from Yahoo Finance."
    )
    series: list[SeriesCoverage] = []
    for spec in price_specs():
        if spec.group == "fx" and not fred_ok:
            series.append(
                SeriesCoverage(
                    id=spec.id,
                    group="fx",
                    name=spec.name,
                    symbol=spec.fred or spec.id,
                    source="FRED",
                    source_url=_fred_url(spec.fred or ""),
                    kind=spec.kind,
                    quote=spec.quote,
                    frequency="daily",
                    start=None,
                    end=None,
                    observations=0,
                    feature_rows=0,
                    feature_path=None,
                    train=False,
                    skip_reason="FRED H.10 CSV was unreachable; ECB does not publish this currency",
                    active=False,
                )
            )
            continue
        series.append(_fetch_spec(spec, root, fred_ok=fred_ok, stooq_ok=stooq_ok))
    series.extend(_fetch_ecb(root, notes))
    series.extend(_fetch_crypto(root, notes, crypto_limit))
    payload: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "notes": notes,
        "series": [record.as_dict() for record in series],
    }
    coverage_path.write_text(json.dumps(payload, indent=2) + "\n")
    trainable = sum(bool(record.train) for record in series)
    print(f"wrote {coverage_path} ({trainable} series marked for training)", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public market history")
    parser.add_argument("--data-dir", default="data/markets")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--crypto-limit", type=int, default=50)
    args = parser.parse_args()
    fetch_all(
        resolve_repo_path(args.data_dir),
        refresh=bool(args.refresh),
        crypto_limit=int(args.crypto_limit),
    )


if __name__ == "__main__":
    main()
