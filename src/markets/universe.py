"""Which series are eligible, and the static stock and commodity specs.

Crypto eligibility is applied in market-cap order. Stablecoins are always
skipped. Wrapped or staked duplicates of another asset are skipped when that
still leaves a full sample; the fetch script can relax that second filter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.markets.parse import as_int

# Symbols that track one dollar (or one euro) rather than a free float.
STABLECOIN_SYMBOLS = frozenset(
    {
        "USDT",
        "USDC",
        "DAI",
        "FDUSD",
        "TUSD",
        "USDE",
        "USDS",
        "PYUSD",
        "USDD",
        "FRAX",
        "LUSD",
        "GUSD",
        "CRVUSD",
        "USD1",
        "USDP",
        "BUSD",
        "EURC",
        "EURT",
        "AEUR",
        "USDB",
        "USD0",
        "RLUSD",
        "UST",
        "USTC",
        "SUSD",
        "MIM",
        "ALUSD",
        "DOLA",
        "USDX",
        "USDN",
        "USDY",
        "USDL",
        "BFUSD",
        "USDG",
        "USDH",
        "USDQ",
        "USDT0",
        "SUSDE",
        "SUSDS",
        "USAT",
    }
)

# Same economic exposure as BTC, ETH, or SOL. Forecasting them again double-counts.
WRAPPED_SYMBOLS = frozenset(
    {
        "WBTC",
        "WETH",
        "STETH",
        "WSTETH",
        "WEETH",
        "WBETH",
        "CBBTC",
        "CBETH",
        "RENBTC",
        "BTCB",
        "WBNB",
        "WSOL",
        "BETH",
        "RETH",
        "METH",
        "EZETH",
        "RSETH",
    }
)

CURRENCY_NAMES = {
    "EUR": "Euro",
    "USD": "US dollar",
    "JPY": "Japanese yen",
    "BGN": "Bulgarian lev",
    "CYP": "Cypriot pound",
    "CZK": "Czech koruna",
    "DKK": "Danish krone",
    "EEK": "Estonian kroon",
    "GBP": "British pound",
    "HUF": "Hungarian forint",
    "LTL": "Lithuanian litas",
    "LVL": "Latvian lats",
    "MTL": "Maltese lira",
    "PLN": "Polish zloty",
    "ROL": "Romanian leu (old)",
    "RON": "Romanian leu",
    "SEK": "Swedish krona",
    "SIT": "Slovenian tolar",
    "SKK": "Slovak koruna",
    "CHF": "Swiss franc",
    "ISK": "Icelandic krona",
    "NOK": "Norwegian krone",
    "HRK": "Croatian kuna",
    "RUB": "Russian ruble",
    "TRL": "Turkish lira (old)",
    "TRY": "Turkish lira",
    "AUD": "Australian dollar",
    "BRL": "Brazilian real",
    "CAD": "Canadian dollar",
    "CNY": "Chinese yuan",
    "HKD": "Hong Kong dollar",
    "IDR": "Indonesian rupiah",
    "ILS": "Israeli shekel",
    "INR": "Indian rupee",
    "KRW": "South Korean won",
    "MXN": "Mexican peso",
    "MYR": "Malaysian ringgit",
    "NZD": "New Zealand dollar",
    "PHP": "Philippine peso",
    "SGD": "Singapore dollar",
    "THB": "Thai baht",
    "ZAR": "South African rand",
    "TWD": "New Taiwan dollar",
}

# Two calendar years, and enough observations for a 20% holdout.
MIN_SPAN_DAYS = 730
MIN_OBSERVATIONS = 250
MIN_CRYPTO_BARS = 730
STALE_DAYS = 10


@dataclass(frozen=True)
class PriceSpec:
    """One equity or commodity series, with a preferred source and a fallback."""

    id: str
    group: str
    name: str
    kind: str
    quote: str
    yahoo: str | None = None
    fred: str | None = None
    fred_name: str | None = None
    fred_kind: str | None = None
    fred_quote: str | None = None

    def label_for(self, source: str) -> tuple[str, str, str]:
        if source == "fred" and self.fred_name:
            return (
                self.fred_name,
                self.fred_kind or self.kind,
                self.fred_quote or self.quote,
            )
        return self.name, self.kind, self.quote


def price_specs() -> list[PriceSpec]:
    """Equities and commodities. FX currencies come from the ECB file instead."""
    futures = "front-month futures"
    return [
        PriceSpec(
            "stock-aapl",
            "stocks",
            "Apple",
            "equity",
            "USD per share",
            yahoo="AAPL",
        ),
        PriceSpec(
            "stock-msft",
            "stocks",
            "Microsoft",
            "equity",
            "USD per share",
            yahoo="MSFT",
        ),
        PriceSpec(
            "stock-nvda",
            "stocks",
            "NVIDIA",
            "equity",
            "USD per share",
            yahoo="NVDA",
        ),
        PriceSpec(
            "stock-sp500",
            "stocks",
            "S&P 500",
            "index",
            "index level",
            yahoo="^GSPC",
            fred="SP500",
            fred_name="S&P 500",
            fred_kind="index",
            fred_quote="index level",
        ),
        PriceSpec(
            "cmdty-gold",
            "commodities",
            "Gold futures (Yahoo GC=F)",
            futures,
            "front-month close",
            yahoo="GC=F",
        ),
        PriceSpec(
            "cmdty-silver",
            "commodities",
            "Silver futures (Yahoo SI=F)",
            futures,
            "front-month close",
            yahoo="SI=F",
        ),
        PriceSpec(
            "cmdty-copper",
            "commodities",
            "Copper futures (Yahoo HG=F)",
            futures,
            "front-month close",
            yahoo="HG=F",
        ),
        PriceSpec(
            "cmdty-aluminum",
            "commodities",
            "Aluminum futures (Yahoo ALI=F)",
            futures,
            "front-month close",
            yahoo="ALI=F",
        ),
        PriceSpec(
            "cmdty-platinum",
            "commodities",
            "Platinum futures (Yahoo PL=F)",
            futures,
            "front-month close",
            yahoo="PL=F",
        ),
        PriceSpec(
            "cmdty-palladium",
            "commodities",
            "Palladium futures (Yahoo PA=F)",
            futures,
            "front-month close",
            yahoo="PA=F",
        ),
        PriceSpec(
            "cmdty-wti",
            "commodities",
            "WTI crude futures (Yahoo CL=F)",
            futures,
            "front-month close",
            yahoo="CL=F",
            fred="DCOILWTICO",
            fred_name="WTI crude (Cushing spot)",
            fred_kind="spot",
            fred_quote="USD per barrel",
        ),
        PriceSpec(
            "cmdty-brent",
            "commodities",
            "Brent crude futures (Yahoo BZ=F)",
            futures,
            "front-month close",
            yahoo="BZ=F",
            fred="DCOILBRENTEU",
            fred_name="Brent crude (spot)",
            fred_kind="spot",
            fred_quote="USD per barrel",
        ),
        PriceSpec(
            "cmdty-natgas",
            "commodities",
            "Natural gas futures (Yahoo NG=F)",
            futures,
            "front-month close",
            yahoo="NG=F",
            fred="DHHNGSP",
            fred_name="Henry Hub natural gas (spot)",
            fred_kind="spot",
            fred_quote="USD per MMBtu",
        ),
        PriceSpec(
            "fx-twd",
            "fx",
            "New Taiwan dollar",
            "reference rate",
            "TWD per 1 USD",
            fred="DEXTWUS",
            fred_name="New Taiwan dollar",
            fred_kind="reference rate",
            fred_quote="TWD per 1 USD",
        ),
    ]


def currency_name(code: str) -> str:
    return CURRENCY_NAMES.get(code, code)


def is_stablecoin(symbol: str, name: str) -> bool:
    if symbol.upper() in STABLECOIN_SYMBOLS:
        return True
    lowered = name.lower()
    return (
        "stablecoin" in lowered or "usd coin" in lowered or lowered.startswith("tether")
    )


def is_wrapped(symbol: str, name: str) -> bool:
    if symbol.upper() in WRAPPED_SYMBOLS:
        return True
    lowered = name.lower()
    return (
        lowered.startswith("wrapped ")
        or "staked ether" in lowered
        or lowered.startswith("lido staked")
        or lowered.startswith("liquid staked")
    )


def dedupe_ranked(coins: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep the best market-cap rank for each ticker."""
    best: dict[str, dict[str, object]] = {}
    for coin in coins:
        symbol = str(coin["symbol"])
        current = best.get(symbol)
        if current is None or as_int(coin["rank"]) < as_int(current["rank"]):
            best[symbol] = coin
    return sorted(best.values(), key=lambda item: as_int(item["rank"]))


def classify_history(
    *,
    observations: int,
    span_days: int,
    feature_rows: int,
    seq_len: int,
    val_split: float,
    min_observations: int = MIN_OBSERVATIONS,
    min_span_days: int = MIN_SPAN_DAYS,
    min_val: int = 30,
    min_train: int = 60,
) -> str | None:
    """Return a skip reason, or None when the series can be trained."""
    if span_days < min_span_days:
        return f"coverage is {span_days} days, shorter than {min_span_days}"
    if observations < min_observations:
        return f"{observations} observations, fewer than {min_observations}"
    n_windows = feature_rows - seq_len
    if n_windows <= 1:
        return "not enough feature rows for a window and a holdout"
    n_val = int(n_windows * val_split)
    n_train = n_windows - n_val
    if val_split > 0 and (n_val == 0 or n_val == n_windows):
        return "validation split leaves an empty train or holdout set"
    if n_val < min_val or n_train < min_train:
        return "holdout or train window is too small for a stable score"
    return None


def binance_candidates(symbol: str) -> list[str]:
    """USDT market, then the 1000-token contract some small coins use.

    Log returns of a 1000-token contract match the token when the scale factor
    is constant, so the contract is a valid return series. The quote is recorded
    separately so the table does not pretend the price is per single token.
    """
    upper = symbol.upper()
    return [f"{upper}USDT", f"1000{upper}USDT"]


@dataclass
class SeriesCoverage:
    """Metadata safe to commit. Prices themselves stay under ``data/``."""

    id: str
    group: str
    name: str
    symbol: str
    source: str
    source_url: str
    kind: str
    quote: str
    frequency: str
    start: str | None
    end: str | None
    observations: int
    feature_rows: int
    feature_path: str | None
    train: bool
    skip_reason: str | None
    active: bool
    nonpositive_prices: int = 0
    price_field: str | None = None
    rank: int | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
