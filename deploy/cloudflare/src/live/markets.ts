import { PRICE_TTL_SECONDS, cachedJson, fetchJson, fetchText } from "./cache";

export type QuoteKind = "stock" | "crypto";

export type Quote = {
  kind: QuoteKind;
  symbol: string;
  name: string;
  price: number;
  changePct?: number | null;
  asOf?: string | null;
  currency: string;
};

const CRYPTO_IDS: Record<string, { id: string; symbol: string; name: string; binance: string; yahoo: string }> = {
  bitcoin: { id: "bitcoin", symbol: "BTC", name: "Bitcoin", binance: "BTCUSDT", yahoo: "BTC-USD" },
  btc: { id: "bitcoin", symbol: "BTC", name: "Bitcoin", binance: "BTCUSDT", yahoo: "BTC-USD" },
  ethereum: { id: "ethereum", symbol: "ETH", name: "Ethereum", binance: "ETHUSDT", yahoo: "ETH-USD" },
  eth: { id: "ethereum", symbol: "ETH", name: "Ethereum", binance: "ETHUSDT", yahoo: "ETH-USD" },
  solana: { id: "solana", symbol: "SOL", name: "Solana", binance: "SOLUSDT", yahoo: "SOL-USD" },
  sol: { id: "solana", symbol: "SOL", name: "Solana", binance: "SOLUSDT", yahoo: "SOL-USD" },
  dogecoin: { id: "dogecoin", symbol: "DOGE", name: "Dogecoin", binance: "DOGEUSDT", yahoo: "DOGE-USD" },
  doge: { id: "dogecoin", symbol: "DOGE", name: "Dogecoin", binance: "DOGEUSDT", yahoo: "DOGE-USD" },
  xrp: { id: "ripple", symbol: "XRP", name: "XRP", binance: "XRPUSDT", yahoo: "XRP-USD" },
  ripple: { id: "ripple", symbol: "XRP", name: "XRP", binance: "XRPUSDT", yahoo: "XRP-USD" },
};

const NAME_TO_TICKER: Record<string, string> = {
  apple: "AAPL",
  microsoft: "MSFT",
  google: "GOOGL",
  alphabet: "GOOGL",
  amazon: "AMZN",
  nvidia: "NVDA",
  tesla: "TSLA",
  meta: "META",
  facebook: "META",
  netflix: "NFLX",
  "s&p": "SPY",
  "s&p 500": "SPY",
  "s & p": "SPY",
  nasdaq: "QQQ",
  dow: "DIA",
};

const TICKER_RE = /\$([A-Za-z]{1,5})\b|\b([A-Z]{1,5})\b/g;

export function parseCrypto(question: string): (typeof CRYPTO_IDS)[string] | null {
  const text = question.toLowerCase();
  if (/\b(crypto|cryptocurrency|coin)\b/.test(text) && !hasNamedCoin(text)) {
    return CRYPTO_IDS.bitcoin;
  }
  for (const key of Object.keys(CRYPTO_IDS).sort((a, b) => b.length - a.length)) {
    if (new RegExp(`\\b${key}\\b`, "i").test(text)) return CRYPTO_IDS[key];
  }
  return null;
}

function hasNamedCoin(text: string): boolean {
  return Object.keys(CRYPTO_IDS).some((key) => new RegExp(`\\b${key}\\b`, "i").test(text));
}

export function parseTicker(question: string): string | null {
  const text = question.toLowerCase();
  for (const [name, ticker] of Object.entries(NAME_TO_TICKER)) {
    if (new RegExp(`\\b${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(text)) {
      return ticker;
    }
  }
  const compact = question.replace(/[^A-Za-z0-9$&\s]/g, " ");
  let match: RegExpExecArray | null;
  const tickers: string[] = [];
  TICKER_RE.lastIndex = 0;
  const tickerScan = /\$([A-Za-z]{1,5})\b|\b([A-Z]{2,5})\b/g;
  while ((match = tickerScan.exec(compact))) {
    tickers.push((match[1] || match[2]).toUpperCase());
  }
  const skip = new Set([
    "HOW", "IS", "THE", "DOING", "WHAT", "WHATS", "PRICE", "STOCK", "STOCKS",
    "MARKET", "MARKETS", "SHARE", "SHARES", "TODAY", "PLEASE", "TELL", "ABOUT",
    "AND", "FOR", "NOW", "LIVE", "QUOTE",
  ]);
  const found = tickers.find((ticker) => !skip.has(ticker));
  if (found) return found;
  if (/\b(stock|stocks|market|markets|share price)\b/i.test(question)) return "SPY";
  return null;
}

function formatMoney(value: number, currency = "USD"): string {
  const abs = Math.abs(value);
  const digits = abs >= 1000 ? 2 : abs >= 1 ? 2 : 4;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatChange(changePct: number | null | undefined): string {
  if (changePct == null || !Number.isFinite(changePct)) return "";
  const rounded = changePct.toFixed(2);
  const direction = changePct > 0 ? "up" : changePct < 0 ? "down" : "unchanged at";
  if (changePct === 0) return " It is unchanged on the day.";
  return ` It is ${direction} ${Math.abs(Number(rounded))}% on the day.`;
}

export function quoteReplyText(quote: Quote): string {
  const when = quote.asOf ? ` as of ${quote.asOf}` : "";
  const change = formatChange(quote.changePct ?? null);
  const noun = quote.kind === "crypto" ? quote.name : quote.symbol;
  return `${noun} is at ${formatMoney(quote.price, quote.currency)}${when}.${change} This is a live ${quote.kind === "crypto" ? "price" : "market quote"}, not financial advice.`;
}


async function firstSuccessful<T>(attempts: Array<() => Promise<T>>): Promise<T> {
  let last: unknown;
  for (const attempt of attempts) {
    try {
      return await attempt();
    } catch (error) {
      last = error;
    }
  }
  throw last instanceof Error ? last : new Error("all quote sources failed");
}

export async function fetchCrypto(coin: (typeof CRYPTO_IDS)[string]): Promise<Quote> {
  const cacheKey = `crypto:${coin.id}`;
  return cachedJson(cacheKey, PRICE_TTL_SECONDS, () =>
    firstSuccessful([
      async () => {
        const url =
          "https://api.coingecko.com/api/v3/simple/price?ids=" +
          encodeURIComponent(coin.id) +
          "&vs_currencies=usd&include_24hr_change=true";
        const payload = await fetchJson<Record<string, { usd?: number; usd_24h_change?: number }>>(url);
        const row = payload[coin.id];
        if (!row || row.usd == null) throw new Error("missing");
        return {
          kind: "crypto",
          symbol: coin.symbol,
          name: coin.name,
          price: row.usd,
          changePct: row.usd_24h_change ?? null,
          asOf: null,
          currency: "USD",
        } satisfies Quote;
      },
      async () => {
        const url = `https://api.binance.com/api/v3/ticker/24hr?symbol=${coin.binance}`;
        const row = await fetchJson<{ lastPrice?: string; priceChangePercent?: string }>(url);
        const price = Number(row.lastPrice);
        if (!Number.isFinite(price)) throw new Error("missing binance price");
        return {
          kind: "crypto",
          symbol: coin.symbol,
          name: coin.name,
          price,
          changePct: row.priceChangePercent != null ? Number(row.priceChangePercent) : null,
          asOf: null,
          currency: "USD",
        } satisfies Quote;
      },
      async () => {
        const url = `https://api.coinbase.com/v2/prices/${encodeURIComponent(coin.symbol)}-USD/spot`;
        const payload = await fetchJson<{ data?: { amount?: string } }>(url);
        const price = Number(payload.data?.amount);
        if (!Number.isFinite(price)) throw new Error("missing coinbase price");
        return {
          kind: "crypto",
          symbol: coin.symbol,
          name: coin.name,
          price,
          changePct: null,
          asOf: null,
          currency: "USD",
        } satisfies Quote;
      },
      async () => {
        const quote = await fetchYahooChart(coin.yahoo);
        return { ...quote, kind: "crypto", symbol: coin.symbol, name: coin.name };
      },
    ]),
  );
}

export function parseStooqCsv(text: string): Quote | null {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return null;
  const header = lines[0].split(",").map((cell) => cell.trim().toLowerCase());
  const row = lines[1].split(",").map((cell) => cell.trim());
  const symbol = row[header.indexOf("symbol")] || row[0];
  const closeIndex = header.indexOf("close") >= 0 ? header.indexOf("close") : 6;
  const dateIndex = header.indexOf("date");
  const price = Number(row[closeIndex]);
  if (!Number.isFinite(price) || /n\/a/i.test(row[closeIndex] || "")) return null;
  const ticker = symbol.replace(/\.US$/i, "").toUpperCase();
  return {
    kind: "stock",
    symbol: ticker,
    name: ticker,
    price,
    changePct: null,
    asOf: dateIndex >= 0 ? row[dateIndex] : null,
    currency: "USD",
  };
}

const MARKET_UA = "Mozilla/5.0 (compatible; NeuralForge/1.0; +https://forge.youneek.dev)";

export async function fetchYahooChart(ticker: string): Promise<Quote> {
  const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker.toUpperCase())}?interval=1d&range=5d`;
  const payload = await fetchJson<{
    chart?: {
      result?: Array<{
        meta?: {
          regularMarketPrice?: number;
          regularMarketTime?: number;
          regularMarketChangePercent?: number;
          symbol?: string;
        };
        indicators?: { quote?: Array<{ close?: Array<number | null> }> };
      }>;
    };
  }>(yahooUrl, { headers: { "User-Agent": MARKET_UA } });
  const result = payload.chart?.result?.[0];
  const price = Number(result?.meta?.regularMarketPrice);
  if (!Number.isFinite(price)) throw new Error("missing yahoo price");
  let changePct =
    result?.meta?.regularMarketChangePercent != null
      ? Number(result.meta.regularMarketChangePercent)
      : null;
  if (changePct == null || !Number.isFinite(changePct)) {
    const closes = result?.indicators?.quote?.[0]?.close?.filter((value): value is number => value != null);
    if (closes && closes.length >= 2 && closes[closes.length - 2]) {
      const previous = closes[closes.length - 2];
      changePct = ((price - previous) / previous) * 100;
    } else {
      changePct = null;
    }
  }
  const asOf = result?.meta?.regularMarketTime
    ? new Date(result.meta.regularMarketTime * 1000).toISOString().slice(0, 10)
    : null;
  return {
    kind: "stock",
    symbol: ticker.toUpperCase(),
    name: ticker.toUpperCase(),
    price,
    changePct,
    asOf,
    currency: "USD",
  };
}

export async function fetchStock(ticker: string): Promise<Quote> {
  const symbol = ticker.toLowerCase();
  const cacheKey = `stock:${symbol}`;
  return cachedJson(cacheKey, PRICE_TTL_SECONDS, async () => {
    try {
      return await fetchYahooChart(ticker);
    } catch {
      const stooqUrl = `https://stooq.com/q/l/?s=${encodeURIComponent(symbol)}.us&f=sd2t2ohlcv&h&e=csv`;
      const stooq = await fetchText(stooqUrl, { headers: { "User-Agent": MARKET_UA } });
      if (!stooq.ok) throw new Error("missing stock price");
      const quote = parseStooqCsv(stooq.text);
      if (!quote) throw new Error("missing stock price");
      return quote;
    }
  });
}

export async function answerQuote(
  question: string,
  prefer: "auto" | "stock" | "crypto" = "auto",
): Promise<{ text: string }> {
  try {
    const crypto = prefer !== "stock" ? parseCrypto(question) : null;
    if (crypto && prefer !== "stock") {
      const quote = await fetchCrypto(crypto);
      return { text: quoteReplyText(quote) };
    }
    const ticker = parseTicker(question);
    if (!ticker) {
      return {
        text: "Tell me a ticker or a coin, for example AAPL or Bitcoin, and I will look up a live price. It will not be financial advice.",
      };
    }
    const quote = await fetchStock(ticker);
    return { text: quoteReplyText(quote) };
  } catch {
    return {
      text: "I could not load a live price right now. Please try again in a moment. This would not have been financial advice anyway.",
    };
  }
}
