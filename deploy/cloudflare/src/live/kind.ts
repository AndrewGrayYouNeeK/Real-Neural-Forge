/**
 * Classify live-data questions so crypto/stock never fall through to weather geocoding.
 */

import { parseCrypto, parseTicker } from "./markets";

export type MarketKind = "crypto" | "stock";

const CRYPTO_WORD =
  /\b(bitcoin|btc|ethereum|eth|solana|sol|doge|dogecoin|xrp|ripple|crypto|cryptocurrency|coin)\b/i;
const STOCK_WORD =
  /\b(stock|stocks|markets?|share\s+price|ticker|aapl|tsla|nvda|msft|googl?|amzn|meta|nflx)\b/i;
const PRICE_CUE = /\b(price|pricing|quote|trading\s+at|trades?\s+at|worth|valued\s+at)\b/i;
const WEATHER_WORD =
  /\b(weather|temperature|temps?|rain|raining|snow|snowing|humidity|humid|hot|cold|windy|degrees|°|umbrella|high|lows?|precip)\b/i;
const FORECAST_PLACE =
  /\bforecast\b.+\b(in|for|at|near)\b.+\b[A-Za-z]{2,}/i;
const HIGH_LOW =
  /\b(high\s+and\s+low|low\s+and\s+high|highs?\s+and\s+lows?|lows?\s+and\s+highs?)\b/i;

/**
 * Detect crypto/stock intent from the user's wording.
 * Checked before weather so "bitcoin at right now" never geocodes.
 */
export function detectMarketKind(question: string): MarketKind | null {
  const text = question.replace(/\s+/g, " ").trim();
  if (!text) return null;

  // Live weather "high/low" must win over stock-ish "high".
  if (HIGH_LOW.test(text) || (/\b(high|low)\b/i.test(text) && /\b(tomorrow|today|tonight|weekend)\b/i.test(text))) {
    return null;
  }

  if (parseCrypto(text) || CRYPTO_WORD.test(text)) return "crypto";

  if (/\$[A-Za-z]{1,5}\b/.test(text) || STOCK_WORD.test(text)) return "stock";

  // "what's AAPL at right now" / "TSLA trading at"
  if (PRICE_CUE.test(text) || /\bat\s+(?:right\s+)?now\b/i.test(text)) {
    const ticker = parseTicker(text);
    if (ticker && !WEATHER_WORD.test(text) && !HIGH_LOW.test(text)) return "stock";
  }

  return null;
}

/** True when the ask is about weather/temperature/rain — not markets. */
export function isWeatherAsk(question: string): boolean {
  const text = question.replace(/\s+/g, " ").trim();
  if (!text) return false;
  if (detectMarketKind(text)) return false;
  if (WEATHER_WORD.test(text) || HIGH_LOW.test(text)) return true;
  // "Forecast tomorrow's high and low in New York" / "forecast for Dallas tomorrow"
  if (FORECAST_PLACE.test(text) && !/\b(starter|series|sample|model|horizon|values?)\b/i.test(text)) {
    return true;
  }
  if (
    /\bforecast\b/i.test(text) &&
    /\b(tomorrow|today|tonight|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i.test(
      text,
    ) &&
    !/\b(starter|series|sample|model|values?)\b/i.test(text)
  ) {
    return true;
  }
  return false;
}

/**
 * Resolve the fetch_live_data kind. Market signals on the original question or
 * tool query override a mistaken model `kind: "weather"`.
 */
export function resolveFetchLiveKind(
  requestedKind: string,
  toolQuery: string,
  originalQuestion: string,
): string {
  const requested = requestedKind.trim().toLowerCase() || "weather";
  if (requested === "moon" || requested === "space" || requested === "solar") {
    return requested;
  }

  const market = detectMarketKind(originalQuestion) || detectMarketKind(toolQuery);
  if (market) return market;

  if (requested === "stock" || requested === "crypto") return requested;

  if (isWeatherAsk(originalQuestion) || isWeatherAsk(toolQuery)) return "weather";

  // Model asked for weather — allow it when the text is not a market ask.
  if (requested === "weather") return "weather";

  return requested;
}
