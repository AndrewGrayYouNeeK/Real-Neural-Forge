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
  /\b(weather|temperature|temps?|rain|raining|snow|snowing|humidity|humid|hot|cold|windy|degrees|°|umbrella)\b/i;

/**
 * Detect crypto/stock intent from the user's wording.
 * Checked before weather so "bitcoin at right now" never geocodes.
 */
export function detectMarketKind(question: string): MarketKind | null {
  const text = question.replace(/\s+/g, " ").trim();
  if (!text) return null;

  if (parseCrypto(text) || CRYPTO_WORD.test(text)) return "crypto";

  if (/\$[A-Za-z]{1,5}\b/.test(text) || STOCK_WORD.test(text)) return "stock";

  // "what's AAPL at right now" / "TSLA trading at"
  if (PRICE_CUE.test(text) || /\bat\s+(?:right\s+)?now\b/i.test(text)) {
    const ticker = parseTicker(text);
    if (ticker && !WEATHER_WORD.test(text)) return "stock";
  }

  return null;
}

/** True when the ask is about weather/temperature/rain — not markets. */
export function isWeatherAsk(question: string): boolean {
  const text = question.replace(/\s+/g, " ").trim();
  if (!text) return false;
  if (detectMarketKind(text)) return false;
  return WEATHER_WORD.test(text);
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

  // Model said weather but wording is not weather and not market — keep requested
  // (caller refuses to geocode non-weather text).
  return requested;
}
