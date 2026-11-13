/**
 * Deterministic tool routing for clear user intents.
 * Used when the model fails to emit a parseable tool call.
 */

import { type AgentToolName, type ParsedToolCall } from "./protocol";

const STARTER_IDS = ["weather", "stocks", "crypto", "near-earth", "lunar"] as const;

function starterFromQuestion(question: string): (typeof STARTER_IDS)[number] {
  const q = question.toLowerCase();
  if (/\b(bitcoin|btc|crypto)\b/.test(q)) return "crypto";
  if (/\b(aapl|stocks?|stock)\b/.test(q)) return "stocks";
  if (/\b(near-?earth|asteroid)\b/.test(q)) return "near-earth";
  if (/\b(moon|lunar)\b/.test(q)) return "lunar";
  return "weather";
}

function horizonFromQuestion(question: string, fallback = 1): number {
  const match =
    question.match(/\b(?:next|horizon(?:\s+of)?|for)\s+(\d{1,2})\b/i) ||
    question.match(/\b(\d{1,2})\s+(?:values?|steps?|days?|points?)\b/i);
  if (!match) return fallback;
  const value = Number(match[1]);
  if (!Number.isFinite(value)) return fallback;
  return Math.min(24, Math.max(1, Math.floor(value)));
}

function call(name: AgentToolName, args: Record<string, unknown>, id: string): ParsedToolCall {
  return { id, name, args };
}

/**
 * Map clear weather / price / forecast / accuracy questions to a tool call.
 * Returns null when the question should stay with the model.
 */
export function routeClearIntent(question: string): ParsedToolCall | null {
  const text = question.replace(/\s+/g, " ").trim();
  if (!text) return null;
  const q = text.toLowerCase();

  // Accuracy / backtest metrics for a starter or model.
  if (/\b(accuracy|accurate|backtest|mae|rmse|mape|baseline|how\s+good|evaluation)\b/.test(q)) {
    return call(
      "get_evaluation_metrics",
      { model_id: starterFromQuestion(text), horizon: horizonFromQuestion(text, 1) },
      "intent_eval",
    );
  }

  // Explicit starter/series forecast (not live weather).
  if (
    /\b(forecast|predict|prediction|next\s+\d+|horizon)\b/.test(q) &&
    /\b(starter|series|sample|values?|steps?|loaded|champion)\b/.test(q)
  ) {
    return call(
      "forecast",
      { series: starterFromQuestion(text), horizon: horizonFromQuestion(text, 1) },
      "intent_forecast",
    );
  }
  if (
    /\b(forecast|predict)\b/.test(q) &&
    /\b(weather|stocks?|crypto|near-?earth|lunar)\s+(starter|series|sample)\b/.test(q)
  ) {
    return call(
      "forecast",
      { series: starterFromQuestion(text), horizon: horizonFromQuestion(text, 1) },
      "intent_forecast",
    );
  }
  // "forecast the next 3 values of the weather starter series"
  if (
    /\bnext\s+\d+\s+values?\b/.test(q) &&
    /\b(weather|stocks?|crypto|near-?earth|lunar|starter|series)\b/.test(q)
  ) {
    return call(
      "forecast",
      { series: starterFromQuestion(text), horizon: horizonFromQuestion(text, 3) },
      "intent_forecast",
    );
  }

  // Live crypto / stock prices.
  if (/\b(bitcoin|btc|ethereum|eth|solana|doge|crypto|cryptocurrency)\b/.test(q)) {
    return call("fetch_live_data", { kind: "crypto", query: text }, "intent_crypto");
  }
  if (
    /\b(stock|stocks|markets?|share\s+price|ticker|aapl|tsla|nvda|msft|goog|amzn)\b/.test(q) ||
    /\$[A-Za-z]{1,5}\b/.test(text)
  ) {
    return call("fetch_live_data", { kind: "stock", query: text }, "intent_stock");
  }

  // Live weather (current or forecast days) — not the weather starter sample.
  if (
    /\b(weather|temperature|temp|rain|forecast|hot|cold|humid)\b/.test(q) &&
    !/\b(starter|sample\s+model|series|backtest|mae)\b/.test(q)
  ) {
    return call("fetch_live_data", { kind: "weather", query: text }, "intent_weather");
  }

  return null;
}
