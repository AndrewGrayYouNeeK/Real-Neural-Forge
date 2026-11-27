/**
 * Agent tools wired to existing forecast, evaluation, live-data, and workspace paths.
 * Tools never invent live weather or prices.
 */

import type { D1Like } from "../live/ai";
import { visitorFromCf, wantsCurrentConditions, type CfIncoming, type VisitorGeo } from "../live/geo";
import { isWeatherAsk, resolveFetchLiveKind } from "../live/kind";
import { answerWeather, fetchDailyHistory, resolveHistoryPlace } from "../live/weather";
import { formatPlace } from "../live/geo";
import { answerQuote } from "../live/markets";
import {
  clearMemory,
  deleteMemoryKey,
  listMemory,
  upsertMemory,
} from "./memory";
import { AGENT_TOOL_NAMES, type AgentToolName } from "./protocol";
import {
  createAlertRule,
  insertAlertEvent,
  listAlertEvents,
  listAlertRules,
  type AlertKind,
} from "./alerts";
import { createReport } from "./reports";
import {
  createScenario,
  getScenarioOwned,
  listScenarios,
  scenarioRerunQuestion,
} from "./scenarios";
import {
  EVAL_CACHE_TTL_SECONDS,
  FORECAST_CACHE_TTL_SECONDS,
  cacheGetOrLoad,
  evalCacheKey,
  forecastCacheKey,
  readStoredBacktest,
  type WaitUntil,
} from "./toolCache";
import { runReadPage, runWebSearch, runWikiSummary } from "./knowledge";

/**
 * Build the weather lookup string from the user's original question and the
 * model's tool query. Current-conditions intent ("right now") must come from
 * the original question — models often drop those words from `query`.
 */
export function weatherQuestionForTool(originalQuestion: string, toolQuery: string): string {
  const original = originalQuestion.replace(/\s+/g, " ").trim();
  const query = toolQuery.replace(/\s+/g, " ").trim();
  const base = query || original;
  if (!base) return original;
  if (wantsCurrentConditions(original) && !wantsCurrentConditions(base)) {
    return `${base} right now`;
  }
  if (wantsCurrentConditions(original)) {
    // Prefer the original question so place + "right now" stay together.
    return original;
  }
  return base;
}

export const MAX_TOOL_CALLS = 8;
export const TOOL_TIMEOUT_MS = 8_000;
/** Per-attempt timeout for forecast/eval (kept under the shared budget). */
export const FORECAST_TOOL_TIMEOUT_MS = 12_000;
/** Hard cap across wake + attempts so /agent/chat does not burn ~20s then give up. */
export const CONTAINER_TOOL_BUDGET_MS = 22_000;
/**
 * simulate_world with encoder auto/jepa trains + backtests a small ensemble (CPU, ~5-10s
 * on the live container), so it gets one longer attempt instead of two short ones.
 */
export const SLOW_SIMULATE_TIMEOUT_MS = 26_000;
export const TURN_TIMEOUT_MS = 45_000;
/** Base wait before the first container retry. */
export const CONTAINER_COLD_START_WAIT_MS = 500;
export const CONTAINER_RETRY_ATTEMPTS = 2;
/** Optional accuracy/backtest must not consume the forecast timeout budget. */
export const ACCURACY_FETCH_TIMEOUT_MS = 2_000;
export const FORECAST_UNAVAILABLE_TEXT =
  "The forecast service is warming up or temporarily unavailable. Please try that forecast again in a moment — I could not load a reliable prediction just now.";
export const EVAL_UNAVAILABLE_TEXT =
  "Model accuracy metrics are temporarily unavailable. Please try again in a moment.";
export const LIVE_DATA_UNAVAILABLE_TEXT =
  "I could not load that live data right now. Please try again in a moment.";

const CONTAINER_RETRY_TOOLS = new Set<AgentToolName>(["forecast", "get_evaluation_metrics"]);

export type ToolResult = {
  ok: boolean;
  name: AgentToolName;
  summary: string;
  data?: unknown;
  error?: string;
  durationMs: number;
};

export type ContainerFetcher = (
  path: string,
  init?: { method?: string; body?: string; headers?: Record<string, string>; signal?: AbortSignal },
) => Promise<{ status: number; json: unknown }>;

export type ToolContext = {
  question: string;
  visitor: VisitorGeo;
  fetchContainer: ContainerFetcher;
  authHeaders: Record<string, string>;
  /** AbortSignal for the current timed attempt (cancels hung container fetches). */
  signal?: AbortSignal;
  /** Optional D1 for stored backtest metrics (0005_backtest) and user memory (0006). */
  db?: D1Like;
  /** Schedule background cache refresh without blocking the chat reply. */
  waitUntil?: WaitUntil;
  /** D1 memory scope from resolveMemoryIdentity (user:/owner/ip:). */
  memoryScope?: string;
  /** Owner session (higher daily knowledge-lookup allowance). */
  owner?: boolean;
  /** True when the account turned memory off in Settings. */
  memoryOff?: boolean;
};

const STARTER_IDS = new Set(["weather", "stocks", "crypto", "near-earth", "lunar"]);

export const TOOL_DEFINITIONS = [
  {
    type: "function" as const,
    name: "forecast",
    description:
      "Run a model forecast for a series. series is a starter id (weather, stocks, crypto, near-earth, lunar), a workspace model id, or 'loaded' for the shared champion. Returns prediction values with interval/confidence and recent accuracy when available. Weather values and errors are already converted to °F.",
    parameters: {
      type: "object",
      properties: {
        series: {
          type: "string",
          description: "Starter id, workspace model id, or 'loaded'.",
        },
        horizon: {
          type: "integer",
          description: "Forecast steps ahead (1-24). Default 1.",
        },
      },
      required: ["series"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "fetch_live_data",
    description:
      "Fetch live weather, stock/crypto prices, moon phase, or near-Earth space data from existing sources. Never invent values. For weather, pass the user's wording in query so 'right now/currently/now' gets current conditions and named days get forecasts.",
    parameters: {
      type: "object",
      properties: {
        kind: {
          type: "string",
          enum: ["weather", "stock", "crypto", "moon", "space", "solar"],
          description: "Which live source to query.",
        },
        query: {
          type: "string",
          description:
            "Place, ticker, coin, or free-text for the live lookup. For weather include time words like 'right now' or 'tomorrow'.",
        },
      },
      required: ["kind"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "get_evaluation_metrics",
    description:
      "Return rolling-origin backtest metrics (MAE, RMSE, MAPE, directional accuracy) and baseline comparison for a starter or workspace model. Weather MAE/RMSE are already converted to °F.",
    parameters: {
      type: "object",
      properties: {
        model_id: {
          type: "string",
          description: "Starter id (weather, stocks, …) or workspace model id.",
        },
        horizon: {
          type: "integer",
          description: "Backtest horizon. Default 1.",
        },
      },
      required: ["model_id"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "compare_horizons",
    description:
      "Compare backtest accuracy across several forecast horizons for one model.",
    parameters: {
      type: "object",
      properties: {
        model_id: { type: "string", description: "Starter or workspace model id." },
        horizons: {
          type: "array",
          items: { type: "integer" },
          description: "Horizons to compare. Default [1, 3, 7].",
        },
      },
      required: ["model_id"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "explain_prediction",
    description:
      "Explain why a forecast looks the way it does: recent trend, seasonality, optional PCA drivers / K-Means regime, plus the point and interval. Weather explanation_facts are already in °F — quote those numbers as °F and never convert again.",
    parameters: {
      type: "object",
      properties: {
        series: {
          type: "string",
          description: "Starter id, workspace model id, or 'loaded'.",
        },
      },
      required: ["series"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "list_user_series",
    description:
      "List the signed-in user's workspace datasets, models, and predictions (including Train-on-my-CSV uploads). Requires an account session.",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "remember_fact",
    description:
      "Store a lasting preference or fact for this user (units, location, favorite_series, name, …). Keys are snake_case. Use when the user asks you to remember something.",
    parameters: {
      type: "object",
      properties: {
        key: {
          type: "string",
          description: "snake_case key, e.g. units, location, favorite_series.",
        },
        value: {
          type: "string",
          description: "Short value to store (max ~200 chars).",
        },
      },
      required: ["key", "value"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "recall_memory",
    description: "List stored preferences/facts for this user, or one key if provided.",
    parameters: {
      type: "object",
      properties: {
        key: {
          type: "string",
          description: "Optional single key to look up.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "forget_memory",
    description:
      "Delete one stored fact by key, or clear all memory when clear_all is true.",
    parameters: {
      type: "object",
      properties: {
        key: { type: "string", description: "Key to forget." },
        clear_all: {
          type: "boolean",
          description: "If true, wipe all facts for this user.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "save_scenario",
    description:
      "Save a named forecast/question scenario (name, question, optional series + horizon) for later re-run.",
    parameters: {
      type: "object",
      properties: {
        name: { type: "string", description: "Short scenario name." },
        question: {
          type: "string",
          description: "The question or prompt to replay.",
        },
        series: {
          type: "string",
          description: "Optional starter/series id (weather, stocks, …).",
        },
        horizon: {
          type: "integer",
          description: "Optional forecast horizon 1-24.",
        },
      },
      required: ["name", "question"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "list_scenarios",
    description: "List this user's saved forecast scenarios.",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "run_scenario",
    description:
      "Load a saved scenario by id or name and return the question/series/horizon to execute (then call forecast if series is set).",
    parameters: {
      type: "object",
      properties: {
        id: { type: "string", description: "Scenario id." },
        name: { type: "string", description: "Scenario name if id unknown." },
      },
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "rename_user_series",
    description: "Rename an uploaded workspace CSV dataset by id. Requires sign-in.",
    parameters: {
      type: "object",
      properties: {
        dataset_id: { type: "string", description: "Workspace dataset id." },
        name: { type: "string", description: "New display name." },
      },
      required: ["dataset_id", "name"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "delete_user_series",
    description: "Delete an uploaded workspace CSV dataset by id. Requires sign-in.",
    parameters: {
      type: "object",
      properties: {
        dataset_id: { type: "string", description: "Workspace dataset id." },
      },
      required: ["dataset_id"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "retrain_user_series",
    description:
      "Queue a capped retrain on an uploaded workspace CSV dataset. Requires sign-in.",
    parameters: {
      type: "object",
      properties: {
        dataset_id: { type: "string", description: "Workspace dataset id." },
        target_column: {
          type: "string",
          description: "Optional target column name.",
        },
      },
      required: ["dataset_id"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "analyze_pca",
    description:
      "Principal component analysis on a multi-column workspace CSV: explained variance and which columns move together. Use when the user asks which columns matter or which features co-vary. Requires a signed-in dataset id from list_user_series or a recent upload.",
    parameters: {
      type: "object",
      properties: {
        dataset_id: {
          type: "string",
          description: "Workspace dataset id (dst_…).",
        },
        n_components: {
          type: "integer",
          description: "Optional number of principal components to keep.",
        },
      },
      required: ["dataset_id"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "analyze_regimes",
    description:
      "K-Means clustering on a workspace CSV: finds regimes/patterns (auto k via silhouette, k≤8), plain labels like calm/volatile or low/high, and points far from any centroid (anomalies). Use for 'what patterns are in my data' or 'anything unusual?'.",
    parameters: {
      type: "object",
      properties: {
        dataset_id: {
          type: "string",
          description: "Workspace dataset id (dst_…).",
        },
        k: {
          type: "integer",
          description: "Optional fixed cluster count (2-8). Omit to auto-pick.",
        },
      },
      required: ["dataset_id"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "what_if",
    description:
      "Shock the last input on a series (or a saved scenario's series) and compare the baseline forecast to the shocked path. Use for counterfactuals like 'what if the last value dropped 10%'. Weather paths, shocks, and deltas are already converted to °F — quote them as °F and never convert again.",
    parameters: {
      type: "object",
      properties: {
        series: {
          type: "string",
          description: "Starter id or workspace model id. Optional when scenario_id is set.",
        },
        scenario_id: {
          type: "string",
          description: "Saved scenario id; uses its series/horizon when set.",
        },
        mode: {
          type: "string",
          enum: ["pct", "abs"],
          description: "pct = percent change on last input; abs = add value. Default pct.",
        },
        value: {
          type: "number",
          description: "Shock size. Default -10 for a 10% drop.",
        },
        horizon: {
          type: "integer",
          description: "Forecast horizon steps. Default from scenario or 1.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "create_report",
    description:
      "Build a shareable one-click summary for a series (forecast, accuracy, regimes) and return a public report id/path.",
    parameters: {
      type: "object",
      properties: {
        series: {
          type: "string",
          description: "Starter series id (weather, stocks, crypto, lunar, near-earth).",
        },
      },
      required: ["series"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "create_alert",
    description:
      "Create an in-app alert rule: threshold on the latest value, or unusual-point (K-Means anomaly) for a series. No email/SMS.",
    parameters: {
      type: "object",
      properties: {
        name: { type: "string", description: "Short alert name." },
        series: { type: "string", description: "Starter series id." },
        kind: {
          type: "string",
          enum: ["threshold", "anomaly"],
          description: "threshold or anomaly.",
        },
        op: {
          type: "string",
          description: "For threshold: >, <, >=, <=, or abs>. Default >.",
        },
        value: {
          type: "number",
          description: "For threshold: numeric threshold.",
        },
      },
      required: ["name", "series", "kind"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "list_alerts",
    description: "List in-app alert rules and recent fired events for this visitor/account.",
    parameters: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "check_alerts",
    description:
      "Run enabled alert rules now (threshold / K-Means anomalies) and return any newly fired in-app events.",
    parameters: {
      type: "object",
      properties: {
        series: {
          type: "string",
          description: "Optional series filter.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "generate_synthetic_series",
    description:
      "Generate a realistic sample series (sine/trend/seasonal/noise/walk, or a tiny diffusion-assisted path) for demos or Train-on-my-CSV.",
    parameters: {
      type: "object",
      properties: {
        pattern: {
          type: "string",
          enum: ["sine", "trend", "seasonal", "noise", "walk", "diffusion"],
          description: "Generator pattern. Default sine.",
        },
        length: {
          type: "integer",
          description: "Number of points (32-2048). Default 256.",
        },
        noise: {
          type: "number",
          description: "Noise scale 0-1. Default 0.05.",
        },
        seed: {
          type: "integer",
          description: "RNG seed. Default 0.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "simulate_world",
    description:
      "Phase 5 world model / simulator. Fits a small learned latent model (PCA state + ridge dynamics + K-Means regime noise; weather also gets an annual seasonal baseline) to a series and rolls a baseline plus up to 3 scenarios forward many steps with p10/p50/p90 ranges (calibrated to cover ~80% in backtests) and an honest backtest vs persistence. Use for multi-step 'what happens over the next N days if…' questions: a one-off shock, a sustained drift per step, a volatility change, or forcing a value at a step. Pass series (starter id) or values (24+ numbers, e.g. from a chart the user attached). Weather numbers, event values, and results are °F — quote them as °F and never convert.",
    parameters: {
      type: "object",
      properties: {
        series: {
          type: "string",
          description: "Starter id: weather, stocks, crypto, lunar, near-earth. Default weather.",
        },
        values: {
          type: "array",
          items: { type: "number" },
          description: "Optional: the user's own series (24-2000 numbers, oldest first) instead of a starter. To fit/train a quick model on numbers the user pasted and backtest it, pass them here; the result includes an honest holdout backtest.",
        },
        values_from: {
          type: "string",
          enum: ["weather_history"],
          description: "Use the daily highs from this turn's weather_history result as the series (instead of copying numbers into values).",
        },
        unit: {
          type: "string",
          description: "Unit label for values (only with values).",
        },
        horizon: {
          type: "integer",
          description: "Steps to roll forward (1-60). Default 14.",
        },
        encoder: {
          type: "string",
          enum: ["pca", "jepa", "auto"],
          description:
            "Latent encoder. pca (default, fastest). auto = backtest PCA vs the self-supervised JEPA encoder on this series and simulate with the winner; use it when the user asks for the JEPA / self-supervised / smartest world model or which encoder is better. jepa = force JEPA.",
        },
        scenarios: {
          type: "array",
          description: "Up to 3 scenarios compared against the baseline.",
          items: {
            type: "object",
            properties: {
              name: { type: "string", description: "Short label, e.g. 'Cold snap'." },
              events: {
                type: "array",
                description: "Up to 4 events.",
                items: {
                  type: "object",
                  properties: {
                    kind: {
                      type: "string",
                      enum: ["shock", "drift", "volatility", "set"],
                      description:
                        "shock = one-off jump at step; drift = add value every step from step on; volatility = multiply noise by value from step on; set = force the value at step.",
                    },
                    step: { type: "integer", description: "1-based step the event starts." },
                    value: { type: "number", description: "Event size (°F for weather)." },
                    mode: {
                      type: "string",
                      enum: ["abs", "pct"],
                      description: "For shock: abs (default) adds value; pct changes by value %.",
                    },
                  },
                  required: ["kind", "value"],
                  additionalProperties: false,
                },
              },
            },
            additionalProperties: false,
          },
        },
      },
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "weather_history",
    description:
      "Past daily high/low temperatures (°F) for a place over the last N days (7-92), plus Open-Meteo's own next-7-day highs. Use for 'last 30 days', 'past few weeks', or as the input series before forecasting or simulating a city's weather. To forecast that city from its history, call simulate_world next with values_from='weather_history'. Values are already °F.",
    parameters: {
      type: "object",
      properties: {
        place: { type: "string", description: "City/place name, e.g. 'Louisville, KY'. Omit to use the visitor's location." },
        days: { type: "integer", description: "How many past days (7-92). Default 30." },
      },
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "web_search",
    description:
      "Search live sources (Wikipedia plus recent news from Wikipedia's Current events, each item linking its news outlet). Use it BEFORE answering any question about current events, recent news, people, places, organizations, dates, statistics, or any fact you are not certain of. Set recent:true for news or anything from the last few weeks. Returns titles, URLs, and snippets to cite.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Short search query (key names and terms, no filler)." },
        recent: { type: "boolean", description: "True for news / current events." },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "wiki_summary",
    description:
      "Plain-language Wikipedia summary of one topic (a person, place, thing, idea, or event), with its URL. Best first step for 'what is / who was / explain X'.",
    parameters: {
      type: "object",
      properties: {
        topic: { type: "string", description: "Article topic, e.g. 'Black hole' or 'Ada Lovelace'." },
      },
      required: ["topic"],
      additionalProperties: false,
    },
  },
  {
    type: "function" as const,
    name: "read_page",
    description:
      "Open one public web page (a URL from web_search results or one the user gave) and read its main text. Use when a snippet is not enough.",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "Full http(s) URL." },
      },
      required: ["url"],
      additionalProperties: false,
    },
  },
];

export function toolsForResponsesApi(): Array<Record<string, unknown>> {
  return TOOL_DEFINITIONS.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
  }));
}

export function toolsForChatCompletions(): Array<Record<string, unknown>> {
  return TOOL_DEFINITIONS.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }));
}

export function toolCatalogForPrompt(): string {
  return TOOL_DEFINITIONS.map(
    (tool) => `- ${tool.name}: ${tool.description}`,
  ).join("\n");
}

function stringArg(args: Record<string, unknown>, key: string, fallback = ""): string {
  const value = args[key];
  return typeof value === "string" ? value.trim() : fallback;
}

function intArg(args: Record<string, unknown>, key: string, fallback: number, min: number, max: number): number {
  const raw = args[key];
  const number = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(number)));
}

function horizonsArg(args: Record<string, unknown>): number[] {
  const raw = args.horizons;
  if (!Array.isArray(raw) || !raw.length) return [1, 3, 7];
  const values = raw
    .map((item) => (typeof item === "number" ? item : Number(item)))
    .filter((item) => Number.isFinite(item))
    .map((item) => Math.min(24, Math.max(1, Math.floor(item))));
  return values.length ? [...new Set(values)].slice(0, 5) : [1, 3, 7];
}

async function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error("tool timeout")), ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** Abort the work when the timer fires so hung container fetches cannot block retries. */
async function withAbortableTimeout<T>(
  run: (signal: AbortSignal) => Promise<T>,
  ms: number,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await Promise.race([
      run(controller.signal),
      new Promise<T>((_resolve, reject) => {
        const onAbort = () => reject(new Error("tool timeout"));
        if (controller.signal.aborted) {
          onAbort();
          return;
        }
        controller.signal.addEventListener("abort", onAbort, { once: true });
      }),
    ]);
  } finally {
    clearTimeout(timer);
    if (!controller.signal.aborted) controller.abort();
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function containerInit(
  ctx: ToolContext,
  init: { method?: string; body?: string; headers?: Record<string, string> } = {},
): { method?: string; body?: string; headers?: Record<string, string>; signal?: AbortSignal } {
  return ctx.signal ? { ...init, signal: ctx.signal } : init;
}

/** Container cold-start / timeout / 5xx errors that deserve retries with backoff. */
export function isContainerColdStartError(message: string): boolean {
  return isRetryableContainerError(message);
}

export function isRetryableContainerError(message: string): boolean {
  const text = message.toLowerCase();
  return (
    text.includes("container is not running") ||
    text.includes("container not running") ||
    text.includes("tool timeout") ||
    text.includes("health check failed") ||
    text.includes("aborted") ||
    text.includes("econnreset") ||
    text.includes("network connection lost") ||
    /\btimeout\b/.test(text) ||
    /\bfailed\s*\(5\d\d\)/.test(text) ||
    /\b(502|503|504)\b/.test(text)
  );
}

export function containerRetryDelayMs(attempt: number, baseMs: number): number {
  return Math.max(0, Math.round(baseMs * 2 ** Math.max(0, attempt)));
}

export function adaptiveToolTimeoutMs(baseTimeoutMs: number, attempt: number): number {
  // Mild growth only — large multipliers blew the whole /agent/chat budget on attempt 0.
  return Math.round(baseTimeoutMs * (1 + attempt * 0.25));
}

/**
 * Optional readiness helper for tests. Production wake uses startAndWaitForPorts
 * on every container fetch (same path as dashboard warmup).
 */
export async function ensureContainerReady(
  fetchContainer: ContainerFetcher,
  options?: { attempts?: number; waitMs?: number },
): Promise<void> {
  const attempts = options?.attempts ?? 2;
  const waitMs = options?.waitMs ?? 200;
  let lastError = "health check failed";
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const result = await withTimeout(
        fetchContainer("/health", { method: "GET" }),
        Math.max(1_000, waitMs * 4),
      );
      if (result.status > 0 && result.status < 400) return;
      lastError = `health check failed (${result.status})`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    if (attempt + 1 < attempts) await sleep(containerRetryDelayMs(attempt, waitMs));
  }
  throw new Error(`container is not running: ${lastError}`);
}

async function withContainerRetries<T>(
  toolName: AgentToolName,
  run: (signal: AbortSignal) => Promise<T>,
  timeoutMs: number,
  options?: { baseWaitMs?: number; attempts?: number; budgetMs?: number },
): Promise<T> {
  const attempts = options?.attempts ?? (CONTAINER_RETRY_TOOLS.has(toolName) ? CONTAINER_RETRY_ATTEMPTS : 1);
  const baseWaitMs = options?.baseWaitMs ?? CONTAINER_COLD_START_WAIT_MS;
  const budgetMs =
    options?.budgetMs ??
    (CONTAINER_RETRY_TOOLS.has(toolName) ? CONTAINER_TOOL_BUDGET_MS : timeoutMs);
  const deadline = Date.now() + budgetMs;
  let lastError: unknown = new Error("tool failed");
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const remaining = deadline - Date.now();
    if (remaining < 1_000) break;
    const attemptTimeout = Math.min(adaptiveToolTimeoutMs(timeoutMs, attempt), remaining);
    try {
      return await withAbortableTimeout(run, attemptTimeout);
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      const retryable = CONTAINER_RETRY_TOOLS.has(toolName) && isRetryableContainerError(message);
      if (!retryable || attempt + 1 >= attempts) break;
      const wait = Math.min(containerRetryDelayMs(attempt, baseWaitMs), Math.max(0, deadline - Date.now() - 500));
      if (wait > 0) await sleep(wait);
    }
  }
  throw lastError;
}

/** Weather starter stores °C; agent tool payloads always expose °F so the model cannot mislabel. */
const C_TO_F_SCALE = 9 / 5;

function celsiusToFahrenheit(celsius: number): number {
  return celsius * C_TO_F_SCALE + 32;
}

/** Absolute error / RMSE scale from °C to °F (no offset). */
function scaleCelsiusDeltaToF(delta: number): number {
  return delta * C_TO_F_SCALE;
}

function roundDisplay(value: number): number {
  return Math.round(value * 100) / 100;
}

function isWeatherSeries(id: unknown): boolean {
  return typeof id === "string" && id.toLowerCase() === "weather";
}

function convertTempFields(
  row: Record<string, unknown>,
  keys: string[],
  convert: (n: number) => number,
): Record<string, unknown> {
  const next = { ...row };
  for (const key of keys) {
    const value = next[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      next[key] = roundDisplay(convert(value));
    }
  }
  return next;
}

function convertForecastRows(rows: unknown): unknown {
  if (!Array.isArray(rows)) return rows;
  return rows.map((row) => {
    if (!row || typeof row !== "object") return row;
    return convertTempFields(row as Record<string, unknown>, ["value", "p10", "p50", "p90", "actual", "predicted"], celsiusToFahrenheit);
  });
}

function convertUncertainty(uncertainty: unknown): unknown {
  if (!uncertainty || typeof uncertainty !== "object") return uncertainty;
  const src = uncertainty as Record<string, unknown>;
  const next: Record<string, unknown> = { ...src };
  const interval = src.interval_80;
  if (interval && typeof interval === "object" && !Array.isArray(interval)) {
    const band = interval as Record<string, unknown>;
    next.interval_80 = {
      ...band,
      lower: convertNumberList(band.lower, celsiusToFahrenheit),
      upper: convertNumberList(band.upper, celsiusToFahrenheit),
    };
  }
  for (const key of ["p10", "p50", "p90", "low", "high"]) {
    const value = src[key];
    if (Array.isArray(value)) {
      next[key] = value.map((item) =>
        typeof item === "number" && Number.isFinite(item) ? roundDisplay(celsiusToFahrenheit(item)) : item,
      );
    } else if (typeof value === "number" && Number.isFinite(value)) {
      next[key] = roundDisplay(celsiusToFahrenheit(value));
    }
  }
  return next;
}

function convertWeatherForecastPayload(payload: Record<string, unknown>): Record<string, unknown> {
  // Idempotent: never °C→°F twice (that produced absurd “high-150s °F” answers).
  if (payload.storage_unit === "°C" && payload.unit === "°F") {
    return payload;
  }
  return {
    ...payload,
    unit: "°F",
    storage_unit: "°C",
    forecast: convertForecastRows(payload.forecast),
    history: convertForecastRows(payload.history),
    uncertainty: convertUncertainty(payload.uncertainty),
  };
}

function convertNumberList(values: unknown, convert: (n: number) => number): unknown {
  if (!Array.isArray(values)) return values;
  return values.map((item) =>
    typeof item === "number" && Number.isFinite(item) ? roundDisplay(convert(item)) : item,
  );
}

/** Weather what-if payloads from the container are °C; expose °F like forecast. */
function convertWeatherWhatIfPayload(payload: Record<string, unknown>): Record<string, unknown> {
  if (payload.storage_unit === "°C" && payload.unit === "°F") {
    return payload;
  }
  const shockRecord =
    payload.shock && typeof payload.shock === "object"
      ? ({ ...(payload.shock as Record<string, unknown>) } as Record<string, unknown>)
      : null;
  if (shockRecord) {
    for (const key of ["before", "after"] as const) {
      const value = shockRecord[key];
      if (typeof value === "number" && Number.isFinite(value)) {
        shockRecord[key] = roundDisplay(celsiusToFahrenheit(value));
      }
    }
    if (shockRecord.mode === "abs" && typeof shockRecord.value === "number") {
      shockRecord.value = roundDisplay(scaleCelsiusDeltaToF(Number(shockRecord.value)));
    }
  }
  const convertPathBlock = (block: unknown): unknown => {
    if (!block || typeof block !== "object") return block;
    const src = block as Record<string, unknown>;
    return {
      ...src,
      p50: convertNumberList(src.p50, celsiusToFahrenheit),
      forecast: convertNumberList(src.forecast, celsiusToFahrenheit),
    };
  };
  const deltaSrc =
    payload.delta && typeof payload.delta === "object"
      ? (payload.delta as Record<string, unknown>)
      : null;
  const meanDelta =
    deltaSrc && typeof deltaSrc.mean === "number" && Number.isFinite(deltaSrc.mean)
      ? roundDisplay(scaleCelsiusDeltaToF(deltaSrc.mean))
      : deltaSrc?.mean;
  const delta = deltaSrc
    ? {
        ...deltaSrc,
        p50: convertNumberList(deltaSrc.p50, scaleCelsiusDeltaToF),
        mean: meanDelta,
        text:
          typeof meanDelta === "number"
            ? rewriteWhatIfDeltaText(String(deltaSrc.text ?? payload.summary ?? ""), meanDelta)
            : deltaSrc.text,
      }
    : payload.delta;
  const summary =
    typeof meanDelta === "number"
      ? rewriteWhatIfDeltaText(String(payload.summary ?? ""), meanDelta)
      : payload.summary;
  return {
    ...payload,
    unit: "°F",
    storage_unit: "°C",
    shock: shockRecord ?? payload.shock,
    baseline: convertPathBlock(payload.baseline),
    shocked: convertPathBlock(payload.shocked),
    delta,
    summary,
    text: typeof summary === "string" ? summary : payload.text,
  };
}

function rewriteWhatIfDeltaText(text: string, meanDeltaF: number): string {
  const abs = Math.abs(meanDeltaF);
  const approx = abs < 1e-9 ? "0" : abs.toPrecision(3);
  if (/leaves the .+ forecast almost unchanged/i.test(text)) {
    return text.replace(/\s*$/, "").replace(/\.$/, "") + " (°F).";
  }
  if (/lifts the /i.test(text)) {
    return text.replace(
      /lifts the (.+?) by about [-+0-9.eE]+(?:\s*\S+)? on average\.?/i,
      `lifts the $1 by about ${approx}°F on average.`,
    );
  }
  if (/lowers the /i.test(text)) {
    return text.replace(
      /lowers the (.+?) by about [-+0-9.eE]+(?:\s*\S+)? on average\.?/i,
      `lowers the $1 by about ${approx}°F on average.`,
    );
  }
  return `${text} (values in °F)`.trim();
}

function convertMetricBag(block: unknown): unknown {
  if (!block || typeof block !== "object") return block;
  const src = block as Record<string, unknown>;
  const next: Record<string, unknown> = { ...src };
  for (const key of ["mae", "rmse"]) {
    const value = next[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      next[key] = roundDisplay(scaleCelsiusDeltaToF(value));
    }
  }
  if (typeof next.mse === "number" && Number.isFinite(next.mse)) {
    next.mse = roundDisplay(next.mse * C_TO_F_SCALE * C_TO_F_SCALE);
  }
  return next;
}

function convertWeatherMetricsPayload(report: Record<string, unknown>): Record<string, unknown> {
  const baselines =
    report.baselines && typeof report.baselines === "object"
      ? Object.fromEntries(
          Object.entries(report.baselines as Record<string, unknown>).map(([name, value]) => [
            name,
            convertMetricBag(value),
          ]),
        )
      : report.baselines;
  return {
    ...report,
    unit: "°F",
    storage_unit: "°C",
    metrics: convertMetricBag(report.metrics),
    baselines,
  };
}

function compactForecast(
  payload: Record<string, unknown>,
  seriesHint?: string,
): Record<string, unknown> {
  const forecast = Array.isArray(payload.forecast) ? payload.forecast.slice(0, 8) : payload.forecast;
  const uncertainty = payload.uncertainty;
  const history = Array.isArray(payload.history) ? payload.history.slice(-4) : undefined;
  const compact: Record<string, unknown> = {
    starter_id: payload.starter_id,
    name: payload.name,
    unit: payload.unit,
    summary: payload.summary,
    horizon: payload.horizon,
    forecast,
    uncertainty,
    history,
    checkpoint_loaded: payload.checkpoint_loaded,
    why: payload.why,
  };
  if (isWeatherSeries(payload.starter_id) || isWeatherSeries(seriesHint)) {
    return convertWeatherForecastPayload(compact);
  }
  return compact;
}

function metricsSummary(report: Record<string, unknown>, modelId?: string): Record<string, unknown> {
  const metrics = report.metrics && typeof report.metrics === "object" ? report.metrics : null;
  const summary: Record<string, unknown> = {
    horizon: report.horizon,
    n: report.n,
    metrics,
    baselines: report.baselines,
    best_baseline: report.best_baseline,
    comparison: report.comparison,
    improvement_pct: report.improvement_pct,
  };
  const weather =
    isWeatherSeries(modelId) ||
    isWeatherSeries(report.model_id) ||
    isWeatherSeries(report.starter_id);
  if (weather) {
    return convertWeatherMetricsPayload(summary);
  }
  return summary;
}

function throwIfContainerStatusError(status: number, label: string): void {
  if (status >= 500) {
    throw new Error(`${label} failed (${status}).`);
  }
  if (status >= 400) {
    throw new Error(`${label} unavailable (${status}).`);
  }
}

async function fetchOptionalAccuracy(
  ctx: ToolContext,
  series: string,
  horizon: number,
): Promise<unknown> {
  // Prefer D1 / Cache API — never spend the forecast budget on a backtest recompute.
  const cached = await cacheGetOrLoad({
    key: evalCacheKey(series, horizon),
    ttlSeconds: EVAL_CACHE_TTL_SECONDS,
    waitUntil: ctx.waitUntil,
    refreshOnHit: false,
    load: async () => {
      const stored = await readStoredBacktest(ctx.db, series);
      if (stored) return metricsSummary(stored, series);
      const summary = await withTimeout(
        ctx.fetchContainer(
          `/evaluation/${encodeURIComponent(series)}`,
          containerInit(ctx, { method: "GET", headers: ctx.authHeaders }),
        ),
        ACCURACY_FETCH_TIMEOUT_MS,
      );
      const report =
        summary.status < 400 &&
        summary.json &&
        typeof summary.json === "object" &&
        (summary.json as { backtest?: unknown }).backtest &&
        typeof (summary.json as { backtest?: unknown }).backtest === "object"
          ? ((summary.json as { backtest: Record<string, unknown> }).backtest)
          : null;
      if (report) return metricsSummary(report, series);
      throw new Error("accuracy unavailable");
    },
  }).catch(() => null);
  return cached?.value ?? null;
}

async function loadStarterForecast(
  ctx: ToolContext,
  series: string,
  horizon: number,
): Promise<Record<string, unknown>> {
  // lite=1 skips the CPU-heavy history replay; same deterministic point forecast.
  const path = `/starters/${encodeURIComponent(series)}/forecast?points=120&lite=1`;
  const result = await ctx.fetchContainer(
    path,
    containerInit(ctx, { method: "GET", headers: ctx.authHeaders }),
  );
  throwIfContainerStatusError(result.status, "Starter forecast");
  const payload = compactForecast(result.json as Record<string, unknown>, series);
  return { series, horizon, ...payload };
}

async function runForecast(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const series = stringArg(args, "series", "weather").toLowerCase() || "weather";
  const horizon = intArg(args, "horizon", 1, 1, 24);

  if (series === "loaded") {
    const recent = await ctx.fetchContainer(
      `/series/recent?points=96`,
      containerInit(ctx, { method: "GET", headers: ctx.authHeaders }),
    );
    throwIfContainerStatusError(recent.status, "Loaded series forecast");
    return { series: "loaded", horizon, ...compactForecast(recent.json as Record<string, unknown>) };
  }
  if (STARTER_IDS.has(series)) {
    const { value, cacheHit } = await cacheGetOrLoad({
      key: forecastCacheKey(series, horizon),
      ttlSeconds: FORECAST_CACHE_TTL_SECONDS,
      waitUntil: ctx.waitUntil,
      refreshOnHit: true,
      load: () => loadStarterForecast(ctx, series, horizon),
    });
    // Accuracy is optional and must not block a cache hit.
    const accuracy = cacheHit
      ? await readStoredBacktest(ctx.db, series).then((row) =>
          row ? metricsSummary(row, series) : null,
        )
      : await fetchOptionalAccuracy(ctx, series, horizon);
    return { ...value, recent_accuracy: accuracy, cache_hit: cacheHit };
  }
  const predict = await ctx.fetchContainer(
    `/workspace/models/${encodeURIComponent(series)}/predict-latest`,
    containerInit(ctx, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: "{}",
    }),
  );
  if (predict.status >= 500) {
    throw new Error(`Workspace forecast failed (${predict.status}).`);
  }
  if (predict.status >= 400) {
    throw new Error("Workspace forecast unavailable. Sign in or use a starter series id.");
  }
  let accuracy: unknown = null;
  try {
    const metrics = await withTimeout(
      ctx.fetchContainer(
        `/workspace/models/${encodeURIComponent(series)}/metrics`,
        containerInit(ctx, { method: "GET", headers: ctx.authHeaders }),
      ),
      ACCURACY_FETCH_TIMEOUT_MS,
    );
    if (metrics.status < 400) accuracy = metrics.json;
  } catch {
    accuracy = null;
  }
  return { series, horizon, prediction: predict.json, recent_accuracy: accuracy };
}

async function runLiveData(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const requested = stringArg(args, "kind", "weather").toLowerCase();
  const query = stringArg(args, "query", ctx.question) || ctx.question;
  // Reclassify: models often pass kind=weather for "bitcoin at right now".
  const kind = resolveFetchLiveKind(requested, query, ctx.question);
  switch (kind) {
    case "weather": {
      // Evaluate wantsCurrentConditions on the user's original question.
      // Do not throw on classifier edge cases — answerWeather handles place/time.
      const weatherAsk = weatherQuestionForTool(
        isWeatherAsk(ctx.question) ? ctx.question : ctx.question || query,
        query,
      );
      const result = await answerWeather(weatherAsk || ctx.question || query, ctx.visitor);
      return {
        kind,
        text: result.text,
        query_used: weatherAsk || ctx.question || query,
        original_question: ctx.question,
        wants_current: wantsCurrentConditions(ctx.question),
        requested_kind: requested,
      };
    }
    case "stock": {
      const result = await answerQuote(query || ctx.question, "stock");
      return { kind, text: result.text, requested_kind: requested };
    }
    case "crypto": {
      const result = await answerQuote(query || ctx.question, "crypto");
      return { kind, text: result.text, requested_kind: requested };
    }
    case "moon": {
      const result = await ctx.fetchContainer("/lunar", { method: "GET", headers: ctx.authHeaders });
      if (result.status >= 400) throw new Error("Lunar feed unavailable.");
      return { kind, data: result.json };
    }
    case "space": {
      const result = await ctx.fetchContainer("/space/near-earth?days=90", {
        method: "GET",
        headers: ctx.authHeaders,
      });
      if (result.status >= 400) throw new Error("Near-Earth feed unavailable.");
      const payload = result.json as { highlights?: unknown; summary?: unknown };
      return { kind, highlights: payload.highlights, summary: payload.summary, data: result.json };
    }
    case "solar": {
      const result = await ctx.fetchContainer("/space/solar-system?days=365", {
        method: "GET",
        headers: ctx.authHeaders,
      });
      if (result.status >= 400) throw new Error("Solar-system feed unavailable.");
      return { kind, data: result.json };
    }
    default:
      throw new Error("Unknown live data kind. Use weather, stock, crypto, moon, space, or solar.");
  }
}

async function loadStarterEvaluation(
  ctx: ToolContext,
  modelId: string,
  horizon: number,
): Promise<Record<string, unknown>> {
  // Prefer already-stored metrics (D1 / GET summary) before any recompute.
  const stored = await readStoredBacktest(ctx.db, modelId);
  if (stored) return { model_id: modelId, ...metricsSummary(stored, modelId) };

  const summary = await ctx.fetchContainer(
    `/evaluation/${encodeURIComponent(modelId)}`,
    containerInit(ctx, { method: "GET", headers: ctx.authHeaders }),
  );
  if (
    summary.status < 400 &&
    summary.json &&
    typeof summary.json === "object" &&
    (summary.json as { backtest?: unknown }).backtest &&
    typeof (summary.json as { backtest?: unknown }).backtest === "object"
  ) {
    return {
      model_id: modelId,
      ...metricsSummary(
        (summary.json as { backtest: Record<string, unknown> }).backtest,
        modelId,
      ),
    };
  }

  // Cold path only: run once and let the container persist to D1.
  const result = await ctx.fetchContainer(
    `/evaluation/${encodeURIComponent(modelId)}/backtest`,
    containerInit(ctx, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({ horizon }),
    }),
  );
  if (result.status >= 500) throw new Error(`Backtest failed (${result.status}).`);
  if (result.status >= 400) throw new Error("Backtest unavailable for that model.");
  return {
    model_id: modelId,
    ...metricsSummary(result.json as Record<string, unknown>, modelId),
  };
}

async function runEvaluation(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const modelId = stringArg(args, "model_id", "weather") || "weather";
  const horizon = intArg(args, "horizon", 1, 1, 24);
  if (STARTER_IDS.has(modelId)) {
    const { value, cacheHit } = await cacheGetOrLoad({
      key: evalCacheKey(modelId, horizon),
      ttlSeconds: EVAL_CACHE_TTL_SECONDS,
      waitUntil: ctx.waitUntil,
      refreshOnHit: false,
      load: () => loadStarterEvaluation(ctx, modelId, horizon),
    });
    return { ...value, cache_hit: cacheHit };
  }
  const metrics = await ctx.fetchContainer(`/workspace/models/${encodeURIComponent(modelId)}/metrics`, {
    method: "GET",
    headers: ctx.authHeaders,
  });
  if (metrics.status >= 400) throw new Error("Workspace metrics unavailable. Sign in or use a starter id.");
  return { model_id: modelId, ...(metrics.json as object) };
}

async function runCompareHorizons(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const modelId = stringArg(args, "model_id", "weather") || "weather";
  const horizons = horizonsArg(args);
  if (!STARTER_IDS.has(modelId)) {
    throw new Error("Horizon comparison is available for shipped sample models (weather, stocks, crypto, near-earth, lunar).");
  }
  const rows: Array<Record<string, unknown>> = [];
  for (const horizon of horizons) {
    // Prefer stored reports; horizon-specific refresh only when missing.
    const result = await ctx.fetchContainer(`/evaluation/${encodeURIComponent(modelId)}/backtest`, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({ horizon, refresh: false }),
    });
    if (result.status >= 400) {
      rows.push({ horizon, error: `unavailable (${result.status})` });
      continue;
    }
    rows.push(metricsSummary(result.json as Record<string, unknown>, modelId));
  }
  return { model_id: modelId, horizons: rows };
}

type ExplainStep = { step: number; p50: number; p10: number | null; p90: number | null };

function fmtNum(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/** Pull the week-ahead path (already in display units) from a forecast tool payload. */
export function explainSteps(forecast: unknown): ExplainStep[] {
  if (!forecast || typeof forecast !== "object") return [];
  const rows = (forecast as { forecast?: unknown }).forecast;
  if (!Array.isArray(rows)) return [];
  const steps: ExplainStep[] = [];
  rows.forEach((row, index) => {
    if (!row || typeof row !== "object") return;
    const record = row as Record<string, unknown>;
    const point = Number(record.p50 ?? record.value);
    if (!Number.isFinite(point)) return;
    const p10 = Number(record.p10);
    const p90 = Number(record.p90);
    steps.push({
      step: Number.isFinite(Number(record.step)) ? Number(record.step) : index + 1,
      p50: roundDisplay(point),
      p10: Number.isFinite(p10) ? roundDisplay(p10) : null,
      p90: Number.isFinite(p90) ? roundDisplay(p90) : null,
    });
  });
  return steps;
}

/** One deterministic, user-ready paragraph for explain_prediction. */
export function explainText(
  label: string,
  explanation: string,
  steps: ExplainStep[],
  unit: string,
  lastObserved: number | null,
): string {
  const parts: string[] = [];
  if (steps.length) {
    const values = steps.map((row) => row.p50);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const span = steps.length > 1 ? `the next ${steps.length} steps` : "the next step";
    const range = lo === hi ? `${fmtNum(lo)}${unit}` : `${fmtNum(lo)}–${fmtNum(hi)}${unit}`;
    parts.push(`${label} forecast for ${span}: ${range}.`);
    const lows = steps.map((row) => row.p10).filter((v): v is number => v != null);
    const highs = steps.map((row) => row.p90).filter((v): v is number => v != null);
    if (lows.length && highs.length) {
      parts.push(`The 80% band runs about ${fmtNum(Math.min(...lows))}–${fmtNum(Math.max(...highs))}${unit}.`);
    }
  }
  if (lastObserved != null) {
    parts.push(`The latest observed value was ${fmtNum(roundDisplay(lastObserved))}${unit}.`);
  }
  if (explanation.trim()) parts.push(`Why: ${explanation.trim()}`);
  return parts.join(" ");
}

async function runExplain(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const series = stringArg(args, "series", "weather") || "weather";
  // Use the full starter horizon so week-ahead paths are available; values are °F for weather.
  const forecast = await runForecast(ctx, {
    series,
    horizon: isWeatherSeries(series) ? 7 : 1,
  });
  const record = forecast && typeof forecast === "object" ? (forecast as Record<string, unknown>) : {};
  const why = record.why;
  let whyPayload: unknown = why;
  let explanation =
    why && typeof why === "object" && typeof (why as { explanation?: unknown }).explanation === "string"
      ? String((why as { explanation: string }).explanation)
      : "";
  const weather = isWeatherSeries(series);
  const alreadyF = record.unit === "°F" && record.storage_unit === "°C";
  const historyRows = Array.isArray(record.history) ? (record.history as Array<Record<string, unknown>>) : [];
  const historyDisplay = historyRows
    .map((row) => Number(row.actual ?? row.value))
    .filter((value) => Number.isFinite(value));
  const steps = explainSteps(forecast);
  if (!explanation) {
    // Degrade gracefully: rebuild from history when the container omitted why.
    // Tool output is °F for weather — convert back to °C for the why API so
    // trend math matches storage scale. Only the text comes back.
    const toStorage = (value: number) => (alreadyF ? (value - 32) * (5 / 9) : value);
    const point = steps[0]?.p50;
    const rebuilt = await ctx.fetchContainer("/analysis/why", {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({
        history: historyDisplay.map(toStorage),
        forecast: typeof point === "number" ? toStorage(point) : null,
        unit: alreadyF ? "°C" : (typeof record.unit === "string" ? record.unit : null),
      }),
    });
    if (rebuilt.status < 400 && rebuilt.json && typeof rebuilt.json === "object") {
      const payload = rebuilt.json as { explanation?: string };
      whyPayload = payload;
      explanation = typeof payload.explanation === "string" ? payload.explanation : "";
    }
  }
  const unit = typeof record.unit === "string" ? record.unit : "";
  const label = typeof record.name === "string" && record.name ? String(record.name) : series;
  const lastObserved = historyDisplay.length ? historyDisplay[historyDisplay.length - 1] : null;
  const text = explainText(label, explanation, steps, unit, lastObserved);
  const drivers =
    whyPayload && typeof whyPayload === "object" && Array.isArray((whyPayload as { drivers?: unknown }).drivers)
      ? (whyPayload as { drivers: unknown[] }).drivers
      : [];
  const weatherNote =
    "All numbers here are already in °F. Quote them exactly as °F; never convert again. Week-ahead values must stay inside the 80% band.";
  const baseNote =
    "Lead with the why explanation; then mention the point, interval, and accuracy using only these facts. Keep it to 2–4 short sentences.";
  const accuracy =
    record.recent_accuracy && typeof record.recent_accuracy === "object"
      ? (record.recent_accuracy as Record<string, unknown>)
      : null;
  // Compact payload so the tool note never gets cut mid-JSON.
  return {
    series,
    unit,
    storage_unit: record.storage_unit,
    explanation,
    drivers,
    text,
    week_ahead: steps,
    last_observed: lastObserved != null ? roundDisplay(lastObserved) : null,
    recent_accuracy: accuracy
      ? {
          comparison: accuracy.comparison,
          metrics: accuracy.metrics,
          unit: accuracy.unit,
          improvement_pct: accuracy.improvement_pct,
        }
      : null,
    explanation_facts: {
      unit,
      storage_unit: record.storage_unit,
      forecast: Array.isArray(record.forecast) ? record.forecast : [],
    },
    note: weather ? `${baseNote} ${weatherNote}` : baseNote,
  };
}

async function runListSeries(ctx: ToolContext): Promise<ToolResult["data"]> {
  const [datasets, models, predictions, starters] = await Promise.all([
    ctx.fetchContainer("/workspace/datasets", { method: "GET", headers: ctx.authHeaders }),
    ctx.fetchContainer("/workspace/models", { method: "GET", headers: ctx.authHeaders }),
    ctx.fetchContainer("/workspace/predictions", { method: "GET", headers: ctx.authHeaders }),
    ctx.fetchContainer("/workspace/starters", { method: "GET", headers: ctx.authHeaders }),
  ]);
  if ([datasets, models, predictions, starters].some((item) => item.status === 401 || item.status === 403)) {
    return {
      signed_in: false,
      message: "Sign in to list your private datasets, models, and Train-on-my-CSV uploads.",
    };
  }
  return {
    signed_in: true,
    datasets: (datasets.json as { datasets?: unknown })?.datasets ?? datasets.json,
    models: (models.json as { models?: unknown })?.models ?? models.json,
    predictions: (predictions.json as { predictions?: unknown })?.predictions ?? predictions.json,
    starters: (starters.json as { starters?: unknown })?.starters ?? starters.json,
  };
}

async function runRenameUserSeries(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const datasetId = stringArg(args, "dataset_id");
  const name = stringArg(args, "name");
  if (!datasetId || !name) {
    return { ok: false, text: "dataset_id and name are required." };
  }
  const result = await ctx.fetchContainer(`/workspace/datasets/${encodeURIComponent(datasetId)}`, {
    method: "PATCH",
    headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (result.status === 401 || result.status === 403) {
    return { signed_in: false, text: "Sign in to rename your uploaded series." };
  }
  if (result.status >= 400) {
    throw new Error(`Rename failed (${result.status}).`);
  }
  return { ok: true, dataset: result.json, text: `Renamed series to ${name}.` };
}

async function runDeleteUserSeries(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const datasetId = stringArg(args, "dataset_id");
  if (!datasetId) return { ok: false, text: "dataset_id is required." };
  const result = await ctx.fetchContainer(`/workspace/datasets/${encodeURIComponent(datasetId)}`, {
    method: "DELETE",
    headers: ctx.authHeaders,
  });
  if (result.status === 401 || result.status === 403) {
    return { signed_in: false, text: "Sign in to delete your uploaded series." };
  }
  if (result.status >= 400) {
    throw new Error(`Delete failed (${result.status}).`);
  }
  return { ok: true, deleted: datasetId, text: `Deleted series ${datasetId}.` };
}

async function runRetrainUserSeries(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const datasetId = stringArg(args, "dataset_id");
  if (!datasetId) return { ok: false, text: "dataset_id is required." };
  const target = stringArg(args, "target_column");
  const body = target ? { target_column: target } : {};
  const result = await ctx.fetchContainer(
    `/workspace/datasets/${encodeURIComponent(datasetId)}/train`,
    {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (result.status === 401 || result.status === 403) {
    return { signed_in: false, text: "Sign in to retrain your uploaded series." };
  }
  if (result.status >= 400) {
    throw new Error(`Retrain failed (${result.status}).`);
  }
  return { ok: true, job: result.json, text: `Queued retrain for ${datasetId}.` };
}

function requireMemoryDb(ctx: ToolContext): { db: D1Like; scope: string } {
  if (!ctx.db) throw new Error("Memory store unavailable.");
  const scope = typeof ctx.memoryScope === "string" ? ctx.memoryScope.trim() : "";
  if (!scope) throw new Error("Memory scope unavailable.");
  return { db: ctx.db, scope };
}

async function runRememberFact(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  if (ctx.memoryOff) return { ok: false, error: "Memory is off.", text: "Memory is turned off for this account." };
  const { db, scope } = requireMemoryDb(ctx);
  const key = stringArg(args, "key");
  const value = stringArg(args, "value");
  const result = await upsertMemory(db, scope, key, value);
  if (!result.ok) {
    return { ok: false, error: result.error, text: result.error };
  }
  return {
    ok: true,
    fact: result.fact,
    text: `Remembered ${result.fact.key}: ${result.fact.value}.`,
  };
}

async function runRecallMemory(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  if (ctx.memoryOff) return { ok: true, facts: [], text: "Memory is turned off for this account." };
  const { db, scope } = requireMemoryDb(ctx);
  const facts = await listMemory(db, scope);
  const key = stringArg(args, "key");
  if (key) {
    const normalized = key.trim().toLowerCase().replace(/[\s-]+/g, "_");
    const match = facts.find((fact) => fact.key === normalized);
    if (!match) {
      return { ok: true, facts: [], text: `No memory stored for ${normalized}.` };
    }
    return {
      ok: true,
      facts: [match],
      text: `${match.key}: ${match.value}`,
    };
  }
  if (!facts.length) {
    return { ok: true, facts: [], text: "No stored preferences yet." };
  }
  return {
    ok: true,
    facts,
    text: facts.map((fact) => `${fact.key}: ${fact.value}`).join("; "),
  };
}

async function runForgetMemory(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const { db, scope } = requireMemoryDb(ctx);
  const clearAll = args.clear_all === true;
  if (clearAll) {
    const deleted = await clearMemory(db, scope);
    return {
      ok: true,
      cleared: deleted,
      text: deleted ? `Cleared ${deleted} memor${deleted === 1 ? "y" : "ies"}.` : "Nothing to clear.",
    };
  }
  const key = stringArg(args, "key");
  if (!key) {
    return { ok: false, error: "Provide key or clear_all.", text: "Provide a key to forget, or clear_all." };
  }
  const removed = await deleteMemoryKey(db, scope, key);
  return {
    ok: removed,
    key,
    text: removed ? `Forgot ${key}.` : `No memory stored for ${key}.`,
  };
}

async function runSaveScenario(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const { db, scope } = requireMemoryDb(ctx);
  const name = stringArg(args, "name");
  const question = stringArg(args, "question") || ctx.question;
  const series = stringArg(args, "series") || null;
  const horizonRaw = args.horizon;
  const horizon =
    typeof horizonRaw === "number" && Number.isFinite(horizonRaw)
      ? Math.min(24, Math.max(1, Math.floor(horizonRaw)))
      : null;
  const result = await createScenario(db, scope, { name, question, series, horizon });
  if (!result.ok) return { ok: false, error: result.error, text: result.error };
  return {
    ok: true,
    scenario: result.scenario,
    text: `Saved scenario “${result.scenario.name}”.`,
  };
}

async function runListScenarios(ctx: ToolContext): Promise<ToolResult["data"]> {
  const { db, scope } = requireMemoryDb(ctx);
  const scenarios = await listScenarios(db, scope);
  if (!scenarios.length) {
    return { ok: true, scenarios: [], text: "No saved scenarios yet." };
  }
  return {
    ok: true,
    scenarios,
    text: scenarios.map((row) => `${row.name} (${row.id})`).join("; "),
  };
}

async function runRunScenario(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const { db, scope } = requireMemoryDb(ctx);
  const id = stringArg(args, "id");
  const name = stringArg(args, "name").toLowerCase();
  let scenario = id ? await getScenarioOwned(db, scope, id) : null;
  if (!scenario && name) {
    const all = await listScenarios(db, scope);
    scenario = all.find((row) => row.name.toLowerCase() === name) || null;
  }
  if (!scenario) {
    return { ok: false, error: "Scenario not found.", text: "Scenario not found." };
  }
  const question = scenarioRerunQuestion(scenario);
  if (scenario.series) {
    const forecast = await runForecast(ctx, {
      series: scenario.series,
      horizon: scenario.horizon || 1,
    });
    return {
      ok: true,
      scenario,
      question,
      forecast,
      text: `Re-ran “${scenario.name}” for ${scenario.series}.`,
    };
  }
  return {
    ok: true,
    scenario,
    question,
    text: `Loaded scenario “${scenario.name}”: ${question}`,
  };
}

async function runWhatIf(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  let series = stringArg(args, "series").toLowerCase();
  let horizon = intArg(args, "horizon", 1, 1, 24);
  const scenarioId = stringArg(args, "scenario_id");
  if (scenarioId) {
    const { db, scope } = requireMemoryDb(ctx);
    const scenario = await getScenarioOwned(db, scope, scenarioId);
    if (!scenario) {
      return { ok: false, text: "Scenario not found." };
    }
    if (!series && scenario.series) series = String(scenario.series).toLowerCase();
    if (scenario.horizon && (!args.horizon || typeof args.horizon !== "number")) {
      horizon = Math.min(24, Math.max(1, Math.floor(scenario.horizon)));
    }
  }
  if (!series) series = "weather";
  const modeRaw = stringArg(args, "mode", "pct").toLowerCase() || "pct";
  const mode = modeRaw === "abs" ? "abs" : "pct";
  const valueRaw = args.value;
  const value =
    typeof valueRaw === "number" && Number.isFinite(valueRaw) ? valueRaw : -10;
  const body = { mode, value, horizon, feature_index: 0 };
  if (STARTER_IDS.has(series)) {
    const result = await ctx.fetchContainer(
      "/analysis/what-if",
      containerInit(ctx, {
        method: "POST",
        headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ starter_id: series, ...body }),
      }),
    );
    throwIfContainerStatusError(result.status, "What-if");
    let payload = result.json as Record<string, unknown>;
    if (isWeatherSeries(series) || isWeatherSeries(payload.starter_id) || isWeatherSeries(payload.series)) {
      payload = convertWeatherWhatIfPayload(payload);
    }
    return {
      ok: true,
      ...payload,
      text:
        typeof payload.summary === "string"
          ? payload.summary
          : typeof payload.text === "string"
            ? payload.text
            : `What-if on ${series} complete.`,
    };
  }
  const result = await ctx.fetchContainer(
    `/workspace/models/${encodeURIComponent(series)}/what-if`,
    containerInit(ctx, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
  if (result.status === 401 || result.status === 403) {
    return { signed_in: false, text: "Sign in to run what-if on a private model." };
  }
  throwIfContainerStatusError(result.status, "What-if");
  const payload = result.json as Record<string, unknown>;
  return {
    ok: true,
    ...payload,
    text:
      typeof payload.summary === "string"
        ? payload.summary
        : `What-if on ${series} complete.`,
  };
}

const SIM_EVENT_KINDS = new Set(["shock", "drift", "volatility", "set"]);
const SIM_ENCODERS = new Set(["pca", "jepa", "auto"]);

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Clamp model-supplied scenarios to what /analysis/simulate accepts. */
export function sanitizeSimScenarios(raw: unknown, horizon: number): Array<Record<string, unknown>> {
  if (!Array.isArray(raw)) return [];
  return raw.slice(0, 3).map((item, index) => {
    const rec = item && typeof item === "object" ? (item as Record<string, unknown>) : {};
    const name =
      typeof rec.name === "string" && rec.name.trim()
        ? rec.name.trim().slice(0, 40)
        : `Scenario ${String.fromCharCode(65 + index)}`;
    const events = (Array.isArray(rec.events) ? rec.events : [])
      .slice(0, 4)
      .map((ev) => (ev && typeof ev === "object" ? (ev as Record<string, unknown>) : {}))
      .filter((ev) => typeof ev.kind === "string" && SIM_EVENT_KINDS.has(ev.kind))
      .map((ev) => {
        const stepRaw = finiteNumber(ev.step);
        const step = Math.min(horizon, Math.max(1, Math.floor(stepRaw ?? 1)));
        let value = finiteNumber(ev.value) ?? 0;
        if (ev.kind === "volatility") value = Math.min(10, Math.max(0, value));
        return { kind: ev.kind, step, value, mode: ev.mode === "pct" ? "pct" : "abs" };
      });
    return { name, events };
  });
}

async function runSimulateWorld(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const horizon = intArg(args, "horizon", 14, 1, 60);
  const encoderArg = stringArg(args, "encoder", "pca").toLowerCase();
  const encoder = SIM_ENCODERS.has(encoderArg) ? encoderArg : "pca";
  // Models sometimes send one `scenario` object instead of a `scenarios` list.
  const scenarioList =
    Array.isArray(args.scenarios) || !args.scenario || typeof args.scenario !== "object"
      ? args.scenarios
      : [args.scenario];
  const scenarios = sanitizeSimScenarios(scenarioList, horizon);
  const rawValues = Array.isArray(args.values)
    ? args.values.filter((v): v is number => typeof v === "number" && Number.isFinite(v))
    : [];
  let body: Record<string, unknown>;
  let series = stringArg(args, "series", "weather").toLowerCase() || "weather";
  const rawDates = Array.isArray(args.dates)
    ? args.dates.filter((d): d is string => typeof d === "string").slice(-2000)
    : [];
  const fromHistory = args._from_history === true;
  if (rawValues.length >= 24) {
    series = "custom";
    const unit = stringArg(args, "unit").slice(0, 32);
    const values = rawValues.slice(-2000);
    body = {
      values,
      ...(unit ? { unit } : {}),
      ...(rawDates.length === values.length ? { dates: rawDates } : {}),
      horizon,
      n_paths: 128,
      scenarios,
      ...(encoder !== "pca" ? { encoder } : {}),
    };
  } else {
    if (!STARTER_IDS.has(series)) {
      return {
        ok: false,
        text: "The simulator runs on the sample series (weather, stocks, crypto, lunar, near-earth) or on 24+ numbers you paste or attach.",
      };
    }
    body = {
      starter_id: series,
      horizon,
      n_paths: 128,
      scenarios,
      ...(isWeatherSeries(series) ? { display_unit: "°F" } : {}),
      ...(encoder !== "pca" ? { encoder } : {}),
    };
  }
  const result = await ctx.fetchContainer(
    "/analysis/simulate",
    containerInit(ctx, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
  throwIfContainerStatusError(result.status, "Simulator");
  const payload = (result.json ?? {}) as Record<string, unknown>;
  const baseline = (payload.baseline ?? {}) as Record<string, unknown>;
  const steps = Number(payload.horizon) || horizon;
  // Loop tool results are capped at 3500 chars: lead with the summary and keep
  // only checkpoint steps (first, middle, last) instead of full paths.
  const checkpoints = [...new Set([1, Math.ceil(steps / 2), steps])];
  const pick = (list: unknown): Record<string, unknown> => {
    const out: Record<string, unknown> = {};
    if (!Array.isArray(list)) return out;
    for (const step of checkpoints) {
      if (step - 1 < list.length) out[`step_${step}`] = list[step - 1];
    }
    return out;
  };
  const compactScenarios = Array.isArray(payload.scenarios)
    ? (payload.scenarios as Array<Record<string, unknown>>).map((sc) => ({
        summary: sc.summary,
        name: sc.name,
        p50: pick(sc.p50),
        p10: pick(sc.p10),
        p90: pick(sc.p90),
        end_delta: sc.end_delta,
        end_regime: sc.end_regime,
      }))
    : [];
  const model = (payload.model ?? {}) as Record<string, unknown>;
  const summary = typeof payload.summary === "string" ? payload.summary : "";
  return {
    ok: true,
    text: summary || `Simulated ${steps} steps on ${series}.`,
    series: payload.series ?? series,
    name: payload.name,
    unit: payload.unit,
    horizon: steps,
    baseline: { p50: pick(baseline.p50), p10: pick(baseline.p10), p90: pick(baseline.p90) },
    scenarios: compactScenarios,
    backtest: payload.backtest,
    model: model.description,
    ...(payload.encoder_comparison ? { encoder_comparison: payload.encoder_comparison } : {}),
    ...(fromHistory ? { from_history: true } : {}),
    // Full paths for charts / CSV; stripped before the model sees the result.
    _full: {
      history: Array.isArray(payload.history) ? (payload.history as unknown[]).slice(-60) : [],
      baseline: { p10: baseline.p10, p50: baseline.p50, p90: baseline.p90 },
      scenarios: Array.isArray(payload.scenarios)
        ? (payload.scenarios as Array<Record<string, unknown>>).map((sc) => ({
            name: sc.name,
            p10: sc.p10,
            p50: sc.p50,
            p90: sc.p90,
          }))
        : [],
    },
  };
}

async function runWeatherHistory(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const days = intArg(args, "days", 30, 7, 92);
  const placeArg =
    stringArg(args, "place") || stringArg(args, "location") || stringArg(args, "city") || stringArg(args, "query");
  const place = await resolveHistoryPlace(placeArg, ctx.question, ctx.visitor);
  const history = await fetchDailyHistory(place, days);
  if (!history.highs.length) throw new Error("No weather history returned.");
  const highs = history.highs;
  const avg = highs.reduce((sum, v) => sum + v, 0) / highs.length;
  const max = Math.max(...highs);
  const min = Math.min(...highs);
  const label = formatPlace(place);
  const first = history.dates[0];
  const last = history.dates[history.dates.length - 1];
  const fmt = (v: number) => `${Math.round(v * 10) / 10}°F`;
  const nextAvg = history.nextHighs.length
    ? history.nextHighs.reduce((sum, v) => sum + v, 0) / history.nextHighs.length
    : null;
  const text =
    `Last ${highs.length} days in ${label} (${first} to ${last}): daily highs averaged ${fmt(avg)}, ` +
    `ranging ${fmt(min)} to ${fmt(max)}; the most recent high was ${fmt(highs[highs.length - 1])}.` +
    (nextAvg != null ? ` Open-Meteo's own next-7-day highs average ${fmt(nextAvg)}.` : "");
  return {
    kind: "weather_history",
    text,
    place: label,
    unit: "°F",
    days: highs.length,
    first_date: first,
    last_date: last,
    highs,
    avg_high: Math.round(avg * 10) / 10,
    min_high: min,
    max_high: max,
    open_meteo_next_highs: history.nextHighs,
    _full: {
      dates: history.dates,
      highs,
      lows: history.lows,
      next_dates: history.nextDates,
      next_highs: history.nextHighs,
    },
  };
}

async function runGenerateSynthetic(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const pattern = stringArg(args, "pattern", "sine") || "sine";
  const length = intArg(args, "length", 256, 32, 2048);
  const noiseRaw = args.noise;
  const noise =
    typeof noiseRaw === "number" && Number.isFinite(noiseRaw)
      ? Math.min(1, Math.max(0, noiseRaw))
      : 0.05;
  const seed = intArg(args, "seed", 0, 0, 1_000_000);
  const result = await ctx.fetchContainer(
    "/analysis/synthetic",
    containerInit(ctx, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({ pattern, length, noise, seed }),
    }),
  );
  throwIfContainerStatusError(result.status, "Synthetic series");
  const payload = result.json as Record<string, unknown>;
  const series = Array.isArray(payload.series) ? payload.series.slice(0, 8) : [];
  return {
    ok: true,
    pattern: payload.pattern,
    method: payload.method,
    length: payload.length,
    noise: payload.noise,
    seed: payload.seed,
    summary: payload.summary,
    preview: series,
    has_csv: typeof payload.csv === "string",
    text:
      typeof payload.summary === "string"
        ? payload.summary
        : `Generated synthetic ${pattern} series.`,
  };
}

async function runCreateAlert(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const { db, scope } = requireMemoryDb(ctx);
  const name = stringArg(args, "name");
  const series = stringArg(args, "series", "weather") || "weather";
  const kindRaw = stringArg(args, "kind", "threshold") || "threshold";
  const kind: AlertKind = kindRaw === "anomaly" ? "anomaly" : "threshold";
  const config: Record<string, unknown> = {};
  if (kind === "threshold") {
    config.op = stringArg(args, "op", ">") || ">";
    const valueRaw = args.value;
    config.value =
      typeof valueRaw === "number" && Number.isFinite(valueRaw) ? valueRaw : 0;
  }
  const result = await createAlertRule(db, scope, { name, series, kind, config });
  if (!result.ok) return { ok: false, text: result.error };
  return {
    ok: true,
    rule: result.rule,
    text: `Created ${kind} alert “${result.rule.name}” on ${series}.`,
  };
}

async function runListAlerts(ctx: ToolContext): Promise<ToolResult["data"]> {
  const { db, scope } = requireMemoryDb(ctx);
  const [rules, events] = await Promise.all([
    listAlertRules(db, scope),
    listAlertEvents(db, scope),
  ]);
  return {
    ok: true,
    rules,
    events,
    text: rules.length
      ? `${rules.length} rule(s), ${events.length} recent event(s).`
      : "No alert rules yet.",
  };
}

async function runCheckAlerts(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const { db, scope } = requireMemoryDb(ctx);
  const seriesFilter = stringArg(args, "series") || null;
  const rules = await listAlertRules(db, scope);
  const fired: Array<Record<string, unknown>> = [];
  for (const rule of rules) {
    if (!rule.enabled) continue;
    if (seriesFilter && rule.series !== seriesFilter) continue;
    const checked = await ctx.fetchContainer(
      "/analysis/alert-check",
      containerInit(ctx, {
        method: "POST",
        headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({
          series: rule.series,
          kind: rule.kind,
          config: rule.config,
        }),
      }),
    );
    if (checked.status >= 400 || !checked.json || typeof checked.json !== "object") {
      continue;
    }
    const payload = checked.json as Record<string, unknown>;
    if (payload.fired !== true) continue;
    const event = await insertAlertEvent(db, scope, rule.id, {
      ...payload,
      rule_name: rule.name,
    });
    fired.push({ rule, event, result: payload });
  }
  return {
    ok: true,
    fired,
    text: fired.length
      ? `Fired ${fired.length} alert(s).`
      : "No alerts fired on this check.",
  };
}

async function runCreateReport(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const series = stringArg(args, "series", "weather").toLowerCase() || "weather";
  const built = await ctx.fetchContainer(
    "/analysis/report",
    containerInit(ctx, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({ series }),
    }),
  );
  throwIfContainerStatusError(built.status, "Report build");
  const snapshot = built.json as Record<string, unknown>;
  if (!ctx.db) {
    return {
      ok: true,
      series,
      snapshot,
      text: "Report snapshot ready (share store unavailable in this environment).",
    };
  }
  const scope =
    typeof ctx.memoryScope === "string" && ctx.memoryScope.trim()
      ? ctx.memoryScope.trim()
      : "visitor";
  const title =
    typeof snapshot.title === "string" && snapshot.title.trim()
      ? snapshot.title
      : `${series} report`;
  const stored = await createReport(ctx.db, scope, {
    series,
    title,
    snapshot,
  });
  if (!stored.ok) return { ok: false, text: stored.error };
  return {
    ok: true,
    report: {
      id: stored.report.id,
      series: stored.report.series,
      title: stored.report.title,
      share_path: `/agent/reports/${stored.report.id}`,
    },
    text: `Shareable report ready: /agent/reports/${stored.report.id}`,
  };
}

async function runAnalyzePca(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const datasetId = stringArg(args, "dataset_id", "");
  if (!datasetId) {
    throw new Error("analyze_pca requires dataset_id (from list_user_series or a recent CSV upload).");
  }
  const nComponents = args.n_components;
  const body: Record<string, unknown> = {};
  if (typeof nComponents === "number" && Number.isFinite(nComponents) && nComponents >= 1) {
    body.n_components = Math.floor(nComponents);
  }
  const result = await ctx.fetchContainer(
    `/workspace/datasets/${encodeURIComponent(datasetId)}/pca`,
    containerInit(ctx, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
  if (result.status === 401 || result.status === 403) {
    throw new Error("Sign in to analyze a private workspace CSV with PCA.");
  }
  if (result.status >= 400) {
    const detail =
      result.json && typeof result.json === "object" && "detail" in (result.json as object)
        ? String((result.json as { detail?: unknown }).detail)
        : `PCA failed (${result.status}).`;
    throw new Error(detail);
  }
  const payload = result.json as Record<string, unknown>;
  return {
    dataset_id: datasetId,
    summary: typeof payload.summary === "string" ? payload.summary : "",
    explained_variance_ratio: payload.explained_variance_ratio,
    groups: payload.groups,
    loadings: payload.loadings,
    feature_names: payload.feature_names,
    n_components: payload.n_components,
  };
}

async function runAnalyzeRegimes(
  ctx: ToolContext,
  args: Record<string, unknown>,
): Promise<ToolResult["data"]> {
  const datasetId = stringArg(args, "dataset_id", "");
  if (!datasetId) {
    throw new Error(
      "analyze_regimes requires dataset_id (from list_user_series or a recent CSV upload).",
    );
  }
  const body: Record<string, unknown> = {};
  const k = args.k;
  if (typeof k === "number" && Number.isFinite(k) && k >= 2 && k <= 8) {
    body.k = Math.floor(k);
  }
  const result = await ctx.fetchContainer(
    `/workspace/datasets/${encodeURIComponent(datasetId)}/kmeans`,
    containerInit(ctx, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
  if (result.status === 401 || result.status === 403) {
    throw new Error("Sign in to cluster a private workspace CSV.");
  }
  if (result.status >= 400) {
    const detail =
      result.json && typeof result.json === "object" && "detail" in (result.json as object)
        ? String((result.json as { detail?: unknown }).detail)
        : `K-Means failed (${result.status}).`;
    throw new Error(detail);
  }
  const payload = result.json as Record<string, unknown>;
  return {
    dataset_id: datasetId,
    summary: typeof payload.summary === "string" ? payload.summary : "",
    k: payload.k,
    k_selection: payload.k_selection,
    regimes: payload.regimes,
    anomaly_count: payload.anomaly_count,
    anomalies: Array.isArray(payload.anomalies)
      ? (payload.anomalies as unknown[]).slice(0, 12)
      : [],
    silhouette: payload.silhouette,
  };
}

function defaultTimeoutForTool(toolName: AgentToolName, override?: number): number {
  if (toolName === "forecast" || toolName === "get_evaluation_metrics" || toolName === "simulate_world") {
    if (typeof override === "number" && Number.isFinite(override)) {
      // Raise the loop's generic 8s default; keep intentionally short test overrides.
      if (override >= TOOL_TIMEOUT_MS) return Math.max(override, FORECAST_TOOL_TIMEOUT_MS);
      return override;
    }
    return FORECAST_TOOL_TIMEOUT_MS;
  }
  if (typeof override === "number" && Number.isFinite(override)) return override;
  return TOOL_TIMEOUT_MS;
}

export async function dispatchTool(
  name: string,
  args: Record<string, unknown>,
  ctx: ToolContext,
  timeoutMs?: number,
  options?: { coldStartWaitMs?: number; retryAttempts?: number },
): Promise<ToolResult> {
  const started = Date.now();
  const toolName = AGENT_TOOL_NAMES.includes(name as AgentToolName) ? (name as AgentToolName) : null;
  if (!toolName) {
    return {
      ok: false,
      name: "forecast",
      summary: "unknown tool",
      error: `Unknown tool: ${name}`,
      durationMs: Date.now() - started,
    };
  }
  const slowSimulate =
    toolName === "simulate_world" &&
    (args.encoder === "auto" || args.encoder === "jepa") &&
    (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs) || timeoutMs >= TOOL_TIMEOUT_MS);
  const effectiveTimeout = slowSimulate
    ? SLOW_SIMULATE_TIMEOUT_MS
    : defaultTimeoutForTool(toolName, timeoutMs);
  const runOnce = (signal: AbortSignal): Promise<unknown> => {
    const keyed: ToolContext = { ...ctx, signal };
    switch (toolName) {
      case "forecast":
        return runForecast(keyed, args);
      case "fetch_live_data":
        return runLiveData(keyed, args);
      case "get_evaluation_metrics":
        return runEvaluation(keyed, args);
      case "compare_horizons":
        return runCompareHorizons(keyed, args);
      case "explain_prediction":
        return runExplain(keyed, args);
      case "list_user_series":
        return runListSeries(keyed);
      case "remember_fact":
        return runRememberFact(keyed, args);
      case "recall_memory":
        return runRecallMemory(keyed, args);
      case "forget_memory":
        return runForgetMemory(keyed, args);
      case "save_scenario":
        return runSaveScenario(keyed, args);
      case "list_scenarios":
        return runListScenarios(keyed);
      case "run_scenario":
        return runRunScenario(keyed, args);
      case "rename_user_series":
        return runRenameUserSeries(keyed, args);
      case "delete_user_series":
        return runDeleteUserSeries(keyed, args);
      case "retrain_user_series":
        return runRetrainUserSeries(keyed, args);
      case "analyze_pca":
        return runAnalyzePca(keyed, args);
      case "analyze_regimes":
        return runAnalyzeRegimes(keyed, args);
      case "what_if":
        return runWhatIf(keyed, args);
      case "create_report":
        return runCreateReport(keyed, args);
      case "create_alert":
        return runCreateAlert(keyed, args);
      case "list_alerts":
        return runListAlerts(keyed);
      case "check_alerts":
        return runCheckAlerts(keyed, args);
      case "generate_synthetic_series":
        return runGenerateSynthetic(keyed, args);
      case "simulate_world":
        return runSimulateWorld(keyed, args);
      case "weather_history":
        return runWeatherHistory(keyed, args);
      case "web_search":
        return runWebSearch(keyed, args);
      case "wiki_summary":
        return runWikiSummary(keyed, args);
      case "read_page":
        return runReadPage(keyed, args);
      default: {
        const unexpected: never = toolName;
        throw new Error(`Unhandled tool: ${String(unexpected)}`);
      }
    }
  };

  try {
    const data = await withContainerRetries(toolName, runOnce, effectiveTimeout, {
      baseWaitMs: options?.coldStartWaitMs ?? CONTAINER_COLD_START_WAIT_MS,
      attempts: slowSimulate ? 1 : options?.retryAttempts,
      budgetMs: slowSimulate ? SLOW_SIMULATE_TIMEOUT_MS : undefined,
    });
    return {
      ok: true,
      name: toolName,
      summary: toolName,
      data,
      durationMs: Date.now() - started,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // Never leave the user with a raw tool-name/error leak — clear degraded answers.
    if (toolName === "forecast") {
      return {
        ok: true,
        name: toolName,
        summary: "forecast_unavailable",
        data: {
          degraded: true,
          text: FORECAST_UNAVAILABLE_TEXT,
          series: stringArg(args, "series", "weather") || "weather",
          horizon: intArg(args, "horizon", 1, 1, 24),
          error: message.slice(0, 240),
        },
        durationMs: Date.now() - started,
      };
    }
    if (toolName === "get_evaluation_metrics") {
      return {
        ok: true,
        name: toolName,
        summary: "evaluation_unavailable",
        data: {
          degraded: true,
          text: EVAL_UNAVAILABLE_TEXT,
          model_id: stringArg(args, "model_id", "weather") || "weather",
          error: message.slice(0, 240),
        },
        durationMs: Date.now() - started,
      };
    }
    if (toolName === "fetch_live_data") {
      return {
        ok: true,
        name: toolName,
        summary: "live_data_unavailable",
        data: {
          degraded: true,
          text: LIVE_DATA_UNAVAILABLE_TEXT,
          error: message.slice(0, 240),
        },
        durationMs: Date.now() - started,
      };
    }
    return {
      ok: false,
      name: toolName,
      summary: toolName,
      error: message.slice(0, 240),
      durationMs: Date.now() - started,
    };
  }
}

export function visitorFromRequest(request: Request): VisitorGeo {
  return visitorFromCf(request.cf as CfIncoming | undefined);
}

export function authHeadersFromRequest(request: Request): Record<string, string> {
  const headers: Record<string, string> = {};
  const authorization = request.headers.get("Authorization");
  if (authorization) headers.Authorization = authorization;
  const cookie = request.headers.get("Cookie");
  if (cookie) headers.Cookie = cookie;
  return headers;
}
