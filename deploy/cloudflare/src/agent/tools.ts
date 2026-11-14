/**
 * Agent tools wired to existing forecast, evaluation, live-data, and workspace paths.
 * Tools never invent live weather or prices.
 */

import type { D1Like } from "../live/ai";
import { visitorFromCf, wantsCurrentConditions, type CfIncoming, type VisitorGeo } from "../live/geo";
import { isWeatherAsk, resolveFetchLiveKind } from "../live/kind";
import { answerWeather } from "../live/weather";
import { answerQuote } from "../live/markets";
import { AGENT_TOOL_NAMES, type AgentToolName } from "./protocol";
import {
  EVAL_CACHE_TTL_SECONDS,
  FORECAST_CACHE_TTL_SECONDS,
  cacheGetOrLoad,
  evalCacheKey,
  forecastCacheKey,
  readStoredBacktest,
  type WaitUntil,
} from "./toolCache";

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

export const MAX_TOOL_CALLS = 4;
export const TOOL_TIMEOUT_MS = 8_000;
/** Per-attempt timeout for forecast/eval (kept under the shared budget). */
export const FORECAST_TOOL_TIMEOUT_MS = 12_000;
/** Hard cap across wake + attempts so /agent/chat does not burn ~20s then give up. */
export const CONTAINER_TOOL_BUDGET_MS = 22_000;
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
  /** Optional D1 for stored backtest metrics (0005_backtest). */
  db?: D1Like;
  /** Schedule background cache refresh without blocking the chat reply. */
  waitUntil?: WaitUntil;
};

const STARTER_IDS = new Set(["weather", "stocks", "crypto", "near-earth", "lunar"]);

export const TOOL_DEFINITIONS = [
  {
    type: "function" as const,
    name: "forecast",
    description:
      "Run a model forecast for a series. series is a starter id (weather, stocks, crypto, near-earth, lunar), a workspace model id, or 'loaded' for the shared champion. Returns prediction values with interval/confidence and recent accuracy when available.",
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
      "Return rolling-origin backtest metrics (MAE, RMSE, MAPE, directional accuracy) and baseline comparison for a starter or workspace model.",
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
      "Explain an existing prediction: point forecast, interval, confidence, and recent accuracy for a series.",
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

function compactForecast(payload: Record<string, unknown>): Record<string, unknown> {
  const forecast = Array.isArray(payload.forecast) ? payload.forecast.slice(0, 8) : payload.forecast;
  const uncertainty = payload.uncertainty;
  const history = Array.isArray(payload.history) ? payload.history.slice(-4) : undefined;
  return {
    starter_id: payload.starter_id,
    name: payload.name,
    unit: payload.unit,
    summary: payload.summary,
    horizon: payload.horizon,
    forecast,
    uncertainty,
    history,
    checkpoint_loaded: payload.checkpoint_loaded,
  };
}

function metricsSummary(report: Record<string, unknown>): Record<string, unknown> {
  const metrics = report.metrics && typeof report.metrics === "object" ? report.metrics : null;
  return {
    horizon: report.horizon,
    n: report.n,
    metrics,
    baselines: report.baselines,
    best_baseline: report.best_baseline,
    comparison: report.comparison,
    improvement_pct: report.improvement_pct,
  };
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
      if (stored) return metricsSummary(stored);
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
      if (report) return metricsSummary(report);
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
  const payload = compactForecast(result.json as Record<string, unknown>);
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
      ? await readStoredBacktest(ctx.db, series).then((row) => (row ? metricsSummary(row) : null))
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
  if (stored) return { model_id: modelId, ...metricsSummary(stored) };

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
      ...metricsSummary((summary.json as { backtest: Record<string, unknown> }).backtest),
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
  return { model_id: modelId, ...metricsSummary(result.json as Record<string, unknown>) };
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
    rows.push(metricsSummary(result.json as Record<string, unknown>));
  }
  return { model_id: modelId, horizons: rows };
}

async function runExplain(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const series = stringArg(args, "series", "weather") || "weather";
  const forecast = await runForecast(ctx, { series, horizon: 1 });
  return {
    series,
    explanation_facts: forecast,
    note: "Explain the point forecast, interval, confidence, and recent accuracy using only these facts.",
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

function defaultTimeoutForTool(toolName: AgentToolName, override?: number): number {
  if (toolName === "forecast" || toolName === "get_evaluation_metrics") {
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
  const effectiveTimeout = defaultTimeoutForTool(toolName, timeoutMs);
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
      default: {
        const unexpected: never = toolName;
        throw new Error(`Unhandled tool: ${String(unexpected)}`);
      }
    }
  };

  try {
    const data = await withContainerRetries(toolName, runOnce, effectiveTimeout, {
      baseWaitMs: options?.coldStartWaitMs ?? CONTAINER_COLD_START_WAIT_MS,
      attempts: options?.retryAttempts,
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
