/**
 * Agent tools wired to existing forecast, evaluation, live-data, and workspace paths.
 * Tools never invent live weather or prices.
 */

import { visitorFromCf, wantsCurrentConditions, type CfIncoming, type VisitorGeo } from "../live/geo";
import { isWeatherAsk, resolveFetchLiveKind } from "../live/kind";
import { answerWeather } from "../live/weather";
import { answerQuote } from "../live/markets";
import { AGENT_TOOL_NAMES, type AgentToolName } from "./protocol";

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
/** Forecast/eval need more headroom: model inference + optional metrics. */
export const FORECAST_TOOL_TIMEOUT_MS = 20_000;
export const TURN_TIMEOUT_MS = 45_000;
/** Base wait before the first container retry (grows with backoff). */
export const CONTAINER_COLD_START_WAIT_MS = 1_500;
export const CONTAINER_RETRY_ATTEMPTS = 3;
/** Optional accuracy/backtest must not consume the forecast timeout budget. */
export const ACCURACY_FETCH_TIMEOUT_MS = 3_000;
export const FORECAST_UNAVAILABLE_TEXT =
  "The forecast service is warming up or temporarily unavailable. Please try that forecast again in a moment — I could not load a reliable prediction just now.";

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
  init?: { method?: string; body?: string; headers?: Record<string, string> },
) => Promise<{ status: number; json: unknown }>;

export type ToolContext = {
  question: string;
  visitor: VisitorGeo;
  fetchContainer: ContainerFetcher;
  authHeaders: Record<string, string>;
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
    text.includes("econnreset") ||
    text.includes("network connection lost") ||
    /\btimeout\b/.test(text) ||
    /\bfailed\s*\(5\d\d\)/.test(text) ||
    /\b(502|503|504)\b/.test(text)
  );
}

export function containerRetryDelayMs(attempt: number, baseMs: number): number {
  // attempt 0 → base, 1 → 2x, 2 → 4x
  return Math.max(0, Math.round(baseMs * 2 ** Math.max(0, attempt)));
}

export function adaptiveToolTimeoutMs(baseTimeoutMs: number, attempt: number): number {
  // Grow timeout on later attempts so a slow-but-alive container can finish.
  return Math.round(baseTimeoutMs * (1 + attempt * 0.5));
}

/**
 * Ping /health so forecast does not race a container that has a port open but
 * is not serving yet.
 */
export async function ensureContainerReady(
  fetchContainer: ContainerFetcher,
  options?: { attempts?: number; waitMs?: number },
): Promise<void> {
  const attempts = options?.attempts ?? CONTAINER_RETRY_ATTEMPTS;
  const waitMs = options?.waitMs ?? 400;
  let lastError = "health check failed";
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const result = await withTimeout(
        fetchContainer("/health", { method: "GET" }),
        Math.max(1_500, waitMs * 4),
      );
      if (result.status > 0 && result.status < 500) {
        const status =
          result.json && typeof result.json === "object"
            ? (result.json as { status?: unknown }).status
            : null;
        if (result.status < 400 && (status === "ok" || status == null || status === "healthy")) {
          return;
        }
        lastError = `health check failed (${result.status})`;
      } else {
        lastError = `health check failed (${result.status})`;
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    if (attempt + 1 < attempts) {
      await sleep(containerRetryDelayMs(attempt, waitMs));
    }
  }
  throw new Error(`container is not running: ${lastError}`);
}

async function withContainerRetries<T>(
  toolName: AgentToolName,
  run: () => Promise<T>,
  timeoutMs: number,
  options?: { baseWaitMs?: number; attempts?: number },
): Promise<T> {
  const attempts = options?.attempts ?? (CONTAINER_RETRY_TOOLS.has(toolName) ? CONTAINER_RETRY_ATTEMPTS : 1);
  const baseWaitMs = options?.baseWaitMs ?? CONTAINER_COLD_START_WAIT_MS;
  let lastError: unknown = new Error("tool failed");
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await withTimeout(run(), adaptiveToolTimeoutMs(timeoutMs, attempt));
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      const retryable = CONTAINER_RETRY_TOOLS.has(toolName) && isRetryableContainerError(message);
      if (!retryable || attempt + 1 >= attempts) break;
      await sleep(containerRetryDelayMs(attempt, baseWaitMs));
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
  try {
    const backtest = await withTimeout(
      ctx.fetchContainer(`/evaluation/${encodeURIComponent(series)}/backtest`, {
        method: "POST",
        headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ horizon }),
      }),
      ACCURACY_FETCH_TIMEOUT_MS,
    );
    if (backtest.status < 400 && backtest.json && typeof backtest.json === "object") {
      return metricsSummary(backtest.json as Record<string, unknown>);
    }
  } catch {
    // Accuracy is optional — never fail the forecast for a slow backtest.
  }
  return null;
}

async function runForecast(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const series = stringArg(args, "series", "weather").toLowerCase() || "weather";
  const horizon = intArg(args, "horizon", 1, 1, 24);
  await ensureContainerReady(ctx.fetchContainer, { attempts: 2, waitMs: 300 });

  if (series === "loaded") {
    const recent = await ctx.fetchContainer(`/series/recent?points=96`, {
      method: "GET",
      headers: ctx.authHeaders,
    });
    throwIfContainerStatusError(recent.status, "Loaded series forecast");
    return { series: "loaded", horizon, ...compactForecast(recent.json as Record<string, unknown>) };
  }
  if (STARTER_IDS.has(series)) {
    const path = `/starters/${encodeURIComponent(series)}/forecast?points=120`;
    const result = await ctx.fetchContainer(path, { method: "GET", headers: ctx.authHeaders });
    throwIfContainerStatusError(result.status, "Starter forecast");
    const payload = compactForecast(result.json as Record<string, unknown>);
    const accuracy = await fetchOptionalAccuracy(ctx, series, horizon);
    return { series, horizon, ...payload, recent_accuracy: accuracy };
  }
  const predict = await ctx.fetchContainer(`/workspace/models/${encodeURIComponent(series)}/predict-latest`, {
    method: "POST",
    headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
    body: "{}",
  });
  if (predict.status >= 500) {
    throw new Error(`Workspace forecast failed (${predict.status}).`);
  }
  if (predict.status >= 400) {
    throw new Error("Workspace forecast unavailable. Sign in or use a starter series id.");
  }
  let accuracy: unknown = null;
  try {
    const metrics = await withTimeout(
      ctx.fetchContainer(`/workspace/models/${encodeURIComponent(series)}/metrics`, {
        method: "GET",
        headers: ctx.authHeaders,
      }),
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
      if (!isWeatherAsk(ctx.question) && !isWeatherAsk(query)) {
        throw new Error(
          "That does not look like a weather question. Ask about weather, temperature, or rain in a place.",
        );
      }
      // Evaluate wantsCurrentConditions on the user's original question.
      const weatherAsk = weatherQuestionForTool(ctx.question, query);
      const result = await answerWeather(weatherAsk, ctx.visitor);
      return {
        kind,
        text: result.text,
        query_used: weatherAsk,
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

async function runEvaluation(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const modelId = stringArg(args, "model_id", "weather") || "weather";
  const horizon = intArg(args, "horizon", 1, 1, 24);
  await ensureContainerReady(ctx.fetchContainer, { attempts: 2, waitMs: 300 });
  if (STARTER_IDS.has(modelId)) {
    const result = await ctx.fetchContainer(`/evaluation/${encodeURIComponent(modelId)}/backtest`, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({ horizon }),
    });
    if (result.status >= 500) throw new Error(`Backtest failed (${result.status}).`);
    if (result.status >= 400) throw new Error("Backtest unavailable for that model.");
    return { model_id: modelId, ...metricsSummary(result.json as Record<string, unknown>) };
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
    const result = await ctx.fetchContainer(`/evaluation/${encodeURIComponent(modelId)}/backtest`, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({ horizon, refresh: true }),
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
  const runOnce = (): Promise<unknown> => {
    switch (toolName) {
      case "forecast":
        return runForecast(ctx, args);
      case "fetch_live_data":
        return runLiveData(ctx, args);
      case "get_evaluation_metrics":
        return runEvaluation(ctx, args);
      case "compare_horizons":
        return runCompareHorizons(ctx, args);
      case "explain_prediction":
        return runExplain(ctx, args);
      case "list_user_series":
        return runListSeries(ctx);
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
    // Forecast: never leave the user with a raw failure — return a clear degraded answer.
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
