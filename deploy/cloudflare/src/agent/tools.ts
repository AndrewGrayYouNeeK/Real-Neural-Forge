/**
 * Agent tools wired to existing forecast, evaluation, live-data, and workspace paths.
 * Tools never invent live weather or prices.
 */

import { visitorFromCf, type CfIncoming, type VisitorGeo } from "../live/geo";
import { answerWeather } from "../live/weather";
import { answerQuote } from "../live/markets";
import { AGENT_TOOL_NAMES, type AgentToolName } from "./protocol";

export const MAX_TOOL_CALLS = 4;
export const TOOL_TIMEOUT_MS = 8_000;
export const TURN_TIMEOUT_MS = 45_000;

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
      "Fetch live weather, stock/crypto prices, moon phase, or near-Earth space data from existing sources. Never invent values.",
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
          description: "Place, ticker, coin, or free-text for the live lookup.",
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

async function runForecast(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const series = stringArg(args, "series", "weather").toLowerCase() || "weather";
  const horizon = intArg(args, "horizon", 1, 1, 24);
  if (series === "loaded") {
    const recent = await ctx.fetchContainer(`/series/recent?points=96`, {
      method: "GET",
      headers: ctx.authHeaders,
    });
    if (recent.status >= 400) {
      throw new Error(
        typeof (recent.json as { detail?: unknown })?.detail === "string"
          ? String((recent.json as { detail: string }).detail)
          : "Loaded series forecast unavailable.",
      );
    }
    return { series: "loaded", horizon, ...compactForecast(recent.json as Record<string, unknown>) };
  }
  if (STARTER_IDS.has(series)) {
    const path = `/starters/${encodeURIComponent(series)}/forecast?points=120`;
    const result = await ctx.fetchContainer(path, { method: "GET", headers: ctx.authHeaders });
    if (result.status >= 400) {
      throw new Error(`Starter forecast failed (${result.status}).`);
    }
    const payload = compactForecast(result.json as Record<string, unknown>);
    let accuracy: unknown = null;
    try {
      const backtest = await ctx.fetchContainer(`/evaluation/${encodeURIComponent(series)}/backtest`, {
        method: "POST",
        headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ horizon }),
      });
      if (backtest.status < 400 && backtest.json && typeof backtest.json === "object") {
        accuracy = metricsSummary(backtest.json as Record<string, unknown>);
      }
    } catch {
      accuracy = null;
    }
    return { series, horizon, ...payload, recent_accuracy: accuracy };
  }
  const predict = await ctx.fetchContainer(`/workspace/models/${encodeURIComponent(series)}/predict-latest`, {
    method: "POST",
    headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
    body: "{}",
  });
  if (predict.status >= 400) {
    throw new Error("Workspace forecast unavailable. Sign in or use a starter series id.");
  }
  let accuracy: unknown = null;
  try {
    const metrics = await ctx.fetchContainer(`/workspace/models/${encodeURIComponent(series)}/metrics`, {
      method: "GET",
      headers: ctx.authHeaders,
    });
    if (metrics.status < 400) accuracy = metrics.json;
  } catch {
    accuracy = null;
  }
  return { series, horizon, prediction: predict.json, recent_accuracy: accuracy };
}

async function runLiveData(ctx: ToolContext, args: Record<string, unknown>): Promise<ToolResult["data"]> {
  const kind = stringArg(args, "kind", "weather").toLowerCase();
  const query = stringArg(args, "query", ctx.question) || ctx.question;
  switch (kind) {
    case "weather": {
      const result = await answerWeather(query, ctx.visitor);
      return { kind, text: result.text };
    }
    case "stock": {
      const result = await answerQuote(query, "stock");
      return { kind, text: result.text };
    }
    case "crypto": {
      const result = await answerQuote(query, "crypto");
      return { kind, text: result.text };
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
  if (STARTER_IDS.has(modelId)) {
    const result = await ctx.fetchContainer(`/evaluation/${encodeURIComponent(modelId)}/backtest`, {
      method: "POST",
      headers: { ...ctx.authHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({ horizon }),
    });
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

export async function dispatchTool(
  name: string,
  args: Record<string, unknown>,
  ctx: ToolContext,
  timeoutMs = TOOL_TIMEOUT_MS,
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
  try {
    const data = await withTimeout(
      (async () => {
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
      })(),
      timeoutMs,
    );
    return {
      ok: true,
      name: toolName,
      summary: toolName,
      data,
      durationMs: Date.now() - started,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
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
