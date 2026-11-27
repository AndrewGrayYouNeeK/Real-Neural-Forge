import { createHash } from "node:crypto";

import { afterEach, describe, expect, it, vi } from "vitest";

/** SHA-256 of the shipped Unhinged prompt; Unhinged must never change. */
const UNHINGED_PROMPT_SHA256 = "dc377e2d40781f22efb30a773dca8a6a598817e7464b3c35efedc5263ec6992b";

import {
  FALLBACK_MODEL,
  NORMAL_TEMPERATURE,
  PRIMARY_MODEL,
  SYSTEM_PROMPT,
  UNHINGED_PROMPT,
  tryReserve,
  type D1Like,
} from "../src/live/ai";
import { routeClearIntent } from "../src/agent/intent";
import {
  AGENT_ANSWER_MAX_TOKENS,
  agentSystemPrompt,
  buildAgentInputs,
  factualFallback,
  guardExplainAnswer,
  hitTokenCap,
  isSmallTalk,
  NORMAL_TOOL_VOICE,
  runAgentTurn,
  trimDanglingFragment,
} from "../src/agent/loop";
import {
  collectToolCalls,
  isDegenerateAnswer,
  parseJsonToolProtocol,
  sanitizeAgentAnswer,
  extractNativeToolCalls,
  tryParseToolJson,
} from "../src/agent/protocol";
import { factualFallback, friendlyToolFailure } from "../src/agent/loop";
import {
  CONTAINER_TOOL_BUDGET_MS,
  FORECAST_TOOL_TIMEOUT_MS,
  FORECAST_UNAVAILABLE_TEXT,
  LIVE_DATA_UNAVAILABLE_TEXT,
  MAX_TOOL_CALLS,
  adaptiveToolTimeoutMs,
  containerRetryDelayMs,
  dispatchTool,
  ensureContainerReady,
  isRetryableContainerError,
  weatherQuestionForTool,
  type ToolContext,
} from "../src/agent/tools";

vi.mock("../src/live/weather", async () => {
  const actual = await vi.importActual<typeof import("../src/live/weather")>("../src/live/weather");
  return {
    ...actual,
    answerWeather: vi.fn(async (question: string) => {
      if (/bitcoin|btc|rightangle/i.test(question)) {
        throw new Error("should not geocode market questions as weather");
      }
      if (/right\s+now|currently|\bnow\b/i.test(question)) {
        return {
          text: "Right now in New York, it is about 54°F with partly cloudy skies. This is live Open-Meteo current conditions, not a trained model.",
        };
      }
      return {
        text: "Tomorrow in New York, the high is about 60°F and the low about 45°F. This is a live Open-Meteo forecast, not a trained model.",
      };
    }),
  };
});

vi.mock("../src/live/markets", async () => {
  const actual = await vi.importActual<typeof import("../src/live/markets")>("../src/live/markets");
  return {
    ...actual,
    answerQuote: vi.fn(async (question: string, prefer: string) => {
      if (/bitcoin|btc/i.test(question) || prefer === "crypto") {
        return {
          text: "Bitcoin is at $104,321.40. This is a live price, not financial advice.",
        };
      }
      return {
        text: "AAPL is at $268.50. This is a live market quote, not financial advice.",
      };
    }),
  };
});

type Row = { day: string; scope: string; requests: number; neurons: number };

class MemoryD1 implements D1Like {
  rows = new Map<string, Row>();

  private key(day: string, scope: string): string {
    return `${day}|${scope}`;
  }

  prepare(sql: string) {
    const run = (values: unknown[]) => this.exec(sql, values);
    return {
      bind: (...values: unknown[]) => ({
        all: async () => run(values),
        first: async () => (await run(values)).results?.[0] ?? null,
        run: async () => run(values),
      }),
      all: async () => run([]),
      run: async () => run([]),
    };
  }

  private exec(sql: string, values: unknown[]) {
    const text = sql.replace(/\s+/g, " ").trim();
    if (text.startsWith("INSERT INTO ai_usage")) {
      const [day, scope] = values as string[];
      const key = this.key(day, scope);
      if (!this.rows.has(key)) this.rows.set(key, { day, scope, requests: 0, neurons: 0 });
      return { results: [], meta: { changes: 1 } };
    }
    if (text.startsWith("SELECT requests, neurons")) {
      const [day, scope] = values as string[];
      const row = this.rows.get(this.key(day, scope));
      return { results: row ? [{ requests: row.requests, neurons: row.neurons }] : [], meta: { changes: 0 } };
    }
    if (text.startsWith("UPDATE ai_usage")) {
      if (text.includes("SET requests")) {
        const [addRequests, addNeurons, day, scope, addNeuronsAgain, maxNeurons, addRequestsAgain, maxRequests] =
          values as number[];
        const row = this.rows.get(this.key(String(day), String(scope)));
        if (!row) return { results: [], meta: { changes: 0 } };
        if (row.neurons + Number(addNeuronsAgain) > Number(maxNeurons)) {
          return { results: [], meta: { changes: 0 } };
        }
        if (maxRequests != null && row.requests + Number(addRequestsAgain) > Number(maxRequests)) {
          return { results: [], meta: { changes: 0 } };
        }
        row.requests += Number(addRequests);
        row.neurons += Number(addNeurons);
        return { results: [], meta: { changes: 1 } };
      }
      const [extra, day, scope] = values as [number, string, string];
      const row = this.rows.get(this.key(day, scope));
      if (!row) return { results: [], meta: { changes: 0 } };
      row.neurons += Number(extra);
      return { results: [], meta: { changes: 1 } };
    }
    throw new Error(`unsupported sql: ${text}`);
  }
}

function toolContext(overrides: Partial<ToolContext> = {}): ToolContext {
  return {
    question: "What is the weather?",
    visitor: {},
    fetchContainer: async () => ({ status: 200, json: {} }),
    authHeaders: {},
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("agent protocol", () => {
  it("parses native Responses function_call items", () => {
    const calls = extractNativeToolCalls({
      output: [
        {
          type: "function_call",
          call_id: "c1",
          name: "forecast",
          arguments: '{"series":"weather","horizon":1}',
        },
        { type: "reasoning", content: [{ type: "reasoning_text", text: "thinking" }] },
        { type: "message", content: [{ type: "output_text", text: "ignore me" }] },
      ],
    });
    expect(calls).toEqual([
      { id: "c1", name: "forecast", args: { series: "weather", horizon: 1 } },
    ]);
  });

  it("parses the JSON tool protocol and ignores reasoning leaks", () => {
    const call = parseJsonToolProtocol('{"tool":"fetch_live_data","args":{"kind":"weather","query":"Dallas"}}');
    expect(call?.name).toBe("fetch_live_data");
    expect(call?.args).toEqual({ kind: "weather", query: "Dallas" });
    const fenced = parseJsonToolProtocol('```json\n{"name":"list_user_series","arguments":{}}\n```');
    expect(fenced?.name).toBe("list_user_series");
    expect(collectToolCalls({ response: "hi" }, '{"tool":"compare_horizons","args":{"model_id":"weather"}}')[0].name).toBe(
      "compare_horizons",
    );
  });

  it("parses tool JSON wrapped in prose, fences, and extra text", () => {
    const prose = parseJsonToolProtocol(
      'Sure — calling the tool now.\n```json\n{"tool":"fetch_live_data","args":{"kind":"weather","query":"New York"}}\n```\nDone.',
    );
    expect(prose?.name).toBe("fetch_live_data");
    expect(prose?.args).toEqual({ kind: "weather", query: "New York" });

    const trailing = tryParseToolJson(
      'I need live weather\n{"tool":"fetch_live_data","args":{"kind":"weather","query":"NYC right now"}}',
    );
    expect(trailing[0]?.name).toBe("fetch_live_data");

    const fromReasoning = collectToolCalls(
      {
        output: [
          {
            type: "reasoning",
            content: [
              {
                type: "reasoning_text",
                text: 'Use {"tool":"forecast","args":{"series":"weather","horizon":3}}',
              },
            ],
          },
          { type: "message", content: [{ type: "output_text", text: "working" }] },
        ],
      },
      "working",
    );
    expect(fromReasoning[0]?.name).toBe("forecast");
    expect(fromReasoning[0]?.args).toEqual({ series: "weather", horizon: 3 });
  });

  it("sanitizes planning and tool JSON from agent output", () => {
    const cleaned = sanitizeAgentAnswer(
      "Need to respond with the forecast\nHigh is 72°F.\n{\"tool\":\"forecast\",\"args\":{\"series\":\"weather\"}}",
    );
    expect(cleaned).toBe("High is 72°F.");
    expect(cleaned).not.toContain("tool");
    expect(cleaned.toLowerCase()).not.toContain("need to respond");
    expect(
      sanitizeAgentAnswer(
        "fetch_live_data failed: That does not look like a weather question. Ask about weather.",
      ),
    ).not.toMatch(/fetch_live_data|does not look like/i);
  });

  it("detects degenerate repeated-character answers", () => {
    expect(isDegenerateAnswer("!!!!!!!!!!!")).toBe(true);
    expect(isDegenerateAnswer("!!!!!!!!")).toBe(true);
    expect(isDegenerateAnswer("")).toBe(true);
    expect(isDegenerateAnswer("ok")).toBe(true);
    expect(isDegenerateAnswer("Right now in New York, it is 54°F.")).toBe(false);
  });
});

describe("agent intent router", () => {
  it("routes clear weather, price, forecast, and accuracy intents", () => {
    expect(routeClearIntent("what's the weather in New York right now")?.name).toBe("fetch_live_data");
    expect(routeClearIntent("what's the weather in New York right now")?.args).toMatchObject({
      kind: "weather",
    });
    expect(routeClearIntent("forecast the next 3 values of the weather starter series")).toEqual({
      id: "intent_forecast",
      name: "forecast",
      args: { series: "weather", horizon: 3 },
    });
    expect(routeClearIntent("How accurate is the weather sample?")?.name).toBe("get_evaluation_metrics");
    expect(routeClearIntent("what is the bitcoin price")?.args).toMatchObject({ kind: "crypto" });
    expect(routeClearIntent("what's bitcoin at right now")?.args).toMatchObject({ kind: "crypto" });
    expect(routeClearIntent("what's AAPL at right now")?.args).toMatchObject({ kind: "stock" });
    expect(routeClearIntent("What is a neural network?")).toBeNull();
  });
});

describe("agent tool dispatch", () => {
  it("dispatches fetch_live_data moon through the container lunar path", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      expect(path).toBe("/lunar");
      return {
        status: 200,
        json: { named_phase: "Full Moon", illumination_percent: 99.1 },
      };
    });
    const result = await dispatchTool(
      "fetch_live_data",
      { kind: "moon" },
      toolContext({ fetchContainer }),
      2000,
    );
    expect(result.name).toBe("fetch_live_data");
    expect(result.ok).toBe(true);
    expect((result.data as { data?: { named_phase?: string } }).data?.named_phase).toBe("Full Moon");
  });

  it("keeps right-now intent from the original question when the model drops it", async () => {
    expect(weatherQuestionForTool("what's the weather in New York right now", "New York")).toMatch(
      /right now/i,
    );
    const result = await dispatchTool(
      "fetch_live_data",
      { kind: "weather", query: "New York" },
      toolContext({ question: "what's the weather in New York right now" }),
    );
    expect(result.ok).toBe(true);
    const data = result.data as {
      text?: string;
      wants_current?: boolean;
      query_used?: string;
    };
    expect(data.wants_current).toBe(true);
    expect(data.query_used).toMatch(/right now/i);
    expect(data.text).toMatch(/Right now/i);
    expect(data.text).toMatch(/54°F/);
    expect(data.text?.toLowerCase()).not.toContain("tomorrow");
  });

  it("routes bitcoin at right now to crypto even when the model asks for weather", async () => {
    const result = await dispatchTool(
      "fetch_live_data",
      { kind: "weather", query: "Rightangle" },
      toolContext({ question: "what's bitcoin at right now" }),
    );
    expect(result.ok).toBe(true);
    const data = result.data as { kind?: string; text?: string; requested_kind?: string };
    expect(data.kind).toBe("crypto");
    expect(data.requested_kind).toBe("weather");
    expect(data.text).toMatch(/Bitcoin/i);
    expect(data.text).toMatch(/\$104,321/);
    expect(data.text?.toLowerCase()).not.toContain("rightangle");
    expect(data.text?.toLowerCase()).not.toContain("kentucky");
    expect(data.text?.toLowerCase()).not.toContain("weather");
  });

  it("retries forecast with backoff after container cold-start / 5xx", async () => {
    expect(FORECAST_TOOL_TIMEOUT_MS).toBe(12_000);
    expect(CONTAINER_TOOL_BUDGET_MS).toBeLessThanOrEqual(25_000);
    expect(isRetryableContainerError("Starter forecast failed (503).")).toBe(true);
    expect(containerRetryDelayMs(0, 100)).toBe(100);
    expect(containerRetryDelayMs(1, 100)).toBe(200);
    expect(adaptiveToolTimeoutMs(12_000, 1)).toBe(15_000);

    let forecastAttempts = 0;
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/starters/weather/forecast")) {
        expect(path).toContain("lite=1");
        forecastAttempts += 1;
        if (forecastAttempts === 1) {
          throw new Error("The container is not running.");
        }
        return {
          status: 200,
          json: {
            starter_id: "weather",
            name: "Central Park",
            unit: "°C",
            forecast: [{ step: 1, value: 15.15 }],
          },
        };
      }
      if (path.includes("/evaluation/")) {
        return {
          status: 200,
          json: { backtest: { horizon: 1, metrics: { mae: 1.1 }, comparison: "ok" } },
        };
      }
      return { status: 404, json: {} };
    });
    const result = await dispatchTool(
      "forecast",
      { series: "weather", horizon: 1 },
      toolContext({ fetchContainer }),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    expect(forecastAttempts).toBe(2);
    // Tool output converts weather °C → °F (15.15°C → 59.27°F).
    expect((result.data as { forecast?: Array<{ value?: number }> }).forecast?.[0]?.value).toBe(59.27);
    expect((result.data as { unit?: string }).unit).toBe("°F");
  });

  it("serves starter forecasts from Cache API on the second call", async () => {
    const memory = new Map<string, Response>();
    vi.stubGlobal("caches", {
      default: {
        match: async (request: Request) => {
          const hit = memory.get(request.url);
          return hit ? hit.clone() : undefined;
        },
        put: async (request: Request, response: Response) => {
          memory.set(request.url, response.clone());
        },
      },
    });
    let forecastHits = 0;
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/starters/weather/forecast")) {
        forecastHits += 1;
        return {
          status: 200,
          json: {
            starter_id: "weather",
            name: "Central Park",
            unit: "°C",
            forecast: [
              { step: 1, value: 15.15 },
              { step: 2, value: 15.52 },
              { step: 3, value: 15.61 },
            ],
          },
        };
      }
      if (path.includes("/evaluation/")) {
        return {
          status: 200,
          json: { backtest: { comparison: "ok", metrics: { mae: 1 } } },
        };
      }
      return { status: 404, json: {} };
    });
    const ctx = toolContext({ fetchContainer });
    const first = await dispatchTool("forecast", { series: "weather", horizon: 3 }, ctx, 2000, {
      coldStartWaitMs: 0,
    });
    const second = await dispatchTool("forecast", { series: "weather", horizon: 3 }, ctx, 2000, {
      coldStartWaitMs: 0,
    });
    expect(first.ok).toBe(true);
    expect(second.ok).toBe(true);
    expect(forecastHits).toBe(1);
    expect((first.data as { cache_hit?: boolean }).cache_hit).toBe(false);
    expect((second.data as { cache_hit?: boolean }).cache_hit).toBe(true);
    // 15.61°C → 60.10°F in tool output.
    expect((second.data as { forecast?: Array<{ value?: number }> }).forecast?.[2]?.value).toBe(60.1);
    expect((second.data as { unit?: string }).unit).toBe("°F");
    vi.unstubAllGlobals();
  });

  it("converts weather what-if paths and deltas to °F", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/analysis/what-if")) {
        return {
          status: 200,
          json: {
            starter_id: "weather",
            series: "weather",
            unit: "°C",
            shock: { mode: "pct", value: -10, before: 10, after: 9 },
            baseline: { p50: [15.0], forecast: [15.0] },
            shocked: { p50: [14.0], forecast: [14.0] },
            delta: {
              p50: [-1.0],
              mean: -1.0,
              text: "A -10.0% shock to the last input lowers the 1-step path by about 1 on average.",
            },
            summary: "A -10.0% shock to the last input lowers the 1-step path by about 1 on average.",
          },
        };
      }
      return { status: 404, json: {} };
    });
    const result = await dispatchTool(
      "what_if",
      { series: "weather", mode: "pct", value: -10, horizon: 1 },
      toolContext({ fetchContainer }),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    const data = result.data as {
      unit?: string;
      shock?: { before?: number; after?: number };
      baseline?: { p50?: number[] };
      delta?: { mean?: number; text?: string };
      summary?: string;
    };
    expect(data.unit).toBe("°F");
    expect(data.shock?.before).toBe(50);
    expect(data.shock?.after).toBe(48.2);
    expect(data.baseline?.p50?.[0]).toBe(59);
    expect(data.delta?.mean).toBe(-1.8);
    expect(data.summary).toMatch(/°F/);
    expect(data.summary).not.toMatch(/by about 1 on average/);
  });

  it("keeps explain_prediction weather facts in °F without double conversion", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/starters/weather/forecast")) {
        return {
          status: 200,
          json: {
            starter_id: "weather",
            name: "Central Park",
            unit: "°C",
            forecast: [
              { step: 1, value: 15.15, p10: 11.0, p50: 15.15, p90: 19.0 },
              { step: 2, value: 15.5, p10: 11.0, p50: 15.5, p90: 19.5 },
            ],
            history: [{ index: 1, actual: 14.4 }],
            why: { explanation: "Recent values have been roughly flat.", drivers: [] },
          },
        };
      }
      if (path.includes("/evaluation/")) {
        return { status: 200, json: { backtest: { comparison: "ok", metrics: { mae: 1 } } } };
      }
      return { status: 404, json: {} };
    });
    const result = await dispatchTool(
      "explain_prediction",
      { series: "weather" },
      toolContext({ fetchContainer }),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    const data = result.data as {
      note?: string;
      explanation_facts?: {
        unit?: string;
        storage_unit?: string;
        forecast?: Array<{ p50?: number; value?: number }>;
      };
    };
    expect(data.explanation_facts?.unit).toBe("°F");
    expect(data.explanation_facts?.storage_unit).toBe("°C");
    // 15.15°C → 59.27°F once (not double-converted into the 130–150s).
    expect(data.explanation_facts?.forecast?.[0]?.value).toBe(59.27);
    expect(data.note).toMatch(/already in °F/i);
  });

  it("keeps explain_prediction week-ahead values inside the °F forecast range", async () => {
    const forecastC = [15.15, 15.52, 15.61, 15.59, 15.54, 15.42, 15.24];
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/starters/weather/forecast")) {
        return {
          status: 200,
          json: {
            starter_id: "weather",
            name: "Central Park temperature",
            unit: "°C",
            forecast: forecastC.map((value, index) => ({
              step: index + 1,
              value,
              p10: value - 4,
              p50: value,
              p90: value + 4,
            })),
            uncertainty: {
              p10: forecastC.map((v) => v - 4),
              p50: forecastC,
              p90: forecastC.map((v) => v + 4),
              interval_80: { lower: forecastC.map((v) => v - 4), upper: forecastC.map((v) => v + 4) },
            },
            history: [{ index: 729, time: "2026-10-31", actual: 14.4 }],
            why: { explanation: "Recent values have been trending lower.", drivers: [] },
          },
        };
      }
      if (path.includes("/evaluation/")) {
        return { status: 200, json: { backtest: { comparison: "ok", metrics: { mae: 3.31 } } } };
      }
      return { status: 404, json: {} };
    });
    const result = await dispatchTool(
      "explain_prediction",
      { series: "weather" },
      toolContext({ fetchContainer }),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    const data = result.data as {
      unit?: string;
      text?: string;
      week_ahead?: Array<{ p10: number | null; p50: number; p90: number | null }>;
      last_observed?: number;
    };
    expect(data.unit).toBe("°F");
    const forecastF = forecastC.map((c) => c * 1.8 + 32);
    const lo = Math.min(...forecastF);
    const hi = Math.max(...forecastF);
    expect(data.week_ahead).toHaveLength(7);
    for (const row of data.week_ahead ?? []) {
      // Sanity: every explain value matches the forecast scale (≈59–60°F, not 130s–160s).
      expect(row.p50).toBeGreaterThanOrEqual(lo - 0.01);
      expect(row.p50).toBeLessThanOrEqual(hi + 0.01);
      expect(row.p10 ?? 0).toBeGreaterThan(40);
      expect(row.p90 ?? 0).toBeLessThan(80);
    }
    expect(data.last_observed).toBeCloseTo(57.92, 1);
    expect(data.text).toMatch(/59\.3–60\.1°F/);
    expect(data.text).not.toMatch(/1[3-6]\d/);
    // Payload stays compact so the tool note is never cut mid-JSON.
    expect(JSON.stringify({ tool: "explain_prediction", ok: true, result: data }).length).toBeLessThan(3500);
    // A double-converted model reply is replaced with the deterministic facts.
    const guarded = guardExplainAnswer(
      "Expect highs in the high-150s to low-160s °F over the week.",
      [result],
    );
    expect(guarded).toBe(data.text);
    const fine = "Highs stay near 59–60°F all week, MAE about 6°F.";
    expect(guardExplainAnswer(fine, [result])).toBe(fine);
  });

  it("converts nested weather interval_80 bands to °F", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/starters/weather/forecast")) {
        return {
          status: 200,
          json: {
            starter_id: "weather",
            unit: "°C",
            forecast: [{ step: 1, value: 15, p10: 11, p50: 15, p90: 19 }],
            uncertainty: { p50: [15], interval_80: { lower: [11], upper: [19] } },
          },
        };
      }
      return { status: 404, json: {} };
    });
    const result = await dispatchTool("forecast", { series: "weather", horizon: 1 }, toolContext({ fetchContainer }), 2000, {
      coldStartWaitMs: 0,
    });
    const band = (result.data as { uncertainty?: { interval_80?: { lower?: number[]; upper?: number[] } } }).uncertainty
      ?.interval_80;
    expect(band?.lower?.[0]).toBe(51.8);
    expect(band?.upper?.[0]).toBe(66.2);
  });

  it("trims replies cut off at the token cap and raises the answer budget", () => {
    const cut =
      "The model sees a cooling trend after a warm spell. Highs sit near 59–60°F. A weak front could tug-";
    expect(trimDanglingFragment(cut)).toBe("The model sees a cooling trend after a warm spell. Highs sit near 59–60°F.");
    expect(trimDanglingFragment("Highs near 60°F. Clear skies all week", true)).toBe("Highs near 60°F.");
    expect(trimDanglingFragment("Highs near 60°F. Clear skies all week")).toBe("Highs near 60°F. Clear skies all week");
    expect(trimDanglingFragment("Hey — what do you want to forge?")).toBe("Hey — what do you want to forge?");
    expect(hitTokenCap({ status: "incomplete", incomplete_details: { reason: "max_output_tokens" } })).toBe(true);
    expect(hitTokenCap({ choices: [{ finish_reason: "length" }] })).toBe(true);
    expect(hitTokenCap({ status: "completed" })).toBe(false);
    const withTools = buildAgentInputs({
      model: PRIMARY_MODEL,
      system: agentSystemPrompt("normal", true),
      question: "Explain the weather forecast",
      history: [],
      toolNotes: ["Tool result for explain_prediction: {}"],
      mode: "normal",
      allowTools: true,
    });
    expect(withTools.max_output_tokens).toBe(AGENT_ANSWER_MAX_TOKENS);
    expect(withTools.reasoning).toEqual({ effort: "low" });
  });

  it("treats high/low place forecasts as weather and never leaks tool errors", async () => {
    const result = await dispatchTool(
      "fetch_live_data",
      { kind: "weather", query: "New York" },
      toolContext({ question: "Forecast tomorrow's high and low in New York" }),
    );
    expect(result.ok).toBe(true);
    const data = result.data as { kind?: string; text?: string };
    expect(data.kind).toBe("weather");
    expect(data.text?.toLowerCase()).not.toContain("fetch_live_data");
    expect(data.text?.toLowerCase()).not.toContain("does not look like");
    expect(friendlyToolFailure("fetch_live_data")).toBe(LIVE_DATA_UNAVAILABLE_TEXT);
    expect(factualFallback([{ ok: false, name: "fetch_live_data", summary: "x", error: "boom", durationMs: 1 }])).not.toContain(
      "fetch_live_data",
    );
  });

  it("returns a clear degraded forecast answer after retries are exhausted", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/health")) return { status: 200, json: { status: "ok" } };
      if (path.includes("/starters/weather/forecast")) {
        throw new Error("tool timeout");
      }
      return { status: 404, json: {} };
    });
    const result = await dispatchTool(
      "forecast",
      { series: "weather", horizon: 1 },
      toolContext({ fetchContainer }),
      5,
      { coldStartWaitMs: 0, retryAttempts: 2 },
    );
    expect(result.ok).toBe(true);
    expect(result.summary).toBe("forecast_unavailable");
    const data = result.data as { degraded?: boolean; text?: string };
    expect(data.degraded).toBe(true);
    expect(data.text).toBe(FORECAST_UNAVAILABLE_TEXT);
  });

  it("readiness ping waits until /health is ok", async () => {
    let hits = 0;
    await ensureContainerReady(
      async () => {
        hits += 1;
        if (hits < 2) return { status: 503, json: { status: "starting" } };
        return { status: 200, json: { status: "ok" } };
      },
      { attempts: 3, waitMs: 0 },
    );
    expect(hits).toBe(2);
  });

  it("retries get_evaluation_metrics after tool timeout", async () => {
    let evalAttempts = 0;
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/health")) return { status: 200, json: { status: "ok" } };
      evalAttempts += 1;
      if (evalAttempts === 1) {
        await new Promise((resolve) => setTimeout(resolve, 40));
        return { status: 200, json: {} };
      }
      if (path.endsWith("/evaluation/weather") || path.includes("/evaluation/weather?")) {
        return {
          status: 200,
          json: {
            backtest: {
              horizon: 1,
              metrics: { mae: 1.1, rmse: 1.4, mape: 0.08, directional_accuracy: 0.66 },
              comparison: "10.0% better than the best simple baseline",
            },
          },
        };
      }
      return {
        status: 200,
        json: {
          horizon: 1,
          metrics: { mae: 1.1, rmse: 1.4, mape: 0.08, directional_accuracy: 0.66 },
          comparison: "10.0% better than the best simple baseline",
        },
      };
    });
    const result = await dispatchTool(
      "get_evaluation_metrics",
      { model_id: "weather" },
      toolContext({ fetchContainer }),
      10,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    expect(evalAttempts).toBeGreaterThanOrEqual(2);
    expect((result.data as { comparison?: string }).comparison).toContain("baseline");
  });

  it("reads evaluation metrics from D1 without hitting the container", async () => {
    vi.stubGlobal("caches", { default: undefined });
    const fetchContainer = vi.fn(async () => ({ status: 500, json: {} }));
    const db = {
      prepare: () => ({
        bind: () => ({
          first: async () => ({
            report_json: JSON.stringify({
              comparison: "12.0% better than the best simple baseline",
              metrics: { mae: 0.9 },
            }),
          }),
          all: async () => ({ results: [] }),
          run: async () => ({}),
        }),
        all: async () => ({ results: [] }),
        run: async () => ({}),
      }),
    };
    const result = await dispatchTool(
      "get_evaluation_metrics",
      { model_id: "weather", horizon: 11 },
      toolContext({ fetchContainer, db }),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    expect(fetchContainer).not.toHaveBeenCalled();
    expect((result.data as { comparison?: string }).comparison).toContain("12.0%");
    // 0.9°C MAE → 1.62°F in tool output.
    expect((result.data as { metrics?: { mae?: number }; unit?: string }).metrics?.mae).toBe(1.62);
    expect((result.data as { unit?: string }).unit).toBe("°F");
  });

  it("converts weather MAE/RMSE to °F in evaluation tool output", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/evaluation/weather") && !path.includes("backtest")) {
        return {
          status: 200,
          json: {
            backtest: {
              horizon: 1,
              metrics: { mae: 3.31, rmse: 4.2 },
              comparison: "beats naive",
            },
          },
        };
      }
      return { status: 404, json: {} };
    });
    const result = await dispatchTool(
      "get_evaluation_metrics",
      { model_id: "weather", horizon: 1 },
      toolContext({ fetchContainer }),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    const data = result.data as { metrics?: { mae?: number; rmse?: number }; unit?: string };
    expect(data.unit).toBe("°F");
    expect(data.metrics?.mae).toBe(5.96);
    expect(data.metrics?.rmse).toBe(7.56);
  });

  it("dispatches forecast to the container starter path", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/health")) return { status: 200, json: { status: "ok" } };
      if (path.includes("/starters/weather/forecast")) {
        expect(path).toContain("lite=1");
        return {
          status: 200,
          json: {
            starter_id: "weather",
            name: "Central Park",
            forecast: [{ step: 1, value: 22, p10: 18, p50: 22, p90: 26 }],
            uncertainty: { confidence: 0.8, calibrated: true, p10: [18], p50: [22], p90: [26] },
          },
        };
      }
      if (path.includes("/evaluation/weather") && !path.includes("backtest")) {
        return {
          status: 200,
          json: {
            backtest: {
              horizon: 1,
              n: 20,
              metrics: { mae: 1.2, rmse: 1.5, mape: 0.1, directional_accuracy: 0.7 },
              comparison: "10.0% better than the best simple baseline",
              best_baseline: "naive",
            },
          },
        };
      }
      return { status: 404, json: { detail: "missing" } };
    });
    const result = await dispatchTool(
      "forecast",
      { series: "weather", horizon: 1 },
      toolContext({ fetchContainer }),
    );
    expect(result.ok).toBe(true);
    expect(fetchContainer).toHaveBeenCalled();
    const data = result.data as { recent_accuracy?: { comparison?: string }; forecast?: unknown };
    expect(data.recent_accuracy?.comparison).toContain("better");
  });

  it("lists user series and reports signed-out cleanly", async () => {
    const result = await dispatchTool(
      "list_user_series",
      {},
      toolContext({
        fetchContainer: async () => ({ status: 401, json: { detail: "Sign in" } }),
      }),
    );
    expect(result.ok).toBe(true);
    expect((result.data as { signed_in?: boolean }).signed_in).toBe(false);
  });

  it("runs analyze_pca against a workspace dataset", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      expect(path).toContain("/workspace/datasets/dst_demo/pca");
      return {
        status: 200,
        json: {
          dataset_id: "dst_demo",
          summary: "PC1 explains 90.0% of the variance. a and b move together.",
          explained_variance_ratio: [0.9, 0.08, 0.02],
          groups: [
            {
              component: 1,
              direction: "same",
              features: ["a", "b"],
              explained_variance_ratio: 0.9,
            },
          ],
          loadings: [],
          feature_names: ["a", "b", "c"],
          n_components: 3,
        },
      };
    });
    const result = await dispatchTool(
      "analyze_pca",
      { dataset_id: "dst_demo" },
      toolContext({ fetchContainer }),
    );
    expect(result.ok).toBe(true);
    expect((result.data as { summary?: string }).summary).toMatch(/move together/);
  });

  it("runs analyze_regimes against a workspace dataset", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      expect(path).toContain("/workspace/datasets/dst_demo/kmeans");
      return {
        status: 200,
        json: {
          dataset_id: "dst_demo",
          summary: "Auto-selected k=3. Regimes: calm days; steady days; volatile days.",
          k: 3,
          k_selection: "silhouette",
          regimes: [
            { cluster: 0, label: "calm days", count: 40, share: 0.4 },
            { cluster: 1, label: "steady days", count: 35, share: 0.35 },
            { cluster: 2, label: "volatile days", count: 25, share: 0.25 },
          ],
          anomaly_count: 2,
          anomalies: [{ index: 9, distance: 3.2 }],
          silhouette: 0.55,
        },
      };
    });
    const result = await dispatchTool(
      "analyze_regimes",
      { dataset_id: "dst_demo" },
      toolContext({ fetchContainer }),
    );
    expect(result.ok).toBe(true);
    expect((result.data as { summary?: string }).summary).toMatch(/Regimes/);
    expect((result.data as { anomaly_count?: number }).anomaly_count).toBe(2);
  });

  it("returns a clear degraded eval answer after container retries are exhausted", async () => {
    const result = await dispatchTool(
      "get_evaluation_metrics",
      { model_id: "weather" },
      toolContext({
        fetchContainer: async () =>
          new Promise((resolve) => {
            setTimeout(() => resolve({ status: 200, json: {} }), 40);
          }),
      }),
      5,
      { coldStartWaitMs: 0, retryAttempts: 2 },
    );
    expect(result.ok).toBe(true);
    expect(result.summary).toBe("evaluation_unavailable");
    const data = result.data as { degraded?: boolean; text?: string };
    expect(data.degraded).toBe(true);
    expect(data.text?.toLowerCase()).toContain("accuracy");
    expect(data.text?.toLowerCase()).not.toContain("get_evaluation_metrics");
  });
});

describe("agent turn caps and fallback", () => {
  it("keeps the normal prompt direct and leaves Unhinged unchanged", () => {
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("answer exactly what was asked");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("lead with the answer");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("secondary");
    expect(agentSystemPrompt("normal", true)).toContain("forecast");
    expect(agentSystemPrompt("normal", true)).toContain(SYSTEM_PROMPT);
    expect(agentSystemPrompt("unhinged", true).startsWith(UNHINGED_PROMPT)).toBe(true);
    expect(UNHINGED_PROMPT).toContain("foul-mouthed comedian");
    expect(UNHINGED_PROMPT).not.toContain("Answer exactly what was asked");
  });

  it("gives the default mode a warm, plain-spoken persona and leaves Unhinged byte-for-byte unchanged", () => {
    const normal = SYSTEM_PROMPT.toLowerCase();
    for (const phrase of [
      "warm",
      "clear",
      "plain-spoken",
      "knowledgeable friend",
      "concise",
      "light dry humor",
      "casual cussing is fine",
      "answer exactly what was asked",
      "answer first",
      "never a ramble",
      "no jargon",
      "corporate help desk",
      "°f",
      "at most one light quip",
      "andrew gray, founder, owner and ceo of youneek",
    ]) {
      expect(normal).toContain(phrase);
    }
    expect(normal).not.toContain("swagger");
    expect(normal).not.toContain("smug");
    expect(normal).toContain("user: what's 2+2");
    expect(agentSystemPrompt("normal", false)).toContain("quick quip");
    expect(agentSystemPrompt("normal", true)).toContain("quick quip");
    expect(agentSystemPrompt("normal", true)).not.toContain("Wit stays light and secondary");
    // Unhinged stays exactly as shipped.
    expect(createHash("sha256").update(UNHINGED_PROMPT).digest("hex")).toBe(UNHINGED_PROMPT_SHA256);
    expect(agentSystemPrompt("unhinged", false).startsWith(UNHINGED_PROMPT)).toBe(true);
    expect(agentSystemPrompt("unhinged", false)).not.toContain("quick quip");
    expect(agentSystemPrompt("unhinged", true)).not.toContain("quick quip");
  });

  it("bumps default temperature only on chat-completions and keeps 120B inputs unchanged", () => {
    const normal = buildAgentInputs({
      model: FALLBACK_MODEL,
      system: agentSystemPrompt("normal", false),
      question: "hey",
      history: [],
      toolNotes: [],
      mode: "normal",
      allowTools: false,
    });
    expect(normal.temperature).toBe(NORMAL_TEMPERATURE);
    const unhinged = buildAgentInputs({
      model: FALLBACK_MODEL,
      system: agentSystemPrompt("unhinged", false),
      question: "hey",
      history: [],
      toolNotes: [],
      mode: "unhinged",
      allowTools: false,
    });
    expect(unhinged.temperature).toBe(0.9);
    const primary = buildAgentInputs({
      model: PRIMARY_MODEL,
      system: agentSystemPrompt("normal", false),
      question: "hey",
      history: [],
      toolNotes: [],
      mode: "normal",
      allowTools: false,
    });
    expect(primary.temperature).toBeUndefined();
    expect(primary.reasoning).toEqual({ effort: "low" });
  });

  it("adds the warm voice reminder to tool notes only in normal mode", async () => {
    const notes: Record<string, string[]> = { normal: [], unhinged: [] };
    for (const mode of ["normal", "unhinged"] as const) {
      let call = 0;
      const ai = {
        async run(_model: string, inputs: Record<string, unknown>) {
          call += 1;
          const input = inputs.input as Array<{ content: string }>;
          notes[mode].push(...input.map((item) => item.content).filter((text) => text.startsWith("Tool result")));
          return call === 1
            ? { response: '{"tool":"get_evaluation_metrics","args":{"model_id":"weather"}}' }
            : { response: "MAE about 5.95°F. Not bad for a robot." };
        },
      };
      await runAgentTurn("how accurate is weather", mode, [], {
        ai,
        db: new MemoryD1(),
        ip: "1.2.3.4",
        owner: false,
        toolContext: toolContext(),
        models: { primary: PRIMARY_MODEL, fallback: FALLBACK_MODEL },
      });
    }
    expect(notes.normal.some((text) => text.includes(NORMAL_TOOL_VOICE.trim()))).toBe(true);
    expect(notes.unhinged.length).toBeGreaterThan(0);
    expect(notes.unhinged.some((text) => text.includes(NORMAL_TOOL_VOICE.trim()))).toBe(false);
  });

  it("builds gpt-oss Responses inputs without a native tools field", () => {
    const inputs = buildAgentInputs({
      model: PRIMARY_MODEL,
      system: agentSystemPrompt("normal", true),
      question: "How accurate is weather?",
      history: [],
      toolNotes: [],
      mode: "normal",
      allowTools: true,
    });
    expect(inputs.input).toBeTruthy();
    expect(inputs.reasoning).toEqual({ effort: "low" });
    expect(inputs.max_output_tokens).toBe(220);
    expect(inputs.tools).toBeUndefined();
    expect(inputs.messages).toBeUndefined();
    const fallback = buildAgentInputs({
      model: FALLBACK_MODEL,
      system: agentSystemPrompt("normal", true),
      question: "How accurate is weather?",
      history: [],
      toolNotes: [],
      mode: "normal",
      allowTools: true,
    });
    expect(fallback.messages).toBeTruthy();
    expect(fallback.tools).toBeTruthy();
  });

  it("stays on 120B when Responses JSON protocol succeeds", async () => {
    const db = new MemoryD1();
    const seen: string[] = [];
    const ai = {
      async run(model: string, inputs: Record<string, unknown>) {
        seen.push(model);
        expect(inputs.tools).toBeUndefined();
        expect(inputs.input).toBeTruthy();
        return { response: "Hey — what do you want to forge?" };
      },
    };
    const result = await runAgentTurn("hey", "normal", [], {
      ai,
      db,
      ip: "9.9.9.9",
      owner: false,
      toolContext: toolContext(),
      models: { primary: PRIMARY_MODEL, fallback: FALLBACK_MODEL },
    });
    expect(seen).toEqual([PRIMARY_MODEL]);
    expect(result.modelId).toBe(PRIMARY_MODEL);
    expect(result.toolsUsed).toEqual([]);
    expect(result.text.toLowerCase()).toContain("forge");
  });

  it("logs the 120B error reason when falling back to 8B", async () => {
    const db = new MemoryD1();
    const logged: string[] = [];
    const original = console.error;
    console.error = (line?: unknown) => {
      logged.push(String(line));
    };
    try {
      const ai = {
        async run(model: string) {
          if (model === PRIMARY_MODEL) {
            const error = new Error("tools field is not supported") as Error & { status?: number };
            error.status = 400;
            throw error;
          }
          return { response: "Nous predicts time series with transformers." };
        },
      };
      const result = await runAgentTurn("What does Nous do?", "normal", [], {
        ai,
        db,
        ip: "4.4.4.4",
        owner: false,
        toolContext: toolContext(),
        models: { primary: PRIMARY_MODEL, fallback: FALLBACK_MODEL },
      });
      expect(result.modelId).toBe(FALLBACK_MODEL);
      expect(result.text.toLowerCase()).toContain("transformer");
      expect(logged.join("\n")).toContain("agent.primary_failed");
      expect(logged.join("\n")).toContain("tools field is not supported");
      expect(logged.join("\n")).toContain("400");
      expect(logged.join("\n")).toContain(FALLBACK_MODEL);
    } finally {
      console.error = original;
    }
  });

  it("does not call tools for small talk like hey", async () => {
    expect(isSmallTalk("hey")).toBe(true);
    expect(isSmallTalk("How accurate is the weather sample?")).toBe(false);
    const db = new MemoryD1();
    let fetched = false;
    const ai = {
      async run(_model: string, inputs: Record<string, unknown>) {
        const system = ((inputs.input || inputs.messages) as { content: string }[])[0].content;
        expect(system.toLowerCase()).toContain("do not call tools");
        expect(system).not.toContain("Valid tools:");
        expect(inputs.tools).toBeUndefined();
        // Even if the model emits a tool call, small talk must ignore it.
        return {
          response: '{"tool":"list_user_series","args":{}}',
        };
      },
    };
    const result = await runAgentTurn("hey", "normal", [], {
      ai,
      db,
      ip: "5.5.5.5",
      owner: false,
      toolContext: toolContext({
        fetchContainer: async () => {
          fetched = true;
          throw new Error("should not fetch for small talk");
        },
      }),
    });
    expect(fetched).toBe(false);
    expect(result.toolsUsed).toEqual([]);
    expect(result.text.toLowerCase()).not.toContain("sign in");
    expect(result.text.toLowerCase()).toContain("forge");
  });

  it("answers without tools when the model returns plain text for a non-tool question", async () => {
    const db = new MemoryD1();
    const ai = {
      async run() {
        return { response: "Need to respond\nA transformer is a sequence model with attention." };
      },
    };
    const result = await runAgentTurn("What is a neural network transformer?", "normal", [], {
      ai,
      db,
      ip: "9.9.9.9",
      owner: false,
      toolContext: toolContext(),
    });
    expect(result.text).toBe("A transformer is a sequence model with attention.");
    expect(result.toolsUsed).toEqual([]);
    expect(result.capped).toBeFalsy();
  });

  it("retries once with a stricter instruction when tool JSON fails to parse", async () => {
    const db = new MemoryD1();
    let calls = 0;
    const prompts: string[] = [];
    const ai = {
      async run(_model: string, inputs: Record<string, unknown>) {
        calls += 1;
        const parts = (inputs.input || inputs.messages) as { content: string }[];
        prompts.push(parts.map((part) => part.content).join("\n"));
        if (calls === 1) {
          return { response: "fetch_live_data { broken" };
        }
        if (calls === 2) {
          return {
            response: '{"tool":"fetch_live_data","args":{"kind":"weather","query":"New York right now"}}',
          };
        }
        return { response: "Right now in New York, it is about 54°F." };
      },
    };
    const result = await runAgentTurn("what's the weather in New York right now", "normal", [], {
      ai,
      db,
      ip: "8.8.4.4",
      owner: false,
      toolContext: toolContext({ question: "what's the weather in New York right now" }),
      models: { primary: PRIMARY_MODEL, fallback: PRIMARY_MODEL },
    });
    expect(calls).toBeGreaterThanOrEqual(2);
    expect(prompts.some((text) => text.includes("not a valid tool JSON"))).toBe(true);
    expect(result.toolsUsed).toContain("fetch_live_data");
    expect(result.text).toMatch(/54°F|Right now/i);
  });

  it("routes clear weather intents when the model emits no parseable tool call", async () => {
    const db = new MemoryD1();
    let calls = 0;
    const ai = {
      async run() {
        calls += 1;
        if (calls === 1) {
          return { response: "I should look that up but here is nothing useful." };
        }
        return { response: "!!!!!!!!!!" };
      },
    };
    const result = await runAgentTurn("what's the weather in New York right now", "normal", [], {
      ai,
      db,
      ip: "8.8.8.8",
      owner: false,
      toolContext: toolContext({ question: "what's the weather in New York right now" }),
      models: { primary: PRIMARY_MODEL, fallback: PRIMARY_MODEL },
    });
    expect(result.toolsUsed).toContain("fetch_live_data");
    expect(result.text).toMatch(/Right now/i);
    expect(result.text).toMatch(/54°F/);
    expect(result.text.toLowerCase()).not.toContain("tomorrow");
    expect(result.text).not.toMatch(/could not finish/i);
  });

  it("parses prose-wrapped tool JSON and falls back when the final answer is degenerate", async () => {
    const db = new MemoryD1();
    let calls = 0;
    const ai = {
      async run() {
        calls += 1;
        if (calls === 1) {
          return {
            response:
              'Okay.\n```json\n{"tool":"forecast","args":{"series":"weather","horizon":3}}\n```',
          };
        }
        return { response: "!!!!!!!!!!!!!!" };
      },
    };
    const result = await runAgentTurn(
      "forecast the next 3 values of the weather starter series",
      "normal",
      [],
      {
        ai,
        db,
        ip: "7.7.7.7",
        owner: false,
        toolContext: toolContext({
          question: "forecast the next 3 values of the weather starter series",
          fetchContainer: async (path: string) => {
            if (path.includes("/health")) return { status: 200, json: { status: "ok" } };
            if (path.includes("/starters/weather/forecast")) {
              return {
                status: 200,
                json: {
                  starter_id: "weather",
                  name: "Central Park",
                  unit: "°C",
                  forecast: [
                    { step: 1, value: 21 },
                    { step: 2, value: 22 },
                    { step: 3, value: 23 },
                  ],
                },
              };
            }
            if (path.includes("/evaluation/")) {
              return {
                status: 200,
                json: { horizon: 3, metrics: { mae: 1.1 }, comparison: "8% better than naive" },
              };
            }
            return { status: 404, json: {} };
          },
        }),
        models: { primary: PRIMARY_MODEL, fallback: PRIMARY_MODEL },
      },
    );
    expect(result.toolsUsed).toContain("forecast");
    // Weather starter values are stored in °C; user-facing copy labels °F.
    expect(result.text).toMatch(/69\.80°F/);
    expect(result.text).not.toMatch(/°C/);
    expect(result.text).not.toMatch(/^!+$/);
    expect(result.text).not.toMatch(/could not finish/i);
  });

  it("logs agent.turn with a Workers Observability message field", async () => {
    const db = new MemoryD1();
    const logged: string[] = [];
    const original = console.log;
    console.log = (line?: unknown) => {
      logged.push(typeof line === "string" ? line : JSON.stringify(line));
    };
    try {
      const ai = {
        async run() {
          return { response: "A short direct answer about transformers." };
        },
      };
      await runAgentTurn("What is attention?", "normal", [], {
        ai,
        db,
        ip: "6.6.6.6",
        owner: false,
        toolContext: toolContext(),
        models: { primary: PRIMARY_MODEL, fallback: PRIMARY_MODEL },
      });
      const turn = logged.map((line) => {
        try {
          return JSON.parse(line) as { message?: string; event?: string };
        } catch {
          return null;
        }
      }).find((item) => item?.message === "agent.turn");
      expect(turn?.message).toBe("agent.turn");
      expect(turn?.event).toBe("agent.turn");
    } finally {
      console.log = original;
    }
  });

  it("builds templated answers from tool results", () => {
    const text = factualFallback([
      {
        ok: true,
        name: "fetch_live_data",
        summary: "fetch_live_data",
        data: { text: "Right now in New York, it is about 54°F with partly cloudy skies." },
        durationMs: 12,
      },
    ]);
    expect(text).toMatch(/54°F/);
  });

  it("uses already-converted °F forecast and MAE from tool payloads", () => {
    const forecastText = factualFallback([
      {
        ok: true,
        name: "forecast",
        summary: "forecast",
        data: {
          series: "weather",
          name: "Central Park",
          unit: "°F",
          forecast: [{ step: 1, value: 59.27 }],
          recent_accuracy: { metrics: { mae: 5.96 }, unit: "°F", comparison: "beats naive" },
        },
        durationMs: 12,
      },
    ]);
    expect(forecastText).toMatch(/59\.27°F/);
    expect(forecastText).toMatch(/5\.96°F/);
    expect(forecastText).not.toMatch(/°C/);

    const evalText = factualFallback([
      {
        ok: true,
        name: "get_evaluation_metrics",
        summary: "get_evaluation_metrics",
        data: {
          model_id: "weather",
          unit: "°F",
          metrics: { mae: 5.96 },
          comparison: "beats naive",
        },
        durationMs: 8,
      },
    ]);
    expect(evalText).toMatch(/5\.96°F/);
    expect(evalText).not.toMatch(/3\.31/);
  });

  it("runs a tool then answers, and counts each model call against quota", async () => {
    const db = new MemoryD1();
    let calls = 0;
    const ai = {
      async run() {
        calls += 1;
        if (calls === 1) {
          return { response: '{"tool":"get_evaluation_metrics","args":{"model_id":"weather"}}' };
        }
        return { response: "The weather sample is 10% better than the best simple baseline." };
      },
    };
    const result = await runAgentTurn("How accurate is weather?", "normal", [], {
      ai,
      db,
      ip: "2.2.2.2",
      owner: false,
      toolContext: toolContext({
        fetchContainer: async () => ({
          status: 200,
          json: {
            horizon: 1,
            metrics: { mae: 1.1, rmse: 1.4, mape: 0.08, directional_accuracy: 0.66 },
            comparison: "10.0% better than the best simple baseline",
            best_baseline: "naive",
          },
        }),
      }),
      models: { primary: "@cf/openai/gpt-oss-120b", fallback: "@cf/openai/gpt-oss-120b" },
    });
    expect(calls).toBe(2);
    expect(result.toolsUsed).toContain("get_evaluation_metrics");
    expect(result.text.toLowerCase()).toContain("baseline");
    const global = [...db.rows.values()].find((row) => row.scope === "global");
    expect((global?.requests || 0) >= 2).toBe(true);
  });

  it("stops after the max tool calls and never exceeds the cap", async () => {
    expect(MAX_TOOL_CALLS).toBe(8);
    const db = new MemoryD1();
    const ai = {
      async run() {
        return { response: '{"tool":"list_user_series","args":{}}' };
      },
    };
    const result = await runAgentTurn("what do I have?", "normal", [], {
      ai,
      db,
      ip: "3.3.3.3",
      owner: true,
      maxToolCalls: 2,
      toolContext: toolContext({
        fetchContainer: async () => ({
          status: 200,
          json: { datasets: [], models: [], predictions: [], starters: [] },
        }),
      }),
      models: { primary: "@cf/meta/llama-3.1-8b-instruct-fp8-fast", fallback: "@cf/meta/llama-3.1-8b-instruct-fp8-fast" },
    });
    expect(result.toolsUsed.length).toBeLessThanOrEqual(2);
    expect(result.text).toBeTruthy();
  });

  it("returns the capped message when the global budget is spent", async () => {
    const db = new MemoryD1();
    const now = new Date("2026-11-13T12:00:00Z");
    await tryReserve({
      db,
      ip: "1.1.1.1",
      owner: true,
      caps: { safety: 80, free: 40, perIpNeurons: 3200, perIpRequests: 40, reserve: 80 },
      now,
    });
    const result = await runAgentTurn("hi", "normal", [], {
      ai: { async run() { return { response: "hello" }; } },
      db,
      ip: "1.1.1.1",
      owner: true,
      now,
      caps: { safety: 80, free: 40, perIpNeurons: 3200, perIpRequests: 40, reserve: 80 },
      toolContext: toolContext(),
    });
    expect(result.capped).toBe(true);
    expect(result.text.toLowerCase()).toContain("daily limit");
  });
});
