import { afterEach, describe, expect, it, vi } from "vitest";

import type { D1Like } from "../src/live/ai";
import { runAgentTurn, promisesFutureWork } from "../src/agent/loop";
import {
  buildArtifacts,
  buildChart,
  chartToCsv,
  missingDeliverables,
  planRequest,
  routeMissingStep,
} from "../src/agent/plan";
import { tryParseToolJson } from "../src/agent/protocol";
import type { ToolContext, ToolResult } from "../src/agent/tools";

const HISTORY_DATES = Array.from({ length: 30 }, (_, i) => `2026-10-${String(i + 1).padStart(2, "0")}`);
const HISTORY_HIGHS = Array.from({ length: 30 }, (_, i) => 60 + Math.round(Math.sin(i / 3) * 80) / 10);

vi.mock("../src/live/weather", async () => {
  const actual = await vi.importActual<typeof import("../src/live/weather")>("../src/live/weather");
  return {
    ...actual,
    resolveHistoryPlace: vi.fn(async () => ({
      name: "Louisville",
      latitude: 38.25,
      longitude: -85.76,
      timezone: "America/Kentucky/Louisville",
      source: "question",
      region: "Kentucky",
      country: "United States",
    })),
    fetchDailyHistory: vi.fn(async () => ({
      timezone: "America/Kentucky/Louisville",
      dates: HISTORY_DATES,
      highs: HISTORY_HIGHS,
      lows: HISTORY_HIGHS.map((v) => v - 15),
      nextDates: ["2026-11-27", "2026-11-28"],
      nextHighs: [50, 52],
    })),
  };
});

type Row = { day: string; scope: string; requests: number; neurons: number };

class MemoryD1 implements D1Like {
  rows = new Map<string, Row>();
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
    const key = (day: unknown, scope: unknown) => `${String(day)}|${String(scope)}`;
    if (text.startsWith("INSERT INTO ai_usage")) {
      const [day, scope] = values as string[];
      if (!this.rows.has(key(day, scope))) this.rows.set(key(day, scope), { day, scope, requests: 0, neurons: 0 });
      return { results: [], meta: { changes: 1 } };
    }
    if (text.startsWith("SELECT requests, neurons")) {
      const row = this.rows.get(key(values[0], values[1]));
      return { results: row ? [{ requests: row.requests, neurons: row.neurons }] : [], meta: { changes: 0 } };
    }
    if (text.startsWith("UPDATE ai_usage")) {
      if (text.includes("SET requests")) {
        const [addRequests, addNeurons, day, scope, addAgain, maxNeurons, reqAgain, maxRequests] = values as number[];
        const row = this.rows.get(key(day, scope));
        if (!row) return { results: [], meta: { changes: 0 } };
        if (row.neurons + Number(addAgain) > Number(maxNeurons)) return { results: [], meta: { changes: 0 } };
        if (maxRequests != null && row.requests + Number(reqAgain) > Number(maxRequests)) {
          return { results: [], meta: { changes: 0 } };
        }
        row.requests += Number(addRequests);
        row.neurons += Number(addNeurons);
        return { results: [], meta: { changes: 1 } };
      }
      const [extra, day, scope] = values as [number, string, string];
      const row = this.rows.get(key(day, scope));
      if (row) row.neurons += Number(extra);
      return { results: [], meta: { changes: row ? 1 : 0 } };
    }
    throw new Error(`unsupported sql: ${text}`);
  }
}

function simPayload(body: Record<string, unknown>) {
  const horizon = Number(body.horizon) || 7;
  const p50 = Array.from({ length: horizon }, (_, i) => 55 - i * 0.2);
  const scenarios = Array.isArray(body.scenarios)
    ? (body.scenarios as Array<{ name?: string }>).map((sc) => ({
        name: sc.name || "Scenario",
        summary: `${sc.name}: ends 2.0°F below baseline.`,
        p50: p50.map((v) => v - 2),
        p10: p50.map((v) => v - 6),
        p90: p50.map((v) => v + 2),
        end_delta: -2,
      }))
    : [];
  return {
    series: body.values ? "custom" : "weather",
    name: body.values ? "Your series" : "Central Park temperature",
    unit: "°F",
    horizon,
    history: Array.from({ length: 10 }, (_, i) => ({ index: i, value: 50 + i })),
    baseline: { p50, p10: p50.map((v) => v - 4), p90: p50.map((v) => v + 4) },
    scenarios,
    backtest: { holdout: 7, origins: 3, mae_model: 4.1, mae_persistence: 4.5, beats_persistence: true, coverage_80: 0.8 },
    model: { description: "latent world model" },
    summary: `Baseline: by step ${horizon} the median is ${p50[horizon - 1].toFixed(1)}°F.`,
  };
}

function context(question: string, simBodies: Array<Record<string, unknown>> = [], failSim = false): ToolContext {
  return {
    question,
    visitor: {},
    authHeaders: {},
    fetchContainer: async (path, init) => {
      if (path === "/analysis/simulate") {
        const body = JSON.parse(String(init?.body || "{}")) as Record<string, unknown>;
        simBodies.push(body);
        if (failSim) return { status: 500, json: { detail: "boom" } };
        return { status: 200, json: simPayload(body) };
      }
      return { status: 200, json: { horizon: 1, metrics: { mae: 3.2 }, comparison: "5% better than naive" } };
    },
  };
}

/** Scripted model: returns replies in order; repeats the last one. */
function scriptedAi(replies: string[]) {
  const inputs: Array<Record<string, unknown>> = [];
  return {
    inputs,
    ai: {
      async run(_model: string, input: Record<string, unknown>) {
        inputs.push(input);
        const reply = replies[Math.min(inputs.length - 1, replies.length - 1)];
        return { response: reply };
      },
    },
  };
}

const MODELS = { primary: "@cf/openai/gpt-oss-120b", fallback: "@cf/openai/gpt-oss-120b" };
const WEATHER_ASK = "get Louisville weather for the last 30 days, forecast the next 7, chart it, then explain";
const SIM_ASK = "simulate a cold snap and compare with normal, then give me a CSV";
const SERIES = Array.from({ length: 30 }, (_, i) => 50 + (i % 7));
const TRAIN_ASK = `train on this series then backtest it: ${SERIES.join(", ")}`;

afterEach(() => {
  vi.clearAllMocks();
});

describe("multi-step planning", () => {
  it("lists every requested deliverable for the three live test asks", () => {
    const weather = planRequest(WEATHER_ASK);
    expect(weather.multiStep).toBe(true);
    expect(weather.deliverables).toEqual(expect.arrayContaining(["history", "forecast", "chart", "explain"]));
    expect(weather.steps.length).toBeGreaterThanOrEqual(4);

    const sim = planRequest(SIM_ASK);
    expect(sim.deliverables).toEqual(expect.arrayContaining(["simulate", "compare", "csv"]));

    const train = planRequest(TRAIN_ASK);
    expect(train.deliverables).toEqual(expect.arrayContaining(["train", "backtest"]));
    // Pasted numbers are a series, not 30 separate steps.
    expect(train.steps.length).toBeLessThanOrEqual(3);
  });

  it("keeps simple asks single-step", () => {
    expect(planRequest("what's the weather tomorrow in Chicago?").multiStep).toBe(false);
    expect(planRequest("bitcoin price right now").multiStep).toBe(false);
  });

  it("spots replies that promise work instead of doing it", () => {
    expect(promisesFutureWork("Got the data. Next I'll chart it and explain.")).toBe(true);
    expect(promisesFutureWork("Let me run the forecast for you.")).toBe(true);
    expect(promisesFutureWork("Highs averaged 58°F; the chart shows a cooling trend.")).toBe(false);
  });

  it("routes the next missing step deterministically", () => {
    const sim = routeMissingStep(["simulate", "compare"], SIM_ASK, []);
    expect(sim?.name).toBe("simulate_world");
    expect((sim?.args.scenarios as Array<{ name: string }>)[0].name).toBe("Cold snap");

    const train = routeMissingStep(["train", "backtest"], TRAIN_ASK, []);
    expect(train?.name).toBe("simulate_world");
    expect((train?.args.values as number[]).length).toBe(30);

    const hist = routeMissingStep(["history", "forecast"], WEATHER_ASK, []);
    expect(hist?.name).toBe("weather_history");
    expect(hist?.args.days).toBe(30);
  });
});

describe("tool-call parsing", () => {
  it("keeps every call when the model emits several JSON objects", () => {
    const calls = tryParseToolJson(
      '{"tool":"weather_history","args":{"place":"Louisville"}}\n{"tool":"get_evaluation_metrics","args":{"model_id":"weather"}}',
    );
    expect(calls.map((c) => c.name)).toEqual(["weather_history", "get_evaluation_metrics"]);
  });

  it("parses a JSON array of parallel calls and drops exact repeats", () => {
    const calls = tryParseToolJson(
      '[{"tool":"forecast","args":{"series":"weather"}},{"tool":"forecast","args":{"series":"weather"}},{"tool":"get_evaluation_metrics","args":{"model_id":"weather"}}]',
    );
    expect(calls.map((c) => c.name)).toEqual(["forecast", "forecast", "get_evaluation_metrics"].slice(0, 3));
  });
});

describe("multi-step agent loop", () => {
  it("finishes fetch -> forecast -> chart -> explain even when the model stops early", async () => {
    const simBodies: Array<Record<string, unknown>> = [];
    const { ai, inputs } = scriptedAi([
      '{"tool":"weather_history","args":{"place":"Louisville, KY","days":30}}',
      "Got 30 days of Louisville highs. Next I'll run the forecast and chart it.",
      '{"tool":"simulate_world","args":{"values_from":"weather_history","horizon":7}}',
      "Louisville's last 30 days averaged about 60°F. The 7-day forecast eases from 55°F toward 53.8°F because the recent cooling trend keeps pulling highs down, with an 80% range of roughly ±4°F. The chart shows both.",
    ]);
    const result = await runAgentTurn(WEATHER_ASK, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "4.4.4.4",
      owner: false,
      models: MODELS,
      toolContext: context(WEATHER_ASK, simBodies),
    });
    expect(result.toolsUsed).toEqual(["weather_history", "simulate_world"]);
    // The simulator ran on Louisville's own history, not the NYC starter.
    expect(simBodies).toHaveLength(1);
    expect(simBodies[0].values).toEqual(HISTORY_HIGHS);
    expect(simBodies[0].dates).toEqual(HISTORY_DATES);
    expect(simBodies[0].starter_id).toBeUndefined();
    expect(result.chart?.series.map((s) => s.name)).toEqual(["Actual", "Baseline (median)"]);
    expect(result.chart?.labels.length).toBe(37);
    expect(result.text).toMatch(/7-day forecast/);
    expect(result.text).not.toMatch(/Heads up/);
    expect(result.steps?.missing).toEqual([]);
    // Chart data never goes back into the prompt.
    expect(JSON.stringify(inputs)).not.toContain("_full");
    expect(inputs.length).toBe(4);
  });

  it("forecasts the fetched city even when the model names the wrong series", async () => {
    const simBodies: Array<Record<string, unknown>> = [];
    const { ai } = scriptedAi([
      '{"tool":"weather_history","args":{"location":"Louisville, KY"}}',
      '[{"tool":"forecast","args":{"series":"weather_history","horizon":7}},{"tool":"simulate_world","args":{"series":"louisville","horizon":7}}]',
      "Louisville highs averaged about 60°F; the 7-day path eases toward 53.8°F because the last two weeks cooled steadily and the model carries that trend forward.",
    ]);
    const result = await runAgentTurn(WEATHER_ASK, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "4.4.4.5",
      owner: false,
      models: MODELS,
      toolContext: context(WEATHER_ASK, simBodies),
    });
    // Both calls collapse into one simulator run on Louisville's own highs.
    expect(simBodies).toHaveLength(1);
    expect(simBodies[0].values).toEqual(HISTORY_HIGHS);
    expect(result.toolsUsed).toEqual(["weather_history", "simulate_world"]);
    expect(result.chart?.series[0].name).toBe("Actual");
    expect(result.steps?.missing).toEqual([]);
  });

  it("runs every call from one reply, including parallel ones", async () => {
    const { ai } = scriptedAi([
      '[{"tool":"weather_history","args":{"days":30}},{"tool":"get_evaluation_metrics","args":{"model_id":"weather"}}]',
      '{"tool":"simulate_world","args":{"values_from":"weather_history","horizon":7}}',
      "Done: history fetched, forecast run, and the backtest says MAE about 3.2°F, 5% better than naive. The trend is gently cooling, which is why the forecast dips.",
    ]);
    const ask = "get Louisville weather for the last 30 days, forecast the next 7, backtest it, then explain";
    const result = await runAgentTurn(ask, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "5.5.5.5",
      owner: false,
      models: MODELS,
      toolContext: context(ask),
    });
    expect(result.toolsUsed).toEqual(["weather_history", "get_evaluation_metrics", "simulate_world"]);
    expect(result.steps?.missing).toEqual([]);
  });

  it("runs the comparison itself when the model only does half and attaches the CSV", async () => {
    const simBodies: Array<Record<string, unknown>> = [];
    const { ai } = scriptedAi([
      '{"tool":"simulate_world","args":{"series":"weather","horizon":14}}',
      "Here's the normal 14-day path. A cold snap scenario wasn't generated.",
      "Here's the normal 14-day path. A cold snap scenario wasn't generated.",
      "Normal path ends near 52.4°F; the cold snap knocks the start down and ends about 2°F lower. CSV attached.",
    ]);
    const result = await runAgentTurn(SIM_ASK, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "6.6.6.6",
      owner: false,
      models: MODELS,
      toolContext: context(SIM_ASK, simBodies),
    });
    // The named cold snap is filled into the model's baseline-only call, so one run covers both.
    expect(simBodies.length).toBe(1);
    const scenarios = simBodies[0].scenarios as Array<{ name: string }>;
    expect(scenarios[0].name).toBe("Cold snap");
    expect(result.csv?.text.split("\n")[0]).toContain("Cold snap");
    expect(result.csv?.text.split("\n")[0]).toContain("Baseline (median)");
    expect(result.steps?.missing).toEqual([]);
    expect(result.text).not.toMatch(/Heads up/);
  });

  it("takes over when the model keeps calling side tools that make no progress", async () => {
    const simBodies: Array<Record<string, unknown>> = [];
    const { ai, inputs } = scriptedAi([
      '{"tool":"list_alerts","args":{}}',
      "Normal path ends near 52.4°F; the cold snap ends about 2°F lower. CSV attached.",
    ]);
    const result = await runAgentTurn(SIM_ASK, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "6.6.6.7",
      owner: false,
      models: MODELS,
      toolContext: context(SIM_ASK, simBodies),
    });
    expect(simBodies).toHaveLength(1);
    expect((simBodies[0].scenarios as Array<{ name: string }>)[0].name).toBe("Cold snap");
    expect(result.csv?.text).toContain("Cold snap");
    expect(result.steps?.missing).toEqual([]);
    expect(inputs.length).toBe(2);
  });

  it("trains + backtests pasted numbers via the simulator", async () => {
    const simBodies: Array<Record<string, unknown>> = [];
    const { ai } = scriptedAi([
      "I can't train on that without an upload.",
      "Fitted a model on your 30 points. Backtest: MAE 4.1 vs 4.5 for persistence, so it edges out the lazy guess, and 80% of holdout points landed in range.",
    ]);
    const result = await runAgentTurn(TRAIN_ASK, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "7.7.7.7",
      owner: false,
      models: MODELS,
      toolContext: context(TRAIN_ASK, simBodies),
    });
    expect(simBodies[0].values).toEqual(SERIES);
    expect(result.toolsUsed).toContain("simulate_world");
    expect(result.steps?.missing).toEqual([]);
    expect(result.text).toMatch(/Backtest/);
  });

  it("reports a step honestly when its tool fails", async () => {
    const { ai } = scriptedAi([
      '{"tool":"simulate_world","args":{"series":"weather","scenarios":[{"name":"Cold snap","events":[{"kind":"shock","value":-15}]}]}}',
      "The simulator is warming up right now.",
    ]);
    const result = await runAgentTurn(SIM_ASK, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "8.8.8.8",
      owner: false,
      models: MODELS,
      toolTimeoutMs: 200,
      toolContext: context(SIM_ASK, [], true),
    });
    expect(result.steps?.missing).toEqual(expect.arrayContaining(["simulate", "csv"]));
    expect(result.text).toMatch(/couldn't finish the simulation/i);
    expect(result.csv).toBeUndefined();
  });

  it("stops a runaway turn at the step cap and neuron budget", async () => {
    let n = 0;
    const ai = {
      async run() {
        n += 1;
        return { response: `{"tool":"generate_synthetic_series","args":{"seed":${n}}}` };
      },
    };
    const db = new MemoryD1();
    const result = await runAgentTurn(WEATHER_ASK, "normal", [], {
      ai,
      db,
      ip: "9.9.9.9",
      owner: false,
      models: MODELS,
      maxModelSteps: 4,
      toolContext: context(WEATHER_ASK),
    });
    // 4 loop steps + at most one tools-off wrap-up call.
    expect(n).toBeLessThanOrEqual(5);
    expect(result.text).toBeTruthy();
    const ipRow = [...db.rows.values()].find((row) => row.scope !== "global");
    expect(ipRow?.requests || 0).toBeLessThanOrEqual(5);
  });

  it("keeps single-step asks to one tool and one answer", async () => {
    const { ai, inputs } = scriptedAi([
      '{"tool":"get_evaluation_metrics","args":{"model_id":"weather"}}',
      "Weather model MAE is about 3.2°F, 5% better than naive.",
    ]);
    const result = await runAgentTurn("How accurate is weather?", "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "10.0.0.1",
      owner: false,
      models: MODELS,
      toolContext: context("How accurate is weather?"),
    });
    expect(inputs.length).toBe(2);
    expect(result.chart).toBeUndefined();
    expect(result.steps).toBeUndefined();
  });
});

describe("artifacts", () => {
  it("builds a chart and CSV from history + simulation", () => {
    const results: ToolResult[] = [
      {
        ok: true,
        name: "weather_history",
        summary: "",
        durationMs: 1,
        data: { place: "Louisville", _full: { dates: ["a", "b"], highs: [50, 52] } },
      },
      {
        ok: true,
        name: "forecast",
        summary: "",
        durationMs: 1,
        data: { unit: "°F", forecast: [{ step: 1, value: 53 }] },
      },
    ];
    const chart = buildChart(results);
    expect(chart?.labels).toEqual(["a", "b", "+1"]);
    const csv = chartToCsv(chart!);
    expect(csv.split("\n")[0]).toBe("label,Actual (°F),Forecast (°F)");
    expect(csv.split("\n")[3]).toBe("+1,,53");
    const plan = planRequest("chart it then export a csv");
    expect(buildArtifacts(plan, results).csv?.filename).toBe("neural-forge-data.csv");
    expect(missingDeliverables(plan, [])).toEqual(["chart", "csv"]);
  });

  it("puts a what_if shocked path next to the simulated baseline in the CSV", () => {
    const results: ToolResult[] = [
      {
        ok: true,
        name: "simulate_world",
        summary: "",
        durationMs: 1,
        data: { unit: "°F", series: "weather", _full: { history: [{ value: 50, date: "d1" }], baseline: { p50: [51, 52], p10: [48, 49], p90: [54, 55] }, scenarios: [] } },
      },
      {
        ok: true,
        name: "what_if",
        summary: "",
        durationMs: 1,
        data: { summary: "lower", unit: "°F", shock: { mode: "abs", value: -15 }, baseline: { p50: [51, 52] }, shocked: { p50: [40, 44] } },
      },
    ];
    // A named "cold snap" needs a real simulated scenario; a generic what-if does not.
    expect(missingDeliverables(planRequest(SIM_ASK), results)).toEqual(["simulate", "compare"]);
    const plan = planRequest("run a what if on the weather and compare with normal, then give me a CSV");
    expect(missingDeliverables(plan, results)).toEqual([]);
    const csv = buildArtifacts(plan, results).csv!.text.split("\n");
    expect(csv[0]).toContain("What-if (-15) (°F)");
    expect(csv[2]).toBe("+1,,51,40,48,54");
  });
});

describe("named scenarios", () => {
  it("fills a cold snap when the model's scenario has no usable event", async () => {
    const simBodies: Array<Record<string, unknown>> = [];
    const { ai } = scriptedAi([
      '{"tool":"simulate_world","args":{"series":"weather","horizon":7,"scenarios":[{"name":"Scenario A","events":[]}]}}',
      "Normal ends near 53.8°F and the cold snap ends about 2°F lower. CSV attached.",
    ]);
    const result = await runAgentTurn(SIM_ASK, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "11.0.0.1",
      owner: false,
      models: MODELS,
      toolContext: context(SIM_ASK, simBodies),
    });
    expect(simBodies).toHaveLength(1);
    const scenarios = simBodies[0].scenarios as Array<{ name: string; events: Array<{ value: number }> }>;
    expect(scenarios[0].name).toBe("Cold snap");
    expect(scenarios[0].events[0].value).toBe(-15);
    expect(result.csv?.text).toContain("Cold snap");
  });

  it("keeps a usable model scenario as sent", async () => {
    const simBodies: Array<Record<string, unknown>> = [];
    const { ai } = scriptedAi([
      '{"tool":"simulate_world","args":{"series":"weather","horizon":7,"scenario":{"name":"Deep freeze","events":[{"kind":"shock","step":1,"value":-25}]}}}',
      "Normal ends near 53.8°F and the deep freeze ends lower. CSV attached.",
    ]);
    await runAgentTurn(SIM_ASK, "normal", [], {
      ai,
      db: new MemoryD1(),
      ip: "11.0.0.2",
      owner: false,
      models: MODELS,
      toolContext: context(SIM_ASK, simBodies),
    });
    const scenarios = simBodies[0].scenarios as Array<{ name: string }>;
    expect(scenarios[0].name).toBe("Deep freeze");
  });
});
