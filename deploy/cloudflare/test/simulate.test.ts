import { describe, expect, it, vi } from "vitest";

import { AGENT_TOOL_NAMES } from "../src/agent/protocol";
import { TOOL_DEFINITIONS, dispatchTool, sanitizeSimScenarios, type ToolContext } from "../src/agent/tools";

function ctxWith(fetchContainer: ToolContext["fetchContainer"]): ToolContext {
  return { question: "simulate", visitor: {}, fetchContainer, authHeaders: {} };
}

function fakePayload(horizon: number, unit: string) {
  const p50 = Array.from({ length: horizon }, (_, i) => 50 + i);
  return {
    series: "weather",
    name: "Central Park temperature",
    unit,
    storage_unit: "°C",
    horizon,
    baseline: { p10: p50.map((v) => v - 5), p50, p90: p50.map((v) => v + 5), regime: [] },
    scenarios: [
      {
        name: "Cold snap",
        events: [{ kind: "shock", step: 1, value: -15, mode: "abs" }],
        p10: p50.map((v) => v - 20),
        p50: p50.map((v) => v - 10),
        p90: p50.map((v) => v),
        delta_p50: p50.map(() => -10),
        end_delta: -10,
        end_regime: "calm (low level)",
        summary: "Cold snap: by step 14 the median is 53.0°F.",
      },
    ],
    backtest: { holdout: 14, origins: 6, mae_model: 4.4, mae_persistence: 3.7, beats_persistence: false },
    model: { description: "Annual seasonal baseline + PCA encoder", seasonal: true, regimes: [] },
    summary: "Baseline: by step 14 the median is 63.0°F.",
  };
}

describe("simulate_world tool", () => {
  it("is registered in the protocol and tool definitions", () => {
    expect(AGENT_TOOL_NAMES).toContain("simulate_world");
    expect(TOOL_DEFINITIONS.some((tool) => tool.name === "simulate_world")).toBe(true);
  });

  it("asks the container for °F on weather and returns a compact payload", async () => {
    const fetchContainer = vi.fn(async (path: string, init?: RequestInit) => {
      expect(path).toBe("/analysis/simulate");
      const body = JSON.parse(String(init?.body));
      expect(body.starter_id).toBe("weather");
      expect(body.display_unit).toBe("°F");
      expect(body.horizon).toBe(14);
      expect(body.scenarios[0].events[0]).toEqual({ kind: "shock", step: 1, value: -15, mode: "abs" });
      return { status: 200, json: fakePayload(14, "°F") };
    });
    const result = await dispatchTool(
      "simulate_world",
      {
        series: "weather",
        horizon: 14,
        scenarios: [{ name: "Cold snap", events: [{ kind: "shock", step: 1, value: -15 }] }],
      },
      ctxWith(fetchContainer),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    const data = result.data as Record<string, any>;
    expect(data.unit).toBe("°F");
    expect(data.text).toContain("°F");
    expect(data.baseline.p50).toEqual({ step_1: 50, step_7: 56, step_14: 63 });
    expect(data.scenarios[0].end_delta).toBe(-10);
    expect(JSON.stringify(data).length).toBeLessThan(3500);
  });

  it("sends custom values without a starter id", async () => {
    const fetchContainer = vi.fn(async (_path: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      expect(body.starter_id).toBeUndefined();
      expect(body.values).toHaveLength(30);
      expect(body.unit).toBe("kWh");
      return { status: 200, json: { ...fakePayload(5, "kWh"), series: "custom" } };
    });
    const values = Array.from({ length: 30 }, (_, i) => i);
    const result = await dispatchTool(
      "simulate_world",
      { values, unit: "kWh", horizon: 5 },
      ctxWith(fetchContainer),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(result.ok).toBe(true);
    expect((result.data as Record<string, unknown>).series).toBe("custom");
  });

  it("passes the JEPA encoder choice through and keeps the comparison", async () => {
    const fetchContainer = vi.fn(async (_path: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      expect(body.encoder).toBe("auto");
      expect(body.starter_id).toBe("lunar");
      return {
        status: 200,
        json: {
          ...fakePayload(7, "illumination fraction"),
          encoder_comparison: {
            winner: "jepa",
            chosen: "jepa",
            pca: { mae: 0.04 },
            jepa: { mae: 0.029 },
            improvement_pct: 27.5,
          },
        },
      };
    });
    const result = await dispatchTool(
      "simulate_world",
      { series: "lunar", horizon: 7, encoder: "AUTO" },
      ctxWith(fetchContainer),
      2000,
      { coldStartWaitMs: 0 },
    );
    const data = result.data as Record<string, any>;
    expect(data.encoder_comparison.chosen).toBe("jepa");
  });

  it("drops unknown encoders back to PCA", async () => {
    const fetchContainer = vi.fn(async (_path: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      expect(body.encoder).toBeUndefined();
      return { status: 200, json: fakePayload(7, "°F") };
    });
    await dispatchTool(
      "simulate_world",
      { series: "weather", horizon: 7, encoder: "gpu-monster" },
      ctxWith(fetchContainer),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(fetchContainer).toHaveBeenCalled();
  });

  it("refuses unknown series without calling the container", async () => {
    const fetchContainer = vi.fn();
    const result = await dispatchTool(
      "simulate_world",
      { series: "my-secret-model" },
      ctxWith(fetchContainer as unknown as ToolContext["fetchContainer"]),
      2000,
      { coldStartWaitMs: 0 },
    );
    expect(fetchContainer).not.toHaveBeenCalled();
    expect((result.data as Record<string, unknown>).ok).toBe(false);
  });

  it("clamps scenarios to the API limits", () => {
    const cleaned = sanitizeSimScenarios(
      [
        { events: [{ kind: "drift", step: 99, value: 1 }, { kind: "bogus", value: 2 }] },
        { name: "x".repeat(80), events: [{ kind: "volatility", step: 0, value: 50 }] },
        {},
        { name: "dropped" },
      ],
      10,
    );
    expect(cleaned).toHaveLength(3);
    expect(cleaned[0]).toEqual({ name: "Scenario A", events: [{ kind: "drift", step: 10, value: 1, mode: "abs" }] });
    expect((cleaned[1].name as string).length).toBe(40);
    expect(cleaned[1].events).toEqual([{ kind: "volatility", step: 1, value: 10, mode: "abs" }]);
    expect(cleaned[2].events).toEqual([]);
  });
});
