import { describe, expect, it } from "vitest";

import { type D1Like } from "../src/live/ai";
import { quickForecast } from "../src/vision/forecast";
import {
  VISION_MODEL,
  VISION_OWNER_DAILY,
  VISION_VISITOR_DAILY,
  answerVision,
  claimVision,
  normalizeSeries,
  parseImageDataUrl,
  parseVisionOutput,
  resolveVisionCaps,
  visionFacts,
  visionInputs,
} from "../src/vision/vision";

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
    if (text.startsWith("SELECT requests")) {
      const [day, scope] = values as string[];
      const row = this.rows.get(this.key(day, scope));
      return {
        results: row ? [{ requests: row.requests, neurons: row.neurons }] : [],
        meta: { changes: 0 },
      };
    }
    if (text.startsWith("UPDATE ai_usage SET requests = requests + 1 WHERE")) {
      const [day, scope, max] = values as [string, string, number];
      const row = this.rows.get(this.key(day, scope));
      if (!row || row.requests + 1 > Number(max)) return { results: [], meta: { changes: 0 } };
      row.requests += 1;
      return { results: [], meta: { changes: 1 } };
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

const PNG = `data:image/png;base64,${"iVBORw0KGgo".padEnd(400, "A")}`;
const NOW = new Date("2026-11-21T12:00:00Z");

const CHART_JSON = JSON.stringify({
  kind: "chart",
  title: "Louisville daily high (°F)",
  description: "A line chart of daily highs.",
  series: {
    name: "daily high",
    unit: "°F",
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    values: [61, 64, 58, 55, 60, 66],
  },
});

function fakeAi(visionText = CHART_JSON) {
  const calls: Array<{ model: string; inputs: Record<string, unknown> }> = [];
  return {
    calls,
    async run(model: string, inputs: Record<string, unknown>) {
      calls.push({ model, inputs });
      if (model === VISION_MODEL) {
        return {
          choices: [{ message: { content: visionText } }],
          usage: { prompt_tokens: 400, completion_tokens: 120, neurons: 7 },
        };
      }
      return {
        output: [{ type: "message", content: [{ type: "output_text", text: "Highs ran 55°F to 66°F; next about 63°F." }] }],
        usage: { prompt_tokens: 300, completion_tokens: 40 },
      };
    },
  };
}

describe("image input", () => {
  it("accepts small data URLs and rejects others", () => {
    expect(parseImageDataUrl(PNG).ok).toBe(true);
    expect(parseImageDataUrl("data:text/plain;base64,QUJD").ok).toBe(false);
    expect(parseImageDataUrl("https://example.com/a.png").ok).toBe(false);
    const huge = `data:image/jpeg;base64,${"A".repeat(2_100_000)}`;
    const result = parseImageDataUrl(huge);
    expect(result.ok).toBe(false);
  });

  it("sends an OpenAI-style image part to the vision model", () => {
    const inputs = visionInputs(PNG, "what is this");
    const messages = inputs.messages as Array<{ content: Array<Record<string, unknown>> }>;
    expect(messages[0].content[1]).toEqual({ type: "image_url", image_url: { url: PNG } });
  });
});

describe("parsing", () => {
  it("normalizes °F series to °C for the API", () => {
    const read = parseVisionOutput({ choices: [{ message: { content: "```json\n" + CHART_JSON + "\n```" } }] });
    expect(read?.series?.unit).toBe("°C");
    expect(read?.series?.source_unit).toBe("°F");
    expect(read?.series?.values[0]).toBeCloseTo(16.111, 2);
    expect(read?.series?.labels).toHaveLength(6);
  });

  it("reads Llama-style parsed response objects and strings with units", () => {
    const read = parseVisionOutput({
      response: { kind: "chart", title: "BTC", description: "Price", series: { name: "price", unit: "$", values: ["$1,200", "1,250", "x", 1300] } },
    });
    expect(read?.series?.values).toEqual([1200, 1250, 1300]);
    expect(read?.series?.unit).toBe("$");
  });

  it("keeps a photo description without a series", () => {
    const read = parseVisionOutput({ choices: [{ message: { content: '{"kind":"photo","description":"A dog on a beach.","series":null}' } }] });
    expect(read?.series).toBeNull();
    expect(read?.description).toContain("dog");
  });

  it("drops series with fewer than two numbers", () => {
    expect(normalizeSeries({ values: [1] })).toBeNull();
  });
});

describe("facts and forecast", () => {
  it("shows temperatures in °F to the user", () => {
    const read = parseVisionOutput({ choices: [{ message: { content: CHART_JSON } }] })!;
    const forecast = quickForecast(read.series!.values);
    const facts = visionFacts(read, forecast);
    expect(facts).toContain("61°F");
    expect(facts).toContain("66°F");
    expect(facts).not.toMatch(/°C/);
  });

  it("forecasts within a sane band of the series", () => {
    const values = [16.1, 17.8, 14.4, 12.8, 15.6, 18.9, 21.1, 20, 17.2, 15];
    const forecast = quickForecast(values)!;
    expect(forecast.horizon).toBeGreaterThanOrEqual(3);
    for (const step of forecast.steps) {
      expect(step.p10).toBeLessThanOrEqual(step.p50);
      expect(step.p90).toBeGreaterThanOrEqual(step.p50);
      expect(step.p50).toBeGreaterThan(5);
      expect(step.p50).toBeLessThan(30);
    }
    expect(quickForecast([1, 2, 3])).toBeNull();
  });
});

describe("caps", () => {
  it("defaults keep owner above visitors", () => {
    const caps = resolveVisionCaps();
    expect(caps.visitor).toBe(VISION_VISITOR_DAILY);
    expect(caps.owner).toBe(VISION_OWNER_DAILY);
    expect(caps.owner).toBeGreaterThan(caps.visitor);
  });

  it("stops a visitor after the daily image cap and lets the owner continue", async () => {
    const db = new MemoryD1();
    const caps = resolveVisionCaps();
    const visitor = { db, owner: false, ip: "1.2.3.4", identityScope: "ip:x", caps, now: NOW };
    for (let index = 0; index < caps.visitor; index += 1) {
      expect((await claimVision(visitor)).ok).toBe(true);
    }
    const blocked = await claimVision(visitor);
    expect(blocked.ok).toBe(false);
    if (!blocked.ok) expect(blocked.reason).toBe("visitor");
    const owner = await claimVision({ ...visitor, owner: true, identityScope: "owner" });
    expect(owner.ok).toBe(true);
  });

  it("applies the shared public cap", async () => {
    const db = new MemoryD1();
    const caps = { visitor: 5, public: 2, owner: 10 };
    const base = { db, owner: false, identityScope: "ip:x", caps, now: NOW };
    expect((await claimVision({ ...base, ip: "1.1.1.1" })).ok).toBe(true);
    expect((await claimVision({ ...base, ip: "2.2.2.2" })).ok).toBe(true);
    const third = await claimVision({ ...base, ip: "3.3.3.3" });
    expect(third.ok).toBe(false);
    if (!third.ok) expect(third.reason).toBe("public");
  });
});

describe("answerVision", () => {
  it("reads a chart, forecasts it, and replies through 120B", async () => {
    const db = new MemoryD1();
    const ai = fakeAi();
    const result = await answerVision(
      { ai, db, owner: false, ip: "5.5.5.5", identityScope: "ip:y", env: {}, chatModels: {}, now: NOW },
      PNG,
      "where is this heading?",
      "normal",
    );
    expect(result.status).toBe(200);
    expect(ai.calls[0].model).toBe(VISION_MODEL);
    expect(ai.calls[1].model).toBe("@cf/openai/gpt-oss-120b");
    expect(result.body.text).toContain("63°F");
    const display = result.body.display_series as { unit: string; values: number[] };
    expect(display.unit).toBe("°F");
    expect(display.values[0]).toBe(61);
    expect(result.body.forecast).toBeTruthy();
    expect(result.body.tools_used).toEqual(["read_image"]);
    const global = [...db.rows.values()].find((row) => row.scope === "global");
    expect(global?.neurons).toBeGreaterThan(0);
  });

  it("rejects a bad image without spending", async () => {
    const db = new MemoryD1();
    const ai = fakeAi();
    const result = await answerVision(
      { ai, db, owner: false, ip: "5.5.5.5", identityScope: "ip:y", env: {}, chatModels: {}, now: NOW },
      "nope",
      "",
      "normal",
    );
    expect(result.status).toBe(400);
    expect(ai.calls).toHaveLength(0);
    expect(db.rows.size).toBe(0);
  });

  it("returns a capped message once the visitor cap is used", async () => {
    const db = new MemoryD1();
    const ai = fakeAi();
    const deps = {
      ai,
      db,
      owner: false,
      ip: "6.6.6.6",
      identityScope: "ip:z",
      env: { NEURAL_FORGE_VISION_VISITOR_DAILY: "1" },
      chatModels: {},
      now: NOW,
    };
    expect((await answerVision(deps, PNG, "", "normal")).body.capped).toBeUndefined();
    const second = await answerVision(deps, PNG, "", "normal");
    expect(second.body.capped).toBe(true);
    expect(ai.calls).toHaveLength(2);
  });
});
