import { describe, expect, it, vi } from "vitest";

import {
  FALLBACK_MODEL,
  PRIMARY_MODEL,
  SYSTEM_PROMPT,
  UNHINGED_PROMPT,
  tryReserve,
  type D1Like,
} from "../src/live/ai";
import {
  agentSystemPrompt,
  buildAgentInputs,
  isSmallTalk,
  runAgentTurn,
} from "../src/agent/loop";
import {
  collectToolCalls,
  parseJsonToolProtocol,
  sanitizeAgentAnswer,
  extractNativeToolCalls,
} from "../src/agent/protocol";
import { MAX_TOOL_CALLS, dispatchTool, type ToolContext } from "../src/agent/tools";

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

  it("sanitizes planning and tool JSON from agent output", () => {
    const cleaned = sanitizeAgentAnswer(
      "Need to respond with the forecast\nHigh is 72°F.\n{\"tool\":\"forecast\",\"args\":{\"series\":\"weather\"}}",
    );
    expect(cleaned).toBe("High is 72°F.");
    expect(cleaned).not.toContain("tool");
    expect(cleaned.toLowerCase()).not.toContain("need to respond");
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

  it("dispatches forecast to the container starter path", async () => {
    const fetchContainer = vi.fn(async (path: string) => {
      if (path.includes("/starters/weather/forecast")) {
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
      if (path.includes("/evaluation/weather/backtest")) {
        return {
          status: 200,
          json: {
            horizon: 1,
            n: 20,
            metrics: { mae: 1.2, rmse: 1.5, mape: 0.1, directional_accuracy: 0.7 },
            comparison: "10.0% better than the best simple baseline",
            best_baseline: "naive",
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

  it("times out slow tools", async () => {
    const result = await dispatchTool(
      "get_evaluation_metrics",
      { model_id: "weather" },
      toolContext({
        fetchContainer: async () =>
          new Promise((resolve) => {
            setTimeout(() => resolve({ status: 200, json: {} }), 50);
          }),
      }),
      5,
    );
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/timeout/i);
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
          return { response: "from 8b after primary failed" };
        },
      };
      const result = await runAgentTurn("How accurate is weather?", "normal", [], {
        ai,
        db,
        ip: "4.4.4.4",
        owner: false,
        toolContext: toolContext(),
        models: { primary: PRIMARY_MODEL, fallback: FALLBACK_MODEL },
      });
      expect(result.modelId).toBe(FALLBACK_MODEL);
      expect(result.text).toContain("8b");
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

  it("answers without tools when the model returns plain text", async () => {
    const db = new MemoryD1();
    const ai = {
      async run() {
        return { response: "Need to respond\nHigh of 70°F in Dallas." };
      },
    };
    const result = await runAgentTurn("weather in Dallas?", "normal", [], {
      ai,
      db,
      ip: "9.9.9.9",
      owner: false,
      toolContext: toolContext(),
    });
    expect(result.text).toBe("High of 70°F in Dallas.");
    expect(result.toolsUsed).toEqual([]);
    expect(result.capped).toBeFalsy();
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
    expect(MAX_TOOL_CALLS).toBe(4);
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
