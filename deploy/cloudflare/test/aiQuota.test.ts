import { describe, expect, it } from "vitest";

import {
  GLOBAL_DAILY_NEURONS,
  PER_IP_DAILY_REQUESTS,
  RESERVE_NEURONS,
  SYSTEM_PROMPT,
  UNHINGED_PROMPT,
  chatMode,
  estimateNeurons,
  extractModelText,
  runChatModel,
  sanitizeFacts,
  sanitizeQuestion,
  systemPromptFor,
  userContent,
  tryReserve,
  type D1Like,
} from "../src/live/ai";

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

describe("AI quota", () => {
  it("allows a first request and blocks after the global cap", async () => {
    const db = new MemoryD1();
    const now = new Date("2026-11-10T12:00:00Z");
    const first = await tryReserve({ db, ip: "1.1.1.1", owner: false, now });
    expect(first.ok).toBe(true);
    const global = [...db.rows.values()].find((row) => row.scope === "global");
    expect(global?.neurons).toBe(RESERVE_NEURONS);
    global!.neurons = GLOBAL_DAILY_NEURONS;
    const blocked = await tryReserve({ db, ip: "8.8.8.8", owner: true, now });
    expect(blocked.ok).toBe(false);
    if (!blocked.ok) expect(blocked.reason).toBe("global");
  });

  it("enforces a per-IP cap but lets the owner skip it within the global cap", async () => {
    const db = new MemoryD1();
    const now = new Date("2026-11-10T12:00:00Z");
    for (let index = 0; index < PER_IP_DAILY_REQUESTS; index += 1) {
      const result = await tryReserve({ db, ip: "9.9.9.9", owner: false, now });
      expect(result.ok).toBe(true);
    }
    const visitor = await tryReserve({ db, ip: "9.9.9.9", owner: false, now });
    expect(visitor.ok).toBe(false);
    if (!visitor.ok) expect(visitor.reason).toBe("ip");
    const owner = await tryReserve({ db, ip: "9.9.9.9", owner: true, now });
    expect(owner.ok).toBe(true);
  });

  it("sanitizes prompts and estimates neurons conservatively", () => {
    expect(sanitizeQuestion("  hello\n\nworld  ".repeat(80)).length).toBeLessThanOrEqual(500);
    expect(estimateNeurons(300, 150)).toBeGreaterThan(0);
    expect(extractModelText({ response: " George Washington " })).toBe("George Washington");
  });

  it("allows casual profanity and still forbids harm and secrets", () => {
    for (const prompt of [SYSTEM_PROMPT, UNHINGED_PROMPT]) {
      const text = prompt.toLowerCase();
      expect(text).not.toMatch(/no (cussing|swearing|profanity)|do not swear|don't swear/);
      expect(text).toContain("secrets");
      expect(text).toContain("hate speech");
      expect(text).toContain("slurs");
      expect(text).toContain("harass real people");
      expect(text).toContain("threats");
      expect(text).toContain("minors");
      expect(text).toContain("illegal instructions");
    }
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("casual profanity is fine");
    const unhinged = UNHINGED_PROMPT.toLowerCase();
    expect(unhinged).toContain("foul-mouthed comedian");
    expect(unhinged).toContain("cuss a lot");
    expect(unhinged).toContain("funny joke in most replies");
    expect(unhinged).toContain("trash-talk and roast");
    expect(unhinged).toContain("their data");
    expect(unhinged).toContain("bad forecasts");
    expect(unhinged).toContain("forecast numbers");
    expect(unhinged).toContain("confidence");
  });

  it("defaults chat mode to normal and only switches the prompt when asked", async () => {
    expect(chatMode(undefined)).toBe("normal");
    expect(chatMode("loud")).toBe("normal");
    expect(chatMode("unhinged")).toBe("unhinged");
    expect(systemPromptFor("normal")).toBe(SYSTEM_PROMPT);
    expect(systemPromptFor("unhinged")).toBe(UNHINGED_PROMPT);
    const seen: string[] = [];
    const ai = {
      async run(_model: string, inputs: Record<string, unknown>) {
        const messages = inputs.messages as { content: string }[];
        seen.push(messages[0].content);
        return { response: "ok" };
      },
    };
    await runChatModel(ai, "hi");
    await runChatModel(ai, "hi", "unhinged");
    expect(seen).toEqual([SYSTEM_PROMPT, UNHINGED_PROMPT]);
    const prompts: string[] = [];
    const rewriting = {
      async run(_model: string, inputs: Record<string, unknown>) {
        const messages = inputs.messages as { content: string }[];
        prompts.push(messages[1].content);
        return { response: "still 72°F" };
      },
    };
    const rewritten = await runChatModel(
      rewriting,
      "hey",
      "unhinged",
      "The high is 72°F with 80% confidence.",
    );
    expect(rewritten.text).toBe("still 72°F");
    expect(prompts[0]).toContain("72°F");
    expect(prompts[0]).toContain("80%");
    expect(prompts[0]).toContain("exactly as written");
    expect(userContent("hey", "   ")).toBe("hey");
    expect(sanitizeFacts(123)).toBe("");
  });
});
