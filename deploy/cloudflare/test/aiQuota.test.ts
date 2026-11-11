import { describe, expect, it } from "vitest";

import {
  GLOBAL_DAILY_NEURONS,
  PER_IP_DAILY_REQUESTS,
  RESERVE_NEURONS,
  SYSTEM_PROMPT,
  estimateNeurons,
  extractModelText,
  sanitizeQuestion,
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
    const prompt = SYSTEM_PROMPT.toLowerCase();
    expect(prompt).toContain("casual profanity is fine");
    expect(prompt).not.toMatch(/no (cussing|swearing|profanity)|do not swear|don't swear/);
    expect(prompt).toContain("secrets");
    expect(prompt).toContain("hate speech");
    expect(prompt).toContain("slurs");
    expect(prompt).toContain("harassment");
    expect(prompt).toContain("threats");
    expect(prompt).toContain("minors");
    expect(prompt).toContain("illegal instructions");
  });
});
