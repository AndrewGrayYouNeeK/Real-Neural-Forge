import { describe, expect, it } from "vitest";

import {
  GLOBAL_DAILY_NEURONS,
  FALLBACK_MODEL,
  FREE_DAILY_NEURONS,
  PER_IP_DAILY_REQUESTS,
  PRIMARY_MODEL,
  RESERVE_NEURONS,
  SAFETY_DAILY_NEURONS,
  SYSTEM_PROMPT,
  UNHINGED_PROMPT,
  UNHINGED_TEMPERATURE,
  chatInputs,
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
    expect(extractModelText({ output_text: "The high is 72°F." })).toBe("The high is 72°F.");
    expect(
      extractModelText({
        output_text: "Need to respond.\nHello. What can I help with?",
        output: [
          {
            type: "reasoning",
            content: [{ type: "reasoning_text", text: "Need to respond." }],
          },
          {
            type: "message",
            content: [{ type: "output_text", text: "Hello. What can I help with?" }],
          },
        ],
      }),
    ).toBe("Hello. What can I help with?");
    expect(
      extractModelText({ response: "Need to respond.\nThe high is 72°F." }),
    ).toBe("The high is 72°F.");
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
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("witty");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("dry humor");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("answer first");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("answer exactly what was asked");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("lead with the answer");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("secondary");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("casual cussing is fine");
    const unhinged = UNHINGED_PROMPT.toLowerCase();
    expect(unhinged).toContain("foul-mouthed comedian");
    expect(unhinged).toContain("cuss a lot");
    expect(unhinged).toContain("funny joke in every reply");
    expect(unhinged).toContain("every reply must cuss a lot");
    for (const word of ["fuck", "shit", "damn", "ass", "hell"]) {
      expect(unhinged).toContain(word);
    }
    expect(unhinged).toContain("no slurs");
    expect(unhinged).toContain("protected group");
    expect(unhinged).toContain("not a threat");
    expect(unhinged).toContain("trash-talk and roast");
    expect(unhinged).toContain("their data");
    expect(unhinged).toContain("bad forecasts");
    expect(unhinged).toContain("forecast numbers");
    expect(unhinged).toContain("confidence");
    expect(unhinged).toContain("user-facing answer");
    expect(unhinged).toContain("i told my calendar to fuck off");
    expect(SYSTEM_PROMPT.toLowerCase()).toContain("user-facing answer");
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
        const messages = (inputs.input || inputs.messages) as { content: string }[];
        seen.push(messages[0].content);
        return { response: "ok" };
      },
    };
    await runChatModel(ai, "hi");
    await runChatModel(ai, "hi", "unhinged");
    expect(seen).toEqual([SYSTEM_PROMPT, UNHINGED_PROMPT]);
    const prompts: string[] = [];
    const temperatures: unknown[] = [];
    const rewriting = {
      async run(_model: string, inputs: Record<string, unknown>) {
        const messages = (inputs.input || inputs.messages) as { content: string }[];
        prompts.push(messages[1].content);
        temperatures.push(inputs.temperature);
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
    expect(prompts[0]).toContain("fuck");
    expect(prompts[0].toLowerCase()).toContain("joke");
    expect(temperatures).toEqual([undefined]);
    expect(userContent("hey", "   ")).toBe("hey");
    expect(sanitizeFacts(123)).toBe("");
  });

  it("uses gpt-oss until that call fails, then the 8B model", async () => {
    const seen: string[] = [];
    const ai = {
      async run(model: string) {
        seen.push(model);
        if (model === PRIMARY_MODEL) throw new Error("capacity");
        return { response: "still here" };
      },
    };
    const result = await runChatModel(ai, "hi");
    expect(seen).toEqual([PRIMARY_MODEL, FALLBACK_MODEL]);
    expect(result.modelId).toBe(FALLBACK_MODEL);
    const primary = chatInputs(PRIMARY_MODEL, UNHINGED_PROMPT, "hey", "unhinged");
    expect(primary.temperature).toBeUndefined();
    expect(primary.messages).toBeUndefined();
    expect(primary.input).toEqual([
      { role: "system", content: UNHINGED_PROMPT },
      { role: "user", content: "hey" },
    ]);
    expect(primary.reasoning).toEqual({ effort: "low" });
    expect(primary.max_output_tokens).toBe(160);
    const fallback = chatInputs(FALLBACK_MODEL, UNHINGED_PROMPT, "hey", "unhinged");
    expect(fallback.temperature).toBe(UNHINGED_TEMPERATURE);
    expect(fallback.messages).toBeTruthy();
    const logged: string[] = [];
    const original = console.error;
    console.error = (line?: unknown) => {
      logged.push(String(line));
    };
    try {
      const failing = {
        async run(model: string) {
          if (model === PRIMARY_MODEL) {
            const error = new Error("responses rejected") as Error & { status?: number };
            error.status = 400;
            throw error;
          }
          return { response: "from 8b" };
        },
      };
      const rescued = await runChatModel(failing, "hey", "unhinged");
      expect(rescued.modelId).toBe(FALLBACK_MODEL);
      expect(rescued.text).toBe("from 8b");
      expect(logged.join("\n")).toContain("responses rejected");
      expect(logged.join("\n")).toContain("400");
      expect(logged.join("\n")).not.toContain("hey");
    } finally {
      console.error = original;
    }
    expect(UNHINGED_PROMPT).toContain("foul-mouthed comedian");
    expect(UNHINGED_PROMPT).toContain("cuss a lot");
    expect(UNHINGED_PROMPT).toContain("User: hey");
    expect(UNHINGED_PROMPT).toContain("I told my calendar to fuck off");
    expect(SAFETY_DAILY_NEURONS).toBe(25000);
    expect(FREE_DAILY_NEURONS).toBe(10000);
    expect(GLOBAL_DAILY_NEURONS).toBe(SAFETY_DAILY_NEURONS);
  });
});
