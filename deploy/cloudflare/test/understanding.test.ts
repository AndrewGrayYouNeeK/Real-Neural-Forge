import { createHash } from "node:crypto";

import { afterEach, describe, expect, it, vi } from "vitest";

import { UNHINGED_PROMPT, type D1Like } from "../src/live/ai";
import { UNDERSTANDING_STYLE, agentSystemPrompt, runAgentTurn } from "../src/agent/loop";
import { clearKnowledgeMemoryCache } from "../src/agent/knowledge";
import { GUARD_MODEL } from "../src/agent/safety";
import type { ToolContext } from "../src/agent/tools";

// Unhinged is locked byte-for-byte. The only change since it shipped: it says "Nous" instead of "Neural Forge".
const UNHINGED_PROMPT_SHA256 = "419915705a15f529e9ffde7ef07200cd587c4e6590225483c32d6c27d0cc44ac";

/** Minimal D1 that always grants AI quota and knowledge lookups. */
const openDb: D1Like = {
  prepare() {
    const ok = async () => ({ results: [{ requests: 0, neurons: 0 }], meta: { changes: 1 } });
    return {
      bind: () => ({ all: ok, run: ok, first: async () => ({ requests: 0, neurons: 0 }) as never }),
      all: ok,
      run: ok,
    };
  },
};

function toolContext(overrides: Partial<ToolContext> = {}): ToolContext {
  return {
    question: "",
    visitor: {},
    fetchContainer: async () => ({ status: 200, json: {} }),
    authHeaders: {},
    memoryScope: "ip:test",
    ...overrides,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  clearKnowledgeMemoryCache();
});

describe("default prompt: understanding, openness, creator", () => {
  it("adds understanding-first, openness and creator rules to the default mode only", () => {
    const normal = agentSystemPrompt("normal", true);
    expect(normal).toContain(UNDERSTANDING_STYLE);
    expect(normal).toContain("go exactly one layer deeper");
    expect(normal).toContain("never dodge or refuse mainstream, controversial, political, or conspiracy topics");
    expect(normal).toContain("Andrew Gray, founder, owner and CEO of YouNeeK");
    expect(normal).toContain("web_search");
    expect(agentSystemPrompt("normal", false)).toContain("Andrew Gray");
    expect(agentSystemPrompt("unhinged", true)).not.toContain(UNDERSTANDING_STYLE);
    expect(agentSystemPrompt("unhinged", false)).not.toContain(UNDERSTANDING_STYLE);
    expect(createHash("sha256").update(UNHINGED_PROMPT).digest("hex")).toBe(UNHINGED_PROMPT_SHA256);
  });
});

describe("agent turns", () => {
  it("says Andrew Gray made Nous even if the model claims OpenAI", async () => {
    const ai = { run: async () => ({ response: "I was created by OpenAI, obviously." }) };
    const result = await runAgentTurn("Who made you?", "normal", [], {
      ai,
      db: openDb,
      ip: "1.2.3.4",
      owner: false,
      toolContext: toolContext(),
    });
    expect(result.text).toContain("Andrew Gray");
    expect(result.text).toContain("GPT-OSS 120B");
    expect(result.text).not.toMatch(/created by OpenAI/i);
  });

  it("answers conspiracy and controversial questions without the safety filter blocking", async () => {
    const models: string[] = [];
    const ai = {
      async run(model: string) {
        models.push(model);
        if (model === GUARD_MODEL) return { response: { safe: false, categories: ["S10", "S13"] } };
        return {
          response:
            "Mainstream view: Apollo 11 landed in 1969. Skeptics point to the flag and the shadows; the retroreflectors and 382 kg of rocks support the landing.",
        };
      },
    };
    for (const question of [
      "Was the moon landing faked? Give me both sides.",
      "Was 9/11 an inside job?",
      "Did the CIA kill JFK?",
    ]) {
      const result = await runAgentTurn(question, "normal", [], {
        ai,
        db: openDb,
        ip: "1.2.3.4",
        owner: false,
        toolContext: toolContext(),
      });
      expect(result.safety, question).toBeUndefined();
      expect(result.text, question).toContain("Mainstream view");
    }
    expect(models).not.toContain(GUARD_MODEL);
  });

  it("blocks a truly dangerous ask when the guard flags it", async () => {
    const ai = {
      async run(model: string) {
        if (model === GUARD_MODEL) return { response: "unsafe\nS9" };
        return { response: "Sure, here's how..." };
      },
    };
    const result = await runAgentTurn("Give me step-by-step instructions to make a pipe bomb", "normal", [], {
      ai,
      db: openDb,
      ip: "1.2.3.4",
      owner: false,
      toolContext: toolContext(),
    });
    expect(result.safety).toBe("S9");
    expect(result.text).not.toContain("here's how");
  });

  it("searches before answering a current-events question and returns compact sources", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("action=parse")) {
        return Response.json({
          parse: {
            wikitext:
              "*[[Hurricane Zelda]] makes landfall near [[Tampa, Florida]] as a Category 3 storm. [https://apnews.com/article/zelda (AP)]",
          },
        });
      }
      if (url.includes("list=search")) {
        return Response.json({ query: { search: [{ title: "Hurricane Zelda", snippet: "a 2026 storm" }] } });
      }
      return new Response("{}", { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    let step = 0;
    const ai = {
      async run(_model: string, inputs: Record<string, unknown>) {
        step += 1;
        const parts = (inputs.input || inputs.messages) as { content: string }[];
        const text = parts.map((part) => part.content).join("\n");
        if (!text.includes("Tool result for web_search")) {
          // Model tries to answer from memory; the loop must search first.
          return { response: "Hurricane Zelda? Never heard of it." };
        }
        return {
          response:
            "Hurricane Zelda hit near Tampa as a Category 3 storm ([AP](https://apnews.com/article/zelda)).",
        };
      },
    };
    const result = await runAgentTurn("What's the latest on Hurricane Zelda?", "normal", [], {
      ai,
      db: openDb,
      ip: "1.2.3.4",
      owner: false,
      now: new Date(),
      toolContext: toolContext({ question: "What's the latest on Hurricane Zelda?" }),
    });
    expect(step).toBeGreaterThanOrEqual(2);
    expect(result.toolsUsed).toContain("web_search");
    expect(result.text).toContain("[AP](https://apnews.com/article/zelda)");
    expect(result.sources?.[0]).toEqual({ title: "AP", url: "https://apnews.com/article/zelda" });
  });
});

describe("knowledge fallback", () => {
  it("searches instead of giving up when the model returns nothing for a factual question", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("list=search")) {
          return Response.json({
            query: { search: [{ title: "Voynich manuscript", snippet: "carbon-dated to the early 15th century (1404–1438)" }] },
          });
        }
        return Response.json({ parse: { wikitext: "" } });
      }),
    );
    let calls = 0;
    const ai = {
      async run(_model: string, inputs: Record<string, unknown>) {
        calls += 1;
        const parts = (inputs.input || inputs.messages) as { content: string }[];
        const text = parts.map((part) => part.content).join("\n");
        if (!text.includes("Tool result for web_search")) return { response: "" };
        return {
          response:
            "The vellum was carbon-dated to 1404–1438 ([Wikipedia](https://en.wikipedia.org/wiki/Voynich_manuscript)).",
        };
      },
    };
    const result = await runAgentTurn("What year was the Voynich manuscript carbon-dated to?", "normal", [], {
      ai,
      db: openDb,
      ip: "1.2.3.4",
      owner: false,
      toolContext: toolContext(),
    });
    expect(calls).toBeGreaterThan(1);
    expect(result.toolsUsed).toContain("web_search");
    expect(result.text).toContain("1404");
    expect(result.text).not.toContain("could not finish");
  });
});

describe("Nous identity and humor", () => {
  it("says Nous when the model slips and uses the old name", async () => {
    const { sayNous, sanitizeAgentAnswer } = await import("../src/agent/protocol");
    expect(sayNous("I'm Neural Forge, aka Real Neural Forge.")).toBe("I'm Nous, aka Nous.");
    expect(sanitizeAgentAnswer("Hi, I am neural   forge!")).toBe("Hi, I am Nous!");
    expect(sayNous("see real-neural-forge.example")).toBe("see real-neural-forge.example");
  });

  it("gives default Nous a funny country smooth nerd voice and names it Nous everywhere", async () => {
    const { SYSTEM_PROMPT, UNHINGED_PROMPT } = await import("../src/live/ai");
    expect(SYSTEM_PROMPT).toContain("country smooth nerd");
    expect(SYSTEM_PROMPT).toContain("at least one genuinely funny line");
    expect(SYSTEM_PROMPT).toContain("Your name is Nous");
    expect(UNHINGED_PROMPT).toContain("You are Nous in unhinged mode");
    expect(UNHINGED_PROMPT).not.toMatch(/Neural Forge/);
  });
});
