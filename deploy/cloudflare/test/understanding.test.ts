import { createHash } from "node:crypto";

import { afterEach, describe, expect, it, vi } from "vitest";

import { UNHINGED_PROMPT, type D1Like } from "../src/live/ai";
import { UNDERSTANDING_STYLE, agentSystemPrompt, runAgentTurn } from "../src/agent/loop";
import { clearKnowledgeMemoryCache } from "../src/agent/knowledge";
import { GUARD_MODEL } from "../src/agent/safety";
import type { ToolContext } from "../src/agent/tools";

const UNHINGED_PROMPT_SHA256 = "dc377e2d40781f22efb30a773dca8a6a598817e7464b3c35efedc5263ec6992b";

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
    expect(normal).toContain("Andrew Gray (YouNeeK)");
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
