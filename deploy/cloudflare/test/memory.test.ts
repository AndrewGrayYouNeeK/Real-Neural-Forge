import { describe, expect, it } from "vitest";

import type { D1Like } from "../src/live/ai";
import {
  MAX_MEMORY_FACTS,
  clearMemory,
  deleteMemoryKey,
  listMemory,
  memoryPromptBlock,
  normalizeMemoryKey,
  normalizeMemoryValue,
  upsertMemory,
} from "../src/agent/memory";
import { AGENT_TOOL_NAMES } from "../src/agent/protocol";
import { dispatchTool, type ToolContext } from "../src/agent/tools";

type FactRow = { scope: string; key: string; value: string; updated_at: string };

class MemoryTableD1 implements D1Like {
  facts = new Map<string, FactRow>();

  private id(scope: string, key: string): string {
    return `${scope}\0${key}`;
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
    if (text.startsWith("SELECT key, value, updated_at FROM user_memory")) {
      const [scope, limit] = values as [string, number];
      const rows = [...this.facts.values()]
        .filter((row) => row.scope === scope)
        .sort((a, b) => a.key.localeCompare(b.key))
        .slice(0, Number(limit) || MAX_MEMORY_FACTS)
        .map((row) => ({ key: row.key, value: row.value, updated_at: row.updated_at }));
      return { results: rows, meta: { changes: 0 } };
    }
    if (text.startsWith("INSERT INTO user_memory")) {
      const [scope, key, value, updatedAt] = values as string[];
      this.facts.set(this.id(scope, key), {
        scope,
        key,
        value,
        updated_at: updatedAt,
      });
      return { results: [], meta: { changes: 1 } };
    }
    if (text.startsWith("DELETE FROM user_memory WHERE scope = ? AND key = ?")) {
      const [scope, key] = values as string[];
      const id = this.id(scope, key);
      const had = this.facts.delete(id);
      return { results: [], meta: { changes: had ? 1 : 0 } };
    }
    if (text.startsWith("DELETE FROM user_memory WHERE scope = ?")) {
      const [scope] = values as string[];
      let changes = 0;
      for (const [id, row] of this.facts) {
        if (row.scope === scope) {
          this.facts.delete(id);
          changes += 1;
        }
      }
      return { results: [], meta: { changes } };
    }
    throw new Error(`unsupported sql: ${text}`);
  }
}

function toolContext(db: MemoryTableD1, scope = "user:alice"): ToolContext {
  return {
    question: "remember my units",
    visitor: {},
    fetchContainer: async () => ({ status: 200, json: {} }),
    authHeaders: {},
    db,
    memoryScope: scope,
  };
}

describe("user memory", () => {
  it("registers memory tools in the agent catalog", () => {
    expect(AGENT_TOOL_NAMES).toContain("remember_fact");
    expect(AGENT_TOOL_NAMES).toContain("recall_memory");
    expect(AGENT_TOOL_NAMES).toContain("forget_memory");
  });

  it("normalizes keys and values", () => {
    expect(normalizeMemoryKey("Favorite Series")).toBe("favorite_series");
    expect(normalizeMemoryKey("1bad")).toBeNull();
    expect(normalizeMemoryValue("  Austin, TX  ")).toBe("Austin, TX");
    expect(normalizeMemoryValue("")).toBeNull();
  });

  it("stores, lists, and clears facts in D1", async () => {
    const db = new MemoryTableD1();
    const saved = await upsertMemory(db, "owner", "units", "°F");
    expect(saved.ok).toBe(true);
    await upsertMemory(db, "owner", "location", "Austin");
    const facts = await listMemory(db, "owner");
    expect(facts.map((row) => row.key)).toEqual(["location", "units"]);
    expect(memoryPromptBlock(facts)).toContain("units: °F");
    expect(await deleteMemoryKey(db, "owner", "units")).toBe(true);
    expect(await clearMemory(db, "owner")).toBe(1);
    expect(await listMemory(db, "owner")).toEqual([]);
  });

  it("dispatches remember/recall/forget tools for owner scope", async () => {
    const db = new MemoryTableD1();
    const ctx = toolContext(db, "owner");
    const remembered = await dispatchTool(
      "remember_fact",
      { key: "favorite_series", value: "weather" },
      ctx,
      2_000,
      { coldStartWaitMs: 0, retryAttempts: 1 },
    );
    expect(remembered.ok).toBe(true);
    expect((remembered.data as { text?: string }).text).toMatch(/favorite_series/);

    const recalled = await dispatchTool("recall_memory", {}, ctx, 2_000, {
      coldStartWaitMs: 0,
      retryAttempts: 1,
    });
    expect(recalled.ok).toBe(true);
    expect((recalled.data as { text?: string }).text).toMatch(/weather/);

    const cleared = await dispatchTool(
      "forget_memory",
      { clear_all: true },
      ctx,
      2_000,
      { coldStartWaitMs: 0, retryAttempts: 1 },
    );
    expect(cleared.ok).toBe(true);
    expect(await listMemory(db, "owner")).toEqual([]);
  });
});
