import { describe, expect, it } from "vitest";

import type { D1Like } from "../src/live/ai";
import { AGENT_TOOL_NAMES } from "../src/agent/protocol";
import {
  createScenario,
  deleteScenario,
  getScenarioOwned,
  listScenarios,
  scenarioRerunQuestion,
} from "../src/agent/scenarios";
import { dispatchTool, type ToolContext } from "../src/agent/tools";

type ScenarioRow = {
  id: string;
  scope: string;
  name: string;
  question: string;
  series: string | null;
  horizon: number | null;
  settings_json: string;
  created_at: string;
  updated_at: string;
};

class ScenarioTableD1 implements D1Like {
  rows = new Map<string, ScenarioRow>();

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
    if (text.startsWith("DELETE FROM saved_scenarios")) {
      const [id, scope] = values as string[];
      const row = this.rows.get(id);
      if (!row || row.scope !== scope) return { results: [], meta: { changes: 0 } };
      this.rows.delete(id);
      return { results: [], meta: { changes: 1 } };
    }
    if (text.startsWith("INSERT INTO saved_scenarios")) {
      const [id, scope, name, question, series, horizon, settingsJson, created, updated] =
        values as Array<string | number | null>;
      this.rows.set(String(id), {
        id: String(id),
        scope: String(scope),
        name: String(name),
        question: String(question),
        series: series == null ? null : String(series),
        horizon: typeof horizon === "number" ? horizon : horizon == null ? null : Number(horizon),
        settings_json: String(settingsJson),
        created_at: String(created),
        updated_at: String(updated),
      });
      return { results: [], meta: { changes: 1 } };
    }
    if (text.startsWith("SELECT") && text.includes("FROM saved_scenarios") && text.includes("WHERE id = ? AND scope = ?")) {
      const [id, scope] = values as string[];
      const row = this.rows.get(id);
      if (!row || row.scope !== scope) return { results: [], meta: { changes: 0 } };
      return { results: [row], meta: { changes: 0 } };
    }
    if (text.startsWith("SELECT") && text.includes("FROM saved_scenarios") && text.includes("WHERE scope = ?")) {
      const [scope, limit] = values as [string, number];
      const rows = [...this.rows.values()]
        .filter((row) => row.scope === scope)
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
        .slice(0, Number(limit) || 40);
      return { results: rows, meta: { changes: 0 } };
    }
    throw new Error(`unsupported sql: ${text}`);
  }
}

function toolContext(db: ScenarioTableD1, scope = "owner"): ToolContext {
  return {
    question: "forecast weather",
    visitor: {},
    fetchContainer: async (path: string) => {
      if (path.includes("/starters/weather/forecast")) {
        return {
          status: 200,
          json: {
            starter_id: "weather",
            name: "Central Park",
            unit: "°C",
            forecast: [{ step: 1, value: 15.15 }],
          },
        };
      }
      if (path.includes("/evaluation/")) {
        return { status: 200, json: { backtest: { comparison: "ok", metrics: { mae: 1 } } } };
      }
      return { status: 404, json: {} };
    },
    authHeaders: {},
    db,
    memoryScope: scope,
  };
}

describe("saved scenarios", () => {
  it("registers scenario tools", () => {
    expect(AGENT_TOOL_NAMES).toContain("save_scenario");
    expect(AGENT_TOOL_NAMES).toContain("list_scenarios");
    expect(AGENT_TOOL_NAMES).toContain("run_scenario");
  });

  it("stores and deletes scenarios in D1", async () => {
    const db = new ScenarioTableD1();
    const created = await createScenario(db, "owner", {
      name: "NYC tomorrow",
      question: "What will the weather be tomorrow?",
      series: "weather",
      horizon: 1,
    });
    expect(created.ok).toBe(true);
    if (!created.ok) return;
    expect(scenarioRerunQuestion(created.scenario)).toMatch(/weather/);
    expect(await listScenarios(db, "owner")).toHaveLength(1);
    expect((await getScenarioOwned(db, "owner", created.scenario.id))?.name).toBe("NYC tomorrow");
    expect(await deleteScenario(db, "owner", created.scenario.id)).toBe(true);
    expect(await listScenarios(db, "owner")).toEqual([]);
  });

  it("dispatches save/list/run scenario tools", async () => {
    const db = new ScenarioTableD1();
    const ctx = toolContext(db);
    const saved = await dispatchTool(
      "save_scenario",
      {
        name: "Weather sample",
        question: "Forecast weather",
        series: "weather",
        horizon: 1,
      },
      ctx,
      2_000,
      { coldStartWaitMs: 0, retryAttempts: 1 },
    );
    expect(saved.ok).toBe(true);
    const listed = await dispatchTool("list_scenarios", {}, ctx, 2_000, {
      coldStartWaitMs: 0,
      retryAttempts: 1,
    });
    expect(listed.ok).toBe(true);
    expect((listed.data as { text?: string }).text).toMatch(/Weather sample/);
    const ran = await dispatchTool(
      "run_scenario",
      { name: "Weather sample" },
      ctx,
      2_000,
      { coldStartWaitMs: 0, retryAttempts: 1 },
    );
    expect(ran.ok).toBe(true);
    expect((ran.data as { forecast?: unknown }).forecast).toBeTruthy();
  });
});
