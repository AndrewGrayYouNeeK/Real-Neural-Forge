/**
 * Saved forecast/question scenarios in D1 (migration 0008).
 * Same scope keys as user memory / chat threads.
 */

import type { D1Like } from "../live/ai";

export const MAX_SCENARIOS = 40;
export const MAX_SCENARIO_NAME_LEN = 80;
export const MAX_SCENARIO_QUESTION_LEN = 500;

export type SavedScenario = {
  id: string;
  name: string;
  question: string;
  series: string | null;
  horizon: number | null;
  settings: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

function newId(): string {
  return `scn_${crypto.randomUUID().replace(/-/g, "").slice(0, 20)}`;
}

export function normalizeScenarioName(raw: string): string | null {
  const name = raw.replace(/\s+/g, " ").trim().slice(0, MAX_SCENARIO_NAME_LEN);
  return name || null;
}

export function normalizeScenarioQuestion(raw: string): string | null {
  const question = raw.replace(/\s+/g, " ").trim().slice(0, MAX_SCENARIO_QUESTION_LEN);
  return question || null;
}

function parseSettings(raw: unknown): Record<string, unknown> {
  if (typeof raw !== "string" || !raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // ignore
  }
  return {};
}

function rowToScenario(row: Record<string, unknown>): SavedScenario | null {
  const id = row.id;
  const name = row.name;
  const question = row.question;
  if (typeof id !== "string" || typeof name !== "string" || typeof question !== "string") {
    return null;
  }
  const series = typeof row.series === "string" && row.series.trim() ? row.series.trim() : null;
  const horizonRaw = row.horizon;
  const horizon =
    typeof horizonRaw === "number" && Number.isFinite(horizonRaw)
      ? Math.min(24, Math.max(1, Math.floor(horizonRaw)))
      : null;
  return {
    id,
    name,
    question,
    series,
    horizon,
    settings: parseSettings(row.settings_json),
    created_at: typeof row.created_at === "string" ? row.created_at : "",
    updated_at: typeof row.updated_at === "string" ? row.updated_at : "",
  };
}

export async function listScenarios(db: D1Like, scope: string): Promise<SavedScenario[]> {
  const result = await db
    .prepare(
      `SELECT id, name, question, series, horizon, settings_json, created_at, updated_at
       FROM saved_scenarios
       WHERE scope = ?
       ORDER BY updated_at DESC
       LIMIT ?`,
    )
    .bind(scope, MAX_SCENARIOS)
    .all();
  const rows = Array.isArray(result.results) ? result.results : [];
  const out: SavedScenario[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const parsed = rowToScenario(row as Record<string, unknown>);
    if (parsed) out.push(parsed);
  }
  return out;
}

export async function getScenarioOwned(
  db: D1Like,
  scope: string,
  scenarioId: string,
): Promise<SavedScenario | null> {
  const result = await db
    .prepare(
      `SELECT id, name, question, series, horizon, settings_json, created_at, updated_at
       FROM saved_scenarios
       WHERE id = ? AND scope = ?
       LIMIT 1`,
    )
    .bind(scenarioId, scope)
    .all();
  const row = result.results?.[0];
  if (!row || typeof row !== "object") return null;
  return rowToScenario(row as Record<string, unknown>);
}

export async function createScenario(
  db: D1Like,
  scope: string,
  input: {
    name: string;
    question: string;
    series?: string | null;
    horizon?: number | null;
    settings?: Record<string, unknown>;
  },
  now = new Date(),
): Promise<{ ok: true; scenario: SavedScenario } | { ok: false; error: string }> {
  const name = normalizeScenarioName(input.name);
  const question = normalizeScenarioQuestion(input.question);
  if (!name) return { ok: false, error: "Scenario name is required." };
  if (!question) return { ok: false, error: "Scenario question is required." };
  const existing = await listScenarios(db, scope);
  if (existing.length >= MAX_SCENARIOS) {
    return { ok: false, error: `Scenario limit reached (max ${MAX_SCENARIOS}).` };
  }
  const id = newId();
  const stamp = now.toISOString();
  const series =
    typeof input.series === "string" && input.series.trim() ? input.series.trim().slice(0, 64) : null;
  const horizon =
    typeof input.horizon === "number" && Number.isFinite(input.horizon)
      ? Math.min(24, Math.max(1, Math.floor(input.horizon)))
      : null;
  const settings = input.settings && typeof input.settings === "object" ? input.settings : {};
  await db
    .prepare(
      `INSERT INTO saved_scenarios
         (id, scope, name, question, series, horizon, settings_json, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .bind(id, scope, name, question, series, horizon, JSON.stringify(settings), stamp, stamp)
    .run();
  return {
    ok: true,
    scenario: {
      id,
      name,
      question,
      series,
      horizon,
      settings,
      created_at: stamp,
      updated_at: stamp,
    },
  };
}

export async function deleteScenario(db: D1Like, scope: string, scenarioId: string): Promise<boolean> {
  const owned = await getScenarioOwned(db, scope, scenarioId);
  if (!owned) return false;
  await db
    .prepare(`DELETE FROM saved_scenarios WHERE id = ? AND scope = ?`)
    .bind(scenarioId, scope)
    .run();
  return true;
}

/** Build a chat question that re-runs the scenario with its stored settings. */
export function scenarioRerunQuestion(scenario: SavedScenario): string {
  if (scenario.series) {
    const horizon = scenario.horizon && scenario.horizon > 1 ? ` for ${scenario.horizon} steps` : "";
    return `Run my saved scenario “${scenario.name}”: forecast ${scenario.series}${horizon}.`;
  }
  return scenario.question;
}
