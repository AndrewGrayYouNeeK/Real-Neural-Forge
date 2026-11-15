/**
 * In-app alert rules + fired events in D1 (migration 0010).
 * Threshold or K-Means anomaly checks; no email/SMS.
 */

import type { D1Like } from "../live/ai";

export const MAX_ALERT_RULES = 40;
export const MAX_ALERT_EVENTS = 100;
export const MAX_ALERT_NAME_LEN = 80;

export type AlertKind = "threshold" | "anomaly";

export type AlertRule = {
  id: string;
  name: string;
  series: string;
  kind: AlertKind;
  config: Record<string, unknown>;
  enabled: boolean;
  created_at: string;
  updated_at: string;
};

export type AlertEvent = {
  id: string;
  rule_id: string;
  fired_at: string;
  payload: Record<string, unknown>;
  acknowledged: boolean;
};

function newId(prefix: string): string {
  return `${prefix}${crypto.randomUUID().replace(/-/g, "").slice(0, 20)}`;
}

function parseJsonObject(raw: unknown): Record<string, unknown> {
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

function rowToRule(row: Record<string, unknown>): AlertRule | null {
  const id = row.id;
  const name = row.name;
  const series = row.series;
  const kind = row.kind;
  if (
    typeof id !== "string" ||
    typeof name !== "string" ||
    typeof series !== "string" ||
    (kind !== "threshold" && kind !== "anomaly")
  ) {
    return null;
  }
  return {
    id,
    name,
    series,
    kind,
    config: parseJsonObject(row.config_json),
    enabled: Number(row.enabled) !== 0,
    created_at: typeof row.created_at === "string" ? row.created_at : "",
    updated_at: typeof row.updated_at === "string" ? row.updated_at : "",
  };
}

function rowToEvent(row: Record<string, unknown>): AlertEvent | null {
  const id = row.id;
  const ruleId = row.rule_id;
  if (typeof id !== "string" || typeof ruleId !== "string") return null;
  return {
    id,
    rule_id: ruleId,
    fired_at: typeof row.fired_at === "string" ? row.fired_at : "",
    payload: parseJsonObject(row.payload_json),
    acknowledged: Number(row.acknowledged) !== 0,
  };
}

export async function listAlertRules(db: D1Like, scope: string): Promise<AlertRule[]> {
  const result = await db
    .prepare(
      `SELECT id, name, series, kind, config_json, enabled, created_at, updated_at
       FROM alert_rules WHERE scope = ?
       ORDER BY updated_at DESC LIMIT ?`,
    )
    .bind(scope, MAX_ALERT_RULES)
    .all();
  const rows = Array.isArray(result.results) ? result.results : [];
  const out: AlertRule[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const parsed = rowToRule(row as Record<string, unknown>);
    if (parsed) out.push(parsed);
  }
  return out;
}

export async function listAlertEvents(db: D1Like, scope: string): Promise<AlertEvent[]> {
  const result = await db
    .prepare(
      `SELECT id, rule_id, fired_at, payload_json, acknowledged
       FROM alert_events WHERE scope = ?
       ORDER BY fired_at DESC LIMIT ?`,
    )
    .bind(scope, MAX_ALERT_EVENTS)
    .all();
  const rows = Array.isArray(result.results) ? result.results : [];
  const out: AlertEvent[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const parsed = rowToEvent(row as Record<string, unknown>);
    if (parsed) out.push(parsed);
  }
  return out;
}

export async function createAlertRule(
  db: D1Like,
  scope: string,
  input: {
    name: string;
    series: string;
    kind: AlertKind;
    config: Record<string, unknown>;
  },
): Promise<{ ok: true; rule: AlertRule } | { ok: false; error: string }> {
  const name = input.name.replace(/\s+/g, " ").trim().slice(0, MAX_ALERT_NAME_LEN);
  const series = input.series.trim().slice(0, 80);
  if (!name || !series) return { ok: false, error: "name and series are required." };
  if (input.kind !== "threshold" && input.kind !== "anomaly") {
    return { ok: false, error: "kind must be threshold or anomaly." };
  }
  const id = newId("alr_");
  const now = new Date().toISOString();
  await db
    .prepare(
      `INSERT INTO alert_rules
       (id, scope, name, series, kind, config_json, enabled, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)`,
    )
    .bind(id, scope, name, series, input.kind, JSON.stringify(input.config || {}), now, now)
    .run();
  await db
    .prepare(
      `DELETE FROM alert_rules
       WHERE scope = ?
         AND id NOT IN (
           SELECT id FROM alert_rules WHERE scope = ?
           ORDER BY updated_at DESC LIMIT ?
         )`,
    )
    .bind(scope, scope, MAX_ALERT_RULES)
    .run();
  return {
    ok: true,
    rule: {
      id,
      name,
      series,
      kind: input.kind,
      config: input.config || {},
      enabled: true,
      created_at: now,
      updated_at: now,
    },
  };
}

export async function deleteAlertRule(
  db: D1Like,
  scope: string,
  ruleId: string,
): Promise<boolean> {
  await db
    .prepare(`DELETE FROM alert_events WHERE scope = ? AND rule_id = ?`)
    .bind(scope, ruleId)
    .run();
  const result = await db
    .prepare(`DELETE FROM alert_rules WHERE scope = ? AND id = ?`)
    .bind(scope, ruleId)
    .run();
  return Number((result as { meta?: { changes?: number } }).meta?.changes || 0) > 0 || true;
}

export async function insertAlertEvent(
  db: D1Like,
  scope: string,
  ruleId: string,
  payload: Record<string, unknown>,
): Promise<AlertEvent> {
  const id = newId("ale_");
  const fired = new Date().toISOString();
  await db
    .prepare(
      `INSERT INTO alert_events (id, scope, rule_id, fired_at, payload_json, acknowledged)
       VALUES (?, ?, ?, ?, ?, 0)`,
    )
    .bind(id, scope, ruleId, fired, JSON.stringify(payload))
    .run();
  await db
    .prepare(
      `DELETE FROM alert_events
       WHERE scope = ?
         AND id NOT IN (
           SELECT id FROM alert_events WHERE scope = ?
           ORDER BY fired_at DESC LIMIT ?
         )`,
    )
    .bind(scope, scope, MAX_ALERT_EVENTS)
    .run();
  return { id, rule_id: ruleId, fired_at: fired, payload, acknowledged: false };
}

export async function acknowledgeAlertEvent(
  db: D1Like,
  scope: string,
  eventId: string,
): Promise<boolean> {
  await db
    .prepare(
      `UPDATE alert_events SET acknowledged = 1
       WHERE scope = ? AND id = ?`,
    )
    .bind(scope, eventId)
    .run();
  return true;
}

export function thresholdFired(
  latest: number,
  config: Record<string, unknown>,
): { fired: boolean; message: string } {
  const value = Number(config.value);
  const op = typeof config.op === "string" ? config.op : ">";
  if (!Number.isFinite(latest) || !Number.isFinite(value)) {
    return { fired: false, message: "Invalid threshold comparison." };
  }
  let fired = false;
  if (op === ">") fired = latest > value;
  else if (op === "<") fired = latest < value;
  else if (op === ">=") fired = latest >= value;
  else if (op === "<=") fired = latest <= value;
  else if (op === "abs>") fired = Math.abs(latest) > value;
  else fired = latest > value;
  return {
    fired,
    message: fired
      ? `Latest value ${latest} crossed threshold ${op} ${value}.`
      : `Latest value ${latest} is within threshold ${op} ${value}.`,
  };
}
