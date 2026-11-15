/**
 * Shareable public report snapshots in D1 (migration 0009).
 * Create is scoped; GET by id is public (share link).
 */

import type { D1Like } from "../live/ai";

export const MAX_REPORTS_PER_SCOPE = 40;
export const MAX_REPORT_TITLE_LEN = 120;

export type PublicReport = {
  id: string;
  series: string;
  title: string;
  snapshot: Record<string, unknown>;
  created_at: string;
};

function newId(): string {
  return `rpt_${crypto.randomUUID().replace(/-/g, "").slice(0, 20)}`;
}

function parseSnapshot(raw: unknown): Record<string, unknown> | null {
  if (typeof raw !== "string" || !raw.trim()) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // ignore
  }
  return null;
}

function rowToReport(row: Record<string, unknown>): PublicReport | null {
  const id = row.id;
  const series = row.series;
  const title = row.title;
  const snapshot = parseSnapshot(row.snapshot_json);
  if (
    typeof id !== "string" ||
    typeof series !== "string" ||
    typeof title !== "string" ||
    !snapshot
  ) {
    return null;
  }
  return {
    id,
    series,
    title,
    snapshot,
    created_at: typeof row.created_at === "string" ? row.created_at : "",
  };
}

export async function createReport(
  db: D1Like,
  scope: string,
  input: { series: string; title: string; snapshot: Record<string, unknown> },
): Promise<{ ok: true; report: PublicReport } | { ok: false; error: string }> {
  const series = input.series.trim().slice(0, 80);
  const title = input.title.replace(/\s+/g, " ").trim().slice(0, MAX_REPORT_TITLE_LEN);
  if (!series || !title) return { ok: false, error: "series and title are required." };
  const id = newId();
  const created = new Date().toISOString();
  await db
    .prepare(
      `INSERT INTO public_reports (id, scope, series, title, snapshot_json, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
    )
    .bind(id, scope, series, title, JSON.stringify(input.snapshot), created)
    .run();
  // Soft cap: keep newest N per scope.
  await db
    .prepare(
      `DELETE FROM public_reports
       WHERE scope = ?
         AND id NOT IN (
           SELECT id FROM public_reports
           WHERE scope = ?
           ORDER BY created_at DESC
           LIMIT ?
         )`,
    )
    .bind(scope, scope, MAX_REPORTS_PER_SCOPE)
    .run();
  return {
    ok: true,
    report: {
      id,
      series,
      title,
      snapshot: input.snapshot,
      created_at: created,
    },
  };
}

export async function getReport(db: D1Like, reportId: string): Promise<PublicReport | null> {
  const result = await db
    .prepare(
      `SELECT id, series, title, snapshot_json, created_at
       FROM public_reports WHERE id = ? LIMIT 1`,
    )
    .bind(reportId)
    .first();
  if (!result || typeof result !== "object") return null;
  return rowToReport(result as Record<string, unknown>);
}

export async function listReports(db: D1Like, scope: string): Promise<PublicReport[]> {
  const result = await db
    .prepare(
      `SELECT id, series, title, snapshot_json, created_at
       FROM public_reports
       WHERE scope = ?
       ORDER BY created_at DESC
       LIMIT ?`,
    )
    .bind(scope, MAX_REPORTS_PER_SCOPE)
    .all();
  const rows = Array.isArray(result.results) ? result.results : [];
  const out: PublicReport[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const parsed = rowToReport(row as Record<string, unknown>);
    if (parsed) out.push(parsed);
  }
  return out;
}
