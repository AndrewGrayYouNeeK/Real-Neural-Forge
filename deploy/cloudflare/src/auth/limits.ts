/** Failed-login lockout stored in D1 (auth_attempts). */

import type { D1Like } from "../live/ai";

export const WINDOW_SECONDS = 15 * 60;
export const LOCK_SECONDS = 15 * 60;
export const EMAIL_LIMIT = 5;
export const IP_LIMIT = 20;
export const MFA_LIMIT = 5;

export async function lockedFor(db: D1Like, keys: string[], now: Date = new Date()): Promise<number> {
  let longest = 0;
  for (const key of keys) {
    const row = await db
      .prepare("SELECT locked_until FROM auth_attempts WHERE key = ?")
      .bind(key)
      .first<{ locked_until?: string | null }>();
    const until = row?.locked_until ? Date.parse(row.locked_until) : NaN;
    if (Number.isFinite(until) && until > now.getTime()) {
      longest = Math.max(longest, Math.ceil((until - now.getTime()) / 1000));
    }
  }
  return longest;
}

export async function recordFailure(db: D1Like, key: string, limit: number, now: Date = new Date()): Promise<void> {
  const row = await db
    .prepare("SELECT failures, first_at FROM auth_attempts WHERE key = ?")
    .bind(key)
    .first<{ failures?: number; first_at?: string }>();
  const first = row?.first_at ? Date.parse(row.first_at) : NaN;
  const inWindow = Number.isFinite(first) && now.getTime() - first < WINDOW_SECONDS * 1000;
  const failures = inWindow ? Number(row?.failures || 0) + 1 : 1;
  const firstAt = inWindow ? (row?.first_at as string) : now.toISOString();
  const lockedUntil = failures >= limit ? new Date(now.getTime() + LOCK_SECONDS * 1000).toISOString() : null;
  await db
    .prepare(
      `INSERT INTO auth_attempts (key, failures, first_at, locked_until) VALUES (?, ?, ?, ?)
       ON CONFLICT(key) DO UPDATE SET failures = excluded.failures,
         first_at = excluded.first_at, locked_until = excluded.locked_until`,
    )
    .bind(key, failures, firstAt, lockedUntil)
    .run();
}

export async function clearFailures(db: D1Like, key: string): Promise<void> {
  await db.prepare("DELETE FROM auth_attempts WHERE key = ?").bind(key).run();
}

export function lockedResponse(seconds: number): Response {
  const minutes = Math.max(1, Math.ceil(seconds / 60));
  return Response.json(
    { detail: `Too many attempts. Try again in about ${minutes} minute${minutes === 1 ? "" : "s"}.`, locked: true },
    { status: 429, headers: { "Retry-After": String(seconds), "Cache-Control": "no-store" } },
  );
}
