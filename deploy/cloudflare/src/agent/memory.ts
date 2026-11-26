/**
 * Per-user memory for agent chat (preferences / facts in D1).
 * Scope: user:<id> for signed-in accounts (incl. the owner),
 * or the same ip:<hash> used by AI quotas for anonymous visitors.
 */

import { ipScope, type D1Like } from "../live/ai";
import { resolveSession } from "../auth/session";

export const MAX_MEMORY_FACTS = 32;
export const MAX_MEMORY_KEY_LEN = 40;
export const MAX_MEMORY_VALUE_LEN = 200;

export type MemoryFact = { key: string; value: string; updated_at: string };

export type MemoryIdentity = {
  scope: string;
  userId: string | null;
  owner: boolean;
  signedIn: boolean;
};

export function normalizeMemoryKey(raw: string): string | null {
  const key = raw.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (!key || key.length > MAX_MEMORY_KEY_LEN) return null;
  if (!/^[a-z][a-z0-9_]{0,39}$/.test(key)) return null;
  return key;
}

export function normalizeMemoryValue(raw: string): string | null {
  const value = raw.replace(/\s+/g, " ").trim();
  if (!value || value.length > MAX_MEMORY_VALUE_LEN) return null;
  return value;
}

export async function resolveMemoryIdentity(
  request: Request,
  options: { db?: D1Like; ip: string | null },
): Promise<MemoryIdentity> {
  // The shared owner token is no longer accepted. Owner = role=owner in D1
  // plus a second factor on this session (see auth/session.ts).
  const session = await resolveSession(request, options.db);
  if (session) {
    return {
      scope: `user:${session.userId}`,
      userId: session.userId,
      owner: session.owner,
      signedIn: true,
    };
  }
  const scope = await ipScope(options.ip);
  return { scope, userId: null, owner: false, signedIn: false };
}

export async function listMemory(db: D1Like, scope: string): Promise<MemoryFact[]> {
  const result = await db
    .prepare(
      `SELECT key, value, updated_at
       FROM user_memory
       WHERE scope = ?
       ORDER BY key ASC
       LIMIT ?`,
    )
    .bind(scope, MAX_MEMORY_FACTS)
    .all();
  const rows = Array.isArray(result.results) ? result.results : [];
  const out: MemoryFact[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const key = (row as { key?: unknown }).key;
    const value = (row as { value?: unknown }).value;
    const updated = (row as { updated_at?: unknown }).updated_at;
    if (typeof key !== "string" || typeof value !== "string") continue;
    out.push({
      key,
      value,
      updated_at: typeof updated === "string" ? updated : "",
    });
  }
  return out;
}

export async function upsertMemory(
  db: D1Like,
  scope: string,
  keyRaw: string,
  valueRaw: string,
  now = new Date(),
): Promise<{ ok: true; fact: MemoryFact } | { ok: false; error: string }> {
  const key = normalizeMemoryKey(keyRaw);
  const value = normalizeMemoryValue(valueRaw);
  if (!key) return { ok: false, error: "Invalid memory key." };
  if (!value) return { ok: false, error: "Invalid memory value." };
  const existing = await listMemory(db, scope);
  if (!existing.some((fact) => fact.key === key) && existing.length >= MAX_MEMORY_FACTS) {
    return { ok: false, error: `Memory is full (max ${MAX_MEMORY_FACTS} facts). Forget one first.` };
  }
  const updatedAt = now.toISOString();
  await db
    .prepare(
      `INSERT INTO user_memory (scope, key, value, updated_at)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(scope, key) DO UPDATE SET
         value = excluded.value,
         updated_at = excluded.updated_at`,
    )
    .bind(scope, key, value, updatedAt)
    .run();
  return { ok: true, fact: { key, value, updated_at: updatedAt } };
}

export async function deleteMemoryKey(db: D1Like, scope: string, keyRaw: string): Promise<boolean> {
  const key = normalizeMemoryKey(keyRaw);
  if (!key) return false;
  const result = await db
    .prepare(`DELETE FROM user_memory WHERE scope = ? AND key = ?`)
    .bind(scope, key)
    .run();
  return (result.meta?.changes || 0) > 0;
}

export async function clearMemory(db: D1Like, scope: string): Promise<number> {
  const result = await db.prepare(`DELETE FROM user_memory WHERE scope = ?`).bind(scope).run();
  return result.meta?.changes || 0;
}

export function memoryPromptBlock(facts: MemoryFact[]): string {
  if (!facts.length) {
    return (
      "User memory: none stored yet. When the user shares a lasting preference " +
      "(units, location, favorite series, name), call remember_fact. " +
      "Use recall_memory / forget_memory when they ask what you know or to clear it."
    );
  }
  const lines = facts.map((fact) => `- ${fact.key}: ${fact.value}`);
  return (
    "User memory (trusted preferences for this visitor — use them when relevant):\n" +
    lines.join("\n") +
    "\nUpdate with remember_fact / forget_memory when the user asks."
  );
}
