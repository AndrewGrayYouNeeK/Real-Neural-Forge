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

/* ---------------------------------------------------------- memory switch */

/** Memory is on unless the account turned it off in Settings. Visitors never get memory. */
export async function memoryEnabled(db: D1Like, scope: string): Promise<boolean> {
  if (!scope.startsWith("user:")) return false;
  try {
    const result = await db.prepare("SELECT enabled FROM memory_settings WHERE scope = ?").bind(scope).all();
    const row = result.results?.[0] as { enabled?: unknown } | undefined;
    if (!row) return true;
    return Number(row.enabled) === 1;
  } catch {
    return true;
  }
}

export async function setMemoryEnabled(db: D1Like, scope: string, enabled: boolean, now = new Date()): Promise<void> {
  await db
    .prepare(
      `INSERT INTO memory_settings (scope, enabled, updated_at) VALUES (?, ?, ?)
       ON CONFLICT(scope) DO UPDATE SET enabled = excluded.enabled, updated_at = excluded.updated_at`,
    )
    .bind(scope, enabled ? 1 : 0, now.toISOString())
    .run();
}

/* ------------------------------------------------------ automatic memory */

const NOT_A_NAME = new Set([
  "back", "later", "tomorrow", "today", "tonight", "when", "if", "at", "on", "in", "a", "an", "the",
  "crazy", "stupid", "dumb", "lazy", "old", "out", "up", "now", "again", "anytime", "sometime", "maybe",
  "not", "just", "so", "very", "really", "here", "there", "please", "asap", "after", "before",
]);

function titleCase(value: string): string {
  return value
    .split(/\s+/)
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : word))
    .join(" ");
}

function trimClause(value: string, max = 80): string {
  return value
    .split(/\s+(?:and|but|so|because|which|who)\s+|[,;:!?]/i)[0]
    .replace(/[.\s]+$/, "")
    .trim()
    .slice(0, max);
}

/**
 * Pull a few lasting facts out of what the user just said, without an AI call.
 * Only clear first-person statements count, so questions and quotes do not.
 */
export function extractMemoryFacts(message: string): Array<{ key: string; value: string }> {
  const text = message.replace(/\s+/g, " ").trim();
  if (!text || text.length > 600) return [];
  const facts: Array<{ key: string; value: string }> = [];
  const add = (key: string, value: string) => {
    const clean = normalizeMemoryValue(value);
    if (clean && !facts.some((fact) => fact.key === key)) facts.push({ key, value: clean });
  };

  const name = text.match(/\b(?:my name is|my name's|call me|i go by)\s+([a-z][a-z'-]{1,29})(?:\s+([a-z][a-z'-]{1,29}))?/i);
  if (name) {
    const first = name[1];
    const second = name[2];
    if (!NOT_A_NAME.has(first.toLowerCase())) {
      const looksLikeSurname = second && /^[A-Z]/.test(second) && !NOT_A_NAME.has(second.toLowerCase());
      add("name", titleCase(looksLikeSurname ? `${first} ${second}` : first));
    }
  }

  const place = text.match(/\bi (?:live in|am based in|'m based in|am from|'m from|moved to)\s+([a-z][a-z .'-]{1,50})/i);
  if (place) {
    const value = trimClause(place[1], 50);
    if (value.length >= 2 && !/^(the future|here|there|nowhere)$/i.test(value)) add("location", titleCase(value));
  }

  const units = text.match(/\b(?:use|prefer|in|switch to)\s+(metric|imperial|celsius|fahrenheit)\b/i);
  if (units && /\b(i|me|my|please|always)\b/i.test(text)) add("units", units[1].toLowerCase());

  const style = text.match(/\bi (?:prefer|like|want) (short|brief|shorter|detailed|longer|simple|simpler|step[- ]by[- ]step) (?:answers|replies|explanations|responses)\b/i);
  if (style) add("answer_style", `${style[1].toLowerCase()} answers`);

  const project = text.match(/\bi(?:'m| am) (?:working on|building|making|writing)\s+([^.!?\n]{3,120})/i);
  if (project) add("working_on", trimClause(project[1]));

  const learning = text.match(/\bi(?:'m| am) (?:learning|studying|trying to understand|trying to learn)\s+([^.!?\n]{3,120})/i);
  if (learning) add("learning", trimClause(learning[1]));

  return facts.filter((fact) => fact.value.length >= 2).slice(0, 4);
}

/** Save the facts from one message. Never throws; returns what was saved. */
export async function autoRemember(db: D1Like, scope: string, message: string): Promise<string[]> {
  const saved: string[] = [];
  for (const fact of extractMemoryFacts(message)) {
    try {
      const result = await upsertMemory(db, scope, fact.key, fact.value);
      if (result.ok) saved.push(fact.key);
    } catch {
      // Best-effort.
    }
  }
  return saved;
}

export const MAX_MEMORY_PROMPT_CHARS = 1400;

/** Memory block for the chat, kept small. Recent chat titles count as ongoing topics. */
export function compactMemoryPrompt(facts: MemoryFact[], recentTopics: string[] = []): string {
  const block = memoryPromptBlock(facts);
  const topics = recentTopics
    .map((topic) => topic.replace(/\s+/g, " ").trim().slice(0, 60))
    .filter(Boolean)
    .slice(0, 3);
  const withTopics = topics.length ? `${block}\nRecent chat topics: ${topics.join("; ")}` : block;
  return withTopics.length > MAX_MEMORY_PROMPT_CHARS ? `${withTopics.slice(0, MAX_MEMORY_PROMPT_CHARS - 1)}…` : withTopics;
}

export const MEMORY_OFF_PROMPT =
  "User memory is turned off for this account. Do not save, recall or mention stored facts, and do not call remember_fact.";
