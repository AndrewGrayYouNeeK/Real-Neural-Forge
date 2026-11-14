/**
 * Persistent chat threads + messages in D1 (migration 0007).
 * Same scope keys as user memory (user:/owner/ip:).
 */

import type { D1Like } from "../live/ai";
import type { HistoryMessage } from "./loop";

export const MAX_CHAT_THREADS = 40;
export const MAX_CHAT_TITLE_LEN = 80;
export const MAX_CHAT_MESSAGE_LEN = 4_000;
export const MAX_CHAT_HISTORY_FOR_MODEL = 12;

export type ChatThreadSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

function newId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, "").slice(0, 20)}`;
}

export function normalizeChatTitle(raw: string): string {
  const title = raw.replace(/\s+/g, " ").trim().slice(0, MAX_CHAT_TITLE_LEN);
  return title || "New chat";
}

export function normalizeChatContent(raw: string): string | null {
  const content = raw.replace(/\s+/g, " ").trim().slice(0, MAX_CHAT_MESSAGE_LEN);
  return content || null;
}

export async function listThreads(db: D1Like, scope: string): Promise<ChatThreadSummary[]> {
  const result = await db
    .prepare(
      `SELECT id, title, created_at, updated_at
       FROM chat_threads
       WHERE scope = ?
       ORDER BY updated_at DESC
       LIMIT ?`,
    )
    .bind(scope, MAX_CHAT_THREADS)
    .all();
  const rows = Array.isArray(result.results) ? result.results : [];
  const out: ChatThreadSummary[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const id = (row as { id?: unknown }).id;
    const title = (row as { title?: unknown }).title;
    const created = (row as { created_at?: unknown }).created_at;
    const updated = (row as { updated_at?: unknown }).updated_at;
    if (typeof id !== "string" || typeof title !== "string") continue;
    out.push({
      id,
      title,
      created_at: typeof created === "string" ? created : "",
      updated_at: typeof updated === "string" ? updated : "",
    });
  }
  return out;
}

export async function getThreadOwned(
  db: D1Like,
  scope: string,
  threadId: string,
): Promise<ChatThreadSummary | null> {
  const result = await db
    .prepare(
      `SELECT id, title, created_at, updated_at
       FROM chat_threads
       WHERE id = ? AND scope = ?
       LIMIT 1`,
    )
    .bind(threadId, scope)
    .all();
  const row = result.results?.[0] as
    | { id?: unknown; title?: unknown; created_at?: unknown; updated_at?: unknown }
    | undefined;
  if (!row || typeof row.id !== "string" || typeof row.title !== "string") return null;
  return {
    id: row.id,
    title: row.title,
    created_at: typeof row.created_at === "string" ? row.created_at : "",
    updated_at: typeof row.updated_at === "string" ? row.updated_at : "",
  };
}

export async function listMessages(db: D1Like, threadId: string): Promise<ChatMessage[]> {
  const result = await db
    .prepare(
      `SELECT id, role, content, created_at
       FROM chat_messages
       WHERE thread_id = ?
       ORDER BY created_at ASC
       LIMIT 200`,
    )
    .bind(threadId)
    .all();
  const rows = Array.isArray(result.results) ? result.results : [];
  const out: ChatMessage[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const id = (row as { id?: unknown }).id;
    const role = (row as { role?: unknown }).role;
    const content = (row as { content?: unknown }).content;
    const created = (row as { created_at?: unknown }).created_at;
    if (typeof id !== "string" || typeof content !== "string") continue;
    if (role !== "user" && role !== "assistant") continue;
    out.push({
      id,
      role,
      content,
      created_at: typeof created === "string" ? created : "",
    });
  }
  return out;
}

export async function createThread(
  db: D1Like,
  scope: string,
  titleRaw: string,
  now = new Date(),
): Promise<ChatThreadSummary> {
  const id = newId("chat");
  const title = normalizeChatTitle(titleRaw);
  const stamp = now.toISOString();
  await db
    .prepare(
      `INSERT INTO chat_threads (id, scope, title, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?)`,
    )
    .bind(id, scope, title, stamp, stamp)
    .run();
  return { id, title, created_at: stamp, updated_at: stamp };
}

export async function touchThread(db: D1Like, threadId: string, now = new Date()): Promise<void> {
  await db
    .prepare(`UPDATE chat_threads SET updated_at = ? WHERE id = ?`)
    .bind(now.toISOString(), threadId)
    .run();
}

export async function appendMessage(
  db: D1Like,
  threadId: string,
  role: "user" | "assistant",
  contentRaw: string,
  now = new Date(),
): Promise<ChatMessage | null> {
  const content = normalizeChatContent(contentRaw);
  if (!content) return null;
  const id = newId("msg");
  const stamp = now.toISOString();
  await db
    .prepare(
      `INSERT INTO chat_messages (id, thread_id, role, content, created_at)
       VALUES (?, ?, ?, ?, ?)`,
    )
    .bind(id, threadId, role, content, stamp)
    .run();
  await touchThread(db, threadId, now);
  return { id, role, content, created_at: stamp };
}

export async function deleteThread(db: D1Like, scope: string, threadId: string): Promise<boolean> {
  const owned = await getThreadOwned(db, scope, threadId);
  if (!owned) return false;
  await db.prepare(`DELETE FROM chat_messages WHERE thread_id = ?`).bind(threadId).run();
  await db.prepare(`DELETE FROM chat_threads WHERE id = ? AND scope = ?`).bind(threadId, scope).run();
  return true;
}

/** Ensure a thread id for this scope; create one titled from the question when missing. */
export async function ensureThreadForTurn(
  db: D1Like,
  scope: string,
  threadId: string | null,
  question: string,
): Promise<ChatThreadSummary> {
  if (threadId) {
    const existing = await getThreadOwned(db, scope, threadId);
    if (existing) return existing;
  }
  return createThread(db, scope, question);
}

export function historyFromMessages(messages: ChatMessage[]): HistoryMessage[] {
  const prior = messages.slice(0, -1); // drop the latest user turn (current question)
  const out: HistoryMessage[] = [];
  for (const item of prior.slice(-MAX_CHAT_HISTORY_FOR_MODEL)) {
    out.push({ role: item.role, content: item.content.slice(0, 500) });
  }
  return out;
}
