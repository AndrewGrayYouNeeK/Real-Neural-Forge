/**
 * Owner-only training-example collector for the Nous LoRA.
 *
 * Every route needs a D1 session with role=owner AND a fresh second factor
 * (session.owner), except the export, which also accepts a read-only bearer
 * token (NOUS_TRAINING_EXPORT_TOKEN) for the weekly GitHub backup.
 *
 *   GET    /owner/training/status           {owner: true, counts}
 *   GET    /owner/training/examples?label=good|rejected|all
 *   POST   /owner/training/examples         {messages, source, tags?, label?, pair_id?}
 *   PUT    /owner/training/examples/:id     {messages?, tags?}
 *   DELETE /owner/training/examples/:id
 *   GET    /owner/training/export?label=good|rejected   (JSONL download)
 *
 * Good examples export in the same chat format as training/nous_chat.jsonl.
 * Rejected ("That was bad") examples export separately and are never mixed in.
 */

import type { D1Like } from "../live/ai";
import { resolveSession } from "../auth/session";
import { json } from "../auth/util";
import { TRAINING_SYSTEM_PROMPT } from "./trainingPrompt";

export type TrainingEnv = { WORKSPACES_DB?: D1Like; NOUS_TRAINING_EXPORT_TOKEN?: string };

export type Turn = { role: "user" | "assistant"; content: string };
export type Label = "good" | "rejected";
export type Source = "save" | "fix" | "manual" | "bad";

export type TrainingExample = {
  id: string;
  messages: Turn[];
  source: Source;
  label: Label;
  tags: string[];
  pair_id: string | null;
  created_at: string;
  updated_at: string;
};

export const SOURCES: Source[] = ["save", "fix", "manual", "bad"];
export const MAX_TURNS = 20;
export const MAX_CONTENT_CHARS = 8000;
export const MAX_TAGS = 10;
export const MAX_EXAMPLES = 10_000;

const SECRET_PATTERNS = [
  /sk-[A-Za-z0-9_-]{16,}/,
  /ghp_[A-Za-z0-9]{20,}/,
  /gh[osu]_[A-Za-z0-9]{20,}/,
  /github_pat_[A-Za-z0-9_]{20,}/,
  /AKIA[0-9A-Z]{16}/,
  /xox[baprs]-[A-Za-z0-9-]{10,}/,
  /Bearer\s+[A-Za-z0-9._-]{20,}/i,
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
  /\b[a-f0-9]{40,}\b/,
  /\bsk_(live|test)_[A-Za-z0-9]{10,}/,
  /\bwhsec_[A-Za-z0-9]{10,}/,
];
const REASONING_MARKERS = /<\/?(think|analysis)>/gi;

export function looksSecret(text: string): boolean {
  return SECRET_PATTERNS.some((pattern) => pattern.test(text));
}

export function normalizeTags(raw: unknown): string[] {
  const list = Array.isArray(raw) ? raw : typeof raw === "string" ? raw.split(/[,\s]+/) : [];
  const out: string[] = [];
  for (const item of list) {
    if (typeof item !== "string") continue;
    const tag = item.trim().toLowerCase().replace(/^#/, "").replace(/[^a-z0-9-]/g, "").slice(0, 24);
    if (tag && !out.includes(tag)) out.push(tag);
    if (out.length >= MAX_TAGS) break;
  }
  return out;
}

/** Turns must alternate user/assistant, start with user and end with assistant. */
export function validateMessages(raw: unknown): { ok: true; messages: Turn[] } | { ok: false; detail: string } {
  if (!Array.isArray(raw) || raw.length < 2) return { ok: false, detail: "Add a question and an answer." };
  if (raw.length > MAX_TURNS * 2) return { ok: false, detail: `Keep it to ${MAX_TURNS} turns or fewer.` };
  const messages: Turn[] = [];
  for (let index = 0; index < raw.length; index += 1) {
    const item = raw[index] as { role?: unknown; content?: unknown } | null;
    const want = index % 2 === 0 ? "user" : "assistant";
    if (!item || item.role !== want) {
      return { ok: false, detail: "Turns must go question, answer, question, answer… and end with an answer." };
    }
    const content = typeof item.content === "string" ? item.content.replace(REASONING_MARKERS, "").replace(/\r\n/g, "\n").trim() : "";
    if (!content) return { ok: false, detail: want === "user" ? "A question is empty." : "An answer is empty." };
    if (content.length > MAX_CONTENT_CHARS) return { ok: false, detail: `Keep each turn under ${MAX_CONTENT_CHARS} characters.` };
    if (looksSecret(content)) return { ok: false, detail: "That looks like it has a key or token in it, so it wasn't saved." };
    messages.push({ role: want, content });
  }
  if (messages[messages.length - 1].role !== "assistant") return { ok: false, detail: "End with an answer." };
  return { ok: true, messages };
}

function parseRow(row: Record<string, unknown>): TrainingExample | null {
  try {
    const messages = JSON.parse(String(row.messages || "[]")) as Turn[];
    const tags = JSON.parse(String(row.tags || "[]")) as string[];
    return {
      id: String(row.id),
      messages: Array.isArray(messages) ? messages : [],
      source: (SOURCES.includes(row.source as Source) ? row.source : "manual") as Source,
      label: row.label === "rejected" ? "rejected" : "good",
      tags: Array.isArray(tags) ? tags : [],
      pair_id: typeof row.pair_id === "string" && row.pair_id ? row.pair_id : null,
      created_at: String(row.created_at || ""),
      updated_at: String(row.updated_at || ""),
    };
  } catch {
    return null;
  }
}

function labelFilter(value: string | null): Label | "all" {
  return value === "good" || value === "rejected" ? value : "all";
}

export async function listExamples(db: D1Like, label: Label | "all"): Promise<TrainingExample[]> {
  const sql =
    label === "all"
      ? "SELECT * FROM training_examples ORDER BY created_at DESC LIMIT ?"
      : "SELECT * FROM training_examples WHERE label = ? ORDER BY created_at DESC LIMIT ?";
  const result = label === "all" ? await db.prepare(sql).bind(MAX_EXAMPLES).all() : await db.prepare(sql).bind(label, MAX_EXAMPLES).all();
  return (result.results || []).map(parseRow).filter((row): row is TrainingExample => row !== null);
}

export async function countExamples(db: D1Like): Promise<{ good: number; rejected: number; total: number }> {
  const result = await db.prepare("SELECT label, COUNT(*) AS n FROM training_examples GROUP BY label").all();
  let good = 0;
  let rejected = 0;
  for (const row of result.results || []) {
    if (row.label === "rejected") rejected += Number(row.n || 0);
    else good += Number(row.n || 0);
  }
  return { good, rejected, total: good + rejected };
}

async function getExample(db: D1Like, id: string): Promise<TrainingExample | null> {
  const row = await db.prepare("SELECT * FROM training_examples WHERE id = ?").bind(id).first<Record<string, unknown>>();
  return row ? parseRow(row) : null;
}

export async function createExample(
  db: D1Like,
  body: { messages?: unknown; source?: unknown; tags?: unknown; label?: unknown; pair_id?: unknown },
  now = new Date(),
): Promise<{ ok: true; example: TrainingExample } | { ok: false; detail: string; status: number }> {
  const checked = validateMessages(body.messages);
  if (!checked.ok) return { ok: false, detail: checked.detail, status: 400 };
  const source: Source = SOURCES.includes(body.source as Source) ? (body.source as Source) : "manual";
  // A thumbs-down is always rejected; everything else is good unless asked.
  const label: Label = source === "bad" || body.label === "rejected" ? "rejected" : "good";
  const tags = normalizeTags(body.tags);
  const { total } = await countExamples(db);
  if (total >= MAX_EXAMPLES) return { ok: false, detail: "The example list is full. Export and delete some first.", status: 409 };
  const id = crypto.randomUUID();
  const stamp = now.toISOString();
  let pairId: string | null = null;
  if (typeof body.pair_id === "string" && /^[0-9a-f-]{36}$/.test(body.pair_id)) {
    const other = await getExample(db, body.pair_id);
    if (other) pairId = other.id;
  }
  await db
    .prepare(
      "INSERT INTO training_examples (id, messages, source, label, tags, pair_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(id, JSON.stringify(checked.messages), source, label, JSON.stringify(tags), pairId, stamp, stamp)
    .run();
  // Link a fix and the bad answer it replaces both ways.
  if (pairId) {
    await db.prepare("UPDATE training_examples SET pair_id = ?, updated_at = ? WHERE id = ? AND pair_id IS NULL").bind(id, stamp, pairId).run();
  }
  const example = await getExample(db, id);
  return example ? { ok: true, example } : { ok: false, detail: "Could not save.", status: 500 };
}

export async function updateExample(
  db: D1Like,
  id: string,
  body: { messages?: unknown; tags?: unknown },
  now = new Date(),
): Promise<{ ok: true; example: TrainingExample } | { ok: false; detail: string; status: number }> {
  const current = await getExample(db, id);
  if (!current) return { ok: false, detail: "Not found.", status: 404 };
  let messages = current.messages;
  if (body.messages !== undefined) {
    const checked = validateMessages(body.messages);
    if (!checked.ok) return { ok: false, detail: checked.detail, status: 400 };
    messages = checked.messages;
  }
  const tags = body.tags !== undefined ? normalizeTags(body.tags) : current.tags;
  await db
    .prepare("UPDATE training_examples SET messages = ?, tags = ?, updated_at = ? WHERE id = ?")
    .bind(JSON.stringify(messages), JSON.stringify(tags), now.toISOString(), id)
    .run();
  const example = await getExample(db, id);
  return example ? { ok: true, example } : { ok: false, detail: "Not found.", status: 404 };
}

export async function deleteExample(db: D1Like, id: string): Promise<boolean> {
  const result = await db.prepare("DELETE FROM training_examples WHERE id = ?").bind(id).run();
  await db.prepare("UPDATE training_examples SET pair_id = NULL WHERE pair_id = ?").bind(id).run();
  return Number(result.meta?.changes || 0) > 0;
}

/** One JSON object per line, oldest first, same chat format as training/nous_chat.jsonl. */
export function toJsonl(examples: TrainingExample[], label: Label): string {
  const rows = examples
    .filter((example) => example.label === label)
    .sort((a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id))
    .map((example) => {
      const messages = [{ role: "system", content: TRAINING_SYSTEM_PROMPT }, ...example.messages];
      const line: Record<string, unknown> = { messages };
      if (label === "rejected") line.label = "rejected";
      line.id = example.id;
      if (example.tags.length) line.tags = example.tags;
      if (example.pair_id) line.pair_id = example.pair_id;
      return JSON.stringify(line);
    });
  return rows.length ? `${rows.join("\n")}\n` : "";
}

async function sameSecret(given: string, secret: string): Promise<boolean> {
  const enc = new TextEncoder();
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(given)),
    crypto.subtle.digest("SHA-256", enc.encode(secret)),
  ]);
  const left = new Uint8Array(a);
  const right = new Uint8Array(b);
  let diff = 0;
  for (let index = 0; index < left.length; index += 1) diff |= left[index] ^ right[index];
  return diff === 0;
}

/** Bearer token for the weekly backup. Only ever grants the read-only export. */
async function exportTokenOk(request: Request, secret: string | undefined): Promise<boolean> {
  if (!secret || secret.length < 32) return false;
  const match = /^Bearer\s+(\S+)$/i.exec(request.headers.get("Authorization") || "");
  if (!match || !match[1].startsWith("nte_")) return false;
  return sameSecret(match[1], secret);
}

async function readBody(request: Request): Promise<Record<string, unknown> | null> {
  try {
    const body = (await request.json()) as unknown;
    return body && typeof body === "object" && !Array.isArray(body) ? (body as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

const PREFIX = "/owner/training/";

export async function handleTrainingRoute(request: Request, env: TrainingEnv): Promise<Response | null> {
  const url = new URL(request.url);
  if (!url.pathname.startsWith(PREFIX)) return null;
  const db = env.WORKSPACES_DB;
  if (!db) return json({ detail: "Not found." }, 404);
  const path = url.pathname.slice(PREFIX.length);

  const isExport = path === "export" && request.method === "GET";
  // Session tokens (also accepted as Bearer) never look like nte_…, so a
  // backup token is checked first and can only ever reach the export.
  const viaToken = isExport && (await exportTokenOk(request, env.NOUS_TRAINING_EXPORT_TOKEN));
  if (!viaToken) {
    const session = await resolveSession(request, db);
    if (!session) return json({ detail: "Sign in to continue.", auth_required: true }, 401);
    // Members and owner sessions without a fresh second factor look the same.
    if (!session.owner) return json({ detail: "Owner only." }, 403);
  }

  if (isExport) {
    const label: Label = url.searchParams.get("label") === "rejected" ? "rejected" : "good";
    const body = toJsonl(await listExamples(db, label), label);
    const name = label === "rejected" ? "rejected.jsonl" : "examples.jsonl";
    return new Response(body, {
      headers: {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Content-Disposition": `attachment; filename="${name}"`,
        "Cache-Control": "no-store",
      },
    });
  }

  if (path === "status" && request.method === "GET") {
    return json({ owner: true, counts: await countExamples(db) });
  }

  if (path === "examples") {
    if (request.method === "GET") {
      const label = labelFilter(url.searchParams.get("label"));
      const [examples, counts] = await Promise.all([listExamples(db, label), countExamples(db)]);
      return json({ examples, counts });
    }
    if (request.method === "POST") {
      const body = await readBody(request);
      if (!body) return json({ detail: "Expected JSON." }, 400);
      const result = await createExample(db, body);
      if (!result.ok) return json({ detail: result.detail }, result.status);
      return json({ ok: true, example: result.example, counts: await countExamples(db) }, 201);
    }
    return json({ detail: "Method not allowed." }, 405);
  }

  const match = /^examples\/([0-9a-f-]{36})$/.exec(path);
  if (match) {
    const id = match[1];
    if (request.method === "PUT" || request.method === "PATCH") {
      const body = await readBody(request);
      if (!body) return json({ detail: "Expected JSON." }, 400);
      const result = await updateExample(db, id, body);
      if (!result.ok) return json({ detail: result.detail }, result.status);
      return json({ ok: true, example: result.example });
    }
    if (request.method === "DELETE") {
      const removed = await deleteExample(db, id);
      if (!removed) return json({ detail: "Not found." }, 404);
      return json({ ok: true, counts: await countExamples(db) });
    }
    return json({ detail: "Method not allowed." }, 405);
  }
  return json({ detail: "Not found." }, 404);
}
