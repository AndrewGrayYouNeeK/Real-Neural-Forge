import { visitorFromCf, type CfIncoming } from "./geo";
import { answerWeather } from "./weather";
import { answerQuote } from "./markets";
import {
  CAP_MESSAGE,
  RESERVE_NEURONS,
  addExtraNeurons,
  ipScope,
  chatMode,
  runChatModel,
  tryReserve,
  type AiLike,
  type ChatMode,
  type D1Like,
} from "./ai";

export type LiveEnv = {
  AI?: AiLike;
  WORKSPACES_DB?: D1Like;
  NEURAL_FORGE_API_TOKEN?: string;
};

const SESSION_COOKIE = "nf_session";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}

function questionFrom(request: Request, url: URL, body: Record<string, unknown> | null): string {
  if (typeof body?.question === "string") return body.question;
  return url.searchParams.get("q") || url.searchParams.get("question") || "";
}

function modeFrom(url: URL, body: Record<string, unknown> | null) {
  const raw = typeof body?.mode === "string" ? body.mode : url.searchParams.get("mode");
  return chatMode(raw);
}

function factsFrom(url: URL, body: Record<string, unknown> | null): string {
  if (typeof body?.facts === "string") return body.facts;
  return url.searchParams.get("facts") || "";
}

function cookieValue(request: Request, name: string): string | null {
  const header = request.headers.get("Cookie") || "";
  for (const part of header.split(";")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const index = trimmed.indexOf("=");
    if (index < 0) continue;
    if (trimmed.slice(0, index) === name) return decodeURIComponent(trimmed.slice(index + 1));
  }
  return null;
}

function bearerToken(request: Request): string | null {
  const header = request.headers.get("Authorization") || "";
  const match = header.match(/^Bearer\s+(.+)$/i);
  return match ? match[1].trim() : null;
}

function timingEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let index = 0; index < left.length; index += 1) {
    diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return diff === 0;
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function isOwnerRequest(request: Request, env: LiveEnv): Promise<boolean> {
  const bearer = bearerToken(request);
  if (bearer && env.NEURAL_FORGE_API_TOKEN && timingEqual(bearer, env.NEURAL_FORGE_API_TOKEN)) {
    return true;
  }
  const token = bearer || cookieValue(request, SESSION_COOKIE);
  if (!token || !env.WORKSPACES_DB) return false;
  try {
    const hash = await sha256Hex(token);
    const now = new Date().toISOString();
    const result = await env.WORKSPACES_DB.prepare(
      `SELECT users.role AS role
       FROM sessions
       JOIN users ON users.id = sessions.user_id
       WHERE sessions.token_hash = ? AND sessions.expires_at > ?
       LIMIT 1`,
    )
      .bind(hash, now)
      .all();
    return result.results?.[0]?.role === "owner";
  } catch {
    return false;
  }
}

function clientIp(request: Request): string | null {
  return request.headers.get("CF-Connecting-IP") || request.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() || null;
}

async function answerAsk(
  request: Request,
  env: LiveEnv,
  question: string,
  mode: ChatMode = "normal",
  facts = "",
): Promise<Response> {
  const clean = question.replace(/\s+/g, " ").trim();
  if (!clean) {
    return json({ text: "Ask me a question and I will try to answer it." }, 400);
  }
  if (!env.AI || !env.WORKSPACES_DB) {
    return json({
      text: "General AI answers are not configured on this server yet. I can still help with live weather, market prices, the moon, and space.",
      capped: true,
    });
  }
  const owner = await isOwnerRequest(request, env);
  const reserved = await tryReserve({
    db: env.WORKSPACES_DB,
    ip: clientIp(request),
    owner,
  });
  if (!reserved.ok) {
    return json({ text: reserved.message, capped: true, reason: reserved.reason });
  }
  try {
    const result = await runChatModel(env.AI, clean, mode, facts);
    if (result.neurons > RESERVE_NEURONS) {
      const extra = result.neurons - RESERVE_NEURONS;
      await addExtraNeurons(env.WORKSPACES_DB, "global", extra);
      if (!owner) await addExtraNeurons(env.WORKSPACES_DB, await ipScope(clientIp(request)), extra);
    }
    return json({ text: result.text, model: true });
  } catch {
    return json({
      text: "I could not reach the language model right now. Please try again in a moment, or ask about the weather, a market price, the moon, or space.",
    });
  }
}

export async function handleLive(request: Request, env: LiveEnv): Promise<Response> {
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: { Allow: "GET, POST, OPTIONS" } });
  }
  let body: Record<string, unknown> | null = null;
  if (request.method === "POST") {
    const text = await request.text();
    if (text) {
      try {
        body = JSON.parse(text) as Record<string, unknown>;
      } catch {
        return json({ text: "I could not read that request." }, 400);
      }
    }
  }
  const question = questionFrom(request, url, body);
  if (path === "/live/weather") {
    if (request.method !== "GET" && request.method !== "POST") return json({ error: "Method not allowed." }, 405);
    const visitor = visitorFromCf(request.cf as CfIncoming | undefined);
    return json(await answerWeather(question, visitor));
  }
  if (path === "/live/quote") {
    if (request.method !== "GET" && request.method !== "POST") return json({ error: "Method not allowed." }, 405);
    const prefer = url.searchParams.get("kind") === "crypto" ? "crypto" : url.searchParams.get("kind") === "stock" ? "stock" : "auto";
    return json(await answerQuote(question, prefer));
  }
  if (path === "/live/ask") {
    if (request.method !== "POST" && request.method !== "GET") return json({ error: "Method not allowed." }, 405);
    return answerAsk(request, env, question, modeFrom(url, body), factsFrom(url, body));
  }
  return json({ error: "Not found." }, 404);
}

export { CAP_MESSAGE };
