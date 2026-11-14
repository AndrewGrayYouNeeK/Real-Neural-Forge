import { getContainer } from "@cloudflare/containers";

import { containerEnv } from "../containerEnv";
import {
  chatMode,
  resolveAiCaps,
  resolveChatModels,
  type AiLike,
  type AiQuotaEnv,
  type ChatMode,
  type D1Like,
} from "../live/ai";
import { runAgentTurn, type HistoryMessage } from "./loop";
import {
  authHeadersFromRequest,
  visitorFromRequest,
  type ContainerFetcher,
} from "./tools";

export type AgentEnv = AiQuotaEnv & {
  AI?: AiLike;
  WORKSPACES_DB?: D1Like;
  NEURAL_FORGE_API_TOKEN?: string;
  NEURAL_FORGE_AI_MODEL?: string;
  NEURAL_FORGE_AI_FALLBACK_MODEL?: string;
  NEURAL_FORGE: Parameters<typeof getContainer>[0];
  FORGE_VOICE_ID?: string;
  NEURAL_FORGE_VOICE_ID?: string;
  [key: string]: unknown;
};

const SESSION_COOKIE = "nf_session";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}

function questionFrom(url: URL, body: Record<string, unknown> | null): string {
  if (typeof body?.question === "string") return body.question;
  return url.searchParams.get("q") || url.searchParams.get("question") || "";
}

function modeFrom(url: URL, body: Record<string, unknown> | null): ChatMode {
  const raw = typeof body?.mode === "string" ? body.mode : url.searchParams.get("mode");
  return chatMode(raw);
}

function historyFrom(body: Record<string, unknown> | null): HistoryMessage[] {
  const raw = body?.history;
  if (!Array.isArray(raw)) return [];
  const out: HistoryMessage[] = [];
  for (const item of raw.slice(-12)) {
    if (!item || typeof item !== "object") continue;
    const role = (item as { role?: unknown }).role;
    const content = (item as { content?: unknown }).content;
    if ((role !== "user" && role !== "assistant") || typeof content !== "string") continue;
    const trimmed = content.replace(/\s+/g, " ").trim().slice(0, 500);
    if (!trimmed) continue;
    out.push({ role, content: trimmed });
  }
  return out;
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

async function isOwnerRequest(request: Request, env: AgentEnv): Promise<boolean> {
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
  return (
    request.headers.get("CF-Connecting-IP") ||
    request.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() ||
    null
  );
}

function makeContainerFetcher(env: AgentEnv, request: Request): ContainerFetcher {
  // Match the dashboard warmup path: wake the same "primary" container on every
  // call via startAndWaitForPorts so sleepAfter cannot leave agent tools hanging.
  return async (path, init = {}) => {
    const container = getContainer(env.NEURAL_FORGE, "primary");
    await container.startAndWaitForPorts({
      ports: 8000,
      startOptions: { envVars: containerEnv(env) },
    });
    const headers = new Headers(init.headers || {});
    if (!headers.has("Accept")) headers.set("Accept", "application/json");
    const auth = authHeadersFromRequest(request);
    for (const [key, value] of Object.entries(auth)) {
      if (!headers.has(key)) headers.set(key, value);
    }
    headers.set("X-Forwarded-Proto", new URL(request.url).protocol.replace(/:$/, ""));
    headers.set("X-Forwarded-Host", new URL(request.url).host);
    headers.delete("X-Neural-Forge-Watch");
    const response = await container.fetch(
      new Request(`http://container${path.startsWith("/") ? path : `/${path}`}`, {
        method: init.method || "GET",
        headers,
        body: init.body,
        signal: init.signal,
      }),
    );
    const text = await response.text();
    let parsed: unknown = null;
    if (text) {
      try {
        parsed = JSON.parse(text) as unknown;
      } catch {
        parsed = { text };
      }
    }
    return { status: response.status, json: parsed };
  };
}

async function answerAgentChat(
  request: Request,
  env: AgentEnv,
  question: string,
  mode: ChatMode,
  history: HistoryMessage[],
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
  const models = resolveChatModels(env);
  const caps = resolveAiCaps(env);
  const ip = clientIp(request);
  try {
    const result = await runAgentTurn(clean, mode, history, {
      ai: env.AI,
      db: env.WORKSPACES_DB,
      ip,
      owner,
      caps,
      models,
      toolContext: {
        question: clean,
        visitor: visitorFromRequest(request),
        fetchContainer: makeContainerFetcher(env, request),
        authHeaders: authHeadersFromRequest(request),
      },
    });
    return json({
      text: result.text,
      model: true,
      model_id: result.modelId,
      voice_resting: result.voiceResting,
      voice_message: result.voiceMessage,
      tools_used: result.toolsUsed,
      capped: result.capped || false,
      reason: result.reason,
    });
  } catch {
    return json({
      text: "I could not reach the language model right now. Please try again in a moment, or ask about the weather, a market price, the moon, or space.",
    });
  }
}

export async function handleAgent(request: Request, env: AgentEnv): Promise<Response> {
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
  if (path === "/agent/chat") {
    if (request.method !== "POST" && request.method !== "GET") {
      return json({ error: "Method not allowed." }, 405);
    }
    return answerAgentChat(
      request,
      env,
      questionFrom(url, body),
      modeFrom(url, body),
      historyFrom(body),
    );
  }
  return json({ error: "Not found." }, 404);
}

