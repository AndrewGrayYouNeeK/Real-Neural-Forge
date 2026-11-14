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
  clearMemory,
  deleteMemoryKey,
  listMemory,
  memoryPromptBlock,
  resolveMemoryIdentity,
  upsertMemory,
} from "./memory";
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

function clientIp(request: Request): string | null {
  return (
    request.headers.get("CF-Connecting-IP") ||
    request.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() ||
    null
  );
}

async function identityFor(
  request: Request,
  env: AgentEnv,
): Promise<Awaited<ReturnType<typeof resolveMemoryIdentity>>> {
  return resolveMemoryIdentity(request, {
    apiToken: env.NEURAL_FORGE_API_TOKEN,
    db: env.WORKSPACES_DB,
    ip: clientIp(request),
  });
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
  waitUntil?: (promise: Promise<unknown>) => void,
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
  const identity = await identityFor(request, env);
  const models = resolveChatModels(env);
  const caps = resolveAiCaps(env);
  const ip = clientIp(request);
  let memoryPrompt = memoryPromptBlock([]);
  try {
    const facts = await listMemory(env.WORKSPACES_DB, identity.scope);
    memoryPrompt = memoryPromptBlock(facts);
  } catch {
    memoryPrompt = memoryPromptBlock([]);
  }
  try {
    const result = await runAgentTurn(clean, mode, history, {
      ai: env.AI,
      db: env.WORKSPACES_DB,
      ip,
      owner: identity.owner,
      caps,
      models,
      memoryPrompt,
      toolContext: {
        question: clean,
        visitor: visitorFromRequest(request),
        fetchContainer: makeContainerFetcher(env, request),
        authHeaders: authHeadersFromRequest(request),
        db: env.WORKSPACES_DB,
        waitUntil,
        memoryScope: identity.scope,
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
      memory_scope: identity.signedIn ? (identity.owner ? "owner" : "user") : "visitor",
    });
  } catch {
    return json({
      text: "I could not reach the language model right now. Please try again in a moment, or ask about the weather, a market price, the moon, or space.",
    });
  }
}

async function handleMemory(
  request: Request,
  env: AgentEnv,
  path: string,
): Promise<Response> {
  if (!env.WORKSPACES_DB) {
    return json({ error: "Memory store is not configured." }, 503);
  }
  const identity = await identityFor(request, env);
  const db = env.WORKSPACES_DB;
  const keyPath = path.replace(/^\/agent\/memory\/?/, "");

  if (request.method === "GET" && !keyPath) {
    const facts = await listMemory(db, identity.scope);
    return json({
      scope: identity.signedIn ? (identity.owner ? "owner" : "user") : "visitor",
      signed_in: identity.signedIn,
      facts,
    });
  }

  if (request.method === "PUT" && !keyPath) {
    let body: Record<string, unknown> = {};
    try {
      const text = await request.text();
      if (text) body = JSON.parse(text) as Record<string, unknown>;
    } catch {
      return json({ error: "Invalid JSON." }, 400);
    }
    const key = typeof body.key === "string" ? body.key : "";
    const value = typeof body.value === "string" ? body.value : "";
    const result = await upsertMemory(db, identity.scope, key, value);
    if (!result.ok) return json({ error: result.error }, 400);
    return json({ fact: result.fact });
  }

  if (request.method === "DELETE" && !keyPath) {
    const deleted = await clearMemory(db, identity.scope);
    return json({ cleared: deleted });
  }

  if (request.method === "DELETE" && keyPath) {
    const removed = await deleteMemoryKey(db, identity.scope, decodeURIComponent(keyPath));
    if (!removed) return json({ error: "Not found." }, 404);
    return json({ deleted: keyPath });
  }

  return json({ error: "Method not allowed." }, 405);
}

export async function handleAgent(
  request: Request,
  env: AgentEnv,
  ctx?: { waitUntil?: (promise: Promise<unknown>) => void },
): Promise<Response> {
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: { Allow: "GET, POST, PUT, DELETE, OPTIONS" },
    });
  }
  if (path === "/agent/memory" || path.startsWith("/agent/memory/")) {
    return handleMemory(request, env, path);
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
      ctx?.waitUntil?.bind(ctx),
    );
  }
  return json({ error: "Not found." }, 404);
}

