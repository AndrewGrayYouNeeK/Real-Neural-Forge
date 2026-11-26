import { capKey, resolveSession } from "../auth/session";
import { visitorFromCf, type CfIncoming } from "./geo";
import { answerWeather } from "./weather";
import { answerQuote } from "./markets";
import {
  CAP_MESSAGE,
  VOICE_RESTING_MESSAGE,
  addExtraNeurons,
  ipScope,
  chatMode,
  resolveAiCaps,
  resolveChatModels,
  runChatModel,
  tryReserve,
  voiceBudget,
  type AiLike,
  type AiQuotaEnv,
  type ChatMode,
  type D1Like,
} from "./ai";

export type LiveEnv = AiQuotaEnv & {
  AI?: AiLike;
  WORKSPACES_DB?: D1Like;
  NEURAL_FORGE_API_TOKEN?: string;
  NEURAL_FORGE_AI_MODEL?: string;
  NEURAL_FORGE_AI_FALLBACK_MODEL?: string;
};


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

async function isOwnerRequest(request: Request, env: LiveEnv): Promise<boolean> {
  // Owner = role=owner in D1 plus a fresh second factor. No shared token.
  const session = await resolveSession(request, env.WORKSPACES_DB);
  return session?.owner === true;
}

async function answerBudget(request: Request, env: LiveEnv): Promise<Response> {
  if (!env.WORKSPACES_DB) return json({ voice_resting: false, voice_message: null });
  const owner = await isOwnerRequest(request, env);
  try {
    const budget = await voiceBudget({
      db: env.WORKSPACES_DB,
      owner,
      caps: resolveAiCaps(env),
    });
    return json({ voice_resting: budget.voiceResting, voice_message: budget.message });
  } catch {
    return json({ voice_resting: false, voice_message: null });
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
  const session = await resolveSession(request, env.WORKSPACES_DB);
  const owner = session?.owner === true;
  const models = resolveChatModels(env);
  const caps = resolveAiCaps(env);
  const ip = capKey(session, clientIp(request));
  const reserved = await tryReserve({
    db: env.WORKSPACES_DB,
    ip,
    owner,
    caps,
  });
  if (!reserved.ok) {
    return json({ text: reserved.message, capped: true, reason: reserved.reason });
  }
  try {
    const result = await runChatModel(env.AI, clean, mode, facts, models);
    if (result.neurons > caps.reserve) {
      const extra = result.neurons - caps.reserve;
      await addExtraNeurons(env.WORKSPACES_DB, "global", extra);
      if (!owner) await addExtraNeurons(env.WORKSPACES_DB, await ipScope(ip), extra);
    }
    let voiceResting = reserved.voiceResting;
    let voiceMessage = voiceResting ? VOICE_RESTING_MESSAGE : null;
    try {
      const budget = await voiceBudget({ db: env.WORKSPACES_DB, owner, caps });
      voiceResting = budget.voiceResting;
      voiceMessage = budget.message;
    } catch {
      // The reserve already recorded the spend. Keep that voice flag.
    }
    return json({
      text: result.text,
      model: true,
      model_id: result.modelId,
      voice_resting: voiceResting,
      voice_message: voiceMessage,
    });
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
  if (path === "/live/budget") {
    if (request.method !== "GET") return json({ error: "Method not allowed." }, 405);
    return answerBudget(request, env);
  }
  if (path === "/live/ask") {
    if (request.method !== "POST" && request.method !== "GET") return json({ error: "Method not allowed." }, 405);
    return answerAsk(request, env, question, modeFrom(url, body), factsFrom(url, body));
  }
  return json({ error: "Not found." }, 404);
}

export { CAP_MESSAGE };
