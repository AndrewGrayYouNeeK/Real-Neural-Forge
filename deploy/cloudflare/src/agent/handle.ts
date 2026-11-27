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
import {
  appendMessage,
  createThread,
  deleteThread,
  ensureThreadForTurn,
  getThreadOwned,
  historyFromMessages,
  listMessages,
  listThreads,
} from "./chats";
import { capKey } from "../auth/session";
import { runAgentTurn, type HistoryMessage } from "./loop";
import {
  MEMORY_OFF_PROMPT,
  autoRemember,
  clearMemory,
  compactMemoryPrompt,
  deleteMemoryKey,
  listMemory,
  memoryEnabled,
  memoryPromptBlock,
  resolveMemoryIdentity,
  setMemoryEnabled,
  upsertMemory,
} from "./memory";
import {
  acknowledgeAlertEvent,
  createAlertRule,
  deleteAlertRule,
  insertAlertEvent,
  listAlertEvents,
  listAlertRules,
  type AlertKind,
} from "./alerts";
import { createReport, getReport, listReports } from "./reports";
import {
  createScenario,
  deleteScenario,
  getScenarioOwned,
  listScenarios,
  scenarioRerunQuestion,
} from "./scenarios";
import {
  answerVision,
  resolveVisionCaps,
  visionAllowance,
  type VisionEnv,
} from "../vision/vision";
import {
  attachFilesToThread,
  fileIdsFrom,
  filePromptBlock,
  filesForTurn,
  loadFileTexts,
  uploadFile,
  type FilesAi,
} from "../abilities/files";
import { generateImage, isImageRequest, serveImage } from "../abilities/imagegen";
import { abilityAllowance, abilityCaps, type R2Like } from "../abilities/limits";
import {
  authHeadersFromRequest,
  visitorFromRequest,
  type ContainerFetcher,
} from "./tools";

export type AgentEnv = AiQuotaEnv & VisionEnv & {
  AI?: AiLike & FilesAi;
  WORKSPACES?: R2Like;
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

function threadIdFrom(body: Record<string, unknown> | null): string | null {
  const raw = body?.thread_id;
  if (typeof raw !== "string") return null;
  const id = raw.trim();
  return id || null;
}

async function answerAgentChat(
  request: Request,
  env: AgentEnv,
  question: string,
  mode: ChatMode,
  history: HistoryMessage[],
  threadId: string | null,
  waitUntil?: (promise: Promise<unknown>) => void,
  fileIds: string[] = [],
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
  // Daily caps are per account once signed in (IP only for anonymous calls).
  const ip = capKey(identity, clientIp(request));
  // Memory is per signed-in account only, and the account can turn it off.
  const memoryOn = identity.signedIn && (await memoryEnabled(env.WORKSPACES_DB, identity.scope));
  let memoryPrompt = memoryOn ? memoryPromptBlock([]) : MEMORY_OFF_PROMPT;
  if (memoryOn) {
    try {
      await autoRemember(env.WORKSPACES_DB, identity.scope, clean);
      const facts = await listMemory(env.WORKSPACES_DB, identity.scope);
      const topics = (await listThreads(env.WORKSPACES_DB, identity.scope))
        .filter((thread) => thread.id !== threadId)
        .map((thread) => thread.title);
      memoryPrompt = compactMemoryPrompt(facts, topics);
    } catch {
      memoryPrompt = memoryPromptBlock([]);
    }
  }

  let activeThreadId: string | null = null;
  let modelHistory = history;
  try {
    const thread = await ensureThreadForTurn(env.WORKSPACES_DB, identity.scope, threadId, clean);
    activeThreadId = thread.id;
    await appendMessage(env.WORKSPACES_DB, thread.id, "user", clean);
    const stored = await listMessages(env.WORKSPACES_DB, thread.id);
    const fromStore = historyFromMessages(stored);
    if (fromStore.length) modelHistory = fromStore;
  } catch {
    // Persistence is best-effort; chat still answers without a saved thread.
    activeThreadId = null;
  }

  if (env.WORKSPACES && !fileIds.length && isImageRequest(clean)) {
    return answerImageRequest(env, identity, ip, clean, activeThreadId);
  }

  if (env.WORKSPACES) {
    try {
      const files = await filesForTurn(env.WORKSPACES_DB, identity.scope, fileIds, activeThreadId);
      if (files.length) {
        const texts = await loadFileTexts(env.WORKSPACES, files);
        const block = filePromptBlock(texts, clean);
        if (block) memoryPrompt = `${memoryPrompt}\n\n${block}`;
        if (activeThreadId && fileIds.length) {
          await attachFilesToThread(env.WORKSPACES_DB, identity.scope, fileIds, activeThreadId);
        }
      }
    } catch {
      // A missing file never blocks the chat answer.
    }
  }

  try {
    const result = await runAgentTurn(clean, mode, modelHistory, {
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
        owner: identity.owner,
        memoryOff: !memoryOn,
      },
    });
    if (activeThreadId && result.text) {
      try {
        await appendMessage(env.WORKSPACES_DB, activeThreadId, "assistant", result.text);
      } catch {
        // Ignore persistence failures after a successful answer.
      }
    }
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
      thread_id: activeThreadId,
      ...(result.chart ? { chart: result.chart } : {}),
      ...(result.csv ? { csv: result.csv } : {}),
      ...(result.steps ? { steps: result.steps } : {}),
      ...(result.sources ? { sources: result.sources } : {}),
    });
  } catch {
    return json({
      text: "I could not reach the language model right now. Please try again in a moment, or ask about the weather, a market price, the moon, or space.",
    });
  }
}

async function answerImageRequest(
  env: AgentEnv,
  identity: Awaited<ReturnType<typeof identityFor>>,
  capKeyValue: string | null,
  question: string,
  threadId: string | null,
): Promise<Response> {
  const db = env.WORKSPACES_DB as D1Like;
  let text: string;
  let capped = false;
  try {
    const result = await generateImage(
      {
        ai: env.AI as AiLike,
        db,
        bucket: env.WORKSPACES as R2Like,
        scope: identity.scope,
        owner: identity.owner,
        capKey: capKeyValue,
        env,
        aiCaps: resolveAiCaps(env),
      },
      question,
    );
    text = result.text;
    capped = Boolean(result.capped);
  } catch {
    text = "The image didn't come out this time. Give it another try in a moment.";
  }
  if (threadId) {
    try {
      await appendMessage(db, threadId, "assistant", text);
    } catch {
      // Persistence is best-effort.
    }
  }
  return json({ text, model: true, capped, tools_used: [], thread_id: threadId });
}

async function handleFiles(request: Request, env: AgentEnv, path: string): Promise<Response> {
  if (!env.WORKSPACES_DB || !env.WORKSPACES) {
    return json({ error: "File reading isn't set up on this server yet." }, 503);
  }
  const identity = await identityFor(request, env);
  if (path === "/agent/files/limits" && request.method === "GET") {
    const allowance = await abilityAllowance({
      db: env.WORKSPACES_DB,
      kind: "files",
      scope: identity.scope,
      owner: identity.owner,
      caps: abilityCaps("files", env),
    });
    return json({ allowance });
  }
  if (path !== "/agent/files") return json({ error: "Not found." }, 404);
  if (request.method !== "POST") return json({ error: "Method not allowed." }, 405);
  const declared = Number(request.headers.get("Content-Length") || 0);
  if (declared > 5 * 1024 * 1024 + 1024) {
    return json({ error: "That file is too big. Keep it under 5 MB." }, 413);
  }
  const bytes = await request.arrayBuffer();
  try {
    const result = await uploadFile(
      {
        db: env.WORKSPACES_DB,
        bucket: env.WORKSPACES,
        ai: env.AI,
        scope: identity.scope,
        owner: identity.owner,
        env,
      },
      request.headers.get("X-File-Name"),
      request.headers.get("Content-Type"),
      bytes,
    );
    return json(result.body, result.status);
  } catch {
    return json({ error: "I couldn't save that file right now. Try again in a moment." }, 500);
  }
}

async function handleImages(request: Request, env: AgentEnv, path: string): Promise<Response> {
  if (!env.WORKSPACES_DB || !env.WORKSPACES) return json({ error: "Not found." }, 404);
  const identity = await identityFor(request, env);
  if (path === "/agent/images/limits" && request.method === "GET") {
    const allowance = await abilityAllowance({
      db: env.WORKSPACES_DB,
      kind: "images",
      scope: identity.scope,
      owner: identity.owner,
      caps: abilityCaps("images", env),
    });
    return json({ allowance });
  }
  if (request.method !== "GET") return json({ error: "Method not allowed." }, 405);
  const id = path.replace(/^\/agent\/images\//, "");
  return serveImage(env.WORKSPACES_DB, env.WORKSPACES, identity.scope, id);
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
      enabled: identity.signedIn ? await memoryEnabled(db, identity.scope) : false,
      facts,
    });
  }

  if (keyPath === "settings" && (request.method === "PUT" || request.method === "POST")) {
    if (!identity.signedIn) return json({ error: "Sign in to use memory." }, 401);
    let enabled: unknown = undefined;
    try {
      enabled = ((await request.json()) as { enabled?: unknown })?.enabled;
    } catch {
      enabled = undefined;
    }
    if (typeof enabled !== "boolean") return json({ error: "Send enabled: true or false." }, 400);
    await setMemoryEnabled(db, identity.scope, enabled);
    return json({ enabled });
  }

  if (request.method === "PUT" && !keyPath && !identity.signedIn) {
    return json({ error: "Sign in to use memory." }, 401);
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

async function handleChats(
  request: Request,
  env: AgentEnv,
  path: string,
  body: Record<string, unknown> | null,
): Promise<Response> {
  if (!env.WORKSPACES_DB) {
    return json({ error: "Chat store is not configured." }, 503);
  }
  const identity = await identityFor(request, env);
  const db = env.WORKSPACES_DB;
  const rest = path.replace(/^\/agent\/chats\/?/, "");

  if (request.method === "GET" && !rest) {
    const threads = await listThreads(db, identity.scope);
    return json({
      scope: identity.signedIn ? (identity.owner ? "owner" : "user") : "visitor",
      threads,
    });
  }

  if (request.method === "POST" && !rest) {
    const title =
      typeof body?.title === "string" ? body.title : typeof body?.question === "string" ? body.question : "New chat";
    const thread = await createThread(db, identity.scope, title);
    return json({ thread });
  }

  if (!rest) {
    return json({ error: "Method not allowed." }, 405);
  }

  const threadId = decodeURIComponent(rest.split("/")[0] || "");
  if (!threadId) return json({ error: "Not found." }, 404);

  if (request.method === "GET") {
    const thread = await getThreadOwned(db, identity.scope, threadId);
    if (!thread) return json({ error: "Not found." }, 404);
    const messages = await listMessages(db, threadId);
    return json({ thread, messages });
  }

  if (request.method === "DELETE") {
    const removed = await deleteThread(db, identity.scope, threadId);
    if (!removed) return json({ error: "Not found." }, 404);
    return json({ deleted: threadId });
  }

  return json({ error: "Method not allowed." }, 405);
}

async function handleScenarios(
  request: Request,
  env: AgentEnv,
  path: string,
  body: Record<string, unknown> | null,
): Promise<Response> {
  if (!env.WORKSPACES_DB) {
    return json({ error: "Scenario store is not configured." }, 503);
  }
  const identity = await identityFor(request, env);
  const db = env.WORKSPACES_DB;
  const rest = path.replace(/^\/agent\/scenarios\/?/, "");

  if (request.method === "GET" && !rest) {
    const scenarios = await listScenarios(db, identity.scope);
    return json({
      scope: identity.signedIn ? (identity.owner ? "owner" : "user") : "visitor",
      scenarios,
    });
  }

  if (request.method === "POST" && !rest) {
    const name = typeof body?.name === "string" ? body.name : "";
    const question = typeof body?.question === "string" ? body.question : "";
    const series = typeof body?.series === "string" ? body.series : null;
    const horizon =
      typeof body?.horizon === "number" && Number.isFinite(body.horizon) ? body.horizon : null;
    const result = await createScenario(db, identity.scope, { name, question, series, horizon });
    if (!result.ok) return json({ error: result.error }, 400);
    return json({ scenario: result.scenario });
  }

  if (!rest) return json({ error: "Method not allowed." }, 405);

  const parts = rest.split("/");
  const scenarioId = decodeURIComponent(parts[0] || "");
  if (!scenarioId) return json({ error: "Not found." }, 404);

  if (request.method === "GET" && parts.length === 1) {
    const scenario = await getScenarioOwned(db, identity.scope, scenarioId);
    if (!scenario) return json({ error: "Not found." }, 404);
    return json({ scenario, rerun_question: scenarioRerunQuestion(scenario) });
  }

  if (request.method === "DELETE" && parts.length === 1) {
    const removed = await deleteScenario(db, identity.scope, scenarioId);
    if (!removed) return json({ error: "Not found." }, 404);
    return json({ deleted: scenarioId });
  }

  return json({ error: "Method not allowed." }, 405);
}

async function handleReports(
  request: Request,
  env: AgentEnv,
  path: string,
  body: Record<string, unknown> | null,
): Promise<Response> {
  if (!env.WORKSPACES_DB) {
    return json({ error: "Report store is not configured." }, 503);
  }
  const db = env.WORKSPACES_DB;
  const rest = path.replace(/^\/agent\/reports\/?/, "");

  // Public share link — no auth required.
  if (request.method === "GET" && rest && !rest.includes("/")) {
    const reportId = decodeURIComponent(rest);
    const report = await getReport(db, reportId);
    if (!report) return json({ error: "Not found." }, 404);
    return json({
      id: report.id,
      series: report.series,
      title: report.title,
      created_at: report.created_at,
      snapshot: report.snapshot,
      share_path: `/agent/reports/${report.id}`,
    });
  }

  const identity = await identityFor(request, env);

  if (request.method === "GET" && !rest) {
    const reports = await listReports(db, identity.scope);
    return json({
      scope: identity.signedIn ? (identity.owner ? "owner" : "user") : "visitor",
      reports: reports.map((row) => ({
        id: row.id,
        series: row.series,
        title: row.title,
        created_at: row.created_at,
        share_path: `/agent/reports/${row.id}`,
      })),
    });
  }

  if (request.method === "POST" && !rest) {
    const series =
      typeof body?.series === "string" && body.series.trim() ? body.series.trim() : "weather";
    let snapshot =
      body?.snapshot && typeof body.snapshot === "object" && !Array.isArray(body.snapshot)
        ? (body.snapshot as Record<string, unknown>)
        : null;
    if (!snapshot) {
      const fetchContainer = makeContainerFetcher(env, request);
      const built = await fetchContainer("/analysis/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ series }),
      });
      if (built.status >= 400 || !built.json || typeof built.json !== "object") {
        return json({ error: `Could not build report (${built.status}).` }, built.status >= 400 ? built.status : 502);
      }
      snapshot = built.json as Record<string, unknown>;
    }
    const title =
      (typeof body?.title === "string" && body.title.trim()) ||
      (typeof snapshot.title === "string" && snapshot.title) ||
      `${series} report`;
    const result = await createReport(db, identity.scope, { series, title, snapshot });
    if (!result.ok) return json({ error: result.error }, 400);
    return json({
      report: {
        id: result.report.id,
        series: result.report.series,
        title: result.report.title,
        created_at: result.report.created_at,
        share_path: `/agent/reports/${result.report.id}`,
      },
      snapshot: result.report.snapshot,
    });
  }

  return json({ error: "Method not allowed." }, 405);
}

async function handleAlerts(
  request: Request,
  env: AgentEnv,
  path: string,
  body: Record<string, unknown> | null,
): Promise<Response> {
  if (!env.WORKSPACES_DB) {
    return json({ error: "Alert store is not configured." }, 503);
  }
  const identity = await identityFor(request, env);
  const db = env.WORKSPACES_DB;
  const rest = path.replace(/^\/agent\/alerts\/?/, "");

  if (request.method === "GET" && !rest) {
    const [rules, events] = await Promise.all([
      listAlertRules(db, identity.scope),
      listAlertEvents(db, identity.scope),
    ]);
    return json({
      scope: identity.signedIn ? (identity.owner ? "owner" : "user") : "visitor",
      rules,
      events,
    });
  }

  if (request.method === "POST" && !rest) {
    const name = typeof body?.name === "string" ? body.name : "";
    const series = typeof body?.series === "string" ? body.series : "weather";
    const kindRaw = typeof body?.kind === "string" ? body.kind : "threshold";
    const kind: AlertKind = kindRaw === "anomaly" ? "anomaly" : "threshold";
    const config =
      body?.config && typeof body.config === "object" && !Array.isArray(body.config)
        ? (body.config as Record<string, unknown>)
        : {};
    const result = await createAlertRule(db, identity.scope, { name, series, kind, config });
    if (!result.ok) return json({ error: result.error }, 400);
    return json({ rule: result.rule });
  }

  if (request.method === "POST" && rest === "check") {
    const rules = await listAlertRules(db, identity.scope);
    const seriesFilter =
      typeof body?.series === "string" && body.series.trim() ? body.series.trim() : null;
    const fetchContainer = makeContainerFetcher(env, request);
    const fired: Array<Record<string, unknown>> = [];
    for (const rule of rules) {
      if (!rule.enabled) continue;
      if (seriesFilter && rule.series !== seriesFilter) continue;
      const checked = await fetchContainer("/analysis/alert-check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          series: rule.series,
          kind: rule.kind,
          config: rule.config,
        }),
      });
      if (checked.status >= 400 || !checked.json || typeof checked.json !== "object") {
        continue;
      }
      const payload = checked.json as Record<string, unknown>;
      if (payload.fired !== true) continue;
      const event = await insertAlertEvent(db, identity.scope, rule.id, {
        ...payload,
        rule_name: rule.name,
      });
      fired.push({ rule, event, result: payload });
    }
    return json({ checked: rules.length, fired });
  }

  if (!rest) return json({ error: "Method not allowed." }, 405);
  const parts = rest.split("/");
  const first = decodeURIComponent(parts[0] || "");
  if (!first) return json({ error: "Not found." }, 404);

  if (request.method === "DELETE" && parts.length === 1) {
    await deleteAlertRule(db, identity.scope, first);
    return json({ deleted: first });
  }

  if (
    request.method === "POST" &&
    parts.length === 3 &&
    parts[0] === "events" &&
    parts[2] === "ack"
  ) {
    const eventId = decodeURIComponent(parts[1] || "");
    await acknowledgeAlertEvent(db, identity.scope, eventId);
    return json({ acknowledged: eventId });
  }

  return json({ error: "Method not allowed." }, 405);
}

async function handleVision(
  request: Request,
  env: AgentEnv,
  path: string,
  body: Record<string, unknown> | null,
): Promise<Response> {
  if (!env.AI || !env.WORKSPACES_DB) {
    return json({ text: "Image reading is not configured on this server yet.", capped: true });
  }
  const identity = await identityFor(request, env);
  const ip = capKey(identity, clientIp(request));
  const caps = resolveVisionCaps(env);
  if (path === "/agent/vision/limits") {
    if (request.method !== "GET") return json({ error: "Method not allowed." }, 405);
    try {
      const allowance = await visionAllowance({
        db: env.WORKSPACES_DB,
        owner: identity.owner,
        ip,
        identityScope: identity.scope,
        caps,
      });
      return json({ allowance, owner: identity.owner });
    } catch {
      return json({ error: "Allowance unavailable." }, 503);
    }
  }
  if (path !== "/agent/vision") return json({ error: "Not found." }, 404);
  if (request.method !== "POST") return json({ error: "Method not allowed." }, 405);
  const question = typeof body?.question === "string" ? body.question : "";
  const mode = chatMode(typeof body?.mode === "string" ? body.mode : null);
  const result = await answerVision(
    {
      ai: env.AI,
      db: env.WORKSPACES_DB,
      owner: identity.owner,
      ip,
      identityScope: identity.scope,
      env,
      chatModels: resolveChatModels(env),
    },
    body?.image,
    question,
    mode,
  );
  let threadId: string | null = null;
  if (result.status === 200 && result.threadText) {
    try {
      const userLine = `[Image] ${question.replace(/\s+/g, " ").trim() || "Read this image"}`;
      const thread = await ensureThreadForTurn(
        env.WORKSPACES_DB,
        identity.scope,
        threadIdFrom(body),
        userLine,
      );
      threadId = thread.id;
      await appendMessage(env.WORKSPACES_DB, thread.id, "user", userLine);
      await appendMessage(env.WORKSPACES_DB, thread.id, "assistant", result.threadText);
    } catch {
      threadId = null;
    }
  }
  return json({ ...result.body, thread_id: threadId }, result.status);
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
  if (path === "/agent/files" || path.startsWith("/agent/files/")) {
    return handleFiles(request, env, path);
  }
  if (path.startsWith("/agent/images/")) {
    return handleImages(request, env, path);
  }
  let body: Record<string, unknown> | null = null;
  if (request.method === "POST" || request.method === "PUT") {
    const text = await request.text();
    if (text) {
      try {
        body = JSON.parse(text) as Record<string, unknown>;
      } catch {
        return json({ text: "I could not read that request." }, 400);
      }
    }
  }
  if (path === "/agent/scenarios" || path.startsWith("/agent/scenarios/")) {
    return handleScenarios(request, env, path, body);
  }
  if (path === "/agent/reports" || path.startsWith("/agent/reports/")) {
    return handleReports(request, env, path, body);
  }
  if (path === "/agent/alerts" || path.startsWith("/agent/alerts/")) {
    return handleAlerts(request, env, path, body);
  }
  if (path === "/agent/vision" || path.startsWith("/agent/vision/")) {
    return handleVision(request, env, path, body);
  }
  if (path === "/agent/chats" || path.startsWith("/agent/chats/")) {
    return handleChats(request, env, path, body);
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
      threadIdFrom(body),
      ctx?.waitUntil?.bind(ctx),
      fileIdsFrom(body),
    );
  }
  return json({ error: "Not found." }, 404);
}

