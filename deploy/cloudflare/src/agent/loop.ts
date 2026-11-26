/**
 * Agentic chat turn: model decides whether to call tools or answer.
 * Every model call reserves against the existing AI neuron caps.
 *
 * GPT-OSS 120B on Workers AI uses the Responses API (`input`, `reasoning`,
 * `max_output_tokens`). Passing a native `tools` array is rejected in
 * production, so 120B uses the JSON tool protocol in the system prompt — the
 * same request shape as `/live/ask`. Native `tools` stay only for the 8B
 * chat-completions fallback.
 */

import {
  NORMAL_TEMPERATURE,
  PRIMARY_MODEL,
  addExtraNeurons,
  chatMode,
  extractModelText,
  ipScope,
  resolveAiCaps,
  resolveChatModels,
  sanitizeQuestion,
  systemPromptFor,
  tokensFromText,
  tryReserve,
  usageNeurons,
  voiceBudget,
  VOICE_RESTING_MESSAGE,
  type AiCaps,
  type AiLike,
  type ChatMode,
  type D1Like,
} from "../live/ai";
import { routeClearIntent } from "./intent";
import {
  buildArtifacts,
  checklistNote,
  deliverableDone,
  hasUsableScenario,
  scenarioFrom,
  missingDeliverables,
  planRequest,
  routeMissingStep,
  shortLabels,
  type ChartSpec,
  type Deliverable,
  type RequestPlan,
} from "./plan";
import {
  collectToolCalls,
  finalAnswerText,
  isDegenerateAnswer,
  looksLikeFailedToolAttempt,
  sanitizeAgentAnswer,
  summarizeArgs,
  type ParsedToolCall,
} from "./protocol";
import {
  MAX_TOOL_CALLS,
  TOOL_TIMEOUT_MS,
  TURN_TIMEOUT_MS,
  dispatchTool,
  toolCatalogForPrompt,
  toolsForChatCompletions,
  type ToolContext,
  type ToolResult,
} from "./tools";

export type HistoryMessage = { role: "user" | "assistant"; content: string };

export type AgentTurnResult = {
  text: string;
  modelId: string;
  neurons: number;
  toolsUsed: string[];
  capped?: boolean;
  reason?: string;
  voiceResting: boolean;
  voiceMessage: string | null;
  /** Chart spec for the frontend line chart (only when the user asked for one). */
  chart?: ChartSpec;
  /** CSV download (only when the user asked for one). */
  csv?: { filename: string; text: string };
  /** Multi-step bookkeeping: what was asked and what never happened. */
  steps?: { planned: Deliverable[]; missing: Deliverable[] };
};

export type AgentLoopDeps = {
  ai: AiLike;
  db: D1Like;
  ip: string | null;
  owner: boolean;
  caps?: AiCaps;
  models?: { primary?: string; fallback?: string };
  toolContext: ToolContext;
  maxToolCalls?: number;
  toolTimeoutMs?: number;
  turnTimeoutMs?: number;
  now?: Date;
  /** Preloaded user memory block appended to the system prompt. */
  memoryPrompt?: string;
  maxModelSteps?: number;
  turnNeuronBudget?: number;
};

const JSON_PROTOCOL =
  "Tools are optional. For greetings, small talk, thanks, or questions that need no live data or model facts, answer in plain language with NO tool call. " +
  "When you do need a tool, reply with ONLY JSON and nothing else: one call " +
  '{"tool":"<name>","args":{...}} ' +
  "or, for independent steps that do not need each other's results, an array " +
  '[{"tool":"<name>","args":{...}},{"tool":"<name>","args":{...}}]. ' +
  "Valid tools:\n" +
  toolCatalogForPrompt() +
  "\nMulti-step requests (for example: get data, then forecast, then chart, then explain): silently list every step the user asked for, " +
  "do them in order (one tool reply per step, or an array for independent steps), and write the final answer only after every step is done. " +
  "Never stop after the first step, and never say you will do something later: do it now with a tool, or say plainly that it cannot be done. " +
  "For simulate / scenario / cold snap / heat wave asks, use simulate_world with a named scenario (it rolls many days and compares against the baseline) rather than a one-step what_if. " +
  "Do not stall on missing details: with no place given, weather tools use the visitor's location; with no series given, use the weather sample. " +
  "Charts and CSV downloads are attached to your reply automatically from the tool data, so never claim you cannot draw a chart or make a file; just make sure the data tool ran. " +
  "\nAfter tool results are provided, answer the user in plain language. " +
  "Never invent live weather or prices. Never show tool JSON, planning, checklists, or hidden reasoning to the user.";

const STRICT_TOOL_RETRY =
  "Your previous reply was not a valid tool JSON object. " +
  "Reply with ONLY this JSON and nothing else (no markdown, no prose): " +
  '{"tool":"<name>","args":{...}} ' +
  "using one of the valid tools. If no tool is needed, reply with a short plain-language answer only.";

/** Short greetings / chit-chat that must never call tools. */
export function isSmallTalk(question: string): boolean {
  const text = question.replace(/\s+/g, " ").trim().toLowerCase();
  if (!text || text.length > 80) return false;
  return /^(hey+|hi+|hello+|yo+|sup+|howdy+)([!.? ]|$)|^(good\s+(morning|afternoon|evening)|thanks|thank you|thx|ty|ok|okay|cool|bye|goodbye|see ya|what's up|whats up|how are you)([!.? ]|$)/i.test(
    text,
  );
}

export function agentSystemPrompt(
  mode: ChatMode,
  allowTools: boolean,
  memoryPrompt?: string,
): string {
  const base = systemPromptFor(mode);
  const memory =
    typeof memoryPrompt === "string" && memoryPrompt.trim()
      ? `\n\n${memoryPrompt.trim()}`
      : "";
  if (!allowTools) {
    if (mode === "unhinged") return `${base}${memory}`;
    return (
      `${base}\n\n` +
      "Directness: answer exactly what was asked. Lead with the answer. Keep it short. " +
      "No rambling, tangents, or unasked extras. Keep the personality: a quick quip or wry line is welcome. " +
      "Do not call tools and do not emit tool JSON." +
      memory
    );
  }
  if (mode === "unhinged") {
    return `${base}\n\n${JSON_PROTOCOL}${memory}`;
  }
  return (
    `${base}\n\n` +
    "Directness: answer exactly what was asked. Lead with the answer. Keep it short. " +
    "No rambling, tangents, or unasked extras. Keep the personality: a quick quip or wry line is welcome.\n\n" +
    JSON_PROTOCOL +
    memory
  );
}

/** Budget for the tool-picking / small-talk step. */
export const AGENT_STEP_MAX_TOKENS = 220;
/**
 * Tool-picking budget for multi-step asks. GPT-OSS spends hidden reasoning
 * tokens on the plan, and simulate_world JSON with scenarios is long; 220 cut
 * those replies off mid-JSON so no tool call parsed and the turn ended early.
 */
export const AGENT_PLAN_STEP_MAX_TOKENS = 900;
/** Hard cap on model calls in one turn (each one reserves against the caps). */
export const MAX_MODEL_STEPS = 8;
/**
 * Per-turn neuron guard (~$0.016 at overage pricing). Daily per-visitor and
 * global caps still apply on top; this only stops one runaway turn.
 */
export const TURN_NEURON_BUDGET = 1500;
/**
 * Budget once tool results are in. GPT-OSS counts hidden reasoning against
 * this cap, so 220 cut explain answers mid-word ("…tug-"). Reasoning stays
 * hidden; the prompt still asks for a short answer.
 */
export const AGENT_ANSWER_MAX_TOKENS = 700;

function usesResponsesApi(model: string): boolean {
  return model.includes("gpt-oss");
}

function historyForModel(history: HistoryMessage[]): HistoryMessage[] {
  return history
    .filter((item) => item.content.trim())
    .slice(-8)
    .map((item) => ({
      role: item.role,
      content: item.content.trim().slice(0, 500),
    }));
}

/**
 * Build Workers AI inputs for one agent model step.
 * GPT-OSS keeps the `/live/ask` Responses shape (no native `tools` — that
 * param is rejected on the binding). Non-gpt-oss may pass chat-completions tools.
 */
export function buildAgentInputs(options: {
  model: string;
  system: string;
  question: string;
  history: HistoryMessage[];
  toolNotes: string[];
  mode: ChatMode;
  allowTools: boolean;
  /** Multi-step plan: give tool-picking steps room for the plan + longer JSON. */
  multiStep?: boolean;
}): Record<string, unknown> {
  const { model, system, question, history, toolNotes, mode, allowTools } = options;
  const prior = historyForModel(history);
  const maxTokens = options.multiStep
    ? Math.max(AGENT_PLAN_STEP_MAX_TOKENS, toolNotes.length ? AGENT_ANSWER_MAX_TOKENS : 0)
    : toolNotes.length
      ? AGENT_ANSWER_MAX_TOKENS
      : AGENT_STEP_MAX_TOKENS;
  if (usesResponsesApi(model)) {
    const input: Array<Record<string, unknown>> = [{ role: "system", content: system }];
    for (const item of prior) {
      input.push({ role: item.role, content: item.content });
    }
    input.push({ role: "user", content: question });
    for (const note of toolNotes) {
      input.push({ role: "user", content: note });
    }
    // Do not attach native `tools` here. Workers AI gpt-oss Responses rejects
    // that field; the JSON protocol in the system prompt is the tool channel.
    return {
      input,
      reasoning: { effort: "low" },
      max_output_tokens: maxTokens,
    };
  }
  const messages: Array<Record<string, unknown>> = [{ role: "system", content: system }];
  for (const item of prior) {
    messages.push({ role: item.role, content: item.content });
  }
  messages.push({ role: "user", content: question });
  for (const note of toolNotes) {
    messages.push({ role: "user", content: note });
  }
  const inputs: Record<string, unknown> = {
    messages,
    max_tokens: maxTokens,
  };
  if (allowTools) inputs.tools = toolsForChatCompletions();
  inputs.temperature = mode === "unhinged" ? 0.9 : NORMAL_TEMPERATURE;
  return inputs;
}

/** Default-mode voice reminder appended to tool results (Unhinged keeps its own prompt). */
export const NORMAL_TOOL_VOICE =
  " Keep your usual witty voice: the facts first, then at most one short quip.";

function toolNote(
  call: ParsedToolCall,
  result: ToolResult,
  mode: ChatMode = "normal",
  multiStep = false,
): string {
  // Never forward internal exception strings to the model — it echoes them.
  // `_full` / `_…` keys hold chart and CSV data the model never needs.
  let safeData = result.data;
  if (safeData && typeof safeData === "object" && !Array.isArray(safeData)) {
    const rest: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(safeData as Record<string, unknown>)) {
      if (key === "error" || key.startsWith("_")) continue;
      rest[key] = value;
    }
    safeData = rest;
  }
  const payload = result.ok
    ? { tool: call.name, ok: true, result: safeData }
    : { tool: call.name, ok: false, error: friendlyToolFailure(call.name) };
  const voice = mode === "unhinged" ? "" : NORMAL_TOOL_VOICE;
  if (multiStep) {
    // The checklist note that follows says whether to call the next tool or answer.
    return `Tool result for ${call.name}:\n${JSON.stringify(payload).slice(0, 3500)}\nUse only these facts. Do not invent values. Never mention tool names or raw errors.${voice}`;
  }
  return `Tool result for ${call.name}:\n${JSON.stringify(payload).slice(0, 3500)}\nUsing only these facts, answer the user directly. Lead with the key number or condition. Do not call another tool unless you still lack a required fact. Do not invent values. Never mention tool names or raw errors.${voice}`;
}

async function reserveOrFail(
  deps: AgentLoopDeps,
  caps: AiCaps,
): Promise<{ ok: true; voiceResting: boolean } | { ok: false; message: string; reason: string }> {
  const reserved = await tryReserve({
    db: deps.db,
    ip: deps.ip,
    owner: deps.owner,
    caps,
    now: deps.now,
  });
  if (!reserved.ok) {
    return { ok: false, message: reserved.message, reason: reserved.reason };
  }
  return { ok: true, voiceResting: reserved.voiceResting };
}

async function settleExtraNeurons(
  deps: AgentLoopDeps,
  caps: AiCaps,
  neurons: number,
): Promise<void> {
  if (neurons <= caps.reserve) return;
  const extra = neurons - caps.reserve;
  await addExtraNeurons(deps.db, "global", extra, deps.now);
  if (!deps.owner) {
    await addExtraNeurons(deps.db, await ipScope(deps.ip), extra, deps.now);
  }
}

type ModelStep = {
  text: string | null;
  calls: ParsedToolCall[];
  neurons: number;
  modelId: string;
  raw: unknown;
};

function errorDetails(error: unknown): { message: string; status: number | null } {
  if (error instanceof Error) {
    const status = (error as { status?: unknown }).status;
    return { message: error.message, status: typeof status === "number" ? status : null };
  }
  if (error && typeof error === "object") {
    const record = error as { message?: unknown; status?: unknown };
    return {
      message: typeof record.message === "string" ? record.message : "primary model failed",
      status: typeof record.status === "number" ? record.status : null,
    };
  }
  return { message: String(error), status: null };
}

async function runModelStep(
  deps: AgentLoopDeps,
  model: string,
  system: string,
  question: string,
  history: HistoryMessage[],
  toolNotes: string[],
  mode: ChatMode,
  allowTools: boolean,
  multiStep = false,
): Promise<ModelStep> {
  const inputs = buildAgentInputs({
    model,
    system,
    question,
    history,
    toolNotes,
    mode,
    allowTools,
    multiStep,
  });
  const raw = await deps.ai.run(model, inputs);
  const text = extractModelText(raw);
  const calls = allowTools ? collectToolCalls(raw, text) : [];
  const promptBits = [system, question, ...toolNotes].join("\n");
  const answerBits = text || JSON.stringify(calls);
  return {
    text,
    calls,
    neurons: usageNeurons(raw, promptBits, answerBits, model),
    modelId: model,
    raw,
  };
}

async function runModelWithFallback(
  deps: AgentLoopDeps,
  system: string,
  question: string,
  history: HistoryMessage[],
  toolNotes: string[],
  mode: ChatMode,
  models: { primary: string; fallback: string },
  allowTools: boolean,
  multiStep = false,
): Promise<ModelStep> {
  try {
    return await runModelStep(
      deps,
      models.primary,
      system,
      question,
      history,
      toolNotes,
      mode,
      allowTools,
      multiStep,
    );
  } catch (error) {
    if (models.fallback === models.primary) throw error;
    const details = errorDetails(error);
    console.error(
      JSON.stringify({
        message: "agent.primary_failed",
        event: "agent.primary_failed",
        model: models.primary,
        error: details.message,
        status: details.status,
        fallback: models.fallback,
      }),
    );
    return runModelStep(
      deps,
      models.fallback,
      system,
      question,
      history,
      toolNotes,
      mode,
      allowTools,
      multiStep,
    );
  }
}

/** User-facing copy for failed tools — never leak tool names or raw errors. */
export function friendlyToolFailure(name: string): string {
  switch (name) {
    case "forecast":
      return "The forecast service is warming up or temporarily unavailable. Please try that forecast again in a moment.";
    case "get_evaluation_metrics":
      return "Model accuracy metrics are temporarily unavailable. Please try again in a moment.";
    case "fetch_live_data":
      return "I could not load that live data right now. Please try again in a moment.";
    default:
      return "I could not finish that with the available data. Please try again in a moment.";
  }
}

/** Build a direct user-facing answer from tool results (no model). */
export function factualFallback(results: ToolResult[]): string {
  const lines: string[] = [];
  for (const result of results) {
    if (!result.ok) {
      lines.push(friendlyToolFailure(result.name));
      continue;
    }
    if (result.data && typeof result.data === "object" && "text" in (result.data as object)) {
      const text = (result.data as { text?: unknown }).text;
      if (typeof text === "string" && text.trim()) {
        lines.push(text.trim());
        continue;
      }
    }
    if (result.name === "forecast" && result.data && typeof result.data === "object") {
      const data = result.data as {
        forecast?: Array<{ value?: unknown; step?: unknown }>;
        name?: unknown;
        unit?: unknown;
        series?: unknown;
        recent_accuracy?: { comparison?: unknown; metrics?: { mae?: unknown }; unit?: unknown };
      };
      const values = Array.isArray(data.forecast)
        ? data.forecast
            .map((row) => (typeof row.value === "number" ? row.value : null))
            .filter((v): v is number => v != null)
        : [];
      if (values.length) {
        const label = typeof data.name === "string" ? data.name : String(data.series || "series");
        // Tool output already converts weather °C → °F; trust unit + values as-is.
        const unit = typeof data.unit === "string" && data.unit ? data.unit : "";
        const displayValues = values.map((value) => {
          const rendered = Number.isInteger(value) ? String(value) : value.toFixed(2);
          return unit ? `${rendered}${unit}` : rendered;
        });
        let line = `Forecast for ${label}: ${displayValues.join(", ")}.`;
        const comparison =
          data.recent_accuracy && typeof data.recent_accuracy.comparison === "string"
            ? data.recent_accuracy.comparison
            : null;
        if (comparison) line += ` Recent accuracy: ${comparison}.`;
        const mae =
          data.recent_accuracy &&
          data.recent_accuracy.metrics &&
          typeof data.recent_accuracy.metrics.mae === "number"
            ? data.recent_accuracy.metrics.mae
            : null;
        if (mae != null) {
          const maeUnit =
            (data.recent_accuracy && typeof data.recent_accuracy.unit === "string"
              ? data.recent_accuracy.unit
              : unit) || "";
          line += ` MAE about ${mae.toFixed(2)}${maeUnit}.`;
        }
        lines.push(line);
        continue;
      }
    }
    if (result.name === "get_evaluation_metrics" && result.data && typeof result.data === "object") {
      const data = result.data as {
        model_id?: unknown;
        comparison?: unknown;
        unit?: unknown;
        metrics?: { mae?: unknown; directional_accuracy?: unknown };
      };
      const modelId = typeof data.model_id === "string" ? data.model_id : "model";
      const mae = data.metrics?.mae;
      const unit = typeof data.unit === "string" && data.unit ? data.unit : "";
      if (typeof mae === "number" && (modelId === "weather" || unit)) {
        const comparison =
          typeof data.comparison === "string" && data.comparison.trim()
            ? ` ${data.comparison.trim()}.`
            : "";
        const maeUnit = unit || (modelId === "weather" ? "°F" : "");
        lines.push(`${modelId} MAE about ${mae.toFixed(2)}${maeUnit}.${comparison}`);
        continue;
      }
      if (typeof data.comparison === "string" && data.comparison.trim()) {
        lines.push(`${modelId}: ${data.comparison}.`);
        continue;
      }
      if (typeof mae === "number") {
        lines.push(`${modelId} MAE: ${mae}${unit}.`);
        continue;
      }
    }
    lines.push(friendlyToolFailure(result.name));
  }
  return lines.join("\n\n");
}

function leadWithToolFacts(modelAnswer: string | null, results: ToolResult[]): string | null {
  const facts = factualFallback(results);
  if (!facts) return modelAnswer;
  // Live weather / prices: prefer the tool text so "right now" current conditions lead.
  const live = results.find(
    (item) =>
      item.ok &&
      item.name === "fetch_live_data" &&
      item.data &&
      typeof item.data === "object" &&
      typeof (item.data as { text?: unknown }).text === "string",
  );
  if (live) {
    const text = String((live.data as { text: string }).text).trim();
    if (!modelAnswer || isDegenerateAnswer(modelAnswer)) return text;
    // If the model wandered into tomorrow while the tool returned current, keep tool text.
    const wantsCurrent =
      live.data &&
      typeof live.data === "object" &&
      (live.data as { wants_current?: unknown }).wants_current === true;
    if (wantsCurrent) return text;
    if (!modelAnswer.toLowerCase().includes(text.slice(0, 24).toLowerCase())) {
      return `${text}${modelAnswer ? ` ${modelAnswer}` : ""}`.trim();
    }
  }
  if (!modelAnswer || isDegenerateAnswer(modelAnswer)) return facts;
  return modelAnswer;
}

/** True when the model stopped because it hit the output-token cap. */
export function hitTokenCap(raw: unknown): boolean {
  if (!raw || typeof raw !== "object") return false;
  const record = raw as Record<string, unknown>;
  if (record.status === "incomplete") return true;
  const details = record.incomplete_details;
  if (details && typeof details === "object" && /max_output_tokens|length/i.test(String((details as { reason?: unknown }).reason ?? ""))) {
    return true;
  }
  const choices = record.choices;
  if (Array.isArray(choices) && choices.some((choice) => choice && typeof choice === "object" && (choice as { finish_reason?: unknown }).finish_reason === "length")) {
    return true;
  }
  return false;
}

/**
 * Drop a dangling, cut-off tail ("…the pressure tug-") so replies end on a
 * complete sentence. Leaves answers alone that already end cleanly or that
 * have no earlier sentence boundary to fall back to.
 */
export function trimDanglingFragment(answer: string, truncated = false): string {
  const text = answer.trimEnd();
  if (!text) return text;
  const endsCleanly = /[.!?…)"'”’\]*_`]$|\p{Extended_Pictographic}$/u.test(text);
  const dangling = /[-–—,;:(]$/.test(text);
  if (endsCleanly && !dangling) return text;
  if (!truncated && !dangling) return text;
  const boundary = Math.max(
    text.lastIndexOf(". "),
    text.lastIndexOf("! "),
    text.lastIndexOf("? "),
    text.lastIndexOf(".\n"),
    text.lastIndexOf("!\n"),
    text.lastIndexOf("?\n"),
  );
  if (boundary < Math.min(40, text.length / 3)) return text;
  return text.slice(0, boundary + 1).trimEnd();
}

/** Temperatures (°F / "60s") in a reply, ignoring percents and small deltas. */
function quotedTemperatures(answer: string): number[] {
  const found: number[] = [];
  const pattern = /(?<![\d.])(-?\d{1,3}(?:\.\d+)?)\s*(?:°|º|deg(?:rees)?\b|s\b|F\b)/gi;
  for (const match of answer.matchAll(pattern)) {
    const value = Number(match[1]);
    if (Number.isFinite(value)) found.push(value);
  }
  return found;
}

/**
 * explain_prediction safety net: if the model quotes weather temperatures far
 * outside the forecast band (e.g. double-converted "high-150s °F"), use the
 * deterministic explain text instead.
 */
export function guardExplainAnswer(answer: string | null, results: ToolResult[]): string | null {
  if (!answer) return answer;
  const explain = [...results]
    .reverse()
    .find((item) => item.ok && item.name === "explain_prediction" && item.data && typeof item.data === "object");
  if (!explain) return answer;
  const data = explain.data as {
    unit?: unknown;
    text?: unknown;
    week_ahead?: Array<{ p10?: number | null; p50?: number; p90?: number | null }>;
    last_observed?: unknown;
  };
  const fallback = typeof data.text === "string" && data.text.trim() ? data.text.trim() : null;
  if (!fallback || data.unit !== "°F" || !Array.isArray(data.week_ahead) || !data.week_ahead.length) {
    return answer;
  }
  const values: number[] = [];
  for (const row of data.week_ahead) {
    for (const key of ["p10", "p50", "p90"] as const) {
      const value = row[key];
      if (typeof value === "number" && Number.isFinite(value)) values.push(value);
    }
  }
  if (typeof data.last_observed === "number") values.push(data.last_observed);
  if (!values.length) return answer;
  const lo = Math.min(...values) - 8;
  const hi = Math.max(...values) + 8;
  // Small numbers are deltas / MAE (e.g. "MAE about 6°F"), not levels.
  const bad = quotedTemperatures(answer).some((value) => Math.abs(value) > 15 && (value < lo || value > hi));
  return bad ? fallback : answer;
}

function logAgentTurn(payload: Record<string, unknown>): void {
  // Workers Observability indexes structured JSON when console.log gets a
  // JSON string with a top-level `message` field (query: message:"agent.turn").
  console.log(
    JSON.stringify({
      message: "agent.turn",
      ...payload,
    }),
  );
}

/** True when a reply promises work instead of doing it ("I'll chart that next"). */
export function promisesFutureWork(answer: string | null): boolean {
  if (!answer) return false;
  return /\b(?:I'll|I will|I'm going to|I am going to|let me|next,? I(?:'ll| will)|give me a (?:sec|second|moment)|stand by|hang tight)\b[^.!?\n]{0,80}\b(?:chart|plot|graph|csv|export|forecast|simulat\w*|run|backtest|train|fetch|pull|grab|explain|compare|calculate|generate)\b/i.test(
    answer,
  );
}

function callKey(call: ParsedToolCall): string {
  const sorted = Object.keys(call.args)
    .sort()
    .map((key) => [key, call.args[key]]);
  return `${call.name}:${JSON.stringify(sorted)}`;
}

/** Drop model-supplied private (`_…`) args; only the loop may set those. */
function publicArgs(args: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args || {})) {
    if (!key.startsWith("_")) out[key] = value;
  }
  return out;
}

function latestHistory(results: ToolResult[]): Record<string, unknown> | null {
  for (let i = results.length - 1; i >= 0; i--) {
    const item = results[i];
    if (item.ok && item.name === "weather_history" && item.data && typeof item.data === "object") {
      return item.data as Record<string, unknown>;
    }
  }
  return null;
}

const NON_WEATHER_SERIES = new Set(["stocks", "crypto", "lunar", "near-earth", "loaded"]);

/** True when a series arg plausibly means "the weather data we just fetched". */
function meansFetchedWeather(series: string): boolean {
  if (!series || series === "weather") return true;
  if (NON_WEATHER_SERIES.has(series)) return false;
  if (/history|highs?|temp/.test(series)) return true;
  // Workspace model ids carry digits; a bare word is usually the place name.
  return !/\d/.test(series);
}

/**
 * Feed this turn's weather_history highs into the simulator when the model
 * asked for it (values_from) or plainly meant "forecast that city" — e.g.
 * series:"weather_history", series:"louisville", or the NYC weather starter
 * after fetching another city's history. A `forecast` call in that situation
 * becomes a simulate_world run on the fetched series.
 */
function injectHistoryValues(
  call: ParsedToolCall,
  results: ToolResult[],
  plan: RequestPlan,
): ParsedToolCall {
  if (call.name !== "simulate_world" && call.name !== "forecast") return call;
  const args = { ...call.args };
  const wantsHistory = args.values_from === "weather_history";
  delete args.values_from;
  const hasValues = Array.isArray(args.values) && args.values.length >= 24;
  const series = typeof args.series === "string" ? args.series.trim().toLowerCase() : "";
  const history = latestHistory(results);
  if (!history || hasValues) return { ...call, args };
  if (!wantsHistory && !(plan.deliverables.includes("history") && meansFetchedWeather(series))) {
    return { ...call, args };
  }
  const full = (history._full || {}) as { dates?: unknown; highs?: unknown };
  const highs = Array.isArray(full.highs) ? (full.highs as unknown[]).filter((v): v is number => typeof v === "number") : [];
  if (highs.length < 24) return { ...call, args };
  const dates = Array.isArray(full.dates) ? (full.dates as unknown[]).map(String) : [];
  delete args.series;
  const horizon = typeof args.horizon === "number" ? args.horizon : Number(args.horizon) || 7;
  return {
    ...call,
    name: "simulate_world",
    args: {
      ...(call.name === "simulate_world" ? args : {}),
      horizon,
      values: highs,
      ...(dates.length === highs.length ? { dates } : {}),
      unit: "°F",
      _from_history: true,
    },
  };
}

const NO_MORE_TOOLS_NOTE =
  "No more tools are available for this turn. Write the final answer now from the tool results above, covering every step the user asked for. " +
  "If a step could not be done, say so plainly in one short line. Do not output tool JSON.";

const FINISH_THE_JOB_NOTE =
  "You stopped before finishing the user's request. Do not answer yet and do not promise to do it later. Reply with ONLY the tool JSON for the next missing step now.";

const WRITE_FULL_ANSWER_NOTE =
  "Every data step is done. Write the complete final answer now, covering every step the user asked for, including a short plain-words explanation. Do not promise further work.";

export async function runAgentTurn(
  question: string,
  mode: ChatMode,
  history: HistoryMessage[],
  deps: AgentLoopDeps,
): Promise<AgentTurnResult> {
  const started = Date.now();
  const turnTimeout = deps.turnTimeoutMs ?? TURN_TIMEOUT_MS;
  const maxTools = deps.maxToolCalls ?? MAX_TOOL_CALLS;
  const maxSteps = deps.maxModelSteps ?? MAX_MODEL_STEPS;
  const neuronBudget = deps.turnNeuronBudget ?? TURN_NEURON_BUDGET;
  const toolTimeout = deps.toolTimeoutMs ?? TOOL_TIMEOUT_MS;
  const clean = sanitizeQuestion(question);
  const caps = deps.caps ?? resolveAiCaps();
  const models = {
    primary: deps.models?.primary?.trim() || resolveChatModels({}).primary,
    fallback: deps.models?.fallback?.trim() || resolveChatModels({}).fallback,
  };
  const allowTools = !isSmallTalk(clean);
  const plan = allowTools ? planRequest(clean) : { steps: [clean], deliverables: [], multiStep: false };
  const multiStep = allowTools && plan.multiStep;
  const system = agentSystemPrompt(mode, allowTools, deps.memoryPrompt);
  const toolNotes: string[] = [];
  const toolLog: Array<Record<string, unknown>> = [];
  const toolsUsed: string[] = [];
  const toolResults: ToolResult[] = [];
  const doneCalls = new Set<string>();
  const executedKeys = new Set<string>();
  let totalNeurons = 0;
  let modelId = PRIMARY_MODEL;
  let voiceResting = false;
  let strictRetryUsed = false;
  let intentRouted = false;
  let lastStepTruncated = false;
  let modelSteps = 0;
  let nudges = 0;
  let plannedRoutes = 0;
  let stopReason = "answered";
  let progressNote: string | null = null;

  const firstReserve = await reserveOrFail(deps, caps);
  if (!firstReserve.ok) {
    logAgentTurn({
      event: "agent.turn",
      user_scope: deps.owner ? "owner" : "ip",
      tools: [],
      tools_used: [],
      tool_count: 0,
      duration_ms: Date.now() - started,
      neurons_estimate: 0,
      model_id: modelId,
      success: false,
      capped: true,
      reason: firstReserve.reason,
      mode: chatMode(mode),
      small_talk: !allowTools,
    });
    return {
      text: firstReserve.message,
      modelId,
      neurons: 0,
      toolsUsed,
      capped: true,
      reason: firstReserve.reason,
      voiceResting: false,
      voiceMessage: null,
    };
  }
  voiceResting = firstReserve.voiceResting;

  let answer: string | null = null;
  let toolCallsThisTurn = 0;

  const notesForModel = (extra?: string): string[] => {
    const notes = [...toolNotes];
    if (progressNote) notes.push(progressNote);
    if (extra) notes.push(extra);
    return notes;
  };

  const refreshProgress = () => {
    if (multiStep) progressNote = checklistNote(plan, toolResults);
  };

  const executeToolCall = async (raw: ParsedToolCall): Promise<void> => {
    let call: ParsedToolCall = { ...raw, args: publicArgs(raw.args) };
    if (
      call.name === "simulate_world" &&
      call.args.values_from === "weather_history" &&
      !latestHistory(toolResults) &&
      toolCallsThisTurn + 1 < maxTools
    ) {
      // The model skipped the fetch it depends on: do it first.
      await executeToolCall({ id: `${call.id}_history`, name: "weather_history", args: { days: 30 } });
    }
    call = injectHistoryValues(call, toolResults, plan);
    if (call.name === "simulate_world" && plan.namedScenario) {
      // "Simulate a cold snap": if the model's scenario has no usable event,
      // run the named one instead of a no-op copy of the baseline.
      const raw = call.args.scenarios ?? call.args.scenario;
      const fallback = scenarioFrom(clean);
      if (fallback && !hasUsableScenario(raw)) {
        const { scenario: _drop, ...rest } = call.args;
        call = { ...call, args: { ...rest, scenarios: [fallback] } };
      }
    }
    doneCalls.add(callKey(raw));
    // A rewritten call (forecast -> simulate on fetched history) can match one already run.
    const executedKey = callKey({ ...call, args: publicArgs(call.args) });
    if (executedKeys.has(executedKey)) return;
    executedKeys.add(executedKey);
    toolCallsThisTurn += 1;
    toolsUsed.push(call.name);
    const result = await dispatchTool(call.name, call.args, deps.toolContext, toolTimeout);
    toolResults.push(result);
    toolNotes.push(toolNote(call, result, mode, multiStep));
    toolLog.push({
      tool: call.name,
      args: summarizeArgs(publicArgs(call.args)),
      duration_ms: result.durationMs,
      ok: result.ok,
      error: result.error || null,
    });
  };

  /** Run every parsed call (not just the first). Independent calls run in parallel. */
  const executeBatch = async (calls: ParsedToolCall[]): Promise<number> => {
    const fresh: ParsedToolCall[] = [];
    const seen = new Set<string>();
    for (const call of calls) {
      const key = callKey(call);
      if (doneCalls.has(key) || seen.has(key)) continue;
      seen.add(key);
      fresh.push(call);
    }
    const room = Math.max(0, maxTools - toolCallsThisTurn);
    const batch = fresh.slice(0, room);
    if (!batch.length) return 0;
    // simulate_world may depend on a weather_history in the same batch.
    const dependent = batch.filter(
      (call) => call.name === "simulate_world" && batch.some((other) => other.name === "weather_history"),
    );
    const independent = batch.filter((call) => !dependent.includes(call));
    if (independent.length === 1) {
      await executeToolCall(independent[0]);
    } else if (independent.length > 1) {
      await Promise.all(independent.map((call) => executeToolCall(call)));
    }
    for (const call of dependent) await executeToolCall(call);
    refreshProgress();
    return batch.length;
  };

  const outOfRoom = (): string | null => {
    if (Date.now() - started >= turnTimeout) return "timeout";
    if (modelSteps >= maxSteps) return "max_steps";
    if (totalNeurons >= neuronBudget) return "neuron_budget";
    return null;
  };

  const modelStep = async (extra?: string) => {
    const step = await runModelWithFallback(
      deps,
      system,
      clean,
      history,
      notesForModel(extra),
      mode,
      models,
      allowTools,
      multiStep,
    );
    modelSteps += 1;
    totalNeurons += step.neurons;
    modelId = step.modelId;
    lastStepTruncated = hitTokenCap(step.raw);
    await settleExtraNeurons(deps, caps, step.neurons);
    return step;
  };

  const proseFrom = (step: { raw: unknown; text: string | null }): string | null => {
    const text =
      finalAnswerText(step.raw) ||
      (step.text && !collectToolCalls(null, step.text).length ? sanitizeAgentAnswer(step.text) : null);
    return text && !isDegenerateAnswer(text) ? text : null;
  };

  let pendingExtra: string | undefined;

  while (true) {
    const room = outOfRoom();
    if (room) {
      stopReason = room;
      break;
    }
    if (modelSteps > 0) {
      const nextReserve = await reserveOrFail(deps, caps);
      if (!nextReserve.ok) {
        answer = factualFallback(toolResults) || nextReserve.message;
        voiceResting = true;
        stopReason = "cap";
        break;
      }
      voiceResting = nextReserve.voiceResting;
    }
    const extra = pendingExtra;
    pendingExtra = undefined;
    const step = await modelStep(extra);

    if (!allowTools) {
      answer = finalAnswerText(step.raw) || (step.text ? sanitizeAgentAnswer(step.text) : null);
      if (!answer || isDegenerateAnswer(answer) || collectToolCalls(null, answer).length) {
        answer =
          mode === "unhinged"
            ? "Hey. What the hell do you want?"
            : "Hey. What are we gonna forge today: forecasts, chaos, or both?";
      }
      break;
    }

    let calls = step.calls;

    // One stricter retry when nothing parsed but the model clearly tried a tool,
    // or when the reply is empty/degenerate and a clear intent applies.
    if (
      !calls.length &&
      !toolResults.length &&
      !strictRetryUsed &&
      (looksLikeFailedToolAttempt(step.text || "") ||
        isDegenerateAnswer(step.text) ||
        (!step.text && routeClearIntent(clean)))
    ) {
      strictRetryUsed = true;
      const retryReserve = await reserveOrFail(deps, caps);
      if (retryReserve.ok) {
        voiceResting = retryReserve.voiceResting;
        const retryStep = await modelStep(STRICT_TOOL_RETRY);
        if (retryStep.calls.length) {
          calls = retryStep.calls;
        } else {
          const retryAnswer =
            finalAnswerText(retryStep.raw) ||
            (retryStep.text ? sanitizeAgentAnswer(retryStep.text) : null);
          if (retryAnswer && !isDegenerateAnswer(retryAnswer) && !collectToolCalls(null, retryAnswer).length) {
            answer = retryAnswer;
          }
        }
      }
    }

    // Deterministic router when the model still produced no tool call.
    if (!calls.length && !toolResults.length && !intentRouted && !answer) {
      const routed = multiStep
        ? routeMissingStep(missingDeliverables(plan, toolResults), clean, toolResults) || routeClearIntent(clean)
        : routeClearIntent(clean);
      if (routed) {
        intentRouted = true;
        calls = [routed];
      }
    }

    if (calls.length) {
      if (toolCallsThisTurn >= maxTools) {
        stopReason = "max_tools";
        answer = null;
        break;
      }
      const missingBefore = multiStep ? missingDeliverables(plan, toolResults).filter((d) => d !== "explain") : [];
      const ran = await executeBatch(calls);
      if (multiStep && ran > 0 && missingBefore.length) {
        // No progress on the checklist (e.g. the model keeps calling side tools):
        // run the next missing step ourselves instead of asking again.
        const missingAfter = missingDeliverables(plan, toolResults).filter((d) => d !== "explain");
        if (missingAfter.length >= missingBefore.length && plannedRoutes < 3 && toolCallsThisTurn < maxTools) {
          const routed = routeMissingStep(missingAfter, clean, toolResults, plan.deliverables);
          if (routed && !doneCalls.has(callKey(routed))) {
            plannedRoutes += 1;
            await executeBatch([routed]);
          }
        }
      }
      if (ran === 0) {
        // Every call was a repeat of one already done: ask for the answer instead.
        pendingExtra = multiStep ? undefined : "You already have those tool results. Answer the user now.";
        if (nudges >= 2) {
          stopReason = "repeat_calls";
          break;
        }
        nudges += 1;
      }
      answer = null;
      continue;
    }

    // The model answered in prose. Before accepting it, check the plan.
    const prose: string | null = answer || proseFrom(step);
    answer = prose;
    if (!multiStep) break;
    const missingData = missingDeliverables(plan, toolResults, prose).filter((item) => item !== "explain");
    const explainMissing = plan.deliverables.includes("explain") && !deliverableDone("explain", toolResults, prose);
    const promised = promisesFutureWork(prose);
    if (!missingData.length && !explainMissing && !promised) break;

    if (missingData.length) {
      if (nudges < 1) {
        nudges += 1;
        pendingExtra = FINISH_THE_JOB_NOTE;
        answer = null;
        continue;
      }
      const routed = plannedRoutes < 3 ? routeMissingStep(missingData, clean, toolResults, plan.deliverables) : null;
      if (routed && toolCallsThisTurn < maxTools && !doneCalls.has(callKey(routed))) {
        plannedRoutes += 1;
        await executeBatch([routed]);
        answer = null;
        continue;
      }
      // Nothing safe left to run: keep the answer and report the gap honestly below.
      break;
    }
    if ((explainMissing || promised) && nudges < 2) {
      nudges += 1;
      pendingExtra = WRITE_FULL_ANSWER_NOTE;
      continue;
    }
    break;
  }

  // Out of steps / tools / time with no final answer: one tools-off wrap-up call.
  if (
    allowTools &&
    (!answer || isDegenerateAnswer(answer)) &&
    toolResults.length &&
    Date.now() - started < turnTimeout + 2_000 &&
    totalNeurons < neuronBudget + caps.reserve * 2 &&
    stopReason !== "cap"
  ) {
    const wrapReserve = await reserveOrFail(deps, caps);
    if (wrapReserve.ok) {
      voiceResting = wrapReserve.voiceResting;
      try {
        const step = await modelStep(NO_MORE_TOOLS_NOTE);
        const wrapped = proseFrom(step);
        if (wrapped) answer = leadWithToolFacts(wrapped, toolResults) || wrapped;
      } catch {
        // Fall through to the factual fallback.
      }
    }
  }

  if (!answer || isDegenerateAnswer(answer)) {
    const fromTools = factualFallback(toolResults);
    if (fromTools) {
      answer = fromTools;
    } else {
      // Never return the generic "could not finish" when a clear tool applies —
      // run the intent router once more as a last resort.
      const routed = routeClearIntent(clean);
      if (routed && toolCallsThisTurn < maxTools && Date.now() - started < turnTimeout) {
        await executeToolCall(routed);
        answer = factualFallback(toolResults);
      }
      if (!answer || isDegenerateAnswer(answer)) {
        answer = fromTools || "I could not finish that answer with the available tools.";
        // Only keep the generic message when no tool obviously applies.
        if (answer.startsWith("I could not finish") && routeClearIntent(clean)) {
          answer = "I could not reach the live data source right now. Please try again in a moment.";
        }
      }
    }
  } else if (toolResults.length) {
    answer = leadWithToolFacts(answer, toolResults) || answer;
  }

  answer = guardExplainAnswer(answer, toolResults);
  answer = answer ? sanitizeAgentAnswer(answer) : answer;
  if (answer) answer = trimDanglingFragment(answer, lastStepTruncated);
  if (isDegenerateAnswer(answer)) {
    const fromTools = factualFallback(toolResults);
    answer = fromTools || "I could not produce an answer right now.";
  }

  // Honest reporting: name any requested step that never happened.
  const missingAtEnd = multiStep
    ? missingDeliverables(plan, toolResults, answer).filter((item) => item !== "explain")
    : [];
  if (missingAtEnd.length && answer) {
    const failed = toolResults.some((item) => !item.ok);
    const already = /\b(couldn't|could not|wasn't able|was not able|can't|cannot|unable)\b/i.test(answer);
    if (!already) {
      answer =
        `${answer}\n\nHeads up: I couldn't finish ${shortLabels(missingAtEnd)} this time` +
        (failed ? " because a data service didn't answer" : "") +
        ". Ask again and I'll retry.";
    }
  }
  const artifacts = allowTools ? buildArtifacts(plan, toolResults) : {};

  let voiceMessage = voiceResting && !deps.owner ? VOICE_RESTING_MESSAGE : null;
  try {
    const budget = await voiceBudget({
      db: deps.db,
      owner: deps.owner,
      caps,
      now: deps.now,
    });
    voiceResting = budget.voiceResting;
    voiceMessage = budget.message;
  } catch {
    // Keep the reserve-derived voice flag.
  }

  const userScope = deps.owner ? "owner" : await ipScope(deps.ip);
  logAgentTurn({
    event: "agent.turn",
    user_scope: userScope,
    tools: toolLog,
    tools_used: toolsUsed,
    tool_count: toolsUsed.length,
    duration_ms: Date.now() - started,
    neurons_estimate: totalNeurons || tokensFromText(clean),
    model_id: modelId,
    success: Boolean(answer) && !isDegenerateAnswer(answer),
    mode: chatMode(mode),
    small_talk: !allowTools,
    intent_routed: intentRouted,
    strict_retry: strictRetryUsed,
    model_steps: modelSteps,
    multi_step: multiStep,
    planned: plan.deliverables,
    missing: missingAtEnd,
    nudges,
    planned_routes: plannedRoutes,
    stop_reason: stopReason,
  });

  return {
    text: answer || "I could not produce an answer right now.",
    modelId,
    neurons: totalNeurons,
    toolsUsed: [...new Set(toolsUsed)],
    voiceResting,
    voiceMessage,
    ...(artifacts.chart ? { chart: artifacts.chart } : {}),
    ...(artifacts.csv ? { csv: artifacts.csv } : {}),
    ...(multiStep ? { steps: { planned: plan.deliverables, missing: missingAtEnd } } : {}),
  };
}
