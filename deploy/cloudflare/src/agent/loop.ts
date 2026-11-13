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
import {
  collectToolCalls,
  finalAnswerText,
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
};

const JSON_PROTOCOL =
  "Tools are optional. For greetings, small talk, thanks, or questions that need no live data or model facts, answer in plain language with NO tool call. " +
  "When you do need a tool, reply with ONLY one JSON object and nothing else: " +
  '{"tool":"<name>","args":{...}}. ' +
  "Valid tools:\n" +
  toolCatalogForPrompt() +
  "\nAfter tool results are provided, answer the user in plain language. " +
  "Never invent live weather or prices. Never show tool JSON, planning, or hidden reasoning to the user. " +
  "Call at most one tool per reply.";

/** Short greetings / chit-chat that must never call tools. */
export function isSmallTalk(question: string): boolean {
  const text = question.replace(/\s+/g, " ").trim().toLowerCase();
  if (!text || text.length > 80) return false;
  return /^(hey+|hi+|hello+|yo+|sup+|howdy+)([!.? ]|$)|^(good\s+(morning|afternoon|evening)|thanks|thank you|thx|ty|ok|okay|cool|bye|goodbye|see ya|what's up|whats up|how are you)([!.? ]|$)/i.test(
    text,
  );
}

export function agentSystemPrompt(mode: ChatMode, allowTools: boolean): string {
  const base = systemPromptFor(mode);
  if (!allowTools) {
    if (mode === "unhinged") return base;
    return (
      `${base}\n\n` +
      "Directness: answer exactly what was asked. Lead with the answer. Keep it short. " +
      "No rambling, tangents, or unasked extras. Wit stays light and secondary. " +
      "Do not call tools and do not emit tool JSON."
    );
  }
  if (mode === "unhinged") {
    return `${base}\n\n${JSON_PROTOCOL}`;
  }
  return (
    `${base}\n\n` +
    "Directness: answer exactly what was asked. Lead with the answer. Keep it short. " +
    "No rambling, tangents, or unasked extras. Wit stays light and secondary.\n\n" +
    JSON_PROTOCOL
  );
}

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
}): Record<string, unknown> {
  const { model, system, question, history, toolNotes, mode, allowTools } = options;
  const prior = historyForModel(history);
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
      max_output_tokens: 220,
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
    max_tokens: 220,
  };
  if (allowTools) inputs.tools = toolsForChatCompletions();
  if (mode === "unhinged") inputs.temperature = 0.9;
  return inputs;
}

function toolNote(call: ParsedToolCall, result: ToolResult): string {
  const payload = result.ok
    ? { tool: call.name, ok: true, result: result.data }
    : { tool: call.name, ok: false, error: result.error || "failed" };
  return `Tool result for ${call.name}:\n${JSON.stringify(payload).slice(0, 3500)}\nUsing only these facts, answer the user. Do not call another tool unless you still lack a required fact.`;
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
): Promise<ModelStep> {
  const inputs = buildAgentInputs({
    model,
    system,
    question,
    history,
    toolNotes,
    mode,
    allowTools,
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
    );
  } catch (error) {
    if (models.fallback === models.primary) throw error;
    const details = errorDetails(error);
    console.error(
      JSON.stringify({
        event: "agent.primary_failed",
        model: models.primary,
        message: details.message,
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
    );
  }
}

function factualFallback(results: ToolResult[]): string {
  const lines: string[] = [];
  for (const result of results) {
    if (!result.ok) {
      lines.push(`${result.name} failed: ${result.error || "unavailable"}`);
      continue;
    }
    if (result.data && typeof result.data === "object" && "text" in (result.data as object)) {
      const text = (result.data as { text?: unknown }).text;
      if (typeof text === "string" && text.trim()) {
        lines.push(text.trim());
        continue;
      }
    }
    lines.push(`${result.name}: ${JSON.stringify(result.data).slice(0, 500)}`);
  }
  return lines.join("\n\n") || "I could not finish that answer with the available tools.";
}

export async function runAgentTurn(
  question: string,
  mode: ChatMode,
  history: HistoryMessage[],
  deps: AgentLoopDeps,
): Promise<AgentTurnResult> {
  const started = Date.now();
  const turnTimeout = deps.turnTimeoutMs ?? TURN_TIMEOUT_MS;
  const maxTools = deps.maxToolCalls ?? MAX_TOOL_CALLS;
  const toolTimeout = deps.toolTimeoutMs ?? TOOL_TIMEOUT_MS;
  const clean = sanitizeQuestion(question);
  const caps = deps.caps ?? resolveAiCaps();
  const models = {
    primary: deps.models?.primary?.trim() || resolveChatModels({}).primary,
    fallback: deps.models?.fallback?.trim() || resolveChatModels({}).fallback,
  };
  const allowTools = !isSmallTalk(clean);
  const system = agentSystemPrompt(mode, allowTools);
  const toolNotes: string[] = [];
  const toolLog: Array<Record<string, unknown>> = [];
  const toolsUsed: string[] = [];
  const toolResults: ToolResult[] = [];
  let totalNeurons = 0;
  let modelId = PRIMARY_MODEL;
  let voiceResting = false;

  const firstReserve = await reserveOrFail(deps, caps);
  if (!firstReserve.ok) {
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

  while (Date.now() - started < turnTimeout) {
    const step = await runModelWithFallback(
      deps,
      system,
      clean,
      history,
      toolNotes,
      mode,
      models,
      allowTools,
    );
    totalNeurons += step.neurons;
    modelId = step.modelId;
    await settleExtraNeurons(deps, caps, step.neurons);

    if (!allowTools) {
      answer = finalAnswerText(step.raw) || (step.text ? sanitizeAgentAnswer(step.text) : null);
      if (!answer || collectToolCalls(null, answer).length) {
        answer =
          mode === "unhinged"
            ? "Hey. What the hell do you want?"
            : "Hey — what do you want to forge?";
      }
      break;
    }

    if (!step.calls.length) {
      answer = finalAnswerText(step.raw) || (step.text ? sanitizeAgentAnswer(step.text) : null);
      break;
    }

    if (toolCallsThisTurn >= maxTools) {
      answer =
        finalAnswerText(step.raw) ||
        (step.text && !collectToolCalls(null, step.text).length
          ? sanitizeAgentAnswer(step.text)
          : null) ||
        factualFallback(toolResults);
      break;
    }

    const call = step.calls[0];
    toolCallsThisTurn += 1;
    toolsUsed.push(call.name);
    const result = await dispatchTool(call.name, call.args, deps.toolContext, toolTimeout);
    toolResults.push(result);
    toolNotes.push(toolNote(call, result));
    toolLog.push({
      tool: call.name,
      args: summarizeArgs(call.args),
      duration_ms: result.durationMs,
      ok: result.ok,
      error: result.error || null,
    });

    if (Date.now() - started >= turnTimeout) {
      answer = factualFallback(toolResults);
      break;
    }

    const nextReserve = await reserveOrFail(deps, caps);
    if (!nextReserve.ok) {
      answer = factualFallback(toolResults);
      voiceResting = true;
      break;
    }
    voiceResting = nextReserve.voiceResting;
  }

  if (!answer) {
    answer = factualFallback(toolResults);
  }
  answer = sanitizeAgentAnswer(answer);

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
  console.log(
    JSON.stringify({
      event: "agent.turn",
      user_scope: userScope,
      tools: toolLog,
      tools_used: toolsUsed,
      tool_count: toolsUsed.length,
      duration_ms: Date.now() - started,
      neurons_estimate: totalNeurons || tokensFromText(clean),
      model_id: modelId,
      success: Boolean(answer),
      mode: chatMode(mode),
      small_talk: !allowTools,
    }),
  );

  return {
    text: answer || "I could not produce an answer right now.",
    modelId,
    neurons: totalNeurons,
    toolsUsed: [...new Set(toolsUsed)],
    voiceResting,
    voiceMessage,
  };
}
