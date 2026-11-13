/**
 * Tool-call parsing for the agent loop.
 *
 * Prefers native Responses / Chat Completions function_call items. Falls back
 * to a single-line JSON protocol the model can emit when native tools fail:
 *   {"tool":"forecast","args":{"series":"weather","horizon":1}}
 */

import { extractModelText, stripLeadingPlanning } from "../live/ai";

export const AGENT_TOOL_NAMES = [
  "forecast",
  "fetch_live_data",
  "get_evaluation_metrics",
  "compare_horizons",
  "explain_prediction",
  "list_user_series",
] as const;

export type AgentToolName = (typeof AGENT_TOOL_NAMES)[number];

export type ParsedToolCall = {
  id: string;
  name: AgentToolName;
  args: Record<string, unknown>;
};

const TOOL_NAME_SET = new Set<string>(AGENT_TOOL_NAMES);

function asToolName(value: unknown): AgentToolName | null {
  if (typeof value !== "string") return null;
  const name = value.trim();
  return TOOL_NAME_SET.has(name) ? (name as AgentToolName) : null;
}

function parseArgs(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return {};
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return {};
    }
  }
  return {};
}

function callFromRecord(record: Record<string, unknown>, index: number): ParsedToolCall | null {
  const name =
    asToolName(record.name) ||
    asToolName(record.tool) ||
    (record.function && typeof record.function === "object"
      ? asToolName((record.function as { name?: unknown }).name)
      : null);
  if (!name) return null;
  const rawArgs =
    record.arguments ??
    record.args ??
    (record.function && typeof record.function === "object"
      ? (record.function as { arguments?: unknown }).arguments
      : undefined);
  const id =
    typeof record.call_id === "string" && record.call_id.trim()
      ? record.call_id.trim()
      : typeof record.id === "string" && record.id.trim()
        ? record.id.trim()
        : `call_${name}_${index}`;
  return { id, name, args: parseArgs(rawArgs) };
}

function callsFromOutput(output: unknown): ParsedToolCall[] {
  if (!Array.isArray(output)) return [];
  const calls: ParsedToolCall[] = [];
  for (const item of output) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const type = typeof record.type === "string" ? record.type : "";
    if (type === "function_call" || type === "tool_call" || type === "custom_tool_call") {
      const parsed = callFromRecord(record, calls.length);
      if (parsed) calls.push(parsed);
      continue;
    }
    if (Array.isArray(record.tool_calls)) {
      for (const nested of record.tool_calls) {
        if (!nested || typeof nested !== "object") continue;
        const parsed = callFromRecord(nested as Record<string, unknown>, calls.length);
        if (parsed) calls.push(parsed);
      }
    }
  }
  return calls;
}

function callsFromChatCompletions(result: Record<string, unknown>): ParsedToolCall[] {
  const choices = result.choices;
  if (!Array.isArray(choices) || !choices[0] || typeof choices[0] !== "object") return [];
  const message = (choices[0] as { message?: { tool_calls?: unknown } }).message;
  if (!message || !Array.isArray(message.tool_calls)) return [];
  const calls: ParsedToolCall[] = [];
  for (const item of message.tool_calls) {
    if (!item || typeof item !== "object") continue;
    const parsed = callFromRecord(item as Record<string, unknown>, calls.length);
    if (parsed) calls.push(parsed);
  }
  return calls;
}

/** Extract native function/tool calls from a Workers AI result. */
export function extractNativeToolCalls(result: unknown): ParsedToolCall[] {
  if (!result || typeof result !== "object") return [];
  const record = result as Record<string, unknown>;
  const fromOutput = callsFromOutput(record.output);
  if (fromOutput.length) return fromOutput;
  return callsFromChatCompletions(record);
}

function tryParseToolJson(raw: string): ParsedToolCall | null {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return null;
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    const record = parsed as Record<string, unknown>;
    if (Array.isArray(record.tool_calls) && record.tool_calls[0]) {
      const first = record.tool_calls[0];
      if (first && typeof first === "object") {
        return callFromRecord(first as Record<string, unknown>, 0);
      }
    }
    return callFromRecord(record, 0);
  } catch {
    return null;
  }
}

/**
 * Parse the JSON tool protocol from model text.
 * Accepts a whole-message JSON object, a fenced ```json block, or the first line.
 */
export function parseJsonToolProtocol(text: string): ParsedToolCall | null {
  const cleaned = text.trim();
  if (!cleaned) return null;
  const direct = tryParseToolJson(cleaned);
  if (direct) return direct;

  const fence = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence?.[1]) {
    const fromFence = tryParseToolJson(fence[1]);
    if (fromFence) return fromFence;
  }

  const firstLine = cleaned.split(/\n/)[0]?.trim() || "";
  if (firstLine !== cleaned) {
    const fromLine = tryParseToolJson(firstLine);
    if (fromLine) return fromLine;
  }
  return null;
}

export function collectToolCalls(result: unknown, text: string | null): ParsedToolCall[] {
  const native = extractNativeToolCalls(result);
  if (native.length) return native;
  if (!text) return [];
  const protocol = parseJsonToolProtocol(text);
  return protocol ? [protocol] : [];
}

const TOOL_JSON_LINE =
  /^\s*\{[\s\S]*"(?:tool|name|tool_calls)"\s*:[\s\S]*\}\s*$/;

/** Remove leaked tool JSON / planning so only the user-facing answer remains. */
export function sanitizeAgentAnswer(text: string): string {
  let value = stripLeadingPlanning(text.trim());
  value = value.replace(/```(?:json)?\s*\{[\s\S]*?"(?:tool|name|tool_calls)"[\s\S]*?\}\s*```/gi, "");
  const lines = value.split("\n");
  const kept = lines.filter((line) => !TOOL_JSON_LINE.test(line.trim()));
  value = kept.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  value = value.replace(/^\s*TOOL_CALL\s*:\s*\S+\s*\{[\s\S]*\}\s*$/gim, "").trim();
  return stripLeadingPlanning(value);
}

export function finalAnswerText(result: unknown): string | null {
  const raw = extractModelText(result);
  if (!raw) return null;
  if (parseJsonToolProtocol(raw)) return null;
  const cleaned = sanitizeAgentAnswer(raw);
  return cleaned || null;
}

export function summarizeArgs(args: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    if (typeof value === "string") {
      out[key] = value.length > 80 ? `${value.slice(0, 77)}...` : value;
    } else if (typeof value === "number" || typeof value === "boolean" || value == null) {
      out[key] = value;
    } else if (Array.isArray(value)) {
      out[key] = value.slice(0, 8);
    } else {
      out[key] = "[object]";
    }
  }
  return out;
}
