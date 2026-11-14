/**
 * Tool-call parsing for the agent loop.
 *
 * Prefers native Responses / Chat Completions function_call items. Falls back
 * to a JSON tool protocol the model can emit when native tools fail:
 *   {"tool":"forecast","args":{"series":"weather","horizon":1}}
 *
 * Parsing tolerates code fences, prose wrappers, and nested brace JSON.
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
const TOOL_NAME_HINT =
  /\b(forecast|fetch_live_data|get_evaluation_metrics|compare_horizons|explain_prediction|list_user_series)\b/;

function asToolName(value: unknown): AgentToolName | null {
  if (typeof value !== "string") return null;
  const name = value.trim();
  return TOOL_NAME_SET.has(name) ? (name as AgentToolName) : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function parseArgs(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return {};
    const parsed = safeJsonParse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
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
  if (!name) {
    // Bare {"forecast":{...}} shape
    for (const key of AGENT_TOOL_NAMES) {
      if (record[key] && typeof record[key] === "object" && !Array.isArray(record[key])) {
        const id =
          typeof record.id === "string" && record.id.trim()
            ? record.id.trim()
            : `call_${key}_${index}`;
        return { id, name: key, args: parseArgs(record[key]) };
      }
    }
    return null;
  }
  const rawArgs =
    record.arguments ??
    record.args ??
    record.parameters ??
    (record.function && typeof record.function === "object"
      ? (record.function as { arguments?: unknown; args?: unknown }).arguments ??
        (record.function as { args?: unknown }).args
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

function stripCodeFences(text: string): string {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json|javascript|js|ts)?\s*([\s\S]*?)\s*```$/i);
  if (fenced) return fenced[1].trim();
  return trimmed.replace(/^```(?:json|javascript|js|ts)?\s*/i, "").replace(/\s*```$/i, "").trim();
}

/** Find balanced {...} or [...] slices, respecting strings. */
function extractBalancedJsonSlices(text: string): string[] {
  const out: string[] = [];
  const openers = new Set(["{", "["]);
  for (let i = 0; i < text.length; i++) {
    const startChar = text[i];
    if (!openers.has(startChar)) continue;
    const closeChar = startChar === "{" ? "}" : "]";
    let depth = 0;
    let inString = false;
    let escape = false;
    for (let j = i; j < text.length; j++) {
      const ch = text[j];
      if (inString) {
        if (escape) {
          escape = false;
          continue;
        }
        if (ch === "\\") {
          escape = true;
          continue;
        }
        if (ch === "\"") inString = false;
        continue;
      }
      if (ch === "\"") {
        inString = true;
        continue;
      }
      if (ch === startChar) depth += 1;
      else if (ch === closeChar) {
        depth -= 1;
        if (depth === 0) {
          out.push(text.slice(i, j + 1));
          i = j;
          break;
        }
      }
    }
  }
  return out;
}

function callsFromParsed(parsed: unknown): ParsedToolCall[] {
  if (Array.isArray(parsed)) {
    return parsed
      .map((item, index) => {
        const record = asRecord(item);
        return record ? callFromRecord(record, index) : null;
      })
      .filter((c): c is ParsedToolCall => c !== null);
  }
  const record = asRecord(parsed);
  if (!record) return [];

  for (const key of ["tool_calls", "tools", "calls", "function_calls"]) {
    if (Array.isArray(record[key])) {
      const nested = callsFromParsed(record[key]);
      if (nested.length) return nested;
    }
  }
  if (record.tool_call && typeof record.tool_call === "object") {
    const one = callFromRecord(asRecord(record.tool_call)!, 0);
    if (one) return [one];
  }
  const one = callFromRecord(record, 0);
  return one ? [one] : [];
}

/**
 * Try to parse one or more tool-call JSON objects from model text.
 * Tolerates code fences, leading/trailing prose, and nested wrappers.
 */
export function tryParseToolJson(text: string): ParsedToolCall[] {
  const cleaned = stripCodeFences(text);
  if (!cleaned) return [];

  const direct = safeJsonParse(cleaned);
  if (direct !== null) {
    const calls = callsFromParsed(direct);
    if (calls.length) return calls;
  }

  // Prefer later slices (models often put prose first, JSON last).
  const slices = extractBalancedJsonSlices(cleaned);
  for (let i = slices.length - 1; i >= 0; i--) {
    const parsed = safeJsonParse(slices[i]);
    if (parsed === null) continue;
    const calls = callsFromParsed(parsed);
    if (calls.length) return calls;
  }

  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start >= 0 && end > start) {
    const parsed = safeJsonParse(cleaned.slice(start, end + 1));
    if (parsed !== null) {
      const calls = callsFromParsed(parsed);
      if (calls.length) return calls;
    }
  }
  return [];
}

/**
 * Parse the JSON tool protocol from model text.
 * Accepts whole-message JSON, fenced blocks, or JSON embedded in prose.
 */
export function parseJsonToolProtocol(text: string): ParsedToolCall | null {
  const calls = tryParseToolJson(text);
  return calls[0] ?? null;
}

/** True when the text looks like a tool attempt that failed to parse. */
export function looksLikeFailedToolAttempt(text: string): boolean {
  const t = text.trim();
  if (!t) return false;
  if (tryParseToolJson(t).length) return false;
  if (TOOL_NAME_HINT.test(t) && (t.includes("{") || t.includes("```") || /"tool"\s*:/.test(t))) {
    return true;
  }
  return /^\s*```/.test(t) && TOOL_NAME_HINT.test(t);
}

/**
 * Collect every text-bearing string from a Workers AI result, including
 * reasoning / message / response fields. Used so tool JSON buried in
 * reasoning still parses.
 */
export function extractAllModelText(result: unknown): string {
  const chunks: string[] = [];
  const push = (value: unknown) => {
    if (typeof value === "string" && value.trim()) chunks.push(value);
  };
  if (!result || typeof result !== "object") {
    push(result);
    return chunks.join("\n");
  }
  const record = result as Record<string, unknown>;
  push(record.response);
  push(record.output_text);
  if (Array.isArray(record.output)) {
    for (const item of record.output) {
      if (!item || typeof item !== "object") continue;
      const entry = item as Record<string, unknown>;
      if (typeof entry.text === "string") push(entry.text);
      if (Array.isArray(entry.content)) {
        for (const part of entry.content) {
          if (!part || typeof part !== "object") continue;
          const block = part as Record<string, unknown>;
          push(block.text);
          push(block.output_text);
          push(block.reasoning_text);
        }
      }
    }
  }
  const choices = record.choices;
  if (Array.isArray(choices) && choices[0] && typeof choices[0] === "object") {
    const message = (choices[0] as { message?: { content?: unknown } }).message;
    if (message && typeof message.content === "string") push(message.content);
  }
  const primary = extractModelText(result);
  if (primary) push(primary);
  return chunks.join("\n");
}

export function collectToolCalls(result: unknown, text: string | null): ParsedToolCall[] {
  const native = extractNativeToolCalls(result);
  if (native.length) return native;
  const candidates = [text, result ? extractAllModelText(result) : null].filter(
    (item): item is string => typeof item === "string" && Boolean(item.trim()),
  );
  for (const candidate of candidates) {
    const calls = tryParseToolJson(candidate);
    if (calls.length) return calls;
  }
  return [];
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
  // Strip "fetch_live_data failed: …" style leaks if the model echoes tool notes.
  value = value.replace(
    /\b(?:fetch_live_data|get_evaluation_metrics|compare_horizons|explain_prediction|list_user_series)\s+failed\s*:\s*[^\n]*/gi,
    "",
  );
  value = value.replace(/\bforecast\s+failed\s*:\s*[^\n]*/gi, "");
  return stripLeadingPlanning(value.replace(/\s{2,}/g, " ").trim());
}

/**
 * Detect empty / too-short / repeated-character garbage (e.g. "!!!!!!!!!").
 */
export function isDegenerateAnswer(text: string | null | undefined): boolean {
  if (text == null) return true;
  const cleaned = sanitizeAgentAnswer(text).replace(/\s+/g, " ").trim();
  if (!cleaned) return true;
  if (cleaned.length < 8) return true;
  // Long run of the same character (or same char with light punctuation).
  const compact = cleaned.replace(/\s+/g, "");
  if (compact.length >= 8 && /^(.)\1+$/.test(compact)) return true;
  const unique = new Set(compact.toLowerCase().split(""));
  if (compact.length >= 12 && unique.size <= 2) return true;
  // Mostly punctuation / symbols.
  const letters = cleaned.replace(/[^a-zA-Z0-9]/g, "");
  if (cleaned.length >= 12 && letters.length < 4) return true;
  return false;
}

export function finalAnswerText(result: unknown): string | null {
  const raw = extractModelText(result);
  if (!raw) return null;
  if (parseJsonToolProtocol(raw)) return null;
  const cleaned = sanitizeAgentAnswer(raw);
  if (!cleaned || isDegenerateAnswer(cleaned)) return null;
  return cleaned;
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
