/**
 * Lightweight safety filter. Blocks ONLY truly dangerous content:
 * - weapons capable of mass harm (explosives, chemical, biological, nuclear, radiological)
 * - child sexual exploitation
 * - operational instructions for serious violence
 * - self-harm methods (answered with support resources instead)
 *
 * Controversial, political, conspiracy, medical, and legal topics are never
 * blocked. To keep cost near zero, Llama Guard runs only when a cheap keyword
 * prefilter sees a possibly dangerous ask (most turns never call it), and the
 * guard's verdict only counts for the narrow categories above.
 */

import type { AiLike } from "../live/ai";

export const GUARD_MODEL = "@cf/meta/llama-guard-3-8b";

/** Llama Guard 3 categories we act on. Everything else (S2, S5–S8, S10, S12–S14) passes. */
export const BLOCKING_CATEGORIES = new Set(["S1", "S4", "S9", "S11"]);

export type SafetyVerdict =
  | { blocked: false; checked: boolean }
  | { blocked: true; checked: true; category: string; reply: string };

// Prefilter: wide enough to catch the dangerous asks, narrow enough that ordinary
// questions (history, news, conspiracies, medicine, law) almost never reach the guard.
const SELF_HARM =
  /\b(kill (?:my ?self|me)|end (?:my|it all)|suicid\w*|self[- ]?harm\w*|cut(?:ting)? myself|hurt myself|overdose|od on|lethal dose|painless (?:way|death)|want to die|hang myself)\b/i;
const MASS_HARM =
  /\b(bomb|explosive|detonat\w*|ied|pipe ?bomb|tatp|petn|rdx|thermite|napalm|nerve agent|sarin|vx|novichok|ricin|anthrax|botulinum|toxin|pathogen|bioweapon|chemical weapon|nuclear (?:bomb|weapon|device)|dirty bomb|enrich(?:ed|ing)? uranium|weapons?[- ]grade|plutonium|mustard gas|chlorine gas|gain[- ]of[- ]function|weaponi[sz]\w*|ghost gun|3d[- ]print\w* (?:gun|firearm)|full[- ]auto|silencer|suppressor)\b/i;
const VIOLENCE =
  /\b(kill|murder|poison|assassinat\w*|kidnap\w*|shoot up|mass shooting|stab|strangle|torture|bury a body|get away with)\b/i;
const CSAM = /\b(child|children|kid|kids|minor|minors|underage|preteen|teen|loli|\d{1,2}[- ]?(?:yo|year[- ]old))\b/i;
const SEXUAL = /\b(sex\w*|nude\w*|naked|porn\w*|erotic\w*|explicit|nsfw)\b/i;
const HOW_TO =
  /\b(how (?:do|can|could|would|should) (?:i|we|you|one|someone)|how to|step[- ]by[- ]step|instructions? (?:for|to|on)|recipe|ingredients?|synthes\w*|at home|diy|homemade|teach me|show me how|walk me through|best way to|easiest way to|cheapest way to|without (?:getting|being) caught|(?:make|build|cook|brew|mix|assemble|grow|culture) (?:a|an|me|some|my own|your own))\b/i;

/** Which risk the text might carry, or null when no guard call is needed. */
export function prefilterRisk(text: string): "self_harm" | "danger" | null {
  const q = text.replace(/\s+/g, " ").trim();
  if (!q) return null;
  if (SELF_HARM.test(q)) return "self_harm";
  if (CSAM.test(q) && SEXUAL.test(q)) return "danger";
  if ((MASS_HARM.test(q) || VIOLENCE.test(q)) && HOW_TO.test(q)) return "danger";
  return null;
}

export const SELF_HARM_REPLY =
  "I'm really glad you said something, and I'm not going to help with ways to get hurt. You deserve support right now. " +
  "In the US you can call or text 988 (Suicide & Crisis Lifeline) any time, or text HOME to 741741. Outside the US, findahelpline.com lists free local lines. " +
  "If you're in immediate danger, call your local emergency number. If you want, tell me what's going on and I'll listen.";

export const DANGER_REPLY =
  "That's one of the few things I won't help with: step-by-step help that could get people seriously hurt. " +
  "I'm happy to talk about the history, the science at a high level, the news, or the debate around it.";

export const CSAM_REPLY = "I won't help with anything that sexualizes minors.";

function replyFor(category: string): string {
  if (category === "S11") return SELF_HARM_REPLY;
  if (category === "S4") return CSAM_REPLY;
  return DANGER_REPLY;
}

/** Parse Llama Guard output: object `{safe, categories}` or text "unsafe\nS9,S1". */
export function parseGuard(raw: unknown): { safe: boolean; categories: string[] } {
  let value: unknown = raw;
  if (value && typeof value === "object" && "response" in (value as object)) {
    value = (value as { response?: unknown }).response;
  }
  if (value && typeof value === "object") {
    const record = value as { safe?: unknown; categories?: unknown };
    const categories = Array.isArray(record.categories)
      ? record.categories.map((item) => String(item).trim().toUpperCase().match(/S\d{1,2}/)?.[0] || "").filter(Boolean)
      : [];
    return { safe: record.safe !== false, categories };
  }
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    if (!text.startsWith("unsafe")) return { safe: true, categories: [] };
    const categories = [...value.toUpperCase().matchAll(/S\d{1,2}/g)].map((m) => m[0]);
    return { safe: false, categories };
  }
  return { safe: true, categories: [] };
}

async function runGuard(ai: AiLike, messages: Array<{ role: "user" | "assistant"; content: string }>): Promise<{ safe: boolean; categories: string[] }> {
  const raw = await ai.run(GUARD_MODEL, {
    messages: messages.map((item) => ({ role: item.role, content: item.content.slice(0, 4000) })),
    max_tokens: 24,
  });
  return parseGuard(raw);
}

function blockingCategory(verdict: { safe: boolean; categories: string[] }, risk: "self_harm" | "danger"): string | null {
  if (verdict.safe) return null;
  const hit = verdict.categories.find((item) => BLOCKING_CATEGORIES.has(item));
  if (!hit) return null;
  // S11 only counts when the prefilter also saw self-harm language, and the
  // other blocking categories only when it saw a how-to for mass harm/violence/CSAM.
  if (hit === "S11" && risk !== "self_harm") return null;
  if (hit !== "S11" && risk !== "danger") return null;
  return hit;
}

/**
 * Check the user's message. Fails open (never blocks) when the guard errors,
 * because a false block on an ordinary question is the worse failure here.
 */
export async function checkInput(ai: AiLike, question: string): Promise<SafetyVerdict> {
  const risk = prefilterRisk(question);
  if (!risk) return { blocked: false, checked: false };
  try {
    const verdict = await runGuard(ai, [{ role: "user", content: question }]);
    const category = blockingCategory(verdict, risk);
    if (category) return { blocked: true, checked: true, category, reply: replyFor(category) };
    return { blocked: false, checked: true };
  } catch {
    return { blocked: false, checked: true };
  }
}

/** Cheap output check: only when the input or answer itself trips the prefilter. */
export async function checkOutput(ai: AiLike, question: string, answer: string): Promise<SafetyVerdict> {
  const risk = prefilterRisk(question) || prefilterRisk(answer);
  if (!risk) return { blocked: false, checked: false };
  try {
    const verdict = await runGuard(ai, [
      { role: "user", content: question },
      { role: "assistant", content: answer },
    ]);
    const category = blockingCategory(verdict, risk);
    if (category) return { blocked: true, checked: true, category, reply: replyFor(category) };
    return { blocked: false, checked: true };
  } catch {
    return { blocked: false, checked: true };
  }
}

// ---------------------------------------------------------------------------
// Creator honesty (default mode never claims another maker).

export const CREATOR_ANSWER =
  "Nous was created by, and is owned by, Andrew Gray, founder, owner and CEO of YouNeeK (pronounced \"unique\"). It runs on GPT-OSS 120B, an open model Andrew chose; OpenAI released that model, but OpenAI and xAI didn't make Nous.";

export function asksAboutCreator(question: string): boolean {
  const q = question.toLowerCase();
  return (
    /\b(who|which company|what company)\b[^?]{0,40}\b(made|make|created|create|built|build|owns?|developed|develop|programmed|trained|behind|runs)\b[^?]{0,20}\b(you|nous|this (?:ai|app|bot|site|thing))\b/.test(q) ||
    /\b(your|nous'?s?)\s+(creator|maker|owner|developer|founder|parent company)\b/.test(q) ||
    /\bare you (?:made|built|created|owned) by\b/.test(q) ||
    /\bwhat (?:ai |llm |language )?model (?:are you|do you (?:use|run on)|powers you)\b/.test(q)
  );
}

/** Make sure a creator answer names Andrew Gray and never credits OpenAI/xAI as the maker. */
export function enforceCreatorAnswer(question: string, answer: string | null): string | null {
  if (!asksAboutCreator(question)) return answer;
  const text = (answer || "").trim();
  // GPT-OSS often emits non-breaking spaces/hyphens; \s covers U+00A0 / U+202F.
  const namesAndrew = /andrew\s+gray/i.test(text);
  const wrongMaker = /\b(?:made|created|built|developed|owned)\s+by\s+(?:openai|xai|x\.ai|anthropic|google|meta)\b|\bi(?:'m| am) (?:chatgpt|grok|gpt-4|claude)\b/i.test(text);
  if (namesAndrew && !wrongMaker) return text;
  if (!text || wrongMaker) return CREATOR_ANSWER;
  return `${CREATOR_ANSWER} ${text}`;
}
