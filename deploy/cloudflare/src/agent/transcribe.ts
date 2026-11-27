// Talk mode hearing. The browser records one spoken turn as a small 16 kHz mono
// WAV and sends it here; Workers AI Whisper turns it into text. This works the
// same in every browser, unlike the built-in speech recognizer (missing or
// flaky on iPhone, Firefox, and installed apps).
//
// Cost: whisper-large-v3-turbo is 46.63 neurons per audio minute, so a 5 second
// question is about 4 neurons. Every call is checked against the shared daily
// neuron cap first, so it can never push the Worker past the $5 plan.

import { resolveAiCaps, tryReserve, type AiCaps, type AiLike, type D1Like } from "../live/ai";

export const STT_MODEL = "@cf/openai/whisper-large-v3-turbo";
export const STT_MAX_BYTES = 1_200_000;
export const STT_MIN_BYTES = 4_000;
export const STT_NEURONS_PER_MINUTE = 46.63;
export const STT_CAP_MESSAGE =
  "The daily voice limit is used up, so I'll keep going in text. Voice is back tomorrow.";

export function toBase64(bytes: Uint8Array): string {
  let binary = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    binary += String.fromCharCode(...bytes.subarray(i, i + step));
  }
  return btoa(binary);
}

// Whisper hears the name "Henad" (said HEN-add) many ways. These spellings are
// never ordinary words, so they always become Henad.
const HENAD_ALWAYS = /\b(?:hen[\s-]+add?|hen[\s-]+nad|henn?add?|hennad|hanadd?|heinad)\b/gi;
// "Henna" / "Hena" / the old name are real words, so only fix them when they
// are used to address the assistant: at the start, or after hey/hi/ok/yo.
const HENAD_ADDRESS = /(^|\b(?:hey|hi|hello|ok|okay|yo)[,\s]+)(?:henna|hena|hennah|nous|noose)\b/gi;

/** Fix the assistant's name in a Talk-mode transcript. */
export function normalizeName(text: string): string {
  return text.replace(HENAD_ALWAYS, "Henad").replace(HENAD_ADDRESS, (_m, lead: string) => `${lead}Henad`);
}

/** Whisper sometimes invents text for silence or noise. Drop those. */
export function cleanTranscript(raw: unknown): string {
  const text = typeof raw === "string" ? raw.replace(/\s+/g, " ").trim() : "";
  if (!text) return "";
  const bare = text.toLowerCase().replace(/[^a-z' ]/g, "").trim();
  const filler = new Set([
    "you",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "bye",
    "uh",
    "um",
    "hmm",
    "mm",
    "the",
  ]);
  if (filler.has(bare)) return "";
  if (/^[\[(].*[\])]$/.test(text)) return "";
  return normalizeName(text).slice(0, 500);
}

export type TranscribeDeps = {
  ai: AiLike;
  db: D1Like;
  ip: string | null;
  caps?: AiCaps;
};

export type TranscribeResult = { status: number; body: Record<string, unknown> };

export async function transcribeAudio(
  deps: TranscribeDeps,
  bytes: Uint8Array,
): Promise<TranscribeResult> {
  if (bytes.length < STT_MIN_BYTES) return { status: 200, body: { text: "" } };
  if (bytes.length > STT_MAX_BYTES) {
    return { status: 413, body: { error: "That recording is too long. Try a shorter question." } };
  }
  const caps = deps.caps ?? resolveAiCaps();
  // 16 kHz 16-bit mono WAV is 32,000 bytes a second. Reserve what this clip costs, plus a little.
  const seconds = Math.max(1, (bytes.length - 44) / 32000);
  const reserveNeurons = Math.ceil((seconds / 60) * STT_NEURONS_PER_MINUTE) + 2;
  // owner: true skips only the per-visitor request count (a Talk turn is two calls);
  // the shared daily neuron cap still applies to every call.
  const reserved = await tryReserve({
    db: deps.db,
    ip: deps.ip,
    owner: true,
    reserveNeurons,
    caps,
  });
  if (!reserved.ok) {
    return { status: 429, body: { error: STT_CAP_MESSAGE, capped: true, reason: reserved.reason } };
  }
  let text = "";
  try {
    const out = (await deps.ai.run(STT_MODEL, { audio: toBase64(bytes), language: "en" })) as {
      text?: unknown;
    } | null;
    text = cleanTranscript(out?.text);
  } catch {
    return { status: 502, body: { error: "I couldn't hear that. Try again." } };
  }
  return { status: 200, body: { text } };
}
