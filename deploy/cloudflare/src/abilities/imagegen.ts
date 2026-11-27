/**
 * Making images from chat ("draw a fox in a spacesuit").
 *
 * Model: Workers AI `@cf/black-forest-labs/flux-1-schnell`, 4 steps, 1024x1024:
 * 4 tiles x 4.8 + 4 steps x 9.6 = about 58 neurons (~$0.0006) per image.
 * Each image reserves neurons in the shared 25,000/day safety cap, so the
 * monthly ceiling does not move.
 *
 * Caps: 5 images/day per account, 30 for the owner, 100/day for everyone
 * (worst case ~5,800 neurons/day).
 *
 * Safety blocks only what is truly illegal or dangerous to make: sexual
 * content involving minors, sexual violence, and explicit sexual images
 * (the last one keeps the site inside Cloudflare's usage terms).
 * Llama Guard runs only when a cheap keyword check sees a risky prompt.
 */

import { tryReserve, resolveAiCaps, type AiLike, type D1Like, type AiCaps } from "../live/ai";
import { abilityCaps, claimAbility, newAbilityId, scopeFolder, type R2Like } from "./limits";

export const IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell";
export const IMAGE_STEPS = 4;
export const IMAGE_NEURONS = 58;
export const GUARD_MODEL = "@cf/meta/llama-guard-3-8b";
export const IMAGE_MARKER = /\[\[image:(img_[a-f0-9]{20})\]\]/g;

const NOUNS =
  "image|images|picture|pictures|pic|pics|photo|photos|drawing|illustration|painting|art|artwork|logo|icon|wallpaper|poster|portrait|sketch|meme|avatar|cartoon|render|graphic|banner|sticker|tattoo design|album cover|thumbnail";
const VERBS = "draw|paint|sketch|illustrate|generate|create|make|design|render|produce|imagine|show me|give me|whip up";

const ASK_IMAGE = new RegExp(`\\b(${VERBS})\\b[^.?!]{0,40}?\\b(${NOUNS})\\b`, "i");
const STARTS_DRAW = /^(?:(?:can|could|would|will) you\s+|please\s+|pls\s+|hey henad,?\s+|henad,?\s+)*(draw|paint|sketch|illustrate)\b/i;
const NOT_IMAGE =
  /\b(chart|graph|plot|forecast|predict|diagram of (?:the )?data|spreadsheet|csv)\b|^\s*(how|why|what|where|when|who|which)\b|\b(how (?:do|can|to|would) (?:i|you|we|one)?\s*(?:draw|make|create|design|paint))\b/i;

export function isImageRequest(question: string): boolean {
  const q = question.replace(/\s+/g, " ").trim();
  if (!q || q.length > 600) return false;
  if (NOT_IMAGE.test(q)) return false;
  return STARTS_DRAW.test(q) || ASK_IMAGE.test(q);
}

/** Turn "can you draw me a picture of a red fox" into "a red fox". */
export function imagePrompt(question: string): string {
  let q = question.replace(/\s+/g, " ").trim();
  q = q.replace(/^(?:(?:hey )?henad,?\s+)?(?:(?:can|could|would|will) you\s+|please\s+|pls\s+|i want you to\s+|i'd like you to\s+)*/i, "");
  const generic = "image|images|picture|pictures|pic|pics|photo|photos|drawing|illustration|painting|art|artwork|render|graphic";
  const lead = new RegExp(
    `^(?:${VERBS})\\s+(?:me\\s+|us\\s+)?(?:(?:(?:a|an|the|some|one|another)\\s+)?(?:(?:quick|cool|nice|new|realistic|cute|simple|detailed)\\s+)?(?:${generic})\\s*(?:of|showing|with|that shows|where|about)?\\s*)?`,
    "i",
  );
  const stripped = q.replace(lead, "").replace(/[?!.]+$/, "").replace(/\s*,?\s*please$/i, "").trim();
  return (stripped.length >= 3 ? stripped : q).slice(0, 500);
}

/* ---------------------------------------------------------------- safety */

const MINOR = /\b(child|children|kid|kids|minor|minors|underage|preteen|pre-teen|teen|teens|teenage\w*|young girl|young boy|little girl|little boy|schoolgirl|schoolboy|toddler|baby|infant|loli|shota|\d{1,2}[- ]?(?:yo|y\/o|years?[- ]old))\b/i;
const SEXUAL = /\b(sex\w*|nude\w*|naked|nsfw|porn\w*|erotic\w*|explicit|topless|lingerie|undress\w*|genital\w*|breasts?|nipples?|boobs?|fetish\w*|hentai|xxx|strip(?:ping|per)?)\b/i;
const VIOLENT_SEX = /\b(rape|raping|molest\w*|non-?consensual|sexual assault)\b/i;

export const IMAGE_CSAM_REPLY = "I won't make anything that sexualizes minors.";
export const IMAGE_BLOCK_REPLY =
  "That's one of the few images I won't make. I'm happy to try a different take on it, though.";

export function parseGuard(raw: unknown): { safe: boolean; categories: string[] } {
  let value: unknown = raw;
  if (value && typeof value === "object" && "response" in (value as object)) {
    value = (value as { response?: unknown }).response;
  }
  if (value && typeof value === "object") {
    const record = value as { safe?: unknown; categories?: unknown };
    const categories = Array.isArray(record.categories)
      ? record.categories
          .map((item) => String(item).toUpperCase().match(/S\d{1,2}/)?.[0] || "")
          .filter(Boolean)
      : [];
    return { safe: record.safe !== false, categories };
  }
  if (typeof value === "string") {
    const text = value.trim();
    if (!text.toLowerCase().startsWith("unsafe")) return { safe: true, categories: [] };
    return { safe: false, categories: [...text.toUpperCase().matchAll(/S\d{1,2}/g)].map((m) => m[0]) };
  }
  return { safe: true, categories: [] };
}

/** S3 sex crimes, S4 child sexual exploitation, S12 explicit sexual content. */
const IMAGE_BLOCKING = new Set(["S3", "S4", "S12"]);

export async function checkImagePrompt(
  ai: AiLike,
  prompt: string,
): Promise<{ blocked: false } | { blocked: true; reply: string }> {
  const sexual = SEXUAL.test(prompt);
  if (sexual && MINOR.test(prompt)) return { blocked: true, reply: IMAGE_CSAM_REPLY };
  if (VIOLENT_SEX.test(prompt) && (sexual || /\b(image|picture|photo|draw)\b/i.test(prompt))) {
    return { blocked: true, reply: IMAGE_BLOCK_REPLY };
  }
  if (!sexual && !VIOLENT_SEX.test(prompt)) return { blocked: false };
  try {
    const verdict = parseGuard(
      await ai.run(GUARD_MODEL, {
        messages: [{ role: "user", content: `Generate an image of: ${prompt}`.slice(0, 2000) }],
        max_tokens: 24,
      }),
    );
    if (!verdict.safe && verdict.categories.some((cat) => IMAGE_BLOCKING.has(cat))) {
      return {
        blocked: true,
        reply: verdict.categories.includes("S4") ? IMAGE_CSAM_REPLY : IMAGE_BLOCK_REPLY,
      };
    }
    return { blocked: false };
  } catch {
    // Explicit wording with no verdict: stay inside the platform terms.
    return { blocked: true, reply: IMAGE_BLOCK_REPLY };
  }
}

/* --------------------------------------------------------------- generate */

export type ImageDeps = {
  ai: AiLike;
  db: D1Like;
  bucket: R2Like;
  scope: string;
  owner: boolean;
  capKey: string | null;
  env?: Record<string, unknown>;
  aiCaps?: AiCaps;
  now?: Date;
};

export type ImageResult = { text: string; imageId?: string; capped?: boolean; blocked?: boolean };

function base64ToBytes(value: string): Uint8Array {
  const clean = value.replace(/^data:[^,]+,/, "").replace(/\s+/g, "");
  const binary = atob(clean);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function imageBytesFrom(result: unknown): Uint8Array | null {
  if (!result) return null;
  if (result instanceof Uint8Array) return result;
  if (result instanceof ArrayBuffer) return new Uint8Array(result);
  if (typeof result === "object") {
    const image = (result as { image?: unknown }).image;
    if (typeof image === "string" && image.length > 100) return base64ToBytes(image);
  }
  return null;
}

function sniffType(bytes: Uint8Array): string {
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return "image/png";
  if (bytes[0] === 0x52 && bytes[1] === 0x49) return "image/webp";
  return "image/jpeg";
}

export async function generateImage(deps: ImageDeps, question: string): Promise<ImageResult> {
  const prompt = imagePrompt(question);
  const safety = await checkImagePrompt(deps.ai, prompt);
  if (safety.blocked) return { text: safety.reply, blocked: true };

  const claim = await claimAbility({
    db: deps.db,
    kind: "images",
    scope: deps.scope,
    owner: deps.owner,
    caps: abilityCaps("images", deps.env),
    now: deps.now,
  });
  if (!claim.ok) {
    return {
      text:
        claim.reason === "user"
          ? `That's your ${claim.limit} images for today. New ones unlock at midnight UTC, and I can still describe or plan the picture in words.`
          : "Image making is resting for today to keep Henad free. Text chat still works.",
      capped: true,
    };
  }
  const reserved = await tryReserve({
    db: deps.db,
    ip: deps.capKey,
    owner: deps.owner,
    reserveNeurons: IMAGE_NEURONS,
    caps: deps.aiCaps ?? resolveAiCaps(deps.env as Record<string, string | undefined>),
    now: deps.now,
  });
  if (!reserved.ok) return { text: reserved.message, capped: true };

  let bytes: Uint8Array | null = null;
  try {
    bytes = imageBytesFrom(await deps.ai.run(IMAGE_MODEL, { prompt, steps: IMAGE_STEPS }));
  } catch {
    bytes = null;
  }
  if (!bytes || bytes.length < 200) {
    return { text: "The image didn't come out this time. Give it another try in a moment." };
  }
  const id = newAbilityId("img");
  const folder = await scopeFolder(deps.scope);
  const type = sniffType(bytes);
  const key = `images/${folder}/${id}.${type === "image/png" ? "png" : type === "image/webp" ? "webp" : "jpg"}`;
  await deps.bucket.put(key, bytes, { httpMetadata: { contentType: type } });
  await deps.db
    .prepare("INSERT INTO generated_images (id, scope, prompt, r2_key, mime, created_at) VALUES (?, ?, ?, ?, ?, ?)")
    .bind(id, deps.scope, prompt, key, type, (deps.now ?? new Date()).toISOString())
    .run();
  const left = Math.max(0, claim.limit - claim.used);
  const tail = left === 0 ? " That was your last one for today." : left <= 2 ? ` ${left} left today.` : "";
  return { text: `Here you go: ${prompt}.${tail}\n[[image:${id}]]`, imageId: id };
}

/** GET /agent/images/<id>: only the person who made it can load it. */
export async function serveImage(
  db: D1Like,
  bucket: R2Like,
  scope: string,
  id: string,
): Promise<Response> {
  if (!/^img_[a-f0-9]{20}$/.test(id)) return new Response("Not found", { status: 404 });
  const row = await db
    .prepare("SELECT r2_key, mime FROM generated_images WHERE id = ? AND scope = ?")
    .bind(id, scope)
    .first<{ r2_key?: string; mime?: string }>();
  if (!row?.r2_key) return new Response("Not found", { status: 404 });
  const object = await bucket.get(row.r2_key);
  if (!object) return new Response("Not found", { status: 404 });
  return new Response(object.body, {
    headers: {
      "Content-Type": row.mime || "image/jpeg",
      "Cache-Control": "private, max-age=86400",
      "Content-Disposition": `inline; filename="henad-${id}.jpg"`,
    },
  });
}
