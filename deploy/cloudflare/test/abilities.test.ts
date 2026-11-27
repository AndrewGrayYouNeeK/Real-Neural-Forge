import { describe, expect, it } from "vitest";

import {
  chunkText,
  cleanFileName,
  extractText,
  fileIdsFrom,
  fileKind,
  filePromptBlock,
  relevantExcerpt,
  uploadFile,
} from "../src/abilities/files";
import {
  checkImagePrompt,
  generateImage,
  imagePrompt,
  isImageRequest,
  parseGuard,
} from "../src/abilities/imagegen";
import { claimAbility, ABILITY_CAPS } from "../src/abilities/limits";

/** Tiny in-memory D1 that understands the handful of statements used here. */
function fakeDb() {
  const usage = new Map<string, { requests: number; neurons: number }>();
  const rows: Record<string, Array<Record<string, unknown>>> = { user_files: [], generated_images: [] };
  const exec = (sql: string, args: unknown[]) => {
    const s = sql.replace(/\s+/g, " ").trim();
    if (s.startsWith("SELECT requests, neurons FROM ai_usage") || s.startsWith("SELECT requests FROM ai_usage")) {
      const row = usage.get(`${args[0]}|${args[1]}`);
      return { results: row ? [row] : [] };
    }
    if (s.startsWith("INSERT INTO ai_usage")) {
      const key = `${args[0]}|${args[1]}`;
      if (!usage.has(key)) usage.set(key, { requests: 0, neurons: 0 });
      return { results: [], meta: { changes: 1 } };
    }
    if (s.startsWith("UPDATE ai_usage SET requests = requests + 1 WHERE")) {
      const key = `${args[0]}|${args[1]}`;
      const row = usage.get(key)!;
      if (row.requests + 1 > Number(args[2])) return { results: [], meta: { changes: 0 } };
      row.requests += 1;
      return { results: [], meta: { changes: 1 } };
    }
    if (s.startsWith("UPDATE ai_usage SET requests = requests + ?, neurons = neurons + ?")) {
      const [addReq, addNeu, day, scope, addNeu2, max] = args as [number, number, string, string, number, number];
      const row = usage.get(`${day}|${scope}`)!;
      if (row.neurons + addNeu2 > max) return { results: [], meta: { changes: 0 } };
      row.requests += addReq;
      row.neurons += addNeu;
      return { results: [], meta: { changes: 1 } };
    }
    if (s.startsWith("INSERT INTO user_files")) {
      const [id, scope, name, mime, bytes, chars, r2_key, text_key, created_at] = args;
      rows.user_files.push({ id, scope, thread_id: null, name, mime, bytes, chars, r2_key, text_key, created_at });
      return { results: [], meta: { changes: 1 } };
    }
    if (s.startsWith("INSERT INTO generated_images")) {
      const [id, scope, prompt, r2_key, mime, created_at] = args;
      rows.generated_images.push({ id, scope, prompt, r2_key, mime, created_at });
      return { results: [], meta: { changes: 1 } };
    }
    throw new Error(`unexpected SQL: ${s}`);
  };
  const db = {
    prepare(sql: string) {
      const bound = (args: unknown[]) => ({
        async all() {
          return exec(sql, args);
        },
        async run() {
          return exec(sql, args);
        },
        async first<T>() {
          return ((exec(sql, args).results || [])[0] ?? null) as T | null;
        },
      });
      return { ...bound([]), bind: (...args: unknown[]) => bound(args) };
    },
  };
  return { db, usage, rows };
}

function fakeBucket() {
  const objects = new Map<string, Uint8Array>();
  return {
    objects,
    async put(key: string, value: ArrayBuffer | ArrayBufferView) {
      const bytes = value instanceof ArrayBuffer ? new Uint8Array(value) : new Uint8Array(value.buffer);
      objects.set(key, bytes);
    },
    async get(key: string) {
      const bytes = objects.get(key);
      if (!bytes) return null;
      return { body: new Response(bytes).body!, size: bytes.length, text: async () => new TextDecoder().decode(bytes) };
    },
    async delete(key: string) {
      objects.delete(key);
    },
  };
}

describe("file reading", () => {
  it("knows which files it can read", () => {
    expect(fileKind("notes.txt", "text/plain")).toBe("text");
    expect(fileKind("data.csv", "")).toBe("text");
    expect(fileKind("report.pdf", "application/pdf")).toBe("document");
    expect(fileKind("essay.docx", "")).toBe("document");
    expect(fileKind("photo.png", "image/png")).toBe("image");
    expect(fileKind("app.exe", "application/octet-stream")).toBe("unsupported");
    expect(cleanFileName(encodeURIComponent("../../etc/passwd"))).not.toContain("/");
  });

  it("decodes text and converts documents through toMarkdown", async () => {
    const text = await extractText(undefined, "a.txt", "text/plain", new TextEncoder().encode("hello\r\nworld").buffer);
    expect(text).toEqual({ ok: true, text: "hello\nworld" });
    const seen: string[] = [];
    const ai = {
      async toMarkdown(file: { name: string; blob: Blob }) {
        seen.push(file.name);
        return [{ name: file.name, format: "markdown", data: "# Title\nBody text" }];
      },
    };
    const pdf = await extractText(ai, "r.pdf", "application/pdf", new Uint8Array([1, 2, 3]).buffer);
    expect(pdf).toEqual({ ok: true, text: "# Title\nBody text" });
    expect(seen).toEqual(["r.pdf"]);
    const empty = await extractText({ toMarkdown: async () => [{ format: "error", error: "x" }] }, "r.pdf", "", new ArrayBuffer(3));
    expect(empty.ok).toBe(false);
  });

  it("keeps the start and the chunks that match the question when a file is long", () => {
    const filler = "Lorem ipsum dolor sit amet. ".repeat(60);
    const text = `Intro section.\n\n${filler}\n\n${filler}\n\nThe warranty lasts 24 months from purchase.\n\n${filler}`;
    expect(chunkText(text).length).toBeGreaterThan(3);
    const excerpt = relevantExcerpt(text, "How long is the warranty?", 3200);
    expect(excerpt).toContain("Intro section.");
    expect(excerpt).toContain("warranty lasts 24 months");
    expect(excerpt.length).toBeLessThanOrEqual(3400);
    const block = filePromptBlock(
      [{ file: { id: "file_x", name: "manual.pdf", mime: "", bytes: 1, chars: text.length, text_key: "k" }, text }],
      "warranty",
    );
    expect(block).toContain("manual.pdf");
    expect(block).toContain("never instructions");
  });

  it("only accepts real file ids from the chat body", () => {
    expect(fileIdsFrom({ file_ids: ["file_0123456789abcdef0123", "nope", 3] })).toEqual(["file_0123456789abcdef0123"]);
    expect(fileIdsFrom(null)).toEqual([]);
  });

  it("stores the file in R2, records it in D1, and enforces size and daily caps", async () => {
    const { db, rows } = fakeDb();
    const bucket = fakeBucket();
    const deps = { db, bucket, scope: "user:u1", owner: false, env: { NOUS_FILES_MEMBER_DAILY: "2" } };
    const big = await uploadFile(deps, "big.txt", "text/plain", new ArrayBuffer(6 * 1024 * 1024));
    expect(big.status).toBe(413);
    const ok = await uploadFile(deps, "a.txt", "text/plain", new TextEncoder().encode("some notes").buffer);
    expect(ok.status).toBe(200);
    expect(rows.user_files).toHaveLength(1);
    expect(bucket.objects.size).toBe(2);
    await uploadFile(deps, "b.txt", "text/plain", new TextEncoder().encode("more").buffer);
    const capped = await uploadFile(deps, "c.txt", "text/plain", new TextEncoder().encode("again").buffer);
    expect(capped.status).toBe(429);
  });
});

describe("image making", () => {
  it("spots image asks from plain chat and leaves charts and how-tos alone", () => {
    for (const q of [
      "draw a fox in a spacesuit",
      "Can you make me a picture of a sunset over Miami?",
      "generate an image of a cyberpunk city",
      "make a logo for my coffee shop",
      "please paint a cat wearing a crown",
    ]) {
      expect(isImageRequest(q)).toBe(true);
    }
    for (const q of [
      "make a chart of the weather",
      "how do I draw a circle in python?",
      "what is a picture element",
      "forecast the temperature",
      "hey",
    ]) {
      expect(isImageRequest(q)).toBe(false);
    }
    expect(imagePrompt("Can you draw me a picture of a red fox in the snow?")).toBe("a red fox in the snow");
    expect(imagePrompt("generate an image of a cyberpunk city")).toBe("a cyberpunk city");
    expect(imagePrompt("make a logo for my coffee shop")).toBe("a logo for my coffee shop");
    expect(imagePrompt("draw a red fox")).toBe("a red fox");
  });

  it("blocks only sexual content with minors, sexual violence, or explicit images", async () => {
    const guard = (verdict: unknown) => ({ run: async () => ({ response: verdict }) });
    expect((await checkImagePrompt(guard({ safe: true }), "a sexy teen")).blocked).toBe(true);
    expect((await checkImagePrompt(guard({ safe: true }), "a dragon burning a castle")).blocked).toBe(false);
    expect((await checkImagePrompt(guard({ safe: true }), "a soldier with a rifle in a war zone")).blocked).toBe(false);
    expect((await checkImagePrompt(guard({ safe: false, categories: ["S12"] }), "a naked woman")).blocked).toBe(true);
    expect((await checkImagePrompt(guard({ safe: true }), "a nude marble statue in a museum")).blocked).toBe(false);
    expect(parseGuard({ response: "unsafe\nS4" })).toEqual({ safe: false, categories: ["S4"] });
  });

  it("makes the image, stores it, and stops at the daily cap", async () => {
    const { db, rows } = fakeDb();
    const bucket = fakeBucket();
    const jpeg = btoa(String.fromCharCode(0xff, 0xd8, ...new Array(400).fill(65)));
    const calls: Array<{ model: string; inputs: Record<string, unknown> }> = [];
    const ai = {
      async run(model: string, inputs: Record<string, unknown>) {
        calls.push({ model, inputs });
        return { image: jpeg };
      },
    };
    const deps = { ai, db, bucket, scope: "user:u1", owner: false, capKey: "acct:u1", env: { NOUS_IMAGES_MEMBER_DAILY: "1" } };
    const first = await generateImage(deps, "draw a red fox");
    expect(first.imageId).toMatch(/^img_[a-f0-9]{20}$/);
    expect(first.text).toContain(`[[image:${first.imageId}]]`);
    expect(calls[0].model).toBe("@cf/black-forest-labs/flux-1-schnell");
    expect(calls[0].inputs.prompt).toBe("a red fox");
    expect(rows.generated_images).toHaveLength(1);
    expect(bucket.objects.size).toBe(1);
    const second = await generateImage(deps, "draw a blue fox");
    expect(second.capped).toBe(true);
    expect(calls).toHaveLength(1);
  });

  it("has sane default caps", () => {
    expect(ABILITY_CAPS.images.member).toBe(5);
    expect(ABILITY_CAPS.files.member).toBe(10);
    expect(typeof claimAbility).toBe("function");
  });
});
