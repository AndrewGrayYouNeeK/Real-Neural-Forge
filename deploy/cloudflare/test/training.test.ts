import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";

import { sha256Hex } from "../src/auth/util";
import { handleTrainingRoute, looksSecret, normalizeTags, toJsonl, validateMessages } from "../src/owner/training";
import { TRAINING_SYSTEM_PROMPT } from "../src/owner/trainingPrompt";
import { SqliteD1 } from "./helpers/sqliteD1";

const EXPORT_TOKEN = `nte_${"a1B2".repeat(12)}`;

async function addUser(db: SqliteD1, id: string, role: string) {
  db.exec("INSERT INTO users (id, email, password_hash, role, created_at) VALUES (?, ?, NULL, ?, ?)", id, `${id}@example.com`, role, new Date().toISOString());
}

async function addSession(db: SqliteD1, userId: string, token: string, mfa: boolean) {
  const hash = await sha256Hex(token);
  db.exec(
    "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
    hash,
    userId,
    new Date(Date.now() + 86_400_000).toISOString(),
    new Date().toISOString(),
  );
  if (mfa) {
    db.exec("INSERT INTO session_mfa (token_hash, user_id, method, verified_at) VALUES (?, ?, 'totp', ?)", hash, userId, new Date().toISOString());
  }
}

function call(db: SqliteD1, method: string, path: string, token?: string, body?: unknown) {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const request = new Request(`https://henad.youneek.dev${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return handleTrainingRoute(request, { WORKSPACES_DB: db, HENAD_TRAINING_EXPORT_TOKEN: EXPORT_TOKEN });
}

const pair = (q: string, a: string) => [
  { role: "user", content: q },
  { role: "assistant", content: a },
];

describe("training system prompt", () => {
  it("matches training/system_prompt.txt exactly", () => {
    const file = readFileSync(join(__dirname, "..", "..", "..", "training", "system_prompt.txt"), "utf8").trim();
    expect(TRAINING_SYSTEM_PROMPT).toBe(file);
  });
});

describe("validation", () => {
  it("needs alternating turns ending with an answer", () => {
    expect(validateMessages(pair("hi", "hey")).ok).toBe(true);
    expect(validateMessages([{ role: "user", content: "hi" }]).ok).toBe(false);
    expect(validateMessages([{ role: "assistant", content: "x" }, { role: "user", content: "y" }]).ok).toBe(false);
    expect(validateMessages([...pair("a", "b"), { role: "user", content: "c" }]).ok).toBe(false);
    expect(validateMessages(pair("a", "   ")).ok).toBe(false);
    expect(validateMessages([...pair("a", "b"), ...pair("c", "d")]).ok).toBe(true);
  });

  it("refuses secrets and strips reasoning tags", () => {
    expect(looksSecret("my key is sk-abcdefghijklmnop1234")).toBe(true);
    expect(looksSecret("ghp_abcdefghijklmnopqrstuvwx")).toBe(true);
    expect(looksSecret("just a normal joke")).toBe(false);
    expect(validateMessages(pair("token?", "sk_live_abcdefghijklmnop")).ok).toBe(false);
    const ok = validateMessages(pair("hi", "<think>hmm</think>Hey there"));
    expect(ok.ok && ok.messages[1].content).toBe("hmmHey there");
  });

  it("normalizes tags", () => {
    expect(normalizeTags(["Funny", "#creator", "funny", "talk mode!"])).toEqual(["funny", "creator", "talkmode"]);
    expect(normalizeTags("funny, talk")).toEqual(["funny", "talk"]);
  });
});

describe("routes", () => {
  let db: SqliteD1;
  beforeEach(async () => {
    db = new SqliteD1();
    await addUser(db, "owner", "owner");
    await addUser(db, "member1", "member");
    await addSession(db, "owner", "owner-token-fresh", true);
    await addSession(db, "owner", "owner-token-stale", false);
    await addSession(db, "member1", "member-token", true);
  });

  it("ignores other paths", async () => {
    expect(await call(db, "GET", "/owner/usage")).toBeNull();
  });

  it("is 401 signed out and 403 for members or owners without a fresh second factor", async () => {
    for (const [method, path] of [
      ["GET", "/owner/training/examples"],
      ["POST", "/owner/training/examples"],
      ["GET", "/owner/training/export"],
      ["GET", "/owner/training/status"],
    ]) {
      expect((await call(db, method, path))!.status).toBe(401);
      expect((await call(db, method, path, "member-token"))!.status).toBe(403);
      expect((await call(db, method, path, "owner-token-stale"))!.status).toBe(403);
    }
    expect(db.rows("SELECT COUNT(*) AS n FROM training_examples")[0].n).toBe(0);
  });

  it("export token only reaches the export", async () => {
    expect((await call(db, "GET", "/owner/training/export", EXPORT_TOKEN))!.status).toBe(200);
    expect((await call(db, "GET", "/owner/training/examples", EXPORT_TOKEN))!.status).toBe(401);
    expect((await call(db, "POST", "/owner/training/examples", EXPORT_TOKEN, { messages: pair("a", "b") }))!.status).toBe(401);
    expect((await call(db, "GET", "/owner/training/export", "nte_wrong-token-wrong-token-wrong-token"))!.status).toBe(401);
  });

  it("saves, fixes, rejects, tags, edits, deletes and exports for the owner", async () => {
    const T = "owner-token-fresh";
    const saved = await (await call(db, "POST", "/owner/training/examples", T, { messages: pair("Who made you?", "Andrew."), source: "save", tags: ["creator"] }))!.json();
    expect(saved.example.label).toBe("good");
    expect(saved.counts).toEqual({ good: 1, rejected: 0, total: 1 });

    const bad = await (await call(db, "POST", "/owner/training/examples", T, { messages: pair("Tell a joke", "No."), source: "bad" }))!.json();
    expect(bad.example.label).toBe("rejected");
    const fix = await (
      await call(db, "POST", "/owner/training/examples", T, { messages: pair("Tell a joke", "Why did the GPU cross the road?"), source: "fix", pair_id: bad.example.id, tags: "funny" })
    )!.json();
    expect(fix.example.label).toBe("good");
    expect(fix.example.pair_id).toBe(bad.example.id);

    const status = await (await call(db, "GET", "/owner/training/status", T))!.json();
    expect(status).toEqual({ owner: true, counts: { good: 2, rejected: 1, total: 3 } });

    const all = await (await call(db, "GET", "/owner/training/examples?label=all", T))!.json();
    expect(all.counts).toEqual({ good: 2, rejected: 1, total: 3 });
    const rejected = await (await call(db, "GET", "/owner/training/examples?label=rejected", T))!.json();
    expect(rejected.examples).toHaveLength(1);
    expect(rejected.examples[0].pair_id).toBe(fix.example.id);

    const edited = await (await call(db, "PUT", `/owner/training/examples/${saved.example.id}`, T, { tags: ["creator", "funny"], messages: pair("Who made you?", "Andrew Gray.") }))!.json();
    expect(edited.example.tags).toEqual(["creator", "funny"]);
    expect(edited.example.messages[1].content).toBe("Andrew Gray.");

    const good = await (await call(db, "GET", "/owner/training/export", T))!.text();
    const lines = good.trim().split("\n").map((line) => JSON.parse(line));
    expect(lines).toHaveLength(2);
    for (const line of lines) {
      expect(line.messages[0]).toEqual({ role: "system", content: TRAINING_SYSTEM_PROMPT });
      expect(line.messages.map((m: { role: string }) => m.role)).toEqual(["system", "user", "assistant"]);
      expect(line.label).toBeUndefined();
    }
    expect(good).not.toContain("No.");

    const badExport = await (await call(db, "GET", "/owner/training/export?label=rejected", EXPORT_TOKEN))!.text();
    const badLines = badExport.trim().split("\n").map((line) => JSON.parse(line));
    expect(badLines).toHaveLength(1);
    expect(badLines[0].label).toBe("rejected");
    expect(badLines[0].messages[2].content).toBe("No.");

    expect((await call(db, "DELETE", `/owner/training/examples/${bad.example.id}`, T))!.status).toBe(200);
    expect((await call(db, "DELETE", `/owner/training/examples/${bad.example.id}`, T))!.status).toBe(404);
    expect(db.rows("SELECT pair_id FROM training_examples WHERE id = ?", fix.example.id)[0].pair_id).toBeNull();
  });

  it("refuses bad input", async () => {
    const T = "owner-token-fresh";
    expect((await call(db, "POST", "/owner/training/examples", T, { messages: pair("x", "ghp_abcdefghijklmnopqrstuvwxyz") }))!.status).toBe(400);
    expect((await call(db, "POST", "/owner/training/examples", T, { messages: [] }))!.status).toBe(400);
    expect((await call(db, "PUT", "/owner/training/examples/00000000-0000-0000-0000-000000000000", T, { tags: [] }))!.status).toBe(404);
  });

  it("empty export is an empty file", () => {
    expect(toJsonl([], "good")).toBe("");
  });
});
