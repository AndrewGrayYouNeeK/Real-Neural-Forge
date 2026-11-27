import { beforeEach, describe, expect, it } from "vitest";

import { handleAuthRoute } from "../src/auth/routes";
import { hashPassword, validatePassword } from "../src/auth/password";
import { RESET_SECONDS } from "../src/auth/recovery";
import { base32Encode, totpAt } from "../src/auth/totp";
import { sha256Hex } from "../src/auth/util";
import { handleOwnerRoute } from "../src/owner/usage";
import { SqliteD1 } from "./helpers/sqliteD1";

type Sent = { to: string; subject: string; text?: string };

function mailer(fail?: string) {
  const sent: Sent[] = [];
  return {
    sent,
    binding: {
      async send(message: { to: string; subject: string; text?: string }) {
        if (fail) throw Object.assign(new Error("nope"), { code: fail });
        sent.push(message);
        return { messageId: "m1" };
      },
    },
  };
}

const forward = async () => new Response("container", { status: 500 });

function post(path: string, body: unknown, headers: Record<string, string> = {}) {
  return new Request(`https://henad.youneek.dev${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.9", ...headers },
    body: JSON.stringify(body),
  });
}

async function addUser(db: SqliteD1, id: string, email: string, role = "member", password = "old-password-123") {
  db.exec(
    "INSERT INTO users (id, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
    id,
    email,
    password ? hashPassword(password) : null,
    role,
    new Date().toISOString(),
  );
}

async function addSession(db: SqliteD1, userId: string, token: string, mfaAt?: string) {
  const hash = await sha256Hex(token);
  db.exec(
    "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
    hash,
    userId,
    new Date(Date.now() + 86_400_000).toISOString(),
    new Date().toISOString(),
  );
  if (mfaAt) {
    db.exec("INSERT INTO session_mfa (token_hash, user_id, method, verified_at) VALUES (?, ?, 'totp', ?)", hash, userId, mfaAt);
  }
}

function linkToken(text: string | undefined, kind: "reset" | "verify"): string {
  const match = String(text || "").match(new RegExp(`#${kind}=([A-Za-z0-9_-]+)`));
  if (!match) throw new Error("no link in email");
  return match[1];
}

describe("password hashing", () => {
  it("matches the container's scrypt format byte for byte", () => {
    const salt = Uint8Array.from(Buffer.from("00112233445566778899aabbccddeeff", "hex"));
    // Computed with Python hashlib.scrypt(n=2**14, r=8, p=1, dklen=32).
    expect(hashPassword("correct horse battery", salt)).toBe(
      "scrypt$16384$8$1$00112233445566778899aabbccddeeff$8cc0de9218d3d6723ae203a934c046cf2bb9aa70723c2cf4b1cf06ca35c89adb",
    );
  });

  it("uses the same password rules as signup", () => {
    expect(validatePassword("short", "a@b.co")).toMatch(/at least 10/);
    expect(validatePassword("a@b.co.uk.x", "A@B.CO.UK.X")).toMatch(/different/);
    expect(validatePassword("a-long-enough-pass", "a@b.co")).toBeNull();
  });
});

describe("forgot password", () => {
  let db: SqliteD1;
  beforeEach(async () => {
    db = new SqliteD1();
    await addUser(db, "usr_a", "member@example.test");
  });

  it("says email is not working, the same way for every address, while sending is off", async () => {
    const mail = mailer();
    const env = { WORKSPACES_DB: db, EMAIL: mail.binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    const known = await handleAuthRoute(post("/auth/password/forgot", { email: "member@example.test" }), env, forward);
    const unknown = await handleAuthRoute(post("/auth/password/forgot", { email: "nobody@example.test" }), env, forward);
    expect(known!.status).toBe(503);
    expect(await known!.json()).toEqual(await unknown!.json());
    expect(mail.sent).toHaveLength(0);
  });

  it("answers identically for real and unknown emails, and only mails the real one", async () => {
    db.exec("INSERT INTO app_settings (key, value, updated_at) VALUES ('email_sending', 'on', 'x')");
    const mail = mailer();
    const env = { WORKSPACES_DB: db, EMAIL: mail.binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    const known = await handleAuthRoute(post("/auth/password/forgot", { email: "Member@Example.test" }), env, forward);
    const unknown = await handleAuthRoute(post("/auth/password/forgot", { email: "nobody@example.test" }), env, forward);
    expect(known!.status).toBe(200);
    expect(unknown!.status).toBe(200);
    expect(await known!.json()).toEqual(await unknown!.json());
    expect(mail.sent.map((m) => m.to)).toEqual(["member@example.test"]);
    const token = linkToken(mail.sent[0].text, "reset");
    // Only a hash is stored, never the token itself.
    const stored = db.rows("SELECT token_hash, expires_at FROM password_resets");
    expect(stored).toHaveLength(1);
    expect(String(stored[0].token_hash)).not.toContain(token);
    const ttl = Date.parse(String(stored[0].expires_at)) - Date.now();
    expect(ttl).toBeGreaterThan((RESET_SECONDS - 60) * 1000);
    expect(ttl).toBeLessThanOrEqual(RESET_SECONDS * 1000);
  });

  it("rate-limits by email and by network", async () => {
    db.exec("INSERT INTO app_settings (key, value, updated_at) VALUES ('email_sending', 'on', 'x')");
    const env = { WORKSPACES_DB: db, EMAIL: mailer().binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    const statuses: number[] = [];
    for (let i = 0; i < 4; i += 1) {
      statuses.push((await handleAuthRoute(post("/auth/password/forgot", { email: "nobody@example.test" }), env, forward))!.status);
    }
    expect(statuses).toEqual([200, 200, 200, 429]);
  });

  it("turns email off by itself when the sending domain is not set up", async () => {
    db.exec("INSERT INTO app_settings (key, value, updated_at) VALUES ('email_sending', 'on', 'x')");
    const env = { WORKSPACES_DB: db, EMAIL: mailer("E_SENDER_DOMAIN_NOT_AVAILABLE").binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    const first = await handleAuthRoute(post("/auth/password/forgot", { email: "member@example.test" }), env, forward);
    expect(first!.status).toBe(200);
    expect(db.rows("SELECT value FROM app_settings WHERE key = 'email_sending'")[0].value).toBe("down:E_SENDER_DOMAIN_NOT_AVAILABLE");
    const next = await handleAuthRoute(post("/auth/password/forgot", { email: "other@example.test" }, { "CF-Connecting-IP": "198.51.100.2" }), env, forward);
    expect(next!.status).toBe(503);
  });
});

describe("password reset", () => {
  let db: SqliteD1;
  let mail: ReturnType<typeof mailer>;
  let env: Record<string, unknown>;

  beforeEach(async () => {
    db = new SqliteD1();
    mail = mailer();
    env = { WORKSPACES_DB: db, EMAIL: mail.binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    db.exec("INSERT INTO app_settings (key, value, updated_at) VALUES ('email_sending', 'on', 'x')");
    await addUser(db, "usr_a", "member@example.test");
    await addSession(db, "usr_a", "member-session");
  });

  async function requestLink(email: string): Promise<string> {
    await handleAuthRoute(post("/auth/password/forgot", { email }), env as never, forward);
    return linkToken(mail.sent.at(-1)?.text, "reset");
  }

  it("changes the password once, signs out everywhere and marks the email confirmed", async () => {
    const token = await requestLink("member@example.test");
    const before = String(db.rows("SELECT password_hash FROM users WHERE id = 'usr_a'")[0].password_hash);
    const ok = await handleAuthRoute(post("/auth/password/reset", { token, password: "brand-new-password" }), env as never, forward);
    expect(ok!.status).toBe(200);
    const after = String(db.rows("SELECT password_hash FROM users WHERE id = 'usr_a'")[0].password_hash);
    expect(after).not.toBe(before);
    expect(after.startsWith("scrypt$16384$8$1$")).toBe(true);
    expect(db.rows("SELECT * FROM sessions WHERE user_id = 'usr_a'")).toHaveLength(0);
    expect(db.rows("SELECT verified_at FROM user_email_status WHERE user_id = 'usr_a'")[0].verified_at).toBeTruthy();
    const again = await handleAuthRoute(post("/auth/password/reset", { token, password: "another-password-1" }), env as never, forward);
    expect(again!.status).toBe(400);
  });

  it("rejects expired links and weak passwords", async () => {
    const token = await requestLink("member@example.test");
    const weak = await handleAuthRoute(post("/auth/password/reset", { token, password: "short" }), env as never, forward);
    expect(weak!.status).toBe(422);
    db.exec("UPDATE password_resets SET expires_at = ?", new Date(Date.now() - 1000).toISOString());
    const late = await handleAuthRoute(post("/auth/password/reset", { token, password: "brand-new-password" }), env as never, forward);
    expect(late!.status).toBe(400);
  });

  it("never sends a reset link for the unclaimed owner", async () => {
    await addUser(db, "owner", "andrew.owner@example.test", "owner", "");
    await handleAuthRoute(post("/auth/password/forgot", { email: "andrew.owner@example.test" }), env as never, forward);
    expect(mail.sent).toHaveLength(0);
  });

  it("still needs the owner's second factor", async () => {
    await addUser(db, "owner", "andrew.owner@example.test", "owner");
    const secretBytes = new TextEncoder().encode("12345678901234567890");
    db.exec("INSERT INTO mfa_totp (user_id, secret, pending_secret, enabled_at, last_step) VALUES ('owner', ?, NULL, 'x', 0)", base32Encode(secretBytes));
    db.exec("INSERT INTO mfa_recovery_codes (user_id, code_hash, used_at, created_at) VALUES ('owner', ?, NULL, 'x')", await sha256Hex("rc:aaaabbbbcccc"));
    const token = await requestLink("andrew.owner@example.test");

    const check = await handleAuthRoute(post("/auth/password/reset/check", { token }), env as never, forward);
    expect(await check!.json()).toEqual({ ok: true, needs_code: true });

    const noCode = await handleAuthRoute(post("/auth/password/reset", { token, password: "owner-new-password" }), env as never, forward);
    expect(noCode!.status).toBe(401);
    const badCode = await handleAuthRoute(post("/auth/password/reset", { token, password: "owner-new-password", code: "000000" }), env as never, forward);
    expect(badCode!.status).toBe(401);
    expect(db.rows("SELECT failures FROM auth_attempts WHERE key = 'mfa:owner'")[0].failures).toBe(1);

    const step = Math.floor(Date.now() / 30_000);
    const code = await totpAt(secretBytes, step);
    const good = await handleAuthRoute(post("/auth/password/reset", { token, password: "owner-new-password", code }), env as never, forward);
    expect(good!.status).toBe(200);
  });

  it("accepts an owner recovery code instead of the app code", async () => {
    await addUser(db, "owner", "andrew.owner@example.test", "owner");
    db.exec("INSERT INTO webauthn_credentials (id, user_id, public_key, alg, sign_count, created_at) VALUES ('cred', 'owner', '{}', -7, 0, 'x')");
    db.exec("INSERT INTO mfa_recovery_codes (user_id, code_hash, used_at, created_at) VALUES ('owner', ?, NULL, 'x')", await sha256Hex("rc:aaaabbbbcccc"));
    const token = await requestLink("andrew.owner@example.test");
    const good = await handleAuthRoute(post("/auth/password/reset", { token, password: "owner-new-password", code: "aaaa-bbbb-cccc" }), env as never, forward);
    expect(good!.status).toBe(200);
    expect(db.rows("SELECT used_at FROM mfa_recovery_codes WHERE user_id = 'owner'")[0].used_at).toBeTruthy();
  });
});

describe("email verification", () => {
  it("sends a confirm link after signup and marks the email confirmed once", async () => {
    const db = new SqliteD1();
    db.exec("INSERT INTO app_settings (key, value, updated_at) VALUES ('email_sending', 'on', 'x')");
    const mail = mailer();
    const env = { WORKSPACES_DB: db, EMAIL: mail.binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    const jobs: Promise<unknown>[] = [];
    const signupForward = async () => {
      await addUser(db, "usr_new", "new@example.test");
      return Response.json({ id: "usr_new", email: "new@example.test", role: "member" }, { status: 201 });
    };
    const signup = await handleAuthRoute(post("/auth/signup", { email: "new@example.test", password: "a-long-password" }), env, signupForward, (job) => jobs.push(job));
    expect(signup!.status).toBe(201);
    await Promise.all(jobs);
    expect(mail.sent).toHaveLength(1);
    const token = linkToken(mail.sent[0].text, "verify");
    const first = await handleAuthRoute(post("/auth/email/verify", { token }), env, forward);
    expect(first!.status).toBe(200);
    const second = await handleAuthRoute(post("/auth/email/verify", { token }), env, forward);
    expect(second!.status).toBe(400);

    await addSession(db, "usr_new", "new-session");
    const status = await handleAuthRoute(
      new Request("https://henad.youneek.dev/auth/email/status", { headers: { Cookie: "nf_session=new-session" } }),
      env,
      forward,
    );
    expect(await status!.json()).toEqual({ email_on: true, verified: true });
  });

  it("does not send anything while email is off", async () => {
    const db = new SqliteD1();
    const mail = mailer();
    const env = { WORKSPACES_DB: db, EMAIL: mail.binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    const jobs: Promise<unknown>[] = [];
    const signupForward = async () => Response.json({ id: "usr_new", email: "new@example.test", role: "member" }, { status: 201 });
    await handleAuthRoute(post("/auth/signup", { email: "new@example.test", password: "a-long-password" }), env, signupForward, (job) => jobs.push(job));
    await Promise.all(jobs);
    expect(mail.sent).toHaveLength(0);
  });
});

describe("owner usage dashboard", () => {
  let db: SqliteD1;
  const env = () => ({ WORKSPACES_DB: db, EMAIL: mailer().binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" });
  const get = (token?: string) =>
    new Request("https://henad.youneek.dev/owner/usage", { headers: token ? { Cookie: `nf_session=${token}` } : {} });

  beforeEach(async () => {
    db = new SqliteD1();
    await addUser(db, "owner", "andrew.owner@example.test", "owner");
    await addUser(db, "usr_a", "member@example.test");
    await addSession(db, "owner", "owner-fresh", new Date().toISOString());
    await addSession(db, "owner", "owner-stale", new Date(Date.now() - 13 * 3600_000).toISOString());
    await addSession(db, "owner", "owner-nomfa");
    await addSession(db, "usr_a", "member-session", new Date().toISOString());
    const today = new Date().toISOString();
    db.exec("INSERT INTO chat_threads (id, scope, title, created_at, updated_at) VALUES ('t1', 'user:usr_a', 'secret topic', ?, ?)", today, today);
    db.exec("INSERT INTO chat_messages (id, thread_id, role, content, created_at) VALUES ('m1', 't1', 'user', 'private words', ?)", today);
    db.exec("INSERT INTO chat_messages (id, thread_id, role, content, created_at) VALUES ('m2', 't1', 'assistant', 'reply', ?)", today);
    db.exec("INSERT INTO ai_usage (day, scope, requests, neurons) VALUES (?, 'global', 9, 12000)", today.slice(0, 10));
    db.exec("INSERT INTO voice_usage (id, user_id, kind, chars, cached, created_at) VALUES ('v1', 'usr_a', 'speak', 500, 0, ?)", today);
    db.exec("INSERT INTO voice_plans (user_id, status, cancel_at_period_end, updated_at) VALUES ('usr_a', 'active', 1, ?)", today);
    db.exec("INSERT INTO voice_grants (ref, user_id, chars, created_at) VALUES ('sub:1', 'usr_a', 1000, ?)", today);
  });

  it("is hidden from visitors, members and owners without a fresh second factor", async () => {
    expect((await handleOwnerRoute(get(), env()))!.status).toBe(401);
    expect((await handleOwnerRoute(get("member-session"), env()))!.status).toBe(404);
    expect((await handleOwnerRoute(get("owner-stale"), env()))!.status).toBe(404);
    expect((await handleOwnerRoute(get("owner-nomfa"), env()))!.status).toBe(404);
    const post = new Request("https://henad.youneek.dev/owner/email", {
      method: "POST",
      headers: { Cookie: "nf_session=member-session" },
      body: JSON.stringify({ action: "test" }),
    });
    expect((await handleOwnerRoute(post, env()))!.status).toBe(404);
  });

  it("gives the owner counts only", async () => {
    const response = await handleOwnerRoute(get("owner-fresh"), env());
    expect(response!.status).toBe(200);
    const body = (await response!.json()) as Record<string, any>;
    const today = body.days.at(-1);
    expect(today).toMatchObject({ signups: 1, active_users: 1, chats: 1, messages: 1, voice_replies: 1, voice_chars: 500, ai_neurons: 12000 });
    expect(today.ai_cost_usd).toBeCloseTo(0.022, 4);
    expect(body.totals.accounts).toBe(1);
    expect(body.subscriptions).toEqual({ active: 1, canceling: 1, new_in_range: 1, canceled_in_range: 0 });
    const text = JSON.stringify(body);
    for (const secret of ["member@example.test", "usr_a", "secret topic", "private words", "andrew.owner"]) {
      expect(text).not.toContain(secret);
    }
  });

  it("turns email on only after a test reaches the owner", async () => {
    const mail = mailer();
    const ownerEnv = { WORKSPACES_DB: db, EMAIL: mail.binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    const request = new Request("https://henad.youneek.dev/owner/email", {
      method: "POST",
      headers: { Cookie: "nf_session=owner-fresh" },
      body: JSON.stringify({ action: "test" }),
    });
    const response = await handleOwnerRoute(request, ownerEnv);
    expect(response!.status).toBe(200);
    expect(mail.sent.map((m) => m.to)).toEqual(["andrew.owner@example.test"]);
    expect(db.rows("SELECT value FROM app_settings WHERE key = 'email_sending'")[0].value).toBe("on");

    const broken = { ...ownerEnv, EMAIL: mailer("E_SENDER_DOMAIN_NOT_AVAILABLE").binding };
    const retry = new Request(request.url, { method: "POST", headers: { Cookie: "nf_session=owner-fresh" }, body: JSON.stringify({ action: "test" }) });
    expect((await handleOwnerRoute(retry, broken))!.status).toBe(502);
    expect(db.rows("SELECT value FROM app_settings WHERE key = 'email_sending'")[0].value).toMatch(/^down:/);
  });
});

describe("shared email for other features (reminders)", () => {
  it("emails only confirmed addresses, only while sending is on, and counts the send", async () => {
    const { emailReminder } = await import("../src/abilities/reminders");
    const db = new SqliteD1();
    const mail = mailer();
    const env = { WORKSPACES_DB: db, EMAIL: mail.binding, HENAD_EMAIL_FROM: "no-reply@youneek.dev" };
    await addUser(db, "usr_a", "member@example.test");
    expect(await emailReminder(db, env, "usr_a", "⏰ Reminder: Stretch.")).toBe(false);
    db.exec("INSERT INTO app_settings (key, value, updated_at) VALUES ('email_sending', 'on', 'x')");
    expect(await emailReminder(db, env, "usr_a", "⏰ Reminder: Stretch.")).toBe(false);
    db.exec("INSERT INTO user_email_status (user_id, email, verified_at) VALUES ('usr_a', 'member@example.test', 'x')");
    expect(await emailReminder(db, env, "usr_a", "⏰ Reminder: Stretch.")).toBe(true);
    expect(mail.sent.map((m) => [m.to, m.subject])).toEqual([["member@example.test", "Reminder: Stretch."]]);
    expect(db.rows("SELECT kind, sent FROM email_log")).toEqual([{ kind: "reminder", sent: 1 }]);
  });
});
