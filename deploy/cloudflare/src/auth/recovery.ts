/**
 * Email verification and password reset (Worker + D1).
 *
 * - Links carry a random token; only its sha256 is stored. Reset links work
 *   once and expire after RESET_SECONDS. Verify links expire after a day.
 * - "Forgot password" answers the same way whether or not the email has an
 *   account, and sends in the background so timing does not tell either.
 * - The owner account still needs its second factor (authenticator code or a
 *   recovery code) to reset. An unclaimed owner never gets a reset link.
 * - A reset signs the account out everywhere.
 */

import type { D1Like } from "../live/ai";
import {
  emailState,
  publicBase,
  resetEmailMessage,
  sendAccountEmail,
  verifyEmailMessage,
  type EmailEnv,
} from "./email";
import { clearFailures, lockedFor, lockedResponse, recordFailure, MFA_LIMIT } from "./limits";
import { hashPassword, validatePassword } from "./password";
import { verifyCode } from "./routes";
import { resolveSession } from "./session";
import { b64urlEncode, clientIp, json, nowIso, randomBytes, sha256Hex } from "./util";

export const RESET_SECONDS = 30 * 60;
export const VERIFY_SECONDS = 24 * 60 * 60;
export const FORGOT_IP_LIMIT = 5;
export const FORGOT_EMAIL_LIMIT = 3;
export const RESET_FAIL_LIMIT = 10;
export const VERIFY_SEND_LIMIT = 3;

export const FORGOT_SENT =
  "If that email has a Nous account, a reset link is on its way. It works once and expires in 30 minutes.";
export const EMAIL_UNAVAILABLE =
  "Password reset by email isn't working yet. Please email andrewgray@youneek.xyz from your account's address and we'll help you get back in.";

export type RecoveryEnv = EmailEnv & { WORKSPACES_DB?: D1Like };
export type WaitUntil = (promise: Promise<unknown>) => void;

type UserRow = { id: string; email: string; role: string; password_hash: string | null };

function newToken(): string {
  return b64urlEncode(randomBytes(32));
}

async function tokenHash(token: string): Promise<string> {
  return sha256Hex(`acct-link:${token}`);
}

function cleanEmail(raw: unknown): string {
  return typeof raw === "string" ? raw.trim().toLowerCase().slice(0, 320) : "";
}

function cleanToken(raw: unknown): string {
  return typeof raw === "string" && /^[A-Za-z0-9_-]{20,100}$/.test(raw) ? raw : "";
}

async function bodyOf(request: Request): Promise<Record<string, unknown>> {
  try {
    const body = await request.json();
    return body && typeof body === "object" ? (body as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

async function userByEmail(db: D1Like, email: string): Promise<UserRow | null> {
  const row = await db
    .prepare("SELECT id, email, role, password_hash FROM users WHERE email = ? LIMIT 1")
    .bind(email)
    .first<UserRow>();
  return row && typeof row.id === "string" ? row : null;
}

async function userById(db: D1Like, id: string): Promise<UserRow | null> {
  const row = await db
    .prepare("SELECT id, email, role, password_hash FROM users WHERE id = ? LIMIT 1")
    .bind(id)
    .first<UserRow>();
  return row && typeof row.id === "string" ? row : null;
}

async function ownerHasSecondFactor(db: D1Like, userId: string): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT (SELECT COUNT(*) FROM mfa_totp WHERE user_id = ? AND secret IS NOT NULL)
            + (SELECT COUNT(*) FROM webauthn_credentials WHERE user_id = ?)
            + (SELECT COUNT(*) FROM mfa_recovery_codes WHERE user_id = ? AND used_at IS NULL) AS n`,
    )
    .bind(userId, userId, userId)
    .first<{ n?: number }>();
  return Number(row?.n || 0) > 0;
}

export async function isEmailVerified(db: D1Like, userId: string, email: string): Promise<boolean> {
  const row = await db
    .prepare("SELECT email, verified_at FROM user_email_status WHERE user_id = ?")
    .bind(userId)
    .first<{ email?: string; verified_at?: string | null }>();
  return Boolean(row?.verified_at) && String(row?.email || "").toLowerCase() === email.toLowerCase();
}

async function markVerified(db: D1Like, userId: string, email: string): Promise<void> {
  await db
    .prepare(
      `INSERT INTO user_email_status (user_id, email, verified_at) VALUES (?, ?, ?)
       ON CONFLICT(user_id) DO UPDATE SET email = excluded.email, verified_at = excluded.verified_at`,
    )
    .bind(userId, email.toLowerCase(), nowIso())
    .run();
}

/** Create a verification link and email it. Caller decides rate limits. */
export async function sendVerification(db: D1Like, env: RecoveryEnv, userId: string, email: string): Promise<boolean> {
  const token = newToken();
  const now = new Date();
  await db.prepare("DELETE FROM email_verifications WHERE user_id = ? OR expires_at < ?").bind(userId, now.toISOString()).run();
  await db
    .prepare(
      "INSERT INTO email_verifications (token_hash, user_id, email, expires_at, used_at, created_at) VALUES (?, ?, ?, ?, NULL, ?)",
    )
    .bind(await tokenHash(token), userId, email.toLowerCase(), new Date(now.getTime() + VERIFY_SECONDS * 1000).toISOString(), now.toISOString())
    .run();
  const link = `${publicBase(env)}/#verify=${token}`;
  const result = await sendAccountEmail(db, env, "verify", email, verifyEmailMessage(link));
  return result.ok;
}

async function issueReset(db: D1Like, env: RecoveryEnv, user: UserRow): Promise<void> {
  const token = newToken();
  const now = new Date();
  await db.prepare("DELETE FROM password_resets WHERE user_id = ? OR expires_at < ?").bind(user.id, now.toISOString()).run();
  await db
    .prepare("INSERT INTO password_resets (token_hash, user_id, expires_at, used_at, created_at) VALUES (?, ?, ?, NULL, ?)")
    .bind(await tokenHash(token), user.id, new Date(now.getTime() + RESET_SECONDS * 1000).toISOString(), now.toISOString())
    .run();
  const link = `${publicBase(env)}/#reset=${token}`;
  await sendAccountEmail(db, env, "reset", user.email, resetEmailMessage(link));
}

async function ipKey(request: Request, kind: string): Promise<string> {
  return `${kind}-ip:${(await sha256Hex(`${kind}:${clientIp(request) || "unknown"}`)).slice(0, 32)}`;
}

export async function forgotPassword(request: Request, env: RecoveryEnv, db: D1Like, waitUntil?: WaitUntil): Promise<Response> {
  const body = await bodyOf(request);
  const email = cleanEmail(body.email);
  if (!email || !email.includes("@")) return json({ detail: "Enter the email you signed up with." }, 422);
  const keys = [await ipKey(request, "forgot"), `forgot-email:${(await sha256Hex(`forgot:${email}`)).slice(0, 32)}`];
  const locked = await lockedFor(db, keys);
  if (locked > 0) return lockedResponse(locked);
  await recordFailure(db, keys[0], FORGOT_IP_LIMIT);
  await recordFailure(db, keys[1], FORGOT_EMAIL_LIMIT);
  // Same answer for every address: sending state is site-wide, not per account.
  if ((await emailState(db, env)) !== "on") {
    return json({ ok: false, email_unavailable: true, detail: EMAIL_UNAVAILABLE }, 503);
  }
  const work = (async () => {
    const user = await userByEmail(db, email);
    if (!user || !user.password_hash) return;
    if (user.role === "owner" && !(await ownerHasSecondFactor(db, user.id))) return;
    await issueReset(db, env, user);
  })().catch(() => {
    console.warn("forgot-password background step failed");
  });
  if (waitUntil) waitUntil(work);
  else await work;
  return json({ ok: true, detail: FORGOT_SENT });
}

type ResetRow = { user_id?: string; expires_at?: string; used_at?: string | null };

async function liveReset(db: D1Like, hash: string): Promise<ResetRow | null> {
  const row = await db
    .prepare("SELECT user_id, expires_at, used_at FROM password_resets WHERE token_hash = ?")
    .bind(hash)
    .first<ResetRow>();
  if (!row?.user_id || row.used_at || !row.expires_at || Date.parse(row.expires_at) <= Date.now()) return null;
  return row;
}

const LINK_EXPIRED = "This reset link is used or expired. Ask for a new one.";

/** GET-like check so the page can ask for a code up front (owner only). */
export async function resetInfo(request: Request, db: D1Like): Promise<Response> {
  const body = await bodyOf(request);
  const token = cleanToken(body.token);
  const failKey = await ipKey(request, "reset");
  const locked = await lockedFor(db, [failKey]);
  if (locked > 0) return lockedResponse(locked);
  const row = token ? await liveReset(db, await tokenHash(token)) : null;
  if (!row) {
    await recordFailure(db, failKey, RESET_FAIL_LIMIT);
    return json({ ok: false, detail: LINK_EXPIRED }, 400);
  }
  const user = await userById(db, String(row.user_id));
  return json({ ok: true, needs_code: user?.role === "owner" });
}

export async function resetPassword(request: Request, db: D1Like): Promise<Response> {
  const body = await bodyOf(request);
  const token = cleanToken(body.token);
  const failKey = await ipKey(request, "reset");
  const locked = await lockedFor(db, [failKey]);
  if (locked > 0) return lockedResponse(locked);
  const hash = token ? await tokenHash(token) : "";
  const row = hash ? await liveReset(db, hash) : null;
  if (!row) {
    await recordFailure(db, failKey, RESET_FAIL_LIMIT);
    return json({ ok: false, detail: LINK_EXPIRED }, 400);
  }
  const user = await userById(db, String(row.user_id));
  if (!user || !user.password_hash) return json({ ok: false, detail: LINK_EXPIRED }, 400);
  const problem = validatePassword(body.password, user.email);
  if (problem) return json({ ok: false, detail: problem }, 422);

  if (user.role === "owner") {
    // Owner security is not weakened by email: a second factor is still required.
    if (!(await ownerHasSecondFactor(db, user.id))) return json({ ok: false, detail: LINK_EXPIRED }, 400);
    const mfaKey = `mfa:${user.id}`;
    const mfaLocked = await lockedFor(db, [mfaKey]);
    if (mfaLocked > 0) return lockedResponse(mfaLocked);
    const code = typeof body.code === "string" ? body.code.slice(0, 64) : "";
    if (!code.trim()) {
      return json(
        { ok: false, needs_code: true, detail: "Enter the 6-digit code from your authenticator app, or one of your recovery codes." },
        401,
      );
    }
    const method = await verifyCode(db, user.id, code);
    if (!method) {
      await recordFailure(db, mfaKey, MFA_LIMIT);
      return json({ ok: false, needs_code: true, detail: "That code did not work." }, 401);
    }
    await clearFailures(db, mfaKey);
  }

  const now = new Date().toISOString();
  const used = await db
    .prepare("UPDATE password_resets SET used_at = ? WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?")
    .bind(now, hash, now)
    .run();
  if (Number(used.meta?.changes || 0) < 1) return json({ ok: false, detail: LINK_EXPIRED }, 400);

  await db.prepare("UPDATE users SET password_hash = ? WHERE id = ?").bind(hashPassword(String(body.password)), user.id).run();
  await db.prepare("DELETE FROM session_mfa WHERE user_id = ?").bind(user.id).run();
  await db.prepare("DELETE FROM sessions WHERE user_id = ?").bind(user.id).run();
  await db.prepare("DELETE FROM password_resets WHERE user_id = ? AND token_hash <> ?").bind(user.id, hash).run();
  // Opening the link proved the inbox, and a fresh start clears login lockout.
  await markVerified(db, user.id, user.email);
  await clearFailures(db, `email:${(await sha256Hex(user.email.trim().toLowerCase())).slice(0, 32)}`);
  await clearFailures(db, failKey);
  return json({ ok: true, detail: "Password changed. Log in with your new password." });
}

export async function confirmVerification(request: Request, db: D1Like): Promise<Response> {
  const body = await bodyOf(request);
  const token = cleanToken(body.token);
  const failKey = await ipKey(request, "verify");
  const locked = await lockedFor(db, [failKey]);
  if (locked > 0) return lockedResponse(locked);
  const hash = token ? await tokenHash(token) : "";
  const now = new Date().toISOString();
  const row = hash
    ? await db
        .prepare("SELECT user_id, email FROM email_verifications WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?")
        .bind(hash, now)
        .first<{ user_id?: string; email?: string }>()
    : null;
  if (!row?.user_id) {
    await recordFailure(db, failKey, RESET_FAIL_LIMIT);
    return json({ ok: false, detail: "This confirm link is used or expired. Send a new one from Settings." }, 400);
  }
  const user = await userById(db, row.user_id);
  if (!user || user.email.toLowerCase() !== String(row.email || "").toLowerCase()) {
    return json({ ok: false, detail: "This confirm link is used or expired. Send a new one from Settings." }, 400);
  }
  await db.prepare("UPDATE email_verifications SET used_at = ? WHERE token_hash = ?").bind(now, hash).run();
  await markVerified(db, user.id, user.email);
  return json({ ok: true, detail: "Email confirmed. Thanks!" });
}

export async function emailStatus(request: Request, env: RecoveryEnv, db: D1Like): Promise<Response> {
  const session = await resolveSession(request, db);
  if (!session) return json({ detail: "Sign in to continue.", auth_required: true }, 401);
  return json({
    email_on: (await emailState(db, env)) === "on",
    verified: await isEmailVerified(db, session.userId, session.email),
  });
}

export async function resendVerification(request: Request, env: RecoveryEnv, db: D1Like): Promise<Response> {
  const session = await resolveSession(request, db);
  if (!session) return json({ detail: "Sign in to continue.", auth_required: true }, 401);
  if (await isEmailVerified(db, session.userId, session.email)) return json({ ok: true, verified: true, detail: "Your email is already confirmed." });
  if ((await emailState(db, env)) !== "on") {
    return json({ ok: false, email_unavailable: true, detail: "Email isn't set up yet. You can keep using Nous." }, 503);
  }
  const key = `verify-send:${session.userId}`;
  const locked = await lockedFor(db, [key]);
  if (locked > 0) return lockedResponse(locked);
  await recordFailure(db, key, VERIFY_SEND_LIMIT);
  const ok = await sendVerification(db, env, session.userId, session.email);
  if (!ok) return json({ ok: false, detail: "We couldn't send the email right now. Try again later." }, 502);
  return json({ ok: true, detail: "Check your inbox for a confirm link." });
}

/** After a successful signup: send the confirm link in the background when email is on. */
export async function afterSignup(response: Response, env: RecoveryEnv, db: D1Like, waitUntil?: WaitUntil): Promise<void> {
  if (!response.ok || !waitUntil) return;
  let body: { id?: unknown; email?: unknown; role?: unknown } = {};
  try {
    body = (await response.clone().json()) as typeof body;
  } catch {
    return;
  }
  const id = typeof body.id === "string" ? body.id : "";
  const email = typeof body.email === "string" ? body.email : "";
  if (!id || !email) return;
  waitUntil(
    (async () => {
      if ((await emailState(db, env)) !== "on") return;
      await sendVerification(db, env, id, email);
    })().catch(() => console.warn("signup verification email failed")),
  );
}

export async function handleRecoveryRoute(
  request: Request,
  env: RecoveryEnv,
  db: D1Like,
  waitUntil?: WaitUntil,
): Promise<Response | null> {
  const path = new URL(request.url).pathname;
  if (path === "/auth/email/status" && request.method === "GET") return emailStatus(request, env, db);
  if (request.method !== "POST") {
    if (["/auth/password/forgot", "/auth/password/reset", "/auth/password/reset/check", "/auth/email/verify", "/auth/email/verify/send"].includes(path)) {
      return json({ detail: "Method not allowed." }, 405);
    }
    return null;
  }
  if (path === "/auth/password/forgot") return forgotPassword(request, env, db, waitUntil);
  if (path === "/auth/password/reset/check") return resetInfo(request, db);
  if (path === "/auth/password/reset") return resetPassword(request, db);
  if (path === "/auth/email/verify") return confirmVerification(request, db);
  if (path === "/auth/email/verify/send") return resendVerification(request, env, db);
  return null;
}
