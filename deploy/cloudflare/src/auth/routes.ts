/**
 * Account-security endpoints handled in the Worker (D1 + WebCrypto):
 * login/signup lockout around the container's password routes, TOTP,
 * recovery codes and WebAuthn passkeys. No secret is ever logged.
 */

import type { D1Like } from "../live/ai";
import { EMAIL_LIMIT, IP_LIMIT, MFA_LIMIT, clearFailures, lockedFor, lockedResponse, recordFailure } from "./limits";
import { resolveSession, type SessionInfo } from "./session";
import { base32Encode, otpauthUrl, verifyTotp } from "./totp";
import { clientIp, json, nowIso, randomBytes, sha256Hex, b64urlEncode, bytesToHex } from "./util";
import { expectedOrigin, rpIdFor, verifyAssertion, verifyRegistration, type StoredCredential } from "./webauthn";

export type AuthEnv = { WORKSPACES_DB?: D1Like };
export type Forward = (request: Request) => Promise<Response>;

const CHALLENGE_SECONDS = 5 * 60;
const RECOVERY_CODES = 10;

async function readJson(request: Request): Promise<Record<string, unknown>> {
  try {
    const body = await request.json();
    return body && typeof body === "object" ? (body as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

function str(value: unknown, max = 4096): string {
  return typeof value === "string" ? value.slice(0, max) : "";
}

async function emailKey(email: string): Promise<string> {
  return `email:${(await sha256Hex(email.trim().toLowerCase())).slice(0, 32)}`;
}

async function ipKey(request: Request): Promise<string> {
  return `ip:${(await sha256Hex(`auth-ip:${clientIp(request) || "unknown"}`)).slice(0, 32)}`;
}

async function passwordRoute(request: Request, db: D1Like, forward: Forward, kind: "login" | "signup"): Promise<Response> {
  const body = await readJson(request.clone());
  const email = str(body.email, 320);
  const keys = [await ipKey(request)];
  if (email) keys.push(await emailKey(email));
  const locked = await lockedFor(db, keys);
  if (locked > 0) return lockedResponse(locked);
  const response = await forward(request);
  if (kind === "login") {
    if (response.status === 401) {
      await recordFailure(db, keys[0], IP_LIMIT);
      if (keys[1]) await recordFailure(db, keys[1], EMAIL_LIMIT);
    } else if (response.ok && keys[1]) {
      await clearFailures(db, keys[1]);
    }
  } else if (!response.ok) {
    // Failed signups (taken email, bad input) count against the IP only.
    await recordFailure(db, keys[0], IP_LIMIT);
  }
  return response;
}

async function mfaEnrolled(db: D1Like, userId: string): Promise<{ totp: boolean; passkeys: number; recovery: number }> {
  const row = await db
    .prepare(
      `SELECT
         (SELECT COUNT(*) FROM mfa_totp WHERE user_id = ? AND secret IS NOT NULL) AS totp,
         (SELECT COUNT(*) FROM webauthn_credentials WHERE user_id = ?) AS passkeys,
         (SELECT COUNT(*) FROM mfa_recovery_codes WHERE user_id = ? AND used_at IS NULL) AS recovery`,
    )
    .bind(userId, userId, userId)
    .first<{ totp?: number; passkeys?: number; recovery?: number }>();
  return {
    totp: Number(row?.totp || 0) > 0,
    passkeys: Number(row?.passkeys || 0),
    recovery: Number(row?.recovery || 0),
  };
}

async function markSessionMfa(db: D1Like, session: SessionInfo, method: string): Promise<void> {
  await db
    .prepare(
      `INSERT INTO session_mfa (token_hash, user_id, method, verified_at) VALUES (?, ?, ?, ?)
       ON CONFLICT(token_hash) DO UPDATE SET method = excluded.method, verified_at = excluded.verified_at`,
    )
    .bind(session.tokenHash, session.userId, method, nowIso())
    .run();
}

function newRecoveryCode(): string {
  const hex = bytesToHex(randomBytes(6));
  return `${hex.slice(0, 4)}-${hex.slice(4, 8)}-${hex.slice(8, 12)}`;
}

function normalizeRecovery(code: string): string {
  return code.toLowerCase().replace(/[^0-9a-f]/g, "");
}

/** Issue fresh recovery codes only when none are left. Returned once, stored hashed. */
async function ensureRecoveryCodes(db: D1Like, userId: string): Promise<string[] | null> {
  const state = await mfaEnrolled(db, userId);
  if (state.recovery > 0) return null;
  await db.prepare("DELETE FROM mfa_recovery_codes WHERE user_id = ?").bind(userId).run();
  const codes: string[] = [];
  for (let index = 0; index < RECOVERY_CODES; index += 1) {
    const code = newRecoveryCode();
    codes.push(code);
    await db
      .prepare("INSERT INTO mfa_recovery_codes (user_id, code_hash, used_at, created_at) VALUES (?, ?, NULL, ?)")
      .bind(userId, await sha256Hex(`rc:${normalizeRecovery(code)}`), nowIso())
      .run();
  }
  return codes;
}

async function canEnroll(db: D1Like, session: SessionInfo): Promise<boolean> {
  if (session.mfaFresh) return true;
  const state = await mfaEnrolled(db, session.userId);
  return !state.totp && state.passkeys === 0;
}

async function newChallenge(db: D1Like, userId: string, kind: string): Promise<{ id: string; challenge: string }> {
  const id = b64urlEncode(randomBytes(16));
  const challenge = b64urlEncode(randomBytes(32));
  await db.prepare("DELETE FROM auth_challenges WHERE expires_at < ?").bind(new Date().toISOString()).run();
  await db
    .prepare("INSERT INTO auth_challenges (id, user_id, kind, challenge, expires_at) VALUES (?, ?, ?, ?, ?)")
    .bind(id, userId, kind, challenge, new Date(Date.now() + CHALLENGE_SECONDS * 1000).toISOString())
    .run();
  return { id, challenge };
}

async function takeChallenge(db: D1Like, id: string, userId: string, kind: string): Promise<string | null> {
  const row = await db
    .prepare("SELECT challenge, expires_at FROM auth_challenges WHERE id = ? AND user_id = ? AND kind = ?")
    .bind(id, userId, kind)
    .first<{ challenge?: string; expires_at?: string }>();
  await db.prepare("DELETE FROM auth_challenges WHERE id = ?").bind(id).run();
  if (!row?.challenge || !row.expires_at || Date.parse(row.expires_at) < Date.now()) return null;
  return row.challenge;
}

async function mfaFailure(db: D1Like, session: SessionInfo, message: string): Promise<Response> {
  await recordFailure(db, `mfa:${session.userId}`, MFA_LIMIT);
  return json({ detail: message }, 401);
}

async function verifyCode(db: D1Like, session: SessionInfo, code: string): Promise<string | null> {
  const totp = await db
    .prepare("SELECT secret, last_step FROM mfa_totp WHERE user_id = ? AND secret IS NOT NULL")
    .bind(session.userId)
    .first<{ secret?: string; last_step?: number }>();
  if (totp?.secret) {
    const step = await verifyTotp(totp.secret, code, Number(totp.last_step || 0));
    if (step !== null) {
      await db.prepare("UPDATE mfa_totp SET last_step = ? WHERE user_id = ?").bind(step, session.userId).run();
      return "totp";
    }
  }
  const normalized = normalizeRecovery(code);
  if (normalized.length === 12) {
    const hash = await sha256Hex(`rc:${normalized}`);
    const result = await db
      .prepare("UPDATE mfa_recovery_codes SET used_at = ? WHERE user_id = ? AND code_hash = ? AND used_at IS NULL")
      .bind(nowIso(), session.userId, hash)
      .run();
    if (Number(result.meta?.changes || 0) > 0) return "recovery";
  }
  return null;
}

export async function handleAuthRoute(request: Request, env: AuthEnv, forward: Forward): Promise<Response | null> {
  const url = new URL(request.url);
  const path = url.pathname;
  const db = env.WORKSPACES_DB;
  if (!db) return null;
  if (request.method === "POST" && path === "/auth/login") return passwordRoute(request, db, forward, "login");
  if (request.method === "POST" && path === "/auth/signup") return passwordRoute(request, db, forward, "signup");
  if (!path.startsWith("/auth/mfa/") && !path.startsWith("/auth/webauthn/")) return null;

  const session = await resolveSession(request, db);
  if (!session) return json({ detail: "Sign in to continue.", auth_required: true }, 401);

  if (path === "/auth/mfa/status" && request.method === "GET") {
    const state = await mfaEnrolled(db, session.userId);
    return json({
      role: session.role,
      mfa_fresh: session.mfaFresh,
      totp: state.totp,
      passkeys: state.passkeys,
      recovery_codes_left: state.recovery,
      passkeys_supported_here: rpIdFor(url) !== null,
    });
  }
  if (request.method !== "POST") return json({ detail: "Method not allowed." }, 405);

  const locked = await lockedFor(db, [`mfa:${session.userId}`]);
  if (locked > 0) return lockedResponse(locked);
  const body = await readJson(request);

  if (path === "/auth/mfa/totp/setup") {
    if (!(await canEnroll(db, session))) return json({ detail: "Confirm your second factor first." }, 403);
    const secret = base32Encode(randomBytes(20));
    await db
      .prepare(
        `INSERT INTO mfa_totp (user_id, secret, pending_secret, enabled_at, last_step) VALUES (?, NULL, ?, NULL, 0)
         ON CONFLICT(user_id) DO UPDATE SET pending_secret = excluded.pending_secret`,
      )
      .bind(session.userId, secret)
      .run();
    return json({ secret, otpauth_url: otpauthUrl(secret, session.email) });
  }

  if (path === "/auth/mfa/totp/confirm") {
    if (!(await canEnroll(db, session))) return json({ detail: "Confirm your second factor first." }, 403);
    const row = await db
      .prepare("SELECT pending_secret FROM mfa_totp WHERE user_id = ?")
      .bind(session.userId)
      .first<{ pending_secret?: string | null }>();
    if (!row?.pending_secret) return json({ detail: "Start authenticator setup first." }, 400);
    const step = await verifyTotp(row.pending_secret, str(body.code, 32), 0);
    if (step === null) return mfaFailure(db, session, "That code did not match. Check the time on your phone and try again.");
    await db
      .prepare("UPDATE mfa_totp SET secret = pending_secret, pending_secret = NULL, enabled_at = ?, last_step = ? WHERE user_id = ?")
      .bind(nowIso(), step, session.userId)
      .run();
    await markSessionMfa(db, session, "totp");
    await clearFailures(db, `mfa:${session.userId}`);
    return json({ ok: true, recovery_codes: await ensureRecoveryCodes(db, session.userId) });
  }

  if (path === "/auth/mfa/verify") {
    const method = await verifyCode(db, session, str(body.code, 64));
    if (!method) return mfaFailure(db, session, "That code did not work.");
    await markSessionMfa(db, session, method);
    await clearFailures(db, `mfa:${session.userId}`);
    const state = await mfaEnrolled(db, session.userId);
    return json({ ok: true, method, recovery_codes_left: state.recovery });
  }

  if (path === "/auth/mfa/recovery/regenerate") {
    if (!session.mfaFresh) return json({ detail: "Confirm your second factor first." }, 403);
    await db.prepare("DELETE FROM mfa_recovery_codes WHERE user_id = ?").bind(session.userId).run();
    return json({ ok: true, recovery_codes: await ensureRecoveryCodes(db, session.userId) });
  }

  const rpId = rpIdFor(url);
  if (!rpId) {
    return json({ detail: "Passkeys work on nous.youneek.dev and nous.youneek.xyz. Use an authenticator code here." }, 400);
  }

  if (path === "/auth/webauthn/register/options") {
    if (!(await canEnroll(db, session))) return json({ detail: "Confirm your second factor first." }, 403);
    const { id, challenge } = await newChallenge(db, session.userId, "register");
    const existing = await db
      .prepare("SELECT id FROM webauthn_credentials WHERE user_id = ?")
      .bind(session.userId)
      .all();
    return json({
      challenge_id: id,
      publicKey: {
        challenge,
        rp: { id: rpId, name: "Nous" },
        user: {
          id: b64urlEncode(new TextEncoder().encode(session.userId)),
          name: session.email,
          displayName: session.email,
        },
        pubKeyCredParams: [
          { type: "public-key", alg: -7 },
          { type: "public-key", alg: -257 },
        ],
        timeout: 60000,
        attestation: "none",
        authenticatorSelection: { residentKey: "preferred", userVerification: "required" },
        excludeCredentials: (existing.results || []).map((row) => ({ type: "public-key", id: String(row.id) })),
      },
    });
  }

  if (path === "/auth/webauthn/register/verify") {
    if (!(await canEnroll(db, session))) return json({ detail: "Confirm your second factor first." }, 403);
    const challenge = await takeChallenge(db, str(body.challenge_id, 64), session.userId, "register");
    if (!challenge) return json({ detail: "That passkey request expired. Try again." }, 400);
    try {
      const credential = await verifyRegistration({
        clientDataJSON: str(body.clientDataJSON, 8192),
        attestationObject: str(body.attestationObject, 16384),
        challenge,
        origin: expectedOrigin(url),
        rpId,
      });
      await db
        .prepare(
          `INSERT INTO webauthn_credentials (id, user_id, public_key, alg, sign_count, created_at, last_used_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)`,
        )
        .bind(credential.id, session.userId, JSON.stringify(credential.publicKey), credential.alg, credential.signCount, nowIso(), nowIso())
        .run();
    } catch {
      return mfaFailure(db, session, "That passkey could not be saved. Try again.");
    }
    await markSessionMfa(db, session, "passkey");
    await clearFailures(db, `mfa:${session.userId}`);
    return json({ ok: true, recovery_codes: await ensureRecoveryCodes(db, session.userId) });
  }

  if (path === "/auth/webauthn/login/options") {
    const rows = await db
      .prepare("SELECT id FROM webauthn_credentials WHERE user_id = ?")
      .bind(session.userId)
      .all();
    const ids = (rows.results || []).map((row) => String(row.id));
    if (!ids.length) return json({ detail: "No passkey is saved for this account yet." }, 400);
    const { id, challenge } = await newChallenge(db, session.userId, "login");
    return json({
      challenge_id: id,
      publicKey: {
        challenge,
        rpId,
        timeout: 60000,
        userVerification: "required",
        allowCredentials: ids.map((credentialId) => ({ type: "public-key", id: credentialId })),
      },
    });
  }

  if (path === "/auth/webauthn/login/verify") {
    const challenge = await takeChallenge(db, str(body.challenge_id, 64), session.userId, "login");
    if (!challenge) return json({ detail: "That passkey request expired. Try again." }, 400);
    const credentialId = str(body.id, 1024);
    const row = await db
      .prepare("SELECT id, public_key, alg, sign_count FROM webauthn_credentials WHERE id = ? AND user_id = ?")
      .bind(credentialId, session.userId)
      .first<{ id?: string; public_key?: string; alg?: number; sign_count?: number }>();
    if (!row?.public_key) return mfaFailure(db, session, "That passkey is not on this account.");
    try {
      const stored: StoredCredential = {
        id: String(row.id),
        publicKey: JSON.parse(row.public_key) as JsonWebKey,
        alg: Number(row.alg),
        signCount: Number(row.sign_count || 0),
      };
      const result = await verifyAssertion({
        credential: stored,
        clientDataJSON: str(body.clientDataJSON, 8192),
        authenticatorData: str(body.authenticatorData, 8192),
        signature: str(body.signature, 2048),
        challenge,
        origin: expectedOrigin(url),
        rpId,
      });
      await db
        .prepare("UPDATE webauthn_credentials SET sign_count = ?, last_used_at = ? WHERE id = ?")
        .bind(result.signCount, nowIso(), stored.id)
        .run();
    } catch {
      return mfaFailure(db, session, "That passkey did not verify.");
    }
    await markSessionMfa(db, session, "passkey");
    await clearFailures(db, `mfa:${session.userId}`);
    return json({ ok: true, method: "passkey" });
  }

  return json({ detail: "Not found." }, 404);
}
