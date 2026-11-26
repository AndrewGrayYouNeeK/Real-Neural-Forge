/**
 * Who is calling, from the D1 session table. The shared owner token is not a
 * login any more: owner powers need role=owner in D1 AND a second factor
 * (passkey, authenticator code or recovery code) on this session in the last
 * OWNER_MFA_SECONDS.
 */

import type { D1Like } from "../live/ai";
import { sessionToken, sha256Hex } from "./util";

export const OWNER_MFA_SECONDS = 12 * 60 * 60;

export type SessionInfo = {
  userId: string;
  email: string;
  role: string;
  tokenHash: string;
  mfaVerifiedAt: string | null;
  /** True only for a role=owner session that passed a second factor recently. */
  owner: boolean;
  mfaFresh: boolean;
};

export function mfaFresh(verifiedAt: string | null, now: Date = new Date()): boolean {
  if (!verifiedAt) return false;
  const stamp = Date.parse(verifiedAt);
  if (!Number.isFinite(stamp)) return false;
  const age = (now.getTime() - stamp) / 1000;
  return age >= 0 && age <= OWNER_MFA_SECONDS;
}

export async function resolveSession(
  request: Request,
  db: D1Like | undefined,
  now: Date = new Date(),
): Promise<SessionInfo | null> {
  const token = sessionToken(request);
  if (!token || !db) return null;
  try {
    const tokenHash = await sha256Hex(token);
    const result = await db
      .prepare(
        `SELECT users.id AS id, users.email AS email, users.role AS role,
                session_mfa.verified_at AS mfa_verified_at
         FROM sessions
         JOIN users ON users.id = sessions.user_id
         LEFT JOIN session_mfa ON session_mfa.token_hash = sessions.token_hash
         WHERE sessions.token_hash = ? AND sessions.expires_at > ?
         LIMIT 1`,
      )
      .bind(tokenHash, now.toISOString())
      .all();
    const row = result.results?.[0] as
      | { id?: unknown; email?: unknown; role?: unknown; mfa_verified_at?: unknown }
      | undefined;
    if (!row || typeof row.id !== "string" || !row.id.trim()) return null;
    const verified = typeof row.mfa_verified_at === "string" ? row.mfa_verified_at : null;
    const fresh = mfaFresh(verified, now);
    const role = typeof row.role === "string" ? row.role : "member";
    return {
      userId: row.id.trim(),
      email: typeof row.email === "string" ? row.email : "",
      role,
      tokenHash,
      mfaVerifiedAt: verified,
      mfaFresh: fresh,
      owner: role === "owner" && fresh,
    };
  } catch {
    return null;
  }
}

/** Daily AI caps are counted per account once signed in, per IP otherwise. */
export function capKey(session: { userId: string | null } | null, ip: string | null): string | null {
  if (session?.userId) return `acct:${session.userId}`;
  return ip;
}
