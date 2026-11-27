/**
 * Carry a signed-in visit from an old host (nous.youneek.* / forge.youneek.*)
 * to the matching henad.youneek.* host.
 *
 * The session cookie (nf_session) is host-only, so the browser never sends the
 * old host's cookie to henad.*. Without help everyone would be signed out by
 * the rename. For a browser page visit that carries a valid session:
 *
 *  1. The old host mints a one-time code (sha256 stored, 2 minutes, single use)
 *     and answers with a tiny page that copies a few local settings (chat
 *     history list, Talk and voice choices) and POSTs them with the code to
 *     https://henad.<tld>/auth/handoff.
 *  2. henad.* checks the code and that the POST came from the matching old
 *     origin, starts a new session for the same account (same expiry, and the
 *     second-factor stamp carries over), sets its own cookie, restores the
 *     settings and goes to the same path and query.
 *
 * Other page and asset loads on an old host get a plain 301; API calls are
 * served in place (see servedOnLegacyHost in gate.ts).
 */

import type { D1Like } from "../live/ai";
import { LEGACY_HOSTS, legacyTarget, movedUrl, servedOnLegacyHost } from "./gate";
import { b64urlEncode, randomBytes, sessionToken, sha256Hex, SESSION_COOKIE } from "./util";

export const HANDOFF_SECONDS = 120;

/** Local settings copied across. Old key name → key name on henad.*. */
export const CARRIED_KEYS: Record<string, string> = {
  neuralForgeChatHistory: "neuralForgeChatHistory",
  neuralForgeTalkPref: "neuralForgeTalkPref",
  neuralForgeChatMode: "neuralForgeChatMode",
  nousVoicePick: "henadVoicePick",
  nousHasAccount: "henadHasAccount",
  henadVoicePick: "henadVoicePick",
  henadHasAccount: "henadHasAccount",
};
const MAX_STATE_CHARS = 64_000;

type HandoffEnv = { WORKSPACES_DB?: D1Like };

function wantsPage(request: Request): boolean {
  if (request.method !== "GET") return false;
  const accept = request.headers.get("Accept") || "";
  return accept.includes("text/html");
}

function htmlEscape(value: string): string {
  return value.replace(/[&<>"']/g, (ch) => `&#${ch.charCodeAt(0)};`);
}

function page(body: string, nonce: string, formAction: string | null, status = 200, extra: Record<string, string> = {}): Response {
  const csp = [
    "default-src 'none'",
    `script-src 'nonce-${nonce}'`,
    "style-src 'unsafe-inline'",
    "base-uri 'none'",
    formAction ? `form-action ${formAction}` : "form-action 'none'",
    "frame-ancestors 'none'",
  ].join("; ");
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="referrer" content="no-referrer"><title>Henad</title><style>body{background:#000;color:#ddd;font-family:-apple-system,Segoe UI,Roboto,sans-serif;display:flex;flex-direction:column;min-height:100vh;min-height:100dvh;align-items:center;justify-content:center;margin:0;padding:0 16px;box-sizing:border-box;text-align:center}a{color:#f5a54a}.site-legal{position:fixed;left:0;right:0;bottom:max(14px,env(safe-area-inset-bottom));margin:0;color:#9aa3b2;opacity:.8;font-size:12px;letter-spacing:.02em}</style></head><body>${body}<footer class="site-legal">YouNeeK™ · © 2026 Andrew Gray</footer></body></html>`;
  return new Response(html, {
    status,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
      "Content-Security-Policy": csp,
      "Referrer-Policy": "no-referrer",
      "X-Content-Type-Options": "nosniff",
      ...extra,
    },
  });
}

async function liveSession(db: D1Like, tokenHash: string, now: Date): Promise<{ userId: string; expiresAt: string } | null> {
  const result = await db
    .prepare("SELECT user_id, expires_at FROM sessions WHERE token_hash = ? AND expires_at > ? LIMIT 1")
    .bind(tokenHash, now.toISOString())
    .all();
  const row = result.results?.[0] as { user_id?: unknown; expires_at?: unknown } | undefined;
  if (!row || typeof row.user_id !== "string" || typeof row.expires_at !== "string") return null;
  return { userId: row.user_id, expiresAt: row.expires_at };
}

/**
 * Old host, signed-in page visit: the hand-off page. Null means "use the plain
 * redirect" (not an old host, not a page visit, no valid session, or no DB).
 */
export async function legacyHandoffPage(request: Request, env: HandoffEnv, now: Date = new Date()): Promise<Response | null> {
  const url = new URL(request.url);
  const target = legacyTarget(url);
  if (!target || servedOnLegacyHost(url, request) || !wantsPage(request) || !env.WORKSPACES_DB) return null;
  const token = sessionToken(request);
  if (!token) return null;
  try {
    const tokenHash = await sha256Hex(token);
    if (!(await liveSession(env.WORKSPACES_DB, tokenHash, now))) return null;
    const code = b64urlEncode(randomBytes(32));
    await env.WORKSPACES_DB
      .prepare(
        "INSERT INTO host_handoffs (code_hash, token_hash, target_host, expires_at, used_at, created_at) VALUES (?, ?, ?, ?, NULL, ?)",
      )
      .bind(await sha256Hex(code), tokenHash, target, new Date(now.getTime() + HANDOFF_SECONDS * 1000).toISOString(), now.toISOString())
      .run();
    const nonce = b64urlEncode(randomBytes(16));
    const moved = movedUrl(url, target);
    const action = `https://${target}/auth/handoff`;
    const next = `${moved.pathname}${moved.search}${moved.hash}`;
    const keys = JSON.stringify(Object.keys(CARRIED_KEYS));
    const body =
      `<form id="h" method="post" action="${htmlEscape(action)}">` +
      `<input type="hidden" name="code" value="${htmlEscape(code)}">` +
      `<input type="hidden" name="next" value="${htmlEscape(next)}">` +
      `<input type="hidden" name="state" id="s" value="">` +
      `<p>Nous is now Henad. Taking you there…</p>` +
      `<noscript><button type="submit">Continue to Henad</button></noscript></form>` +
      `<script nonce="${nonce}">(function(){var o={};try{${keys}.forEach(function(k){var v=localStorage.getItem(k);if(v!==null)o[k]=v;});}catch(e){}` +
      `try{document.querySelector("[name=next]").value+=location.hash;}catch(e){}var s=JSON.stringify(o);if(s.length>${MAX_STATE_CHARS})s="{}";document.getElementById("s").value=s;document.getElementById("h").submit();})();</script>`;
    return page(body, nonce, `https://${target}`);
  } catch {
    return null;
  }
}

function safeNext(raw: unknown): string {
  if (typeof raw !== "string" || !raw.startsWith("/") || raw.startsWith("//") || raw.includes("\\")) return "/";
  try {
    const parsed = new URL(raw, "https://henad.invalid");
    if (parsed.origin !== "https://henad.invalid") return "/";
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch {
    return "/";
  }
}

function carriedState(raw: unknown): Record<string, string> {
  if (typeof raw !== "string" || !raw || raw.length > MAX_STATE_CHARS) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const out: Record<string, string> = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      const mapped = CARRIED_KEYS[key];
      if (mapped && typeof value === "string" && !(mapped in out && key !== mapped)) out[mapped] = value;
    }
    return out;
  } catch {
    return {};
  }
}

function sessionCookie(token: string, maxAge: number, secure: boolean): string {
  return `${SESSION_COOKIE}=${token}; Max-Age=${maxAge}; Path=/; HttpOnly;${secure ? " Secure;" : ""} SameSite=Strict`;
}

/** POST /auth/handoff on henad.*: redeem the code, set this host's cookie, restore settings. */
export async function handleHandoff(request: Request, env: HandoffEnv, now: Date = new Date()): Promise<Response | null> {
  const url = new URL(request.url);
  if (url.pathname !== "/auth/handoff") return null;
  const host = url.hostname.toLowerCase();
  const home = `${url.protocol}//${url.host}/`;
  const fail = () =>
    new Response(null, { status: 303, headers: { Location: home, "Cache-Control": "no-store" } });
  if (request.method !== "POST") return fail();
  // Only the matching old origin may post here, so another site can't sign a
  // visitor into someone else's account (login CSRF).
  const origin = (request.headers.get("Origin") || "").toLowerCase();
  const allowed = Object.entries(LEGACY_HOSTS)
    .filter(([, target]) => target === host)
    .map(([old]) => `https://${old}`);
  if (!allowed.includes(origin) || !env.WORKSPACES_DB) return fail();
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return fail();
  }
  const code = form.get("code");
  if (typeof code !== "string" || code.length < 20 || code.length > 100) return fail();
  const db = env.WORKSPACES_DB;
  try {
    const codeHash = await sha256Hex(code);
    // Claim the code atomically: only one request can flip used_at.
    const claim = await db
      .prepare("UPDATE host_handoffs SET used_at = ? WHERE code_hash = ? AND used_at IS NULL AND expires_at > ? AND target_host = ?")
      .bind(now.toISOString(), codeHash, now.toISOString(), host)
      .run();
    const changes = (claim as { meta?: { changes?: number } }).meta?.changes ?? 0;
    if (changes !== 1) return fail();
    const rows = await db.prepare("SELECT token_hash FROM host_handoffs WHERE code_hash = ? LIMIT 1").bind(codeHash).all();
    const oldHash = (rows.results?.[0] as { token_hash?: unknown } | undefined)?.token_hash;
    if (typeof oldHash !== "string") return fail();
    const session = await liveSession(db, oldHash, now);
    if (!session) return fail();
    const token = b64urlEncode(randomBytes(32));
    const newHash = await sha256Hex(token);
    await db
      .prepare("INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)")
      .bind(newHash, session.userId, session.expiresAt, now.toISOString())
      .run();
    const mfa = await db
      .prepare("SELECT method, verified_at FROM session_mfa WHERE token_hash = ? LIMIT 1")
      .bind(oldHash)
      .all();
    const mfaRow = mfa.results?.[0] as { method?: unknown; verified_at?: unknown } | undefined;
    if (mfaRow && typeof mfaRow.method === "string" && typeof mfaRow.verified_at === "string") {
      await db
        .prepare("INSERT OR REPLACE INTO session_mfa (token_hash, user_id, method, verified_at) VALUES (?, ?, ?, ?)")
        .bind(newHash, session.userId, mfaRow.method, mfaRow.verified_at)
        .run();
    }
    const maxAge = Math.max(60, Math.floor((Date.parse(session.expiresAt) - now.getTime()) / 1000));
    const next = safeNext(form.get("next"));
    const state = carriedState(form.get("state"));
    const nonce = b64urlEncode(randomBytes(16));
    // JSON inside a script: escape "<" so the data can't close the tag.
    const data = JSON.stringify({ state, next }).replace(/</g, "\\u003c");
    const body =
      `<p>Welcome to Henad.</p><script nonce="${nonce}">(function(){var d=${data};` +
      `try{Object.keys(d.state).forEach(function(k){if(localStorage.getItem(k)===null)localStorage.setItem(k,d.state[k]);});}catch(e){}` +
      `location.replace(d.next);})();</script><noscript><a href="${htmlEscape(next)}">Continue</a></noscript>`;
    return page(body, nonce, null, 200, { "Set-Cookie": sessionCookie(token, maxAge, url.protocol === "https:") });
  } catch {
    return fail();
  }
}
