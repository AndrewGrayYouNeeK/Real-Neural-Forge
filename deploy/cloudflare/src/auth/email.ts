/**
 * Account emails (verify address, reset password) through the Cloudflare
 * Email Sending Workers binding. Sending is OFF until the owner turns it on
 * (app_settings.email_sending = 'on'), which only happens after a test email
 * reaches the owner. Setup errors switch it back off by themselves, so the
 * site never pretends to send mail it cannot deliver.
 */

import type { D1Like } from "../live/ai";
import { nowIso } from "./util";

export type EmailBinding = {
  send(message: {
    to: string;
    from: string | { email: string; name: string };
    subject: string;
    html?: string;
    text?: string;
  }): Promise<{ messageId?: string }>;
};

export type EmailEnv = {
  EMAIL?: EmailBinding;
  NOUS_EMAIL_FROM?: string;
  NEURAL_FORGE_PUBLIC_URL?: string;
};

export type EmailState = "on" | "off" | "down";

/** Error codes that mean the sending domain or account is not set up. */
const SETUP_ERRORS = new Set([
  "E_SENDER_NOT_VERIFIED",
  "E_SENDER_DOMAIN_NOT_AVAILABLE",
  "E_SENDER_DOMAIN_NOT_CONFIGURED",
  "E_EMAIL_SENDING_NOT_READY",
  "E_RECIPIENT_PLAN_RESTRICTED",
  "E_ACCOUNT_SUSPENDED",
]);

export function emailConfigured(env: EmailEnv): boolean {
  return Boolean(env.EMAIL && typeof env.EMAIL.send === "function" && senderAddress(env));
}

export function senderAddress(env: EmailEnv): string {
  const from = (env.NOUS_EMAIL_FROM || "").trim();
  return /^[^@\s]+@youneek\.(dev|xyz)$/i.test(from) ? from : "";
}

export async function readSetting(db: D1Like, key: string): Promise<string | null> {
  try {
    const row = await db.prepare("SELECT value FROM app_settings WHERE key = ?").bind(key).first<{ value?: string }>();
    return typeof row?.value === "string" ? row.value : null;
  } catch {
    return null;
  }
}

export async function writeSetting(db: D1Like, key: string, value: string): Promise<void> {
  await db
    .prepare(
      `INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
    )
    .bind(key, value, nowIso())
    .run();
}

export async function emailState(db: D1Like, env: EmailEnv): Promise<EmailState> {
  if (!emailConfigured(env)) return "off";
  const value = await readSetting(db, "email_sending");
  if (value === "on") return "on";
  if (value && value.startsWith("down")) return "down";
  return "off";
}

async function countEmail(db: D1Like, kind: string, ok: boolean): Promise<void> {
  try {
    const day = new Date().toISOString().slice(0, 10);
    await db
      .prepare(
        `INSERT INTO email_log (day, kind, sent, failed) VALUES (?, ?, ?, ?)
         ON CONFLICT(day, kind) DO UPDATE SET sent = sent + excluded.sent, failed = failed + excluded.failed`,
      )
      .bind(day, kind, ok ? 1 : 0, ok ? 0 : 1)
      .run();
  } catch {
    // Counting is best-effort.
  }
}

export type SendResult = { ok: true } | { ok: false; code: string };

export async function sendAccountEmail(
  db: D1Like,
  env: EmailEnv,
  /** Short label for the dashboard counts, e.g. "verify", "reset", "reminder". */
  kind: string,
  to: string,
  message: { subject: string; text: string; html: string },
  options: { force?: boolean } = {},
): Promise<SendResult> {
  if (!emailConfigured(env)) return { ok: false, code: "E_NOT_CONFIGURED" };
  if (!options.force && (await emailState(db, env)) !== "on") return { ok: false, code: "E_TURNED_OFF" };
  try {
    await env.EMAIL!.send({
      to,
      from: { email: senderAddress(env), name: "Nous" },
      subject: message.subject,
      text: message.text,
      html: message.html,
    });
    await countEmail(db, kind, true);
    return { ok: true };
  } catch (error) {
    const code = String((error as { code?: unknown })?.code || "E_UNKNOWN");
    // Log the code only; never the address or the link.
    console.warn(`account email (${kind}) failed: ${code}`);
    await countEmail(db, kind, false);
    if (SETUP_ERRORS.has(code) && !options.force) {
      try {
        await writeSetting(db, "email_sending", `down:${code}`);
      } catch {
        // ignore
      }
    }
    return { ok: false, code };
  }
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

function layout(title: string, lead: string, button: string, link: string, foot: string): { text: string; html: string } {
  const text = `${title}\n\n${lead}\n\n${link}\n\n${foot}\n\n— Nous · The architecture of understanding.`;
  const html = `<!doctype html><html><body style="margin:0;background:#0b0b10;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#eee">
<div style="max-width:480px;margin:0 auto;padding:32px 24px">
<p style="font-size:22px;font-weight:700;margin:0 0 16px">${escapeHtml(title)}</p>
<p style="font-size:16px;line-height:1.5;margin:0 0 24px">${escapeHtml(lead)}</p>
<p style="margin:0 0 24px"><a href="${escapeHtml(link)}" style="display:inline-block;background:#fff;color:#000;text-decoration:none;font-weight:600;padding:12px 20px;border-radius:999px">${escapeHtml(button)}</a></p>
<p style="font-size:13px;line-height:1.5;color:#aaa;margin:0 0 8px">${escapeHtml(foot)}</p>
<p style="font-size:12px;color:#777;word-break:break-all;margin:0 0 24px">${escapeHtml(link)}</p>
<p style="font-size:12px;color:#777;margin:0">Nous · The architecture of understanding.</p>
</div></body></html>`;
  return { text, html };
}

export function verifyEmailMessage(link: string): { subject: string; text: string; html: string } {
  return {
    subject: "Confirm your email for Nous",
    ...layout(
      "Confirm your email",
      "Tap the button to confirm this is your email address for Nous.",
      "Confirm email",
      link,
      "If you didn't make a Nous account, you can ignore this email.",
    ),
  };
}

export function resetEmailMessage(link: string): { subject: string; text: string; html: string } {
  return {
    subject: "Reset your Nous password",
    ...layout(
      "Reset your password",
      "Tap the button to choose a new password. The link works once and expires in 30 minutes.",
      "Choose a new password",
      link,
      "If you didn't ask for this, ignore this email. Your password stays the same.",
    ),
  };
}

export function testEmailMessage(): { subject: string; text: string; html: string } {
  return {
    subject: "Nous email test",
    text: "This is a test from Nous. Email is working.\n\n— Nous",
    html: "<p>This is a test from Nous. Email is working.</p><p>— Nous</p>",
  };
}

/** Links always point at the public site, never at the request's Host header. */
export function publicBase(env: EmailEnv): string {
  const raw = (env.NEURAL_FORGE_PUBLIC_URL || "").trim().replace(/\/+$/, "");
  return /^https:\/\/nous\.youneek\.(dev|xyz)$/.test(raw) ? raw : "https://nous.youneek.dev";
}

/**
 * The address to email for an account, or null unless that address is
 * confirmed (user_email_status matches the current users.email).
 */
export async function verifiedAddress(db: D1Like, userId: string): Promise<string | null> {
  try {
    const row = await db
      .prepare(
        `SELECT users.email AS email, user_email_status.email AS verified_email, user_email_status.verified_at AS verified_at
         FROM users JOIN user_email_status ON user_email_status.user_id = users.id
         WHERE users.id = ?`,
      )
      .bind(userId)
      .first<{ email?: string; verified_email?: string; verified_at?: string | null }>();
    const to = String(row?.email || "");
    if (!row?.verified_at || !to || to.toLowerCase() !== String(row.verified_email || "").toLowerCase()) return null;
    return to;
  } catch {
    return null;
  }
}

/**
 * Shared way for any feature (reminders, alerts…) to email one account:
 * only when sending is on and the address is confirmed. Counts toward the
 * owner dashboard and switches sending off on setup errors, like account mail.
 */
export async function sendToVerifiedUser(
  db: D1Like,
  env: EmailEnv,
  kind: string,
  userId: string,
  message: { subject: string; text: string; html: string },
): Promise<SendResult> {
  if ((await emailState(db, env)) !== "on") return { ok: false, code: "E_TURNED_OFF" };
  const to = await verifiedAddress(db, userId);
  if (!to) return { ok: false, code: "E_NOT_VERIFIED" };
  return sendAccountEmail(db, env, kind, to, message);
}
