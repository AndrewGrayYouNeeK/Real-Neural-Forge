/**
 * Owner-only usage dashboard data. Counts only: no emails, ids, messages or
 * IP hashes ever leave this module. Every route here requires a D1 session
 * with role=owner AND a fresh second factor (session.owner).
 */

import type { D1Like } from "../live/ai";
import { emailConfigured, emailState, sendAccountEmail, testEmailMessage, writeSetting, type EmailEnv } from "../auth/email";
import { resolveSession } from "../auth/session";
import { json } from "../auth/util";

/** Workers AI: 10,000 neurons/day free, then $0.011 per 1,000 neurons. */
export const AI_FREE_NEURONS_PER_DAY = 10_000;
export const AI_USD_PER_1K_NEURONS = 0.011;
/** Rough ElevenLabs cost per 1,000 spoken characters (your plan may differ). */
export const VOICE_USD_PER_1K_CHARS = 0.3;
/** Email Sending on Workers Paid: 3,000/month included, then $0.35 per 1,000. */
export const EMAIL_FREE_PER_MONTH = 3_000;

export type OwnerEnv = EmailEnv & { WORKSPACES_DB?: D1Like };

export type DayRow = {
  day: string;
  signups: number;
  active_users: number;
  chats: number;
  messages: number;
  voice_replies: number;
  voice_chars: number;
  ai_requests: number;
  ai_neurons: number;
  ai_cost_usd: number;
  emails: number;
};

export type UsageSummary = {
  days: DayRow[];
  totals: {
    accounts: number;
    active_7d: number;
    active_30d: number;
    chats: number;
    messages: number;
    voice_replies: number;
    voice_chars: number;
    ai_cost_usd: number;
    voice_cost_usd: number;
    emails: number;
  };
  subscriptions: { active: number; canceling: number; new_in_range: number; canceled_in_range: number };
  email: { state: "on" | "off" | "down"; configured: boolean };
  notes: string[];
};

function dayList(days: number, now: Date): string[] {
  const out: string[] = [];
  for (let index = days - 1; index >= 0; index -= 1) {
    out.push(new Date(now.getTime() - index * 86_400_000).toISOString().slice(0, 10));
  }
  return out;
}

async function rows(db: D1Like, sql: string, ...values: unknown[]): Promise<Record<string, unknown>[]> {
  try {
    const result = await db.prepare(sql).bind(...values).all();
    return Array.isArray(result.results) ? result.results : [];
  } catch {
    // A table that does not exist yet just counts as zero.
    return [];
  }
}

async function one(db: D1Like, sql: string, ...values: unknown[]): Promise<number> {
  const found = await rows(db, sql, ...values);
  const value = found[0] ? Object.values(found[0])[0] : 0;
  const number = Number(value || 0);
  return Number.isFinite(number) ? number : 0;
}

function byDay(found: Record<string, unknown>[], field = "n"): Map<string, number> {
  const map = new Map<string, number>();
  for (const row of found) {
    if (typeof row.d === "string") map.set(row.d, Number(row[field] || 0));
  }
  return map;
}

export function aiCostUsd(neurons: number): number {
  const billable = Math.max(0, neurons - AI_FREE_NEURONS_PER_DAY);
  return Math.round((billable / 1000) * AI_USD_PER_1K_NEURONS * 10_000) / 10_000;
}

export async function usageSummary(db: D1Like, env: EmailEnv, daysWanted = 30, now = new Date()): Promise<UsageSummary> {
  const span = Math.min(90, Math.max(7, Math.floor(daysWanted) || 30));
  const days = dayList(span, now);
  const since = days[0];
  const since7 = days[Math.max(0, days.length - 7)];

  const signups = byDay(
    await rows(db, "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n FROM users WHERE id <> 'owner' AND substr(created_at, 1, 10) >= ? GROUP BY d", since),
  );
  const chats = byDay(
    await rows(db, "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n FROM chat_threads WHERE substr(created_at, 1, 10) >= ? GROUP BY d", since),
  );
  const messages = byDay(
    await rows(
      db,
      "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n FROM chat_messages WHERE role = 'user' AND substr(created_at, 1, 10) >= ? GROUP BY d",
      since,
    ),
  );
  const active = byDay(
    await rows(
      db,
      `SELECT substr(m.created_at, 1, 10) AS d, COUNT(DISTINCT t.scope) AS n
       FROM chat_messages m JOIN chat_threads t ON t.id = m.thread_id
       WHERE m.role = 'user' AND substr(m.created_at, 1, 10) >= ? GROUP BY d`,
      since,
    ),
  );
  const voiceRows = await rows(
    db,
    "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS n, SUM(CASE WHEN cached = 1 THEN 0 ELSE chars END) AS c FROM voice_usage WHERE substr(created_at, 1, 10) >= ? GROUP BY d",
    since,
  );
  const voiceReplies = byDay(voiceRows);
  const voiceChars = byDay(voiceRows, "c");
  const aiRows = await rows(db, "SELECT day AS d, requests AS r, neurons AS n FROM ai_usage WHERE scope = 'global' AND day >= ?", since);
  const aiNeurons = byDay(aiRows);
  const aiRequests = byDay(aiRows, "r");
  const emails = byDay(await rows(db, "SELECT day AS d, SUM(sent) AS n FROM email_log WHERE day >= ? GROUP BY day", since));

  const dayRows: DayRow[] = days.map((day) => {
    const neurons = aiNeurons.get(day) || 0;
    return {
      day,
      signups: signups.get(day) || 0,
      active_users: active.get(day) || 0,
      chats: chats.get(day) || 0,
      messages: messages.get(day) || 0,
      voice_replies: voiceReplies.get(day) || 0,
      voice_chars: voiceChars.get(day) || 0,
      ai_requests: aiRequests.get(day) || 0,
      ai_neurons: neurons,
      ai_cost_usd: aiCostUsd(neurons),
      emails: emails.get(day) || 0,
    };
  });

  const activeSince = (start: string) =>
    one(
      db,
      `SELECT COUNT(DISTINCT t.scope) AS n FROM chat_messages m JOIN chat_threads t ON t.id = m.thread_id
       WHERE m.role = 'user' AND substr(m.created_at, 1, 10) >= ?`,
      start,
    );
  const sum = (field: keyof DayRow) => dayRows.reduce((total, row) => total + Number(row[field] || 0), 0);
  const voiceCharsTotal = sum("voice_chars");

  return {
    days: dayRows,
    totals: {
      accounts: await one(db, "SELECT COUNT(*) AS n FROM users WHERE id <> 'owner'"),
      active_7d: await activeSince(since7),
      active_30d: await activeSince(days[Math.max(0, days.length - 30)]),
      chats: sum("chats"),
      messages: sum("messages"),
      voice_replies: sum("voice_replies"),
      voice_chars: voiceCharsTotal,
      ai_cost_usd: Math.round(sum("ai_cost_usd") * 100) / 100,
      voice_cost_usd: Math.round((voiceCharsTotal / 1000) * VOICE_USD_PER_1K_CHARS * 100) / 100,
      emails: sum("emails"),
    },
    subscriptions: {
      active: await one(db, "SELECT COUNT(*) AS n FROM voice_plans WHERE status IN ('active', 'trialing')"),
      canceling: await one(db, "SELECT COUNT(*) AS n FROM voice_plans WHERE status IN ('active', 'trialing') AND cancel_at_period_end = 1"),
      new_in_range: await one(
        db,
        "SELECT COUNT(*) AS n FROM (SELECT user_id, MIN(created_at) AS first_at FROM voice_grants GROUP BY user_id) WHERE substr(first_at, 1, 10) >= ?",
        since,
      ),
      canceled_in_range: await one(
        db,
        "SELECT COUNT(*) AS n FROM voice_plans WHERE status IN ('canceled', 'unpaid', 'incomplete_expired') AND substr(updated_at, 1, 10) >= ?",
        since,
      ),
    },
    email: { state: await emailState(db, env), configured: emailConfigured(env) },
    notes: [
      `AI cost: Workers AI includes ${AI_FREE_NEURONS_PER_DAY.toLocaleString("en-US")} neurons a day, then $${AI_USD_PER_1K_NEURONS} per 1,000.`,
      `Voice cost is a rough guess at $${VOICE_USD_PER_1K_CHARS.toFixed(2)} per 1,000 spoken characters. Your ElevenLabs plan decides the real number.`,
      `Email: ${EMAIL_FREE_PER_MONTH.toLocaleString("en-US")} a month are included with Workers Paid.`,
    ],
  };
}

async function ownerOnly(request: Request, db: D1Like): Promise<{ email: string } | Response> {
  const session = await resolveSession(request, db);
  if (!session) return json({ detail: "Sign in to continue.", auth_required: true }, 401);
  if (!session.owner) {
    // Members and owner sessions without a fresh second factor look the same.
    return json({ detail: "Not found." }, 404);
  }
  return { email: session.email };
}

export async function handleOwnerRoute(request: Request, env: OwnerEnv): Promise<Response | null> {
  const url = new URL(request.url);
  if (url.pathname !== "/owner/usage" && url.pathname !== "/owner/email") return null;
  const db = env.WORKSPACES_DB;
  if (!db) return json({ detail: "Not found." }, 404);
  const owner = await ownerOnly(request, db);
  if (owner instanceof Response) return owner;

  if (url.pathname === "/owner/usage" && request.method === "GET") {
    const days = Number(url.searchParams.get("days") || 30);
    return json(await usageSummary(db, env, days));
  }
  if (url.pathname === "/owner/email" && request.method === "POST") {
    let action = "";
    try {
      const body = (await request.json()) as { action?: unknown };
      action = typeof body?.action === "string" ? body.action : "";
    } catch {
      action = "";
    }
    if (action === "off") {
      await writeSetting(db, "email_sending", "off");
      return json({ ok: true, state: "off" });
    }
    if (action === "test") {
      // The test goes only to the owner's own address. Email turns on only if it sends.
      const result = await sendAccountEmail(db, env, "test", owner.email, testEmailMessage(), { force: true });
      if (!result.ok) {
        await writeSetting(db, "email_sending", `down:${result.code}`);
        return json({ ok: false, state: "down", code: result.code, detail: "The test email did not send. Finish the email setup steps, then try again." }, 502);
      }
      await writeSetting(db, "email_sending", "on");
      return json({ ok: true, state: "on", detail: "Test email sent. Email is on." });
    }
    return json({ detail: "Unknown action." }, 400);
  }
  return json({ detail: "Method not allowed." }, 405);
}
