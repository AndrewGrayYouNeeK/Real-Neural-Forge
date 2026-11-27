/**
 * Reminders and scheduled check-ins from plain chat:
 *   "remind me to call mom tomorrow at 5pm"
 *   "remind me in 20 minutes to check the oven"
 *   "check in with me every evening at 8 about my workout"
 *   "what are my reminders?" / "cancel my reminders"
 *
 * Times are read in the person's own time zone (sent by the browser). A cheap
 * rule-based parser handles the usual phrasings; only odd ones fall back to the
 * small Llama 3.1 8B model (~5 neurons). Reminders live in D1 (`reminders`)
 * and a Worker cron trigger (every minute) fires them: the message is added
 * to the chat it came from, queued for the open app (in-app pop-up and an
 * optional browser notification), and emailed when account email is switched
 * on and the address is verified.
 *
 * Caps: 10 new reminders/day per account (owner 50), 25 active at once.
 */

import { tryReserve, type AiLike, type D1Like } from "../live/ai";
import { abilityCaps, claimAbility, newAbilityId } from "./limits";
import { publicBase, sendToVerifiedUser, type EmailEnv } from "../auth/email";

export const DEFAULT_TZ = "America/New_York";
export const MAX_ACTIVE_REMINDERS = 25;
export const PARSE_MODEL = "@cf/meta/llama-3.1-8b-instruct-fp8-fast";
export const MAX_REMINDER_TEXT = 200;
const MINUTE = 60_000;
const DAY = 86_400_000;

export type Repeat = "none" | "daily" | "weekly" | "weekdays";

/* ------------------------------------------------------------ intents */

export function reminderIntent(question: string): "create" | "list" | "cancel" | null {
  const q = question.replace(/\s+/g, " ").trim().toLowerCase();
  if (!q || q.length > 400) return null;
  // "remind me what X is" is a question, not a reminder.
  if (/\bremind me (what|who|how|why|where|which|of|again)\b/.test(q) && !/\b(in \d|tomorrow|tonight|at \d|every)\b/.test(q)) {
    return null;
  }
  if (/\b(cancel|delete|remove|clear|stop|turn off)\b.{0,30}\b(reminders?|check-?ins?)\b/.test(q)) return "cancel";
  if (
    /\b(what|which|show|list|see|any)\b.{0,25}\b(my )?(reminders|check-?ins)\b/.test(q) ||
    /^(my )?(reminders|check-?ins)\??$/.test(q) ||
    /\bdo i have any reminders\b/.test(q)
  ) {
    return "list";
  }
  if (
    /^(?:(?:hey )?nous,?\s+)?(?:(?:can|could|would|will) you\s+|please\s+)*(remind me|set (?:a |an )?reminder|check in (?:with|on) me|ping me|nudge me)\b/.test(q) ||
    /\b(remind me (to|about|that|in|at|on|tomorrow|tonight|every)|set (?:a |an )?reminder (to|for)|check in (with|on) me)\b/.test(q)
  ) {
    return "create";
  }
  return null;
}

/* ------------------------------------------------------ time zones */

export function validTimeZone(tz: unknown): string {
  if (typeof tz !== "string" || !tz.trim() || tz.length > 64) return DEFAULT_TZ;
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: tz.trim() }).format(new Date());
    return tz.trim();
  } catch {
    return DEFAULT_TZ;
  }
}

type Local = { y: number; m: number; d: number; h: number; min: number; dow: number };

const WEEKDAYS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"];

export function localParts(at: number, tz: string): Local {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    year: "numeric",
    month: "numeric",
    day: "numeric",
    hour: "numeric",
    minute: "numeric",
    weekday: "long",
    hourCycle: "h23",
  }).formatToParts(new Date(at));
  const get = (type: string) => parts.find((part) => part.type === type)?.value || "";
  return {
    y: Number(get("year")),
    m: Number(get("month")),
    d: Number(get("day")),
    h: Number(get("hour")) % 24,
    min: Number(get("minute")),
    dow: WEEKDAYS.indexOf(get("weekday").toLowerCase()),
  };
}

/** UTC ms for a wall-clock time in `tz` (handles DST by re-checking once). */
export function zonedToUtc(y: number, m: number, d: number, h: number, min: number, tz: string): number {
  const wall = Date.UTC(y, m - 1, d, h, min);
  let guess = wall;
  for (let i = 0; i < 2; i += 1) {
    const p = localParts(guess, tz);
    const seen = Date.UTC(p.y, p.m - 1, p.d, p.h, p.min);
    guess += wall - seen;
  }
  return guess;
}

function addDays(local: { y: number; m: number; d: number }, days: number): { y: number; m: number; d: number } {
  const date = new Date(Date.UTC(local.y, local.m - 1, local.d + days));
  return { y: date.getUTCFullYear(), m: date.getUTCMonth() + 1, d: date.getUTCDate() };
}

export function formatWhen(at: number, tz: string, now: number): string {
  const today = localParts(now, tz);
  const that = localParts(at, tz);
  const time = new Intl.DateTimeFormat("en-US", { timeZone: tz, hour: "numeric", minute: "2-digit" }).format(new Date(at));
  const tomorrow = addDays(today, 1);
  let day: string;
  if (that.y === today.y && that.m === today.m && that.d === today.d) day = "today";
  else if (that.y === tomorrow.y && that.m === tomorrow.m && that.d === tomorrow.d) day = "tomorrow";
  else {
    day = new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      weekday: "short",
      month: "short",
      day: "numeric",
      ...(that.y !== today.y ? { year: "numeric" } : {}),
    }).format(new Date(at));
  }
  const zone = new Intl.DateTimeFormat("en-US", { timeZone: tz, timeZoneName: "short" })
    .formatToParts(new Date(at))
    .find((part) => part.type === "timeZoneName")?.value;
  return `${day} at ${time}${zone ? ` ${zone}` : ""}`;
}

/* ----------------------------------------------------------- parsing */

export type ParsedReminder = { dueAt: number; repeat: Repeat; text: string; checkIn: boolean };

const NUM_WORDS: Record<string, number> = {
  a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  fifteen: 15, twenty: 20, thirty: 30, forty: 40, "forty-five": 45, sixty: 60, ninety: 90, half: 0.5, couple: 2, few: 3,
};
const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

function num(word: string): number | null {
  const n = Number(word);
  if (Number.isFinite(n)) return n;
  return NUM_WORDS[word] ?? null;
}

function clockFrom(text: string): { h: number; min: number; match: string; ampm: boolean } | null {
  const named = text.match(/\b(?:at\s+)?(noon|midday|midnight)\b/);
  if (named) return { h: named[1] === "midnight" ? 0 : 12, min: 0, match: named[0], ampm: true };
  const m = text.match(/\b(?:at|by|around|@)?\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?|am|pm)\b/);
  if (m) {
    let h = Number(m[1]) % 12;
    if (m[3].startsWith("p")) h += 12;
    return { h, min: Number(m[2] || 0), match: m[0], ampm: true };
  }
  const h24 = text.match(/\b(?:at|by|around|@)\s*(\d{1,2}):(\d{2})\b/);
  if (h24 && Number(h24[1]) < 24 && Number(h24[2]) < 60) return { h: Number(h24[1]), min: Number(h24[2]), match: h24[0], ampm: Number(h24[1]) >= 13 || h24[1].startsWith("0") };
  const bare = text.match(/\bat\s+(\d{1,2})\b(?!\s*(?:minutes?|mins?|hours?|days?|%))/);
  if (bare && Number(bare[1]) >= 1 && Number(bare[1]) <= 12) {
    // "at 5" means the next 5 o'clock that makes sense: 1-6 → pm, 7-11 → am, 12 → noon.
    const n = Number(bare[1]);
    return { h: n === 12 ? 12 : n <= 6 ? n + 12 : n, min: 0, match: bare[0], ampm: false };
  }
  return null;
}

const PART_OF_DAY: Array<[RegExp, number]> = [
  [/\b(?:in the |this |tomorrow |every )?morning\b/, 9],
  [/\b(?:in the |this |tomorrow |every )?afternoon\b/, 15],
  [/\b(?:in the |this |tomorrow |every )?evening\b/, 18],
  [/\btonight\b|\bevery night\b|\bat night\b/, 20],
];

/** Rule-based parser for the common phrasings; null when unsure. */
export function parseReminder(question: string, tz: string, now: number): ParsedReminder | null {
  const original = question.replace(/\s+/g, " ").trim();
  let text = ` ${original.toLowerCase()} `;
  const strip = (piece: string) => {
    text = text.replace(piece, " ");
  };
  const checkIn = /\bcheck in (with|on) me\b/.test(text);
  const today = localParts(now, tz);

  // Repeats.
  let repeat: Repeat = "none";
  let weeklyDay: number | null = null;
  const weekdaysMatch = text.match(/\b(every weekday|on weekdays|each weekday|weekdays)\b/);
  const everyDow = text.match(/\b(?:every|each|on) (sunday|monday|tuesday|wednesday|thursday|friday|saturday)s?\b/);
  const daily = text.match(/\b(every day|everyday|daily|each day|every morning|every evening|every night|every afternoon)\b/);
  if (weekdaysMatch) {
    repeat = "weekdays";
    strip(weekdaysMatch[0]);
  } else if (everyDow && /\b(every|each)\b/.test(everyDow[0])) {
    repeat = "weekly";
    weeklyDay = WEEKDAYS.indexOf(everyDow[1]);
    strip(everyDow[0]);
  } else if (daily) {
    repeat = "daily";
    if (/^every (day|everyday)$|^daily$|^each day$/.test(daily[0])) strip(daily[0]);
  }

  let due: number | null = null;

  // Relative: "in 20 minutes", "in an hour", "in half an hour", "in 2 days".
  const rel = text.match(/\bin (?:about |around )?(\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty|forty-five|sixty|ninety|half an?|a couple(?: of)?|a few|couple(?: of)?|few) (minutes?|mins?|hours?|hrs?|days?|weeks?)\b/);
  if (rel && repeat === "none") {
    const word = rel[1].replace(/^half an?$/, "half").replace(/^a couple(?: of)?$|^couple(?: of)?$/, "couple").replace(/^a few$/, "few");
    const amount = num(word);
    const unit = rel[2];
    if (amount != null) {
      const ms = unit.startsWith("m") ? MINUTE : unit.startsWith("h") ? 60 * MINUTE : unit.startsWith("d") ? DAY : 7 * DAY;
      due = now + amount * ms;
      strip(rel[0]);
    }
  }

  if (due == null) {
    let dayOffset: number | null = null;
    let explicitDate: { y: number; m: number; d: number } | null = null;
    const tomorrow = text.match(/\b(tomorrow|tmrw|tmr)\b/);
    const todayWord = text.match(/\b(today|tonight|this (?:morning|afternoon|evening))\b/);
    const dow = text.match(/\b(?:on |this |next )?(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b/);
    const monthDate = text.match(/\b(?:on )?(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? (\d{1,2})(?:st|nd|rd|th)?(?:,? (\d{4}))?\b/);
    const slashDate = text.match(/\b(?:on )?(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?\b/);
    if (tomorrow) {
      dayOffset = 1;
      strip(tomorrow[0]);
    } else if (monthDate) {
      const m = MONTHS.indexOf(monthDate[1].slice(0, 3)) + 1;
      const d = Number(monthDate[2]);
      let y = monthDate[3] ? Number(monthDate[3]) : today.y;
      if (!monthDate[3] && (m < today.m || (m === today.m && d < today.d))) y += 1;
      explicitDate = { y, m, d };
      strip(monthDate[0]);
    } else if (slashDate) {
      const m = Number(slashDate[1]);
      const d = Number(slashDate[2]);
      if (m >= 1 && m <= 12 && d >= 1 && d <= 31) {
        let y = slashDate[3] ? Number(slashDate[3].length === 2 ? `20${slashDate[3]}` : slashDate[3]) : today.y;
        if (!slashDate[3] && (m < today.m || (m === today.m && d < today.d))) y += 1;
        explicitDate = { y, m, d };
        strip(slashDate[0]);
      }
    } else if (dow && repeat === "none") {
      const target = WEEKDAYS.indexOf(dow[1]);
      let diff = (target - today.dow + 7) % 7;
      if (diff === 0 || /\bnext\b/.test(dow[0])) diff = diff === 0 ? 7 : diff;
      dayOffset = diff;
      strip(dow[0]);
    } else if (todayWord) {
      dayOffset = 0;
    }

    let partHour: number | null = null;
    for (const [pattern, hour] of PART_OF_DAY) {
      const hit = text.match(pattern);
      if (hit) {
        partHour = hour;
        strip(hit[0]);
        break;
      }
    }
    const clock = clockFrom(text);
    let h: number | null = null;
    let min = 0;
    if (clock) {
      h = clock.h;
      min = clock.min;
      if (!clock.ampm && partHour != null) {
        h = clock.h % 12 + (partHour >= 15 ? 12 : 0);
      }
      strip(clock.match);
    } else if (partHour != null) {
      h = partHour;
    }
    if (todayWord) strip(todayWord[0]);
    if (h == null && (dayOffset != null || explicitDate || repeat !== "none")) h = 9;
    if (h == null) return null;

    if (repeat === "weekly" && weeklyDay != null) {
      dayOffset = (weeklyDay - today.dow + 7) % 7;
    }
    let base = explicitDate ?? addDays(today, dayOffset ?? 0);
    due = zonedToUtc(base.y, base.m, base.d, h, min, tz);
    if (due <= now + 30_000) {
      if (explicitDate) return null;
      const step = repeat === "weekly" || (dayOffset != null && dayOffset > 1) ? 7 : 1;
      if (dayOffset == null || dayOffset === 0 || repeat !== "none") {
        base = addDays(localParts(due, tz), step);
        due = zonedToUtc(base.y, base.m, base.d, h, min, tz);
      } else {
        return null;
      }
    }
    if (repeat === "weekdays") {
      while ([0, 6].includes(localParts(due, tz).dow)) {
        const next = addDays(localParts(due, tz), 1);
        due = zonedToUtc(next.y, next.m, next.d, h, min, tz);
      }
    }
  }

  const body = cleanReminderText(text, checkIn);
  if (!body) return null;
  return { dueAt: due, repeat, text: body, checkIn };
}

export function cleanReminderText(raw: string, checkIn = false): string {
  let text = raw
    .replace(/^\s*(?:(?:hey )?nous,?\s+)?(?:(?:can|could|would|will) you\s+|please\s+)*/i, "")
    .replace(/\b(remind me|set (?:a |an )?reminder|ping me|nudge me|check in (?:with|on) me)\b/i, " ")
    .replace(/^\s*(to|that|about|for|on|at)\b/i, " ")
    .replace(/\s+(to|that|about|for|on|at|in|and)\s*$/i, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^(to|that|about|for)\s+/i, "")
    .replace(/[?.!,\s]+$/, "")
    .replace(/\bplease\b/gi, "")
    .replace(/\bmy\b/gi, "your")
    .replace(/\bi\b/g, "you")
    .replace(/\bme\b/gi, "you")
    .replace(/\s+/g, " ")
    .trim();
  if (!text && checkIn) text = "how are you doing";
  return text.slice(0, MAX_REMINDER_TEXT);
}

/** Fallback for odd phrasings: the small model returns local time + text as JSON. */
export async function parseWithModel(
  ai: AiLike,
  question: string,
  tz: string,
  now: number,
): Promise<ParsedReminder | null> {
  const local = localParts(now, tz);
  const stamp = `${local.y}-${String(local.m).padStart(2, "0")}-${String(local.d).padStart(2, "0")} ${String(local.h).padStart(2, "0")}:${String(local.min).padStart(2, "0")} (${WEEKDAYS[local.dow]})`;
  const raw = (await ai.run(PARSE_MODEL, {
    messages: [
      {
        role: "system",
        content:
          'Extract a reminder. Reply with JSON only: {"date":"YYYY-MM-DD","time":"HH:MM","repeat":"none|daily|weekly|weekdays","text":"what to remind about, short, second person"}. ' +
          'If no time of day is given use 09:00. If there is no date or time at all, reply {"date":null}.',
      },
      { role: "user", content: `Now: ${stamp}\nMessage: ${question.slice(0, 300)}` },
    ],
    max_tokens: 80,
    temperature: 0,
  })) as { response?: unknown };
  const reply = typeof raw?.response === "string" ? raw.response : typeof raw?.response === "object" ? JSON.stringify(raw.response) : "";
  const match = reply.match(/\{[\s\S]*\}/);
  if (!match) return null;
  let data: { date?: unknown; time?: unknown; repeat?: unknown; text?: unknown };
  try {
    data = JSON.parse(match[0]);
  } catch {
    return null;
  }
  const date = typeof data.date === "string" ? data.date.match(/^(\d{4})-(\d{2})-(\d{2})$/) : null;
  const time = typeof data.time === "string" ? data.time.match(/^(\d{1,2}):(\d{2})$/) : null;
  if (!date) return null;
  const h = time ? Number(time[1]) : 9;
  const min = time ? Number(time[2]) : 0;
  const dueAt = zonedToUtc(Number(date[1]), Number(date[2]), Number(date[3]), h, min, tz);
  if (!Number.isFinite(dueAt) || dueAt <= now) return null;
  const repeat = ["daily", "weekly", "weekdays"].includes(String(data.repeat)) ? (String(data.repeat) as Repeat) : "none";
  const text = typeof data.text === "string" ? data.text.replace(/\s+/g, " ").trim().slice(0, MAX_REMINDER_TEXT) : "";
  if (!text) return null;
  return { dueAt, repeat, text, checkIn: /\bcheck in\b/i.test(question) };
}

/* --------------------------------------------------------- storage */

export type ReminderDeps = {
  db: D1Like;
  ai?: AiLike;
  scope: string;
  userId: string | null;
  owner: boolean;
  capKey: string | null;
  threadId: string | null;
  tz: string;
  env?: Record<string, unknown>;
  now?: number;
};

export type ReminderReply = { text: string; reminderId?: string };

function repeatLabel(repeat: Repeat): string {
  if (repeat === "daily") return " and every day after that";
  if (repeat === "weekly") return " and every week after that";
  if (repeat === "weekdays") return " and every weekday after that";
  return "";
}

export async function createReminder(deps: ReminderDeps, question: string): Promise<ReminderReply> {
  const now = deps.now ?? Date.now();
  if (!deps.userId) return { text: "Sign in and I can set reminders for you." };
  let parsed = parseReminder(question, deps.tz, now);
  if (!parsed && deps.ai) {
    try {
      const reserved = await tryReserve({ db: deps.db, ip: deps.capKey, owner: deps.owner, reserveNeurons: 8 });
      if (reserved.ok) parsed = await parseWithModel(deps.ai, question, deps.tz, now);
    } catch {
      parsed = null;
    }
  }
  if (!parsed) {
    return {
      text: "Sure, when should I remind you? Try something like \"tomorrow at 9am\", \"in 2 hours\", or \"every weekday at 8am\".",
    };
  }
  if (parsed.dueAt < now + 30_000) return { text: "That time has already passed. When should I remind you instead?" };
  if (parsed.dueAt > now + 366 * DAY) return { text: "I can set reminders up to a year out. Want to pick a closer date?" };
  const active = await deps.db
    .prepare("SELECT COUNT(*) AS n FROM reminders WHERE scope = ? AND status = 'pending'")
    .bind(deps.scope)
    .first<{ n?: number }>();
  if (Number(active?.n || 0) >= MAX_ACTIVE_REMINDERS) {
    return { text: `You already have ${MAX_ACTIVE_REMINDERS} reminders waiting. Say "cancel my reminders" to clear some out first.` };
  }
  const claim = await claimAbility({
    db: deps.db,
    kind: "reminders",
    scope: deps.scope,
    owner: deps.owner,
    caps: abilityCaps("reminders", deps.env),
    now: new Date(now),
  });
  if (!claim.ok) {
    return { text: "You've set the most reminders I can add in one day. New ones open up at midnight UTC." };
  }
  const id = newAbilityId("rem");
  await deps.db
    .prepare(
      `INSERT INTO reminders (id, scope, user_id, thread_id, text, kind, due_at, repeat, tz, status, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)`,
    )
    .bind(
      id,
      deps.scope,
      deps.userId,
      deps.threadId,
      parsed.text,
      parsed.checkIn ? "checkin" : "reminder",
      parsed.dueAt,
      parsed.repeat,
      deps.tz,
      new Date(now).toISOString(),
    )
    .run();
  const when = formatWhen(parsed.dueAt, deps.tz, now);
  const what = parsed.checkIn ? `check in with you about ${parsed.text}` : `remind you to ${parsed.text}`;
  return {
    text: `Got it. I'll ${what} ${when}${repeatLabel(parsed.repeat)}. It'll pop up here in Nous.\n[[reminder:${id}]]`,
    reminderId: id,
  };
}

export async function listReminders(deps: ReminderDeps): Promise<ReminderReply> {
  const now = deps.now ?? Date.now();
  const result = await deps.db
    .prepare("SELECT text, kind, due_at, repeat, tz FROM reminders WHERE scope = ? AND status = 'pending' ORDER BY due_at ASC LIMIT 25")
    .bind(deps.scope)
    .all();
  const rows = result.results || [];
  if (!rows.length) return { text: "You don't have any reminders set. Just say something like \"remind me to stretch at 3pm\"." };
  const lines = rows.map((row) => {
    const tz = validTimeZone(row.tz);
    const label = row.kind === "checkin" ? `Check in: ${row.text}` : String(row.text);
    return `• ${label}: ${formatWhen(Number(row.due_at), tz, now)}${repeatLabel(row.repeat as Repeat).replace(" and ", ", then ")}`;
  });
  return { text: `Here's what's coming up:\n${lines.join("\n")}` };
}

export async function cancelReminders(deps: ReminderDeps, question: string): Promise<ReminderReply> {
  const q = question.toLowerCase();
  const all = /\b(all|every|my reminders|the reminders|reminders)\b/.test(q) && !/\breminder (about|to|for)\b/.test(q);
  if (all) {
    const result = await deps.db
      .prepare("UPDATE reminders SET status = 'cancelled' WHERE scope = ? AND status = 'pending'")
      .bind(deps.scope)
      .run();
    const n = Number(result.meta?.changes || 0);
    return { text: n ? `Done. I cancelled ${n === 1 ? "your reminder" : `all ${n} reminders`}.` : "You didn't have any reminders set." };
  }
  const topic = q.replace(/^.*?\breminders?\b\s*(about|to|for|on)?\s*/, "").replace(/[?.!]+$/, "").trim();
  const result = await deps.db
    .prepare("SELECT id, text FROM reminders WHERE scope = ? AND status = 'pending' ORDER BY due_at ASC LIMIT 25")
    .bind(deps.scope)
    .all();
  const rows = result.results || [];
  const words = topic.split(/\s+/).filter((word) => word.length > 2);
  const hit = rows.find((row) => words.length && words.every((word) => String(row.text).toLowerCase().includes(word))) ||
    (rows.length === 1 ? rows[0] : undefined);
  if (!hit) {
    return { text: rows.length ? "Which one? Say \"what are my reminders\" to see them, or \"cancel all my reminders\"." : "You didn't have any reminders set." };
  }
  await deps.db.prepare("UPDATE reminders SET status = 'cancelled' WHERE id = ? AND scope = ?").bind(hit.id, deps.scope).run();
  return { text: `Done. I cancelled the reminder to ${hit.text}.` };
}

/* ----------------------------------------------------------- firing */

export function nextDue(dueAt: number, repeat: Repeat, tz: string, now: number): number | null {
  if (repeat === "none") return null;
  const start = localParts(dueAt, tz);
  let day = { y: start.y, m: start.m, d: start.d };
  for (let i = 0; i < 800; i += 1) {
    day = addDays(day, repeat === "weekly" ? 7 : 1);
    const at = zonedToUtc(day.y, day.m, day.d, start.h, start.min, tz);
    if (at <= now) continue;
    if (repeat === "weekdays") {
      const dow = new Date(Date.UTC(day.y, day.m - 1, day.d)).getUTCDay();
      if (dow === 0 || dow === 6) continue;
    }
    return at;
  }
  return null;
}

export function reminderMessage(row: { text: unknown; kind?: unknown }): string {
  const text = String(row.text || "").trim();
  if (row.kind === "checkin") return `👋 Checking in: ${text}. How's it going?`;
  return `⏰ Reminder: ${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

/**
 * Email the reminder when account email is live: the EMAIL binding and a
 * youneek sender exist, the owner switched sending on (app_settings), and the
 * person verified their address. Any missing piece just skips email.
 */
export async function emailReminder(
  db: D1Like,
  env: Record<string, unknown>,
  userId: string | null,
  message: string,
): Promise<boolean> {
  if (!userId) return false;
  try {
    const mailEnv = env as EmailEnv;
    const base = publicBase(mailEnv);
    const result = await sendToVerifiedUser(db, mailEnv, "reminder", userId, {
      subject: message.replace(/^\S+\s/, "").slice(0, 120),
      text: `${message}\n\nOpen Nous: ${base}\n\n— Nous · The architecture of understanding.`,
      html: `<!doctype html><html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0b10;color:#eee;margin:0"><div style="max-width:480px;margin:0 auto;padding:32px 24px"><p style="font-size:20px;font-weight:700;margin:0 0 20px">${escapeHtml(message)}</p><p><a href="${escapeHtml(base)}" style="display:inline-block;background:#fff;color:#000;text-decoration:none;font-weight:600;padding:12px 20px;border-radius:999px">Open Nous</a></p><p style="font-size:12px;color:#777;margin-top:24px">Nous · The architecture of understanding.</p></div></body></html>`,
    });
    return result.ok;
  } catch {
    return false;
  }
}

/** Cron: fire everything due. Returns how many reminders fired. */
export async function fireDueReminders(
  db: D1Like,
  env: Record<string, unknown>,
  appendToThread: (threadId: string, text: string) => Promise<void>,
  now = Date.now(),
  limit = 100,
): Promise<number> {
  const result = await db
    .prepare(
      "SELECT id, scope, user_id, thread_id, text, kind, due_at, repeat, tz FROM reminders WHERE status = 'pending' AND due_at <= ? ORDER BY due_at ASC LIMIT ?",
    )
    .bind(now, limit)
    .all();
  let fired = 0;
  for (const row of result.results || []) {
    const id = String(row.id);
    const repeat = (["daily", "weekly", "weekdays"].includes(String(row.repeat)) ? row.repeat : "none") as Repeat;
    const tz = validTimeZone(row.tz);
    const next = nextDue(Number(row.due_at), repeat, tz, now);
    // Claim the row first so an overlapping cron run can't fire it twice.
    const claimed = next
      ? await db
          .prepare("UPDATE reminders SET due_at = ?, fired_at = ? WHERE id = ? AND status = 'pending' AND due_at = ?")
          .bind(next, new Date(now).toISOString(), id, row.due_at)
          .run()
      : await db
          .prepare("UPDATE reminders SET status = 'fired', fired_at = ? WHERE id = ? AND status = 'pending' AND due_at = ?")
          .bind(new Date(now).toISOString(), id, row.due_at)
          .run();
    if (!Number(claimed.meta?.changes || 0)) continue;
    const message = reminderMessage(row as { text: unknown; kind?: unknown });
    await db
      .prepare("INSERT INTO reminder_deliveries (id, reminder_id, scope, text, fired_at, delivered_at) VALUES (?, ?, ?, ?, ?, NULL)")
      .bind(newAbilityId("rdl"), id, row.scope, message, new Date(now).toISOString())
      .run();
    if (typeof row.thread_id === "string" && row.thread_id) {
      try {
        await appendToThread(row.thread_id, message);
      } catch {
        // The pop-up still shows even if the chat is gone.
      }
    }
    await emailReminder(db, env, typeof row.user_id === "string" ? row.user_id : null, message);
    fired += 1;
  }
  return fired;
}

/** In-app pop-ups not yet shown to this person. */
export async function pendingDeliveries(db: D1Like, scope: string): Promise<Array<{ id: string; text: string; fired_at: string }>> {
  const result = await db
    .prepare(
      "SELECT id, text, fired_at FROM reminder_deliveries WHERE scope = ? AND delivered_at IS NULL ORDER BY fired_at ASC LIMIT 20",
    )
    .bind(scope)
    .all();
  return (result.results || []).map((row) => ({ id: String(row.id), text: String(row.text), fired_at: String(row.fired_at) }));
}

export async function acknowledgeDeliveries(db: D1Like, scope: string, ids: string[], now = new Date()): Promise<number> {
  let n = 0;
  for (const id of ids.slice(0, 50)) {
    if (!/^rdl_[a-f0-9]{20}$/.test(id)) continue;
    const result = await db
      .prepare("UPDATE reminder_deliveries SET delivered_at = ? WHERE id = ? AND scope = ? AND delivered_at IS NULL")
      .bind(now.toISOString(), id, scope)
      .run();
    n += Number(result.meta?.changes || 0);
  }
  return n;
}

/** Drop delivery rows and finished reminders older than 30 days. */
export async function cleanupReminders(db: D1Like, now = Date.now()): Promise<void> {
  const cutoff = new Date(now - 30 * DAY).toISOString();
  await db.prepare("DELETE FROM reminder_deliveries WHERE fired_at < ?").bind(cutoff).run();
  await db.prepare("DELETE FROM reminders WHERE status != 'pending' AND created_at < ?").bind(cutoff).run();
}
