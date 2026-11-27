import { describe, expect, it } from "vitest";

import {
  acknowledgeDeliveries,
  cancelReminders,
  createReminder,
  fireDueReminders,
  formatWhen,
  listReminders,
  localParts,
  nextDue,
  parseReminder,
  pendingDeliveries,
  reminderIntent,
  validTimeZone,
  zonedToUtc,
} from "../src/abilities/reminders";

const TZ = "America/New_York";
// Thu Nov 26 2026, 9:40 PM in New York (02:40 UTC Nov 27).
const NOW = Date.UTC(2026, 10, 27, 2, 40);

function at(parsedDue: number) {
  const p = localParts(parsedDue, TZ);
  return `${p.y}-${p.m}-${p.d} ${p.h}:${String(p.min).padStart(2, "0")}`;
}

/** Minimal D1 for reminders + usage counters. */
function fakeDb() {
  const usage = new Map<string, number>();
  const reminders: Array<Record<string, unknown>> = [];
  const deliveries: Array<Record<string, unknown>> = [];
  const exec = (sql: string, a: unknown[]) => {
    const s = sql.replace(/\s+/g, " ").trim();
    if (s.startsWith("SELECT requests FROM ai_usage")) {
      const n = usage.get(`${a[0]}|${a[1]}`);
      return { results: n == null ? [] : [{ requests: n }] };
    }
    if (s.startsWith("INSERT INTO ai_usage")) {
      const key = `${a[0]}|${a[1]}`;
      if (!usage.has(key)) usage.set(key, 0);
      return { results: [], meta: { changes: 1 } };
    }
    if (s.startsWith("UPDATE ai_usage SET requests = requests + 1")) {
      const key = `${a[0]}|${a[1]}`;
      const n = usage.get(key) || 0;
      if (n + 1 > Number(a[2])) return { results: [], meta: { changes: 0 } };
      usage.set(key, n + 1);
      return { results: [], meta: { changes: 1 } };
    }
    if (s.startsWith("SELECT COUNT(*) AS n FROM reminders")) {
      return { results: [{ n: reminders.filter((r) => r.scope === a[0] && r.status === "pending").length }] };
    }
    if (s.startsWith("INSERT INTO reminders")) {
      const [id, scope, user_id, thread_id, text, kind, due_at, repeat, tz, created_at] = a;
      reminders.push({ id, scope, user_id, thread_id, text, kind, due_at, repeat, tz, status: "pending", created_at });
      return { results: [], meta: { changes: 1 } };
    }
    if (s.startsWith("SELECT text, kind, due_at, repeat, tz FROM reminders") || s.startsWith("SELECT id, text FROM reminders")) {
      return { results: reminders.filter((r) => r.scope === a[0] && r.status === "pending") };
    }
    if (s.startsWith("UPDATE reminders SET status = 'cancelled' WHERE scope = ?")) {
      let n = 0;
      for (const r of reminders) if (r.scope === a[0] && r.status === "pending") (r.status = "cancelled"), (n += 1);
      return { results: [], meta: { changes: n } };
    }
    if (s.startsWith("UPDATE reminders SET status = 'cancelled' WHERE id = ?")) {
      const r = reminders.find((x) => x.id === a[0] && x.scope === a[1]);
      if (r) r.status = "cancelled";
      return { results: [], meta: { changes: r ? 1 : 0 } };
    }
    if (s.startsWith("SELECT id, scope, user_id, thread_id, text, kind, due_at, repeat, tz FROM reminders WHERE status = 'pending' AND due_at <= ?")) {
      return { results: reminders.filter((r) => r.status === "pending" && Number(r.due_at) <= Number(a[0])) };
    }
    if (s.startsWith("UPDATE reminders SET due_at = ?")) {
      const r = reminders.find((x) => x.id === a[2] && x.status === "pending" && x.due_at === a[3]);
      if (r) r.due_at = a[0];
      return { results: [], meta: { changes: r ? 1 : 0 } };
    }
    if (s.startsWith("UPDATE reminders SET status = 'fired'")) {
      const r = reminders.find((x) => x.id === a[1] && x.status === "pending" && x.due_at === a[2]);
      if (r) r.status = "fired";
      return { results: [], meta: { changes: r ? 1 : 0 } };
    }
    if (s.startsWith("INSERT INTO reminder_deliveries")) {
      deliveries.push({ id: a[0], reminder_id: a[1], scope: a[2], text: a[3], fired_at: a[4], delivered_at: null });
      return { results: [], meta: { changes: 1 } };
    }
    if (s.startsWith("SELECT id, text, fired_at FROM reminder_deliveries")) {
      return { results: deliveries.filter((d) => d.scope === a[0] && d.delivered_at == null) };
    }
    if (s.startsWith("UPDATE reminder_deliveries SET delivered_at")) {
      const d = deliveries.find((x) => x.id === a[1] && x.scope === a[2] && x.delivered_at == null);
      if (d) d.delivered_at = a[0];
      return { results: [], meta: { changes: d ? 1 : 0 } };
    }
    if (s.startsWith("SELECT value FROM app_settings")) return { results: [] };
    throw new Error(`unexpected SQL: ${s}`);
  };
  const db = {
    prepare(sql: string) {
      const bound = (args: unknown[]) => ({
        all: async () => exec(sql, args),
        run: async () => exec(sql, args),
        first: async <T>() => ((exec(sql, args).results || [])[0] ?? null) as T | null,
      });
      return { ...bound([]), bind: (...args: unknown[]) => bound(args) };
    },
  };
  return { db, reminders, deliveries };
}

describe("reminder intents", () => {
  it("spots reminder asks and leaves ordinary questions alone", () => {
    expect(reminderIntent("remind me to call mom tomorrow at 5pm")).toBe("create");
    expect(reminderIntent("Can you remind me in 20 minutes to check the oven?")).toBe("create");
    expect(reminderIntent("check in with me every evening at 8 about my workout")).toBe("create");
    expect(reminderIntent("what are my reminders?")).toBe("list");
    expect(reminderIntent("cancel all my reminders")).toBe("cancel");
    expect(reminderIntent("remind me what photosynthesis is")).toBeNull();
    expect(reminderIntent("what's the weather tomorrow")).toBeNull();
  });
});

describe("reading times in the person's time zone", () => {
  it("handles relative times", () => {
    const r = parseReminder("remind me in 20 minutes to check the oven", TZ, NOW)!;
    expect(r.dueAt).toBe(NOW + 20 * 60_000);
    expect(r.text).toBe("check the oven");
    expect(parseReminder("remind me to stretch in an hour", TZ, NOW)!.dueAt).toBe(NOW + 3_600_000);
    expect(parseReminder("remind me in half an hour to stretch", TZ, NOW)!.dueAt).toBe(NOW + 1_800_000);
  });

  it("handles days and clock times", () => {
    const r = parseReminder("remind me to call mom tomorrow at 5pm", TZ, NOW)!;
    expect(at(r.dueAt)).toBe("2026-11-27 17:00");
    expect(r.text).toBe("call mom");
    expect(at(parseReminder("remind me at 10:30 pm to lock the door", TZ, NOW)!.dueAt)).toBe("2026-11-26 22:30");
    // 9 AM already passed today, so it rolls to tomorrow.
    expect(at(parseReminder("remind me at 9am to take my vitamins", TZ, NOW)!.dueAt)).toBe("2026-11-27 9:00");
    expect(at(parseReminder("remind me on monday to pay rent", TZ, NOW)!.dueAt)).toBe("2026-11-30 9:00");
    expect(at(parseReminder("remind me on Dec 5 at 3pm about the dentist", TZ, NOW)!.dueAt)).toBe("2026-12-5 15:00");
    expect(parseReminder("remind me on Dec 5 at 3pm about the dentist", TZ, NOW)!.text).toBe("the dentist");
    expect(at(parseReminder("remind me tomorrow morning to email Sam", TZ, NOW)!.dueAt)).toBe("2026-11-27 9:00");
  });

  it("handles repeating check-ins", () => {
    const r = parseReminder("check in with me every evening at 8 about my workout", TZ, NOW)!;
    expect(r.repeat).toBe("daily");
    expect(r.checkIn).toBe(true);
    expect(at(r.dueAt)).toBe("2026-11-27 20:00");
    expect(r.text).toBe("your workout");
    const w = parseReminder("remind me every weekday at 8am to stand up", TZ, NOW)!;
    expect(w.repeat).toBe("weekdays");
    expect(at(w.dueAt)).toBe("2026-11-27 8:00");
    expect(parseReminder("remind me to do something", TZ, NOW)).toBeNull();
  });

  it("converts wall-clock times across daylight saving", () => {
    const summer = zonedToUtc(2026, 7, 1, 9, 0, TZ);
    expect(new Date(summer).toISOString()).toBe("2026-07-01T13:00:00.000Z");
    const winter = zonedToUtc(2026, 12, 1, 9, 0, TZ);
    expect(new Date(winter).toISOString()).toBe("2026-12-01T14:00:00.000Z");
    expect(validTimeZone("Not/AZone")).toBe(TZ);
    expect(formatWhen(NOW + 20 * 60_000, TZ, NOW)).toMatch(/^today at 10:00 PM EST$/);
    const daily = nextDue(zonedToUtc(2027, 3, 13, 8, 0, TZ), "daily", TZ, zonedToUtc(2027, 3, 13, 8, 1, TZ));
    expect(at(daily!)).toBe("2027-3-14 8:00");
    const weekday = nextDue(zonedToUtc(2026, 11, 27, 8, 0, TZ), "weekdays", TZ, zonedToUtc(2026, 11, 27, 8, 1, TZ));
    expect(at(weekday!)).toBe("2026-11-30 8:00");
  });
});

describe("storing and firing reminders", () => {
  it("saves, lists, fires once, delivers in-app, and cancels", async () => {
    const { db, reminders } = fakeDb();
    const deps = { db, scope: "user:u1", userId: "u1", owner: false, capKey: "acct:u1", threadId: "thr_1", tz: TZ, now: NOW };
    const made = await createReminder(deps, "remind me in 20 minutes to check the oven");
    expect(made.text).toContain("remind you to check the oven today at 10:00 PM");
    expect(made.text).toMatch(/\[\[reminder:rem_[a-f0-9]{20}\]\]/);
    expect((await listReminders(deps)).text).toContain("check the oven");

    const appended: string[] = [];
    const append = async (thread: string, text: string) => {
      appended.push(`${thread}:${text}`);
    };
    expect(await fireDueReminders(db, {}, append, NOW + 10 * 60_000)).toBe(0);
    expect(await fireDueReminders(db, {}, append, NOW + 21 * 60_000)).toBe(1);
    expect(await fireDueReminders(db, {}, append, NOW + 22 * 60_000)).toBe(0);
    expect(appended).toEqual(["thr_1:⏰ Reminder: Check the oven."]);
    const due = await pendingDeliveries(db, "user:u1");
    expect(due.map((d) => d.text)).toEqual(["⏰ Reminder: Check the oven."]);
    expect(await acknowledgeDeliveries(db, "user:u1", due.map((d) => d.id))).toBe(1);
    expect(await pendingDeliveries(db, "user:u1")).toEqual([]);

    await createReminder(deps, "check in with me every evening at 8 about my workout");
    const daily = reminders.find((r) => r.repeat === "daily")!;
    const firstDue = Number(daily.due_at);
    expect(await fireDueReminders(db, {}, append, firstDue + 1000)).toBe(1);
    expect(Number(daily.due_at)).toBe(firstDue + 86_400_000);
    expect(daily.status).toBe("pending");
    expect((await cancelReminders(deps, "cancel all my reminders")).text).toContain("cancelled");
    expect(daily.status).toBe("cancelled");
  });

  it("asks for a time when it can't find one and respects the daily cap", async () => {
    const { db } = fakeDb();
    const deps = {
      db,
      scope: "user:u2",
      userId: "u2",
      owner: false,
      capKey: "acct:u2",
      threadId: null,
      tz: TZ,
      now: NOW,
      env: { HENAD_REMINDERS_MEMBER_DAILY: "1" },
    };
    expect((await createReminder(deps, "remind me to water the plants")).text).toContain("when should I remind you");
    expect((await createReminder(deps, "remind me in 5 minutes to breathe")).text).toContain("Got it");
    expect((await createReminder(deps, "remind me in 6 minutes to breathe")).text).toContain("most reminders");
  });
});
