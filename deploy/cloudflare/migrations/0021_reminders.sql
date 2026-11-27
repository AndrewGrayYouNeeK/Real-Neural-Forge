-- Reminders and scheduled check-ins set from chat ("remind me ...").
-- due_at is UTC milliseconds; tz is the person's IANA time zone so repeats
-- keep the same local time across daylight-saving changes.
-- A Worker cron trigger (every minute) fires due rows into reminder_deliveries
-- (in-app pop-ups) and the chat thread they came from.

CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    user_id TEXT,
    thread_id TEXT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'reminder',
    due_at INTEGER NOT NULL,
    repeat TEXT NOT NULL DEFAULT 'none',
    tz TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    fired_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reminders_status_due ON reminders (status, due_at);
CREATE INDEX IF NOT EXISTS idx_reminders_scope_status ON reminders (scope, status, due_at);

CREATE TABLE IF NOT EXISTS reminder_deliveries (
    id TEXT PRIMARY KEY,
    reminder_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    text TEXT NOT NULL,
    fired_at TEXT NOT NULL,
    delivered_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reminder_deliveries_scope ON reminder_deliveries (scope, delivered_at, fired_at);
