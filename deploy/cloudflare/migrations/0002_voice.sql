-- GQ voice metering. The same script is applied to local SQLite and to D1.
-- Balances are per account. Usage is the audit trail for every spend.
-- Audio bytes live in the same blob store as workspace files, under
-- voice/cache/<sha256>.mp3, so identical text never costs a second call.

CREATE TABLE IF NOT EXISTS voice_balances (
    user_id TEXT PRIMARY KEY,
    free_remaining INTEGER NOT NULL DEFAULT 0,
    purchased_remaining INTEGER NOT NULL DEFAULT 0,
    spent INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_usage (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    chars INTEGER NOT NULL,
    cached INTEGER NOT NULL DEFAULT 0,
    free_after INTEGER,
    purchased_after INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_purchases (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    pack_id TEXT NOT NULL,
    chars INTEGER NOT NULL,
    session_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_daily (
    day TEXT PRIMARY KEY,
    chars INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_voice_usage_user ON voice_usage(user_id);