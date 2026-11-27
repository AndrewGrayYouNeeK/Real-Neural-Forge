-- Email verification, password reset, the memory switch and the email
-- on/off setting. Additive only: no existing table changes shape.
-- Link tokens are stored only as sha256 hashes.

CREATE TABLE IF NOT EXISTS email_verifications (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    email TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_email_verifications_user ON email_verifications(user_id);

CREATE TABLE IF NOT EXISTS user_email_status (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    verified_at TEXT
);

CREATE TABLE IF NOT EXISTS password_resets (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_password_resets_user ON password_resets(user_id);

-- Site-wide switches. email_sending: 'on', 'off' (default) or 'down:<code>'.
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Per-account memory switch (missing row = on).
CREATE TABLE IF NOT EXISTS memory_settings (
    scope TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);

-- Daily account-email counts for the owner dashboard (no addresses).
CREATE TABLE IF NOT EXISTS email_log (
    day TEXT NOT NULL,
    kind TEXT NOT NULL,
    sent INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, kind)
);
