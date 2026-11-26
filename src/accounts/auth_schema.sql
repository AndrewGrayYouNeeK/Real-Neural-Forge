-- Account security: owner second factor, passkeys, recovery codes, login lockout.
-- The same script is applied to local SQLite and to D1 (migration 0011).

CREATE TABLE IF NOT EXISTS session_mfa (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    method TEXT NOT NULL,
    verified_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mfa_totp (
    user_id TEXT PRIMARY KEY,
    secret TEXT,
    pending_secret TEXT,
    enabled_at TEXT,
    last_step INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mfa_recovery_codes (
    user_id TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    used_at TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, code_hash)
);

CREATE TABLE IF NOT EXISTS webauthn_credentials (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    public_key TEXT NOT NULL,
    alg INTEGER NOT NULL,
    sign_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_webauthn_user ON webauthn_credentials(user_id);

CREATE TABLE IF NOT EXISTS auth_challenges (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    challenge TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_attempts (
    key TEXT PRIMARY KEY,
    failures INTEGER NOT NULL,
    first_at TEXT NOT NULL,
    locked_until TEXT
);

-- One-time owner setup link. Only the sha256 of the token is stored; the
-- link itself is written to a file on the operator's machine.
CREATE TABLE IF NOT EXISTS owner_setup (
    token_hash TEXT PRIMARY KEY,
    expires_at TEXT NOT NULL,
    used_at TEXT
);
