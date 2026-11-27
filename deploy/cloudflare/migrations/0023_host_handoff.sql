-- One-time codes that carry a signed-in session from an old host
-- (nous.youneek.* / forge.youneek.*) to the matching henad.youneek.* host.
-- The session cookie is host-only, so the old cookie is never sent to the new
-- host. Only the sha256 of each code is stored; codes expire after 2 minutes
-- and work once.
CREATE TABLE IF NOT EXISTS host_handoffs (
    code_hash TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL,
    target_host TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    created_at TEXT NOT NULL
);
