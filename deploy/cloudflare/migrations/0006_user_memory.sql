-- Per-user (or IP/owner) chat memory: preferences and facts the agent can reuse.
-- Additive. Scoped so owner, signed-in members, and anonymous visitors each get a bucket.

CREATE TABLE IF NOT EXISTS user_memory (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (scope, key)
);

CREATE INDEX IF NOT EXISTS idx_user_memory_scope ON user_memory (scope);
