-- Chat abilities: files people attach and images Nous makes.
-- Additive. Bytes live in R2 (neural-forge-workspaces); these rows record the
-- owner scope (user:<id>) so only that person can read them back.

CREATE TABLE IF NOT EXISTS user_files (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    thread_id TEXT,
    name TEXT NOT NULL,
    mime TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    chars INTEGER NOT NULL,
    r2_key TEXT NOT NULL,
    text_key TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_files_scope_thread
    ON user_files (scope, thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_files_created
    ON user_files (created_at);

CREATE TABLE IF NOT EXISTS generated_images (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    prompt TEXT NOT NULL,
    r2_key TEXT NOT NULL,
    mime TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_generated_images_scope_created
    ON generated_images (scope, created_at DESC);
