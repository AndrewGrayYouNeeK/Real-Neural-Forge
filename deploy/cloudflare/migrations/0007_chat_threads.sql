-- Persistent agent chat threads per memory scope (user / owner / visitor IP).
-- Additive. Messages restore context for /agent/chat when a sidebar chat is reopened.

CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_threads_scope_updated
    ON chat_threads (scope, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created
    ON chat_messages (thread_id, created_at ASC);
