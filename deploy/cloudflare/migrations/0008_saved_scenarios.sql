-- Named forecast/question scenarios per memory scope. Additive.
-- Re-run from the Memory panel or via agent tools.

CREATE TABLE IF NOT EXISTS saved_scenarios (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    name TEXT NOT NULL,
    question TEXT NOT NULL,
    series TEXT,
    horizon INTEGER,
    settings_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_saved_scenarios_scope_updated
    ON saved_scenarios (scope, updated_at DESC);
