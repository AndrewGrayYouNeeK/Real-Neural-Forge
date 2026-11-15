-- In-app alert rules and fired events. Additive.
-- No email/SMS — panel + agent tools only.

CREATE TABLE IF NOT EXISTS alert_rules (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    name TEXT NOT NULL,
    series TEXT NOT NULL,
    kind TEXT NOT NULL,
    config_json TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alert_rules_scope_updated
    ON alert_rules (scope, updated_at DESC);

CREATE TABLE IF NOT EXISTS alert_events (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    fired_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_alert_events_scope_fired
    ON alert_events (scope, fired_at DESC);
