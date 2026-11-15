-- Shareable series report snapshots. Additive.
-- Public GET by id; create stores a frozen JSON snapshot (forecast/accuracy/regimes).

CREATE TABLE IF NOT EXISTS public_reports (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    series TEXT NOT NULL,
    title TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_public_reports_scope_created
    ON public_reports (scope, created_at DESC);
