-- Daily Workers AI usage caps. Limits reset by UTC date (00:00 UTC).
-- scope is 'global' or 'ip:<hex>'; raw IP addresses are not stored.

CREATE TABLE IF NOT EXISTS ai_usage (
    day TEXT NOT NULL,
    scope TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    neurons INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, scope)
);
