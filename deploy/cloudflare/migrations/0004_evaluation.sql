-- Issued forecasts and the empirical coverage of their intervals.
-- Additive: this does not alter the existing predictions table.
-- horizon is the lead, in steps (1 = the next observation).
-- coverage_80 is the share of actuals inside [p10, p90].
-- p10_hit_rate is the share with actual >= p10 (nominal about 0.90).
-- p90_hit_rate is the share with actual <= p90 (nominal about 0.90).
-- Windows are the most recent scored rows for that model and lead.

CREATE TABLE IF NOT EXISTS forecast_outcomes (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    issued_at TEXT NOT NULL,
    target_time TEXT,
    p10 REAL,
    p50 REAL NOT NULL,
    p90 REAL,
    actual REAL,
    scored_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_forecast_outcomes_model
    ON forecast_outcomes(model_id, horizon, issued_at);

CREATE INDEX IF NOT EXISTS idx_forecast_outcomes_unscored
    ON forecast_outcomes(model_id, target_time, actual);

CREATE TABLE IF NOT EXISTS coverage_windows (
    model_id TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    window_size INTEGER NOT NULL,
    n INTEGER NOT NULL,
    coverage_80 REAL,
    p10_hit_rate REAL,
    p90_hit_rate REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (model_id, horizon, window_size)
);
