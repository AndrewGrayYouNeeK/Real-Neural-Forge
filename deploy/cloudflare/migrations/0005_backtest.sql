-- Latest rolling-origin backtest for a model. Additive.
-- The report JSON holds MAE, RMSE, MAPE, directional accuracy, coverage,
-- the three simple baselines, and the comparison sentence.

CREATE TABLE IF NOT EXISTS backtest_reports (
    model_id TEXT PRIMARY KEY,
    report_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
