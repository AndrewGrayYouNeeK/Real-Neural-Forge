-- Nous Plus subscription: GQ voice included as a monthly allowance (no packs).

-- Nous Plus: the one paid plan. GQ voice is included as a monthly allowance.
-- One row per subscriber; voice_grants makes each billing period's allowance
-- land exactly once no matter which Stripe event or sync reaches us first.
CREATE TABLE IF NOT EXISTS voice_plans (
    user_id TEXT PRIMARY KEY,
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    status TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT,
    cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
    allowance_total INTEGER NOT NULL DEFAULT 0,
    allowance_remaining INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_grants (
    ref TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    chars INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_voice_plans_sub ON voice_plans(stripe_subscription_id);
