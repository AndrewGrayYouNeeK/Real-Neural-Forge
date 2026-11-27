-- Owner-only training examples for the Nous LoRA (see training/README.md).
-- messages is a JSON array of {role, content} turns (user/assistant only; the
-- system prompt is added at export time from training/system_prompt.txt).
-- source: save (kept as-is), fix (rewritten answer), manual (typed from
-- scratch), bad (thumbs-down). label: good (SFT data) or rejected (never mixed
-- into SFT; exported separately). pair_id links a rejected answer to the good
-- fix written for it. tags is a JSON array of short words. No secrets: the
-- Worker refuses text that looks like a key or token.

CREATE TABLE IF NOT EXISTS training_examples (
    id TEXT PRIMARY KEY,
    messages TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    label TEXT NOT NULL DEFAULT 'good',
    tags TEXT NOT NULL DEFAULT '[]',
    pair_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_examples_label ON training_examples (label, created_at);
