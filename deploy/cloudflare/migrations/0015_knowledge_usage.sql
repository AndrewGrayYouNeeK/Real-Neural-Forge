-- Daily live-knowledge lookups (web_search / wiki_summary / read_page).
-- scope is 'global', 'owner', 'user:<id>' or 'ip:<hex>'; raw IPs are not stored.
CREATE TABLE IF NOT EXISTS knowledge_usage (
    day TEXT NOT NULL,
    scope TEXT NOT NULL,
    lookups INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, scope)
);
