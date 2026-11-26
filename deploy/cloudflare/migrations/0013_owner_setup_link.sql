-- Owner claim now needs a one-time setup link (token hash in owner_setup).
-- The owner row had an old, never-usable password hash; clear it so the
-- owner can set a password through the setup link, and end old sessions.

CREATE TABLE IF NOT EXISTS owner_setup (
    token_hash TEXT PRIMARY KEY,
    expires_at TEXT NOT NULL,
    used_at TEXT
);

UPDATE users
SET email = 'andrewgray@youneek.xyz', password_hash = NULL
WHERE id = 'owner'
  AND NOT EXISTS (
    SELECT 1 FROM users AS other
    WHERE other.email = 'andrewgray@youneek.xyz' AND other.id <> 'owner'
  )
  AND NOT EXISTS (SELECT 1 FROM webauthn_credentials WHERE user_id = 'owner')
  AND NOT EXISTS (SELECT 1 FROM mfa_totp WHERE user_id = 'owner' AND secret IS NOT NULL);

DELETE FROM sessions WHERE user_id = 'owner';
