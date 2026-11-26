-- Owner becomes a normal email + password account with a required second factor.
-- Move the unclaimed owner row to the new address and end every old owner session
-- (those came from the retired shared owner token).

UPDATE users
SET email = 'andrewgray@youneek.xyz'
WHERE id = 'owner'
  AND password_hash IS NULL
  AND NOT EXISTS (
    SELECT 1 FROM users AS other
    WHERE other.email = 'andrewgray@youneek.xyz' AND other.id <> 'owner'
  );

DELETE FROM sessions WHERE user_id = 'owner';
