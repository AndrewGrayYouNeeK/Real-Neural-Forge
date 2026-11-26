"""Who is calling.

The owner is a normal email + password account (id ``owner``, role ``owner``).
Owner powers only unlock on a session that has passed a second factor
(passkey, authenticator code, or recovery code) within ``OWNER_MFA_SECONDS``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

OWNER_ID = "owner"
OWNER_EMAIL_ENV = "NEURAL_FORGE_OWNER_EMAIL"
DEFAULT_OWNER_EMAIL = "andrewgray@youneek.xyz"
SESSION_COOKIE = "nf_session"
SESSION_SECONDS = 14 * 24 * 60 * 60
# A second-factor check keeps owner powers unlocked for this long.
OWNER_MFA_SECONDS = 12 * 60 * 60


def owner_email() -> str:
    raw = os.environ.get(OWNER_EMAIL_ENV, "").strip().lower()
    return raw or DEFAULT_OWNER_EMAIL


@dataclass(frozen=True)
class Actor:
    """A signed-in visitor, or the owner when the server is still open locally."""

    user_id: str
    email: str
    role: str
    via: str
    # "ok" (second factor passed), "required" (enrolled, not passed on this
    # session), "setup" (owner must enrol), or "none" (members).
    mfa: str = "none"

    @property
    def is_owner(self) -> bool:
        if self.user_id != OWNER_ID or self.role != "owner":
            return False
        return self.via == "local" or self.mfa == "ok"

    def public(self) -> dict[str, str]:
        return {
            "id": self.user_id,
            "email": self.email,
            "role": self.role,
            "auth": self.via,
            "mfa": self.mfa,
            "owner_powers": "yes" if self.is_owner else "no",
        }
