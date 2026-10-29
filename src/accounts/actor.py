"""Who is calling. The owner is Andrew's existing API token."""

from __future__ import annotations

import os
from dataclasses import dataclass

OWNER_ID = "owner"
OWNER_EMAIL_ENV = "NEURAL_FORGE_OWNER_EMAIL"
DEFAULT_OWNER_EMAIL = "andrew@youneek.xyz"
SESSION_COOKIE = "nf_session"
SESSION_SECONDS = 14 * 24 * 60 * 60


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

    @property
    def is_owner(self) -> bool:
        return self.user_id == OWNER_ID and self.role == "owner"

    def public(self) -> dict[str, str]:
        return {
            "id": self.user_id,
            "email": self.email,
            "role": self.role,
            "auth": self.via,
        }
