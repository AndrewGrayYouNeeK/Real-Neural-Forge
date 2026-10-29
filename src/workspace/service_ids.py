"""Short ids for workspace rows. Kept separate so the runner can import them."""

from __future__ import annotations

import secrets


def new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(8)
