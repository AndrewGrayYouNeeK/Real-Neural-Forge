"""Test helper: a signed-in owner session that already passed its second factor."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from src.accounts.actor import OWNER_ID, owner_email
from src.workspace.runtime import runtime
from src.workspace.store import WorkspaceStore, utc_now


def owner_session_token(
    store: WorkspaceStore | None = None, *, mfa: bool = True
) -> str:
    target = store if store is not None else runtime.store
    assert target is not None, "owner sessions need a workspace store"
    target.ensure_owner(owner_email())
    token = secrets.token_urlsafe(24)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    target.create_session(token_hash, OWNER_ID, expires)
    if mfa:
        target.mark_session_mfa(token_hash, OWNER_ID, "test", utc_now())
    return token


def owner_headers(
    store: WorkspaceStore | None = None, *, mfa: bool = True
) -> dict[str, str]:
    return {"Authorization": f"Bearer {owner_session_token(store, mfa=mfa)}"}
