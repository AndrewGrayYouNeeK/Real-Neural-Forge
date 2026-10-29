"""Process-wide handle the API lifespan fills in."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.workspace.store import WorkspaceStore


class Runtime:
    """Workspace store for this process, plus whether the owner job is running."""

    def __init__(self) -> None:
        self.store: WorkspaceStore | None = None
        self.error: str | None = None
        self.config: dict[str, Any] | None = None
        self.owner_busy: Callable[[], bool] = lambda: False
        self.member_job_id: str | None = None


runtime = Runtime()
