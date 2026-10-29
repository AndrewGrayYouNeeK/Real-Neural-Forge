"""Errors the workspace routes turn into HTTP responses."""

from __future__ import annotations


class WorkspaceError(Exception):
    """A request failed with an HTTP status and a client-facing detail."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
