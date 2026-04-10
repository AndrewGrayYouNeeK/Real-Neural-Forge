"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Health check")
async def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}
