"""HTTP routes for GQ voice status, playback, the subscription, and the Stripe webhook."""

from __future__ import annotations

from typing import Any, NoReturn

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.accounts.actor import Actor
from src.voice.config import resolve_checkout_base_url
from src.voice.service import (
    create_checkout,
    create_portal,
    handle_webhook,
    speak,
    sync_checkout,
    voice_status,
)
from src.workspace.errors import WorkspaceError
from src.workspace.runtime import runtime
from src.workspace.service import resolve_actor

router = APIRouter()

_SIGN_UP_PROMPT = "Create a free account to hear GQ. Text on Nous stays free."


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)


class SyncRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=200)


def _raise(exc: WorkspaceError) -> NoReturn:
    headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
    raise HTTPException(
        status_code=exc.status, detail=exc.detail, headers=headers
    ) from exc


def _voice_actor(request: Request, authorization: str | None) -> Actor:
    """Resolve the caller, with a sign-up prompt instead of a bare 401."""
    try:
        return resolve_actor(request, authorization)
    except WorkspaceError as exc:
        if exc.status == 401:
            _raise(WorkspaceError(401, _SIGN_UP_PROMPT))
        _raise(exc)


def _optional_actor(request: Request, authorization: str | None) -> Actor | None:
    try:
        return resolve_actor(request, authorization)
    except WorkspaceError:
        return None


def _app_config() -> dict[str, Any] | None:
    config = runtime.config
    return config if isinstance(config, dict) else None


@router.get("/voice/status", tags=["voice"])
def get_voice_status(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    try:
        return voice_status(_optional_actor(request, authorization), _app_config())
    except WorkspaceError as exc:
        _raise(exc)


@router.post("/voice/speak", tags=["voice"])
def voice_speak(
    body: SpeakRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> Response:
    actor = _voice_actor(request, authorization)
    try:
        result = speak(actor, body.text, _app_config())
    except WorkspaceError as exc:
        _raise(exc)
    headers = {
        "X-Neural-Forge-Voice-Cache": "hit" if result["cached"] else "miss",
        "X-Neural-Forge-Voice-Chars": str(result["chars"]),
        "Cache-Control": "no-store",
    }
    return Response(content=result["audio"], media_type="audio/mpeg", headers=headers)


def _base_url(request: Request) -> str:
    return (
        resolve_checkout_base_url(
            forwarded_proto=request.headers.get("x-forwarded-proto"),
            forwarded_host=request.headers.get("x-forwarded-host"),
            request_base_url=str(request.base_url).rstrip("/"),
        )
        or ""
    )


@router.post("/voice/checkout", tags=["voice"])
def voice_checkout(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Start the one Nous Plus checkout. GQ voice comes with the subscription."""
    actor = _voice_actor(request, authorization)
    try:
        return create_checkout(actor, _app_config(), _base_url(request))
    except WorkspaceError as exc:
        _raise(exc)


@router.post("/voice/portal", tags=["voice"])
def voice_portal(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Stripe's customer portal, to update the card or cancel."""
    actor = _voice_actor(request, authorization)
    try:
        return create_portal(actor, _app_config(), _base_url(request))
    except WorkspaceError as exc:
        _raise(exc)


@router.post("/voice/sync", tags=["voice"])
def voice_sync(
    body: SyncRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Confirm a finished checkout so voice works the moment the buyer is back."""
    actor = _voice_actor(request, authorization)
    try:
        return sync_checkout(actor, body.session_id, _app_config())
    except WorkspaceError as exc:
        _raise(exc)


@router.post("/voice/stripe/webhook", tags=["voice"])
async def voice_stripe_webhook(request: Request) -> dict[str, Any]:
    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    try:
        return handle_webhook(payload, signature, _app_config())
    except WorkspaceError as exc:
        _raise(exc)
