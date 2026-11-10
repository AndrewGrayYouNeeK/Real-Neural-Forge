"""HTTP routes for GQ voice status, playback, checkout, and the Stripe webhook."""

from __future__ import annotations

from typing import Any, NoReturn

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.accounts.actor import Actor
from src.voice.config import resolve_checkout_base_url
from src.voice.service import create_checkout, handle_webhook, speak, voice_status
from src.workspace.errors import WorkspaceError
from src.workspace.runtime import runtime
from src.workspace.service import resolve_actor

router = APIRouter()

_SIGN_UP_PROMPT = "Create a free account to hear GQ. Text on Neural Forge stays free."


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)


class CheckoutRequest(BaseModel):
    pack_id: str


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


@router.post("/voice/checkout", tags=["voice"])
def voice_checkout(
    body: CheckoutRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _voice_actor(request, authorization)
    base = resolve_checkout_base_url(
        forwarded_proto=request.headers.get("x-forwarded-proto"),
        forwarded_host=request.headers.get("x-forwarded-host"),
        request_base_url=str(request.base_url).rstrip("/"),
    )
    try:
        return create_checkout(actor, body.pack_id, _app_config(), base or "")
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
