"""GQ voice: metered read-aloud in the owner's cloned voice.

Text on Neural Forge is always free. Only audio playback spends voice
credit: each account gets a small one-time free allowance. Additional
packs are listed in the app. Checkout may run in Stripe test mode and
will not charge a real card. Signed-out visitors get no GQ. The owner
account is unlimited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import stripe

from src.accounts.actor import Actor
from src.voice.config import (
    VoiceConfig,
    VoicePack,
    api_key,
    load_voice_config,
    pack_by_id,
    stripe_is_test_mode,
    stripe_secret,
    voice_enabled,
    webhook_secret,
)
from src.voice.store import VoiceStore, utc_day
from src.voice.text import (
    NORMALIZER_VERSION,
    speech_text,
    split_for_speech,
    truncate_speech,
)
from src.workspace.bindings import StorageError
from src.workspace.errors import WorkspaceError
from src.workspace.service import require_store
from src.workspace.store import SqlError

logger = logging.getLogger("uvicorn.error")

SPEAK_USAGE_KIND = "speak"
PURCHASE_USAGE_KIND = "purchase"

_ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_OUTPUT_FORMAT = "mp3_44100_128"
_SYNTHESIS_TIMEOUT_SECONDS = 60


def voice_store() -> VoiceStore:
    store = require_store()
    return VoiceStore(store.sql, store.blobs)


def _cache_digest(cfg: VoiceConfig, text: str) -> str:
    """Key on the voice, the model, the delivery settings, and the normalized text.

    Old clips were cached from the raw string, so a settings or normalizer
    change must miss that cache instead of replaying a stuttering take.
    """
    material = "\n".join(
        (
            cfg.voice_id,
            cfg.model_id,
            NORMALIZER_VERSION,
            cfg.settings.cache_token(),
            cfg.apply_text_normalization,
            text,
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _tts_body(
    text: str,
    cfg: VoiceConfig,
    *,
    previous_text: str,
    next_text: str,
    previous_request_ids: list[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "text": text,
        "model_id": cfg.model_id,
        "voice_settings": cfg.settings.as_dict(),
        "apply_text_normalization": cfg.apply_text_normalization,
    }
    if previous_text:
        body["previous_text"] = previous_text
    if next_text:
        body["next_text"] = next_text
    if previous_request_ids:
        body["previous_request_ids"] = previous_request_ids[-3:]
    return body


def _id3v2_skip(data: bytes) -> int:
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    size = (
        ((data[6] & 0x7F) << 21)
        | ((data[7] & 0x7F) << 14)
        | ((data[8] & 0x7F) << 7)
        | (data[9] & 0x7F)
    )
    end = 10 + size
    return end if end <= len(data) else 0


def concat_mp3(parts: list[bytes]) -> bytes:
    """Join same-format MP3 clips. Later ID3 tags are dropped so playback is one stream."""
    if not parts:
        return b""
    audio = bytearray(parts[0])
    for part in parts[1:]:
        audio.extend(part[_id3v2_skip(part) :])
    return bytes(audio)


def synthesize_speech(text: str, cfg: VoiceConfig) -> bytes:
    """Call ElevenLabs text-to-speech. The API key comes only from the environment.

    Long replies are split on sentence boundaries. Each chunk is synthesized
    with the surrounding text and the previous request ids, then joined into
    one MP3 so the caller still receives a single clip.
    """
    key = api_key()
    if key is None:
        raise WorkspaceError(
            503,
            "GQ voice is not configured. Everything on the site stays free to read.",
        )
    chunks = split_for_speech(text)
    if not chunks:
        raise WorkspaceError(422, "Text to speak must not be empty.")
    url = (
        _ELEVENLABS_URL.format(voice_id=urllib.parse.quote(cfg.voice_id, safe=""))
        + f"?output_format={_OUTPUT_FORMAT}"
    )
    parts: list[bytes] = []
    request_ids: list[str] = []
    for index, chunk in enumerate(chunks):
        previous = chunks[index - 1] if index else ""
        following = chunks[index + 1] if index + 1 < len(chunks) else ""
        body = _tts_body(
            chunk,
            cfg,
            previous_text=previous[-200:],
            next_text=following[:200],
            previous_request_ids=request_ids,
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "xi-api-key": key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=_SYNTHESIS_TIMEOUT_SECONDS
            ) as hit:
                audio: bytes = hit.read()
                request_id = hit.headers.get("request-id")
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            raise WorkspaceError(
                502,
                "GQ voice is temporarily unavailable. Text on the site is unaffected.",
            ) from exc
        if not audio:
            raise WorkspaceError(
                502,
                "GQ voice returned no audio. Text on the site is unaffected.",
            )
        parts.append(audio)
        if isinstance(request_id, str) and request_id.strip():
            request_ids.append(request_id.strip())
    return concat_mp3(parts)


def _require_available(cfg: VoiceConfig) -> None:
    if not voice_enabled():
        raise WorkspaceError(
            503,
            "GQ voice is not available right now. "
            "Everything on the site stays free to read.",
        )


def _member_row(store: VoiceStore, actor: Actor, cfg: VoiceConfig) -> dict[str, int]:
    return store.ensure_account(actor.user_id, cfg.free_chars)


def voice_status(actor: Actor | None, config: dict[str, Any] | None) -> dict[str, Any]:
    """Balance, usage, and pack catalogue. Signed-out visitors can look, not listen."""
    cfg = load_voice_config(config)
    available = voice_enabled()
    packs = [pack.as_dict() for pack in cfg.packs]
    checkout_available = stripe_secret() is not None
    checkout_test_mode = stripe_is_test_mode()
    if actor is None:
        return {
            "signed_in": False,
            "voice": available,
            "owner_unlimited": False,
            "checkout_available": checkout_available,
            "checkout_test_mode": checkout_test_mode,
            "max_chars_per_request": cfg.max_chars_per_request,
            "packs": packs,
            "message": (
                "Create a free account to hear GQ. Reading the site in text stays free."
            ),
        }
    store = voice_store()
    if actor.is_owner:
        return {
            "signed_in": True,
            "role": "owner",
            "voice": available,
            "owner_unlimited": True,
            "checkout_available": False,
            "checkout_test_mode": checkout_test_mode,
            "max_chars_per_request": cfg.max_chars_per_request,
            "packs": packs,
            "balance": None,
            "message": "The owner account is unlimited.",
            "usage": [],
        }
    row = _member_row(store, actor, cfg)
    day = utc_day()
    return {
        "signed_in": True,
        "role": "member",
        "voice": available,
        "owner_unlimited": False,
        "checkout_available": checkout_available,
        "checkout_test_mode": checkout_test_mode,
        "max_chars_per_request": cfg.max_chars_per_request,
        "packs": packs,
        "balance": {
            "free": row["free_remaining"],
            "purchased": row["purchased_remaining"],
            "total": row["free_remaining"] + row["purchased_remaining"],
            "spent": row["spent"],
        },
        "daily": {
            "used": store.daily_chars(day),
            "limit": cfg.daily_char_limit,
        },
        "usage": store.usage_history(actor.user_id, 20),
        "message": "Text on Neural Forge is always free. Only GQ voice spends credit.",
    }


def _unavailable_voice() -> WorkspaceError:
    return WorkspaceError(
        503,
        "GQ voice is temporarily unavailable. Text on the site is unaffected.",
    )


def speak(actor: Actor, text: str, config: dict[str, Any] | None) -> dict[str, Any]:
    """Synthesize (or replay) audio, charging member credit only on a fresh call."""
    cfg = load_voice_config(config)
    _require_available(cfg)
    cleaned = truncate_speech(speech_text(text), cfg.max_chars_per_request)
    if not cleaned:
        raise WorkspaceError(422, "Text to speak must not be empty.")
    if len(cleaned) > cfg.max_chars_per_request:
        raise WorkspaceError(
            422,
            "Voice requests are limited to "
            f"{cfg.max_chars_per_request} characters. This one is {len(cleaned)}.",
        )
    store = voice_store()
    digest = _cache_digest(cfg, cleaned)
    try:
        cached = store.cache_get(digest)
        if cached is not None:
            # Same text, same audio: the owner pays nothing twice and neither do visitors.
            return {
                "audio": cached,
                "cached": True,
                "chars": len(cleaned),
                "balance": None if actor.is_owner else store.account(actor.user_id),
            }
        day = utc_day()
        if actor.is_owner:
            balance = None
        else:
            if cfg.daily_char_limit:
                if store.daily_chars(day) + len(cleaned) > cfg.daily_char_limit:
                    raise WorkspaceError(
                        429,
                        "The site's daily GQ voice budget is spent. Try again tomorrow. "
                        "Text stays free.",
                    )
                if len(cleaned) > cfg.daily_char_limit:
                    raise WorkspaceError(
                        422,
                        "The request is longer than the site's daily GQ voice budget.",
                    )
            row = _member_row(store, actor, cfg)
            total = row["free_remaining"] + row["purchased_remaining"]
            if total < len(cleaned):
                raise WorkspaceError(
                    402,
                    "You are out of GQ voice credit. Buy a credit pack on the "
                    "Voice & credits page. Text on Neural Forge stays free.",
                )
    except (SqlError, StorageError) as exc:
        raise _unavailable_voice() from exc
    audio = synthesize_speech(cleaned, cfg)
    try:
        store.cache_put(digest, audio)
        if actor.is_owner:
            store.record_usage(
                actor.user_id,
                SPEAK_USAGE_KIND,
                len(cleaned),
                cached=False,
                balance=None,
            )
            return {
                "audio": audio,
                "cached": False,
                "chars": len(cleaned),
                "balance": None,
            }
        balance = store.spend(actor.user_id, len(cleaned))
        if balance is None:  # pragma: no cover - the check above makes this a race
            raise WorkspaceError(
                402,
                "You are out of GQ voice credit. Buy a credit pack on the "
                "Voice & credits page. Text on Neural Forge stays free.",
            )
        store.add_daily_chars(day, len(cleaned))
        store.record_usage(
            actor.user_id, SPEAK_USAGE_KIND, len(cleaned), cached=False, balance=balance
        )
        return {
            "audio": audio,
            "cached": False,
            "chars": len(cleaned),
            "balance": balance,
        }
    except (SqlError, StorageError):
        # The audio is ready; a cache or ledger write should not drop the reply.
        return {
            "audio": audio,
            "cached": False,
            "chars": len(cleaned),
            "balance": None,
        }


def create_checkout_session(
    pack: VoicePack, actor: Actor, secret: str, base_url: str
) -> dict[str, str]:
    """Open a Stripe Checkout session. One-time payment, no subscription."""
    stripe.api_key = secret
    origin = base_url.rstrip("/")
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[
                {
                    "quantity": 1,
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": int(round(pack.price_usd * 100)),
                        "product_data": {
                            "name": f"Neural Forge GQ voice — {pack.name}",
                            "description": (
                                f"{pack.chars:,} characters read aloud in GQ. "
                                "Text on the site stays free."
                            ),
                        },
                    },
                }
            ],
            metadata={"nf_user_id": actor.user_id, "nf_pack_id": pack.id},
            client_reference_id=actor.user_id,
            success_url=f"{origin}/#voice",
            cancel_url=f"{origin}/#voice",
            # The connected Stripe account has Managed Payments on by
            # default, which rejects inline price_data unless a product
            # tax_code is set. These are one-time digital voice credits,
            # not a Managed Payments sale, so disable it on the session.
            managed_payments={"enabled": False},
        )
    except stripe.StripeError as exc:
        logger.exception("Stripe checkout session failed")
        raise WorkspaceError(502, _safe_stripe_message(exc)) from exc
    url = getattr(session, "url", None) or (
        session.get("url") if isinstance(session, dict) else None
    )
    session_id = getattr(session, "id", None) or (
        session.get("id") if isinstance(session, dict) else None
    )
    if not url or not session_id:
        raise WorkspaceError(502, "Stripe did not return a checkout URL.")
    return {"id": str(session_id), "url": str(url)}


def _safe_stripe_message(exc: stripe.StripeError) -> str:
    """Surface Stripe's public error text without leaking keys or payloads."""
    raw = ""
    user = getattr(exc, "user_message", None)
    if isinstance(user, str) and user.strip():
        raw = user.strip()
    else:
        body = getattr(exc, "json_body", None) or {}
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            raw = error["message"].strip()
        elif exc.args:
            raw = str(exc.args[0]).strip()
    raw = re.sub(r"(sk|rk|pk|whsec)_([a-zA-Z]+_)?[A-Za-z0-9]+", r"\1_[REDACTED]", raw)
    if not raw:
        return "Stripe could not start checkout. Try again in a moment."
    return f"Stripe could not start checkout: {raw}"


def create_checkout(
    actor: Actor, pack_id: str, config: dict[str, Any] | None, base_url: str
) -> dict[str, Any]:
    cfg = load_voice_config(config)
    pack = pack_by_id(cfg, pack_id)
    if pack is None:
        raise WorkspaceError(404, "Unknown voice credit pack.")
    secret = stripe_secret()
    if secret is None:
        return {
            "available": False,
            "detail": (
                "Buying GQ voice credit is coming soon. Text on the site stays free."
            ),
        }
    if not base_url:
        raise WorkspaceError(
            503,
            "Voice checkout is missing a public site URL.",
        )
    session = create_checkout_session(pack, actor, secret, base_url)
    return {
        "available": True,
        "id": session["id"],
        "url": session["url"],
        "pack": pack.as_dict(),
    }


def handle_webhook(
    payload: bytes, signature: str | None, config: dict[str, Any] | None
) -> dict[str, Any]:
    """Credit a paid pack. The Stripe signature must verify before anything moves."""
    secret = webhook_secret()
    if secret is None:
        raise WorkspaceError(503, "Voice purchases are not configured.")
    if not signature:
        raise WorkspaceError(400, "Missing Stripe signature header.")
    try:
        stripe.Webhook.construct_event(payload, signature, secret)
    except stripe.SignatureVerificationError as exc:
        raise WorkspaceError(400, "Invalid Stripe signature.") from exc
    except ValueError as exc:
        raise WorkspaceError(400, "Invalid webhook payload.") from exc
    try:
        event = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceError(400, "Invalid webhook payload.") from exc
    if not isinstance(event, dict) or event.get("type") != "checkout.session.completed":
        return {"status": "ignored"}
    session = event.get("data", {}).get("object", {})
    metadata = session.get("metadata") or {}
    user_id = metadata.get("nf_user_id")
    pack_id = metadata.get("nf_pack_id")
    session_id = session.get("id")
    if not isinstance(user_id, str) or not user_id:
        return {"status": "ignored"}
    if not isinstance(pack_id, str) or not isinstance(session_id, str):
        return {"status": "ignored"}
    cfg = load_voice_config(config)
    pack = pack_by_id(cfg, pack_id)
    if pack is None:
        return {"status": "ignored"}
    store = voice_store()
    store.ensure_account(user_id, 0)
    if not store.credit_purchase(user_id, pack.id, pack.chars, session_id):
        return {"status": "already_processed", "session_id": session_id}
    store.record_usage(
        user_id,
        PURCHASE_USAGE_KIND,
        pack.chars,
        cached=False,
        balance=store.account(user_id),
    )
    return {
        "status": "credited",
        "chars": pack.chars,
        "user_id": user_id,
        "pack_id": pack.id,
    }
