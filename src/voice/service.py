"""GQ voice: metered read-aloud in the owner's cloned voice.

Text on Nous is always free. Only audio playback spends characters. Each
account gets a small one-time free allowance. The paid Nous Plus subscription
includes a monthly voice allowance, so there is nothing else to buy and voice
just works once someone subscribes. Checkout may run in Stripe test mode and
will not charge a real card. Signed-out visitors get no GQ. The owner account
is unlimited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import stripe

from src.accounts.actor import Actor
from src.voice.config import (
    VoiceConfig,
    VoicePlan,
    api_key,
    load_voice_config,
    stripe_is_test_mode,
    stripe_secret,
    voice_enabled,
    webhook_secret,
)
from src.voice.store import ACTIVE_STATUSES, VoiceStore, parse_stamp, utc_day
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

_REFRESH_EVERY = timedelta(minutes=10)
_SUBSCRIPTION_EVENTS = frozenset(
    {
        "checkout.session.completed",
        "invoice.paid",
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }
)

SPEAK_USAGE_KIND = "speak"
PLAN_USAGE_KIND = "plan"

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


def _iso(stamp: Any) -> str | None:
    """A Stripe unix timestamp as an ISO UTC string."""
    if isinstance(stamp, bool) or not isinstance(stamp, int | float):
        return None
    return datetime.fromtimestamp(int(stamp), tz=timezone.utc).isoformat()


def _plain(value: Any) -> dict[str, Any]:
    """A Stripe object as a plain dict."""
    if isinstance(value, dict) and not hasattr(value, "to_dict"):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted
    return dict(value) if isinstance(value, dict) else {}


def _field(value: Any, name: str) -> Any:
    """A field of a Stripe object, a plain dict, or a stand-in with attributes."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _subscription_period(sub: dict[str, Any]) -> tuple[Any, Any]:
    """Period bounds live on the subscription or, on newer API versions, its items."""
    start, end = sub.get("current_period_start"), sub.get("current_period_end")
    if start is None or end is None:
        items = (sub.get("items") or {}).get("data") or []
        if items and isinstance(items[0], dict):
            start = items[0].get("current_period_start", start)
            end = items[0].get("current_period_end", end)
    return start, end


def _id_of(value: Any) -> str | None:
    """A Stripe reference that may be an id string or an expanded object."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        return str(value["id"])
    return None


def apply_stripe_subscription(
    store: VoiceStore,
    cfg: VoiceConfig,
    subscription: Any,
    user_id: str | None = None,
) -> str | None:
    """Store a Stripe subscription for its user. Returns the user id, or None."""
    sub = _plain(subscription)
    sub_id = _id_of(sub.get("id"))
    if not sub_id:
        return None
    metadata = sub.get("metadata") or {}
    uid = user_id or metadata.get("nf_user_id")
    if not isinstance(uid, str) or not uid:
        known = store.plan_by_subscription(sub_id)
        uid = str(known["user_id"]) if known else None
    if not uid:
        return None
    start, end = _subscription_period(sub)
    start_iso, end_iso = _iso(start), _iso(end)
    if not start_iso or not end_iso:
        return None
    store.apply_subscription(
        uid,
        customer_id=_id_of(sub.get("customer")),
        subscription_id=sub_id,
        status=str(sub.get("status") or "incomplete"),
        period_start=start_iso,
        period_end=end_iso,
        cancel_at_period_end=bool(
            sub.get("cancel_at_period_end") or sub.get("cancel_at")
        ),
        monthly_chars=cfg.plan.monthly_chars,
    )
    return uid


def _refresh_from_stripe(
    store: VoiceStore, cfg: VoiceConfig, row: dict[str, Any]
) -> None:
    """Pull the subscription once its paid period has passed without a renewal event."""
    secret = stripe_secret()
    sub_id = row.get("stripe_subscription_id")
    if secret is None or not sub_id:
        return
    end = parse_stamp(row.get("period_end"))
    updated = parse_stamp(row.get("updated_at"))
    now = datetime.now(timezone.utc)
    if end is None or end > now:
        return
    if updated is not None and now - updated < _REFRESH_EVERY:
        return
    stripe.api_key = secret
    try:
        sub = stripe.Subscription.retrieve(str(sub_id))
        apply_stripe_subscription(store, cfg, sub, str(row["user_id"]))
    except stripe.StripeError:
        logger.warning("Could not refresh a Nous Plus subscription from Stripe")
        store.touch_plan(str(row["user_id"]))


def current_plan(
    store: VoiceStore, cfg: VoiceConfig, user_id: str
) -> dict[str, Any] | None:
    """The subscriber row, renewed from Stripe when a period has quietly rolled over."""
    row = store.plan(user_id)
    if row is None:
        return None
    if row["status"] in ACTIVE_STATUSES:
        _refresh_from_stripe(store, cfg, row)
        row = store.plan(user_id)
    return row


def _plan_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "active": row["active"],
        "status": row["status"],
        "period_end": row["period_end"],
        "cancel_at_period_end": row["cancel_at_period_end"],
        "allowance_total": row["allowance_total"],
        "allowance_remaining": row["allowance_remaining"] if row["active"] else 0,
        "can_manage": bool(row.get("stripe_customer_id")),
    }


def voice_status(actor: Actor | None, config: dict[str, Any] | None) -> dict[str, Any]:
    """Balance, plan, and usage. Signed-out visitors can look, not listen."""
    cfg = load_voice_config(config)
    available = voice_enabled()
    plan = cfg.plan.as_dict()
    checkout_available = stripe_secret() is not None
    checkout_test_mode = stripe_is_test_mode()
    if actor is None:
        return {
            "signed_in": False,
            "voice": available,
            "owner_unlimited": False,
            "subscribed": False,
            "checkout_available": checkout_available,
            "checkout_test_mode": checkout_test_mode,
            "max_chars_per_request": cfg.max_chars_per_request,
            "voices": cfg.choices(),
            "plan": plan,
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
            "subscribed": False,
            "checkout_available": False,
            "checkout_test_mode": checkout_test_mode,
            "max_chars_per_request": cfg.max_chars_per_request,
            "voices": cfg.choices(),
            "plan": plan,
            "balance": None,
            "message": "The owner account is unlimited.",
            "usage": [],
        }
    row = _member_row(store, actor, cfg)
    sub = current_plan(store, cfg, actor.user_id)
    subscribed = bool(sub and sub["active"])
    plan_left = sub["allowance_remaining"] if subscribed and sub else 0
    day = utc_day()
    if subscribed:
        message = f"{cfg.plan.name} includes GQ voice. Text is always free."
    else:
        message = (
            f"Text on Nous is always free. {cfg.plan.name} adds GQ voice: "
            f"{cfg.plan.monthly_chars:,} characters every month."
        )
    return {
        "signed_in": True,
        "role": "member",
        "voice": available,
        "owner_unlimited": False,
        "subscribed": subscribed,
        "subscription": _plan_summary(sub),
        "checkout_available": checkout_available and not subscribed,
        "checkout_test_mode": checkout_test_mode,
        "max_chars_per_request": cfg.max_chars_per_request,
        "voices": cfg.choices(),
        "plan": plan,
        "balance": {
            "free": row["free_remaining"],
            "purchased": row["purchased_remaining"],
            "plan": plan_left,
            "total": row["free_remaining"] + row["purchased_remaining"] + plan_left,
            "spent": row["spent"],
        },
        "daily": {
            "used": store.daily_chars(day),
            "limit": cfg.daily_char_limit,
        },
        "usage": store.usage_history(actor.user_id, 20),
        "message": message,
    }


def _out_of_voice(cfg: VoiceConfig, sub: dict[str, Any] | None) -> str:
    if sub is not None:
        renews = str(sub.get("period_end") or "")[:10]
        when = f" It renews on {renews}." if renews else ""
        return (
            f"You have used this month's {cfg.plan.monthly_chars:,} characters of "
            f"GQ voice.{when} Text on Nous stays free."
        )
    return (
        "You are out of free GQ voice. Subscribe to "
        f"{cfg.plan.name} (${cfg.plan.price_usd:.0f}/month) and voice is included. "
        "Text on Nous stays free."
    )


def _unavailable_voice() -> WorkspaceError:
    return WorkspaceError(
        503,
        "GQ voice is temporarily unavailable. Text on the site is unaffected.",
    )


def speak(
    actor: Actor,
    text: str,
    config: dict[str, Any] | None,
    voice: str | None = None,
) -> dict[str, Any]:
    """Synthesize (or replay) audio, charging member credit only on a fresh call.

    ``voice`` picks GQ (the default) or the second voice. Both draw on the same
    character balance, and each voice has its own cached clips.
    """
    cfg = load_voice_config(config)
    chosen = cfg.voice_id_for(voice)
    if chosen is None:
        raise WorkspaceError(422, "Pick GQ or " + cfg.alt_voice_name + ".")
    cfg = replace(cfg, voice_id=chosen)
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
        subscribed = False
        if actor.is_owner:
            balance = None
        else:
            row = _member_row(store, actor, cfg)
            sub = current_plan(store, cfg, actor.user_id)
            subscribed = bool(sub and sub["active"])
            plan_left = sub["allowance_remaining"] if subscribed and sub else 0
            # Subscribers pay for their own voice, so the site-wide daily cap
            # that protects the free tier never gets in their way.
            if cfg.daily_char_limit and not subscribed:
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
            total = row["free_remaining"] + row["purchased_remaining"] + plan_left
            if total < len(cleaned):
                raise WorkspaceError(
                    402, _out_of_voice(cfg, sub if subscribed else None)
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
        # A subscriber's monthly allowance goes first, then the free allowance.
        from_plan = store.spend_plan(actor.user_id, len(cleaned)) if subscribed else 0
        remainder = len(cleaned) - from_plan
        balance = store.account(actor.user_id)
        if remainder:
            spent = store.spend(actor.user_id, remainder)
            if spent is None:  # pragma: no cover - the check above makes this a race
                raise WorkspaceError(402, _out_of_voice(cfg, None))
            balance = spent
        if not subscribed:
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
    plan: VoicePlan, actor: Actor, secret: str, base_url: str
) -> dict[str, str]:
    """Open a Stripe Checkout session for the one plan: a monthly subscription."""
    stripe.api_key = secret
    origin = base_url.rstrip("/")
    owner_meta = {"nf_user_id": actor.user_id, "nf_plan_id": plan.id}
    params: dict[str, Any] = {
        "mode": "subscription",
        "line_items": [
            {
                "quantity": 1,
                "price_data": {
                    "currency": "usd",
                    "unit_amount": int(round(plan.price_usd * 100)),
                    "recurring": {"interval": "month"},
                    "product_data": {
                        "name": plan.name,
                        "description": (
                            f"GQ voice included: {plan.monthly_chars:,} characters "
                            "every month. Text on Nous is always free."
                        ),
                    },
                },
            }
        ],
        "metadata": owner_meta,
        "subscription_data": {"metadata": owner_meta},
        "client_reference_id": actor.user_id,
        # Stripe fills {CHECKOUT_SESSION_ID}; the app uses it to turn voice on
        # the moment the buyer returns, without waiting for the webhook.
        "success_url": f"{origin}/?subscribed={{CHECKOUT_SESSION_ID}}#voice",
        "cancel_url": f"{origin}/#voice",
        # The connected Stripe account has Managed Payments on by default,
        # which rejects inline price_data unless a product tax_code is set.
        # Disable it on the session, as the one-time checkout did.
        "managed_payments": {"enabled": False},
    }
    if "@" in actor.email and actor.user_id != "owner":
        params["customer_email"] = actor.email
    try:
        session = stripe.checkout.Session.create(**params)
    except stripe.StripeError as exc:
        logger.exception("Stripe checkout session failed")
        raise WorkspaceError(502, _safe_stripe_message(exc)) from exc
    url, session_id = _field(session, "url"), _field(session, "id")
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
    actor: Actor, config: dict[str, Any] | None, base_url: str
) -> dict[str, Any]:
    """One Subscribe button, one Stripe Checkout. Nothing else is sold."""
    cfg = load_voice_config(config)
    if actor.is_owner:
        raise WorkspaceError(409, "The owner account already has unlimited voice.")
    secret = stripe_secret()
    if secret is None:
        return {
            "available": False,
            "detail": (
                f"Subscribing to {cfg.plan.name} is coming soon. "
                "Text on the site stays free."
            ),
        }
    if not base_url:
        raise WorkspaceError(503, "Checkout is missing a public site URL.")
    try:
        sub = current_plan(voice_store(), cfg, actor.user_id)
    except (SqlError, StorageError) as exc:
        raise _unavailable_voice() from exc
    if sub is not None and sub["active"]:
        raise WorkspaceError(
            409, f"You are already subscribed to {cfg.plan.name}. Voice is included."
        )
    session = create_checkout_session(cfg.plan, actor, secret, base_url)
    return {
        "available": True,
        "id": session["id"],
        "url": session["url"],
        "plan": cfg.plan.as_dict(),
    }


def create_portal(
    actor: Actor, config: dict[str, Any] | None, base_url: str
) -> dict[str, Any]:
    """Stripe's customer portal: update the card, or cancel."""
    load_voice_config(config)
    secret = stripe_secret()
    if secret is None or not base_url:
        raise WorkspaceError(503, "Subscription management is not available right now.")
    try:
        row = voice_store().plan(actor.user_id)
    except (SqlError, StorageError) as exc:
        raise _unavailable_voice() from exc
    customer = row.get("stripe_customer_id") if row else None
    if not customer:
        raise WorkspaceError(404, "There is no subscription on this account yet.")
    stripe.api_key = secret
    try:
        portal = stripe.billing_portal.Session.create(
            customer=str(customer), return_url=f"{base_url.rstrip('/')}/#voice"
        )
    except stripe.StripeError as exc:
        logger.exception("Stripe portal session failed")
        raise WorkspaceError(502, _safe_stripe_message(exc)) from exc
    url = _field(portal, "url")
    if not url:
        raise WorkspaceError(502, "Stripe did not return a portal URL.")
    return {"url": str(url)}


def sync_checkout(
    actor: Actor, session_id: str | None, config: dict[str, Any] | None
) -> dict[str, Any]:
    """Turn voice on right after checkout, without waiting for the webhook.

    Stripe is asked for the session itself, so a made-up session id gets
    nothing: it must belong to this account and be a completed subscription.
    """
    cfg = load_voice_config(config)
    secret = stripe_secret()
    store = voice_store()
    if session_id and secret is not None and not actor.is_owner:
        stripe.api_key = secret
        try:
            session = _plain(stripe.checkout.Session.retrieve(session_id))
        except stripe.StripeError as exc:
            raise WorkspaceError(404, "That checkout session was not found.") from exc
        owner = (session.get("metadata") or {}).get("nf_user_id")
        if (
            owner == actor.user_id
            and session.get("mode") == "subscription"
            and session.get("status") == "complete"
        ):
            sub_id = _id_of(session.get("subscription"))
            if sub_id:
                try:
                    sub = stripe.Subscription.retrieve(sub_id)
                except stripe.StripeError as exc:
                    raise WorkspaceError(
                        502, "Stripe could not confirm the subscription yet."
                    ) from exc
                apply_stripe_subscription(store, cfg, sub, actor.user_id)
    return voice_status(actor, config)


def _invoice_subscription_id(invoice: dict[str, Any]) -> str | None:
    """Invoices name their subscription directly or, on newer API versions, under parent."""
    direct = _id_of(invoice.get("subscription"))
    if direct:
        return direct
    parent = invoice.get("parent") or {}
    details = parent.get("subscription_details") or {}
    return _id_of(details.get("subscription"))


def handle_webhook(
    payload: bytes, signature: str | None, config: dict[str, Any] | None
) -> dict[str, Any]:
    """Keep subscriptions current. The Stripe signature must verify first."""
    secret = webhook_secret()
    if secret is None:
        raise WorkspaceError(503, "Subscriptions are not configured.")
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
    if not isinstance(event, dict):
        return {"status": "ignored"}
    kind = event.get("type")
    obj = event.get("data", {}).get("object", {})
    if not isinstance(obj, dict) or kind not in _SUBSCRIPTION_EVENTS:
        return {"status": "ignored"}
    cfg = load_voice_config(config)
    store = voice_store()
    stripe.api_key = stripe_secret() or ""
    try:
        if kind in {
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
        }:
            user_id = apply_stripe_subscription(store, cfg, obj)
        else:
            if kind == "checkout.session.completed":
                if obj.get("mode") != "subscription":
                    return {"status": "ignored"}
                sub_id = _id_of(obj.get("subscription"))
                hint = (obj.get("metadata") or {}).get("nf_user_id")
            else:  # invoice.paid: a renewal
                sub_id = _invoice_subscription_id(obj)
                hint = None
            if not sub_id:
                return {"status": "ignored"}
            sub = stripe.Subscription.retrieve(sub_id)
            user_id = apply_stripe_subscription(
                store, cfg, sub, hint if isinstance(hint, str) else None
            )
    except stripe.StripeError as exc:
        # Stripe retries on a 5xx, so a transient lookup failure heals itself.
        raise WorkspaceError(
            502, "Could not read the subscription from Stripe."
        ) from exc
    if user_id is None:
        return {"status": "ignored"}
    return {"status": "updated", "user_id": user_id}
