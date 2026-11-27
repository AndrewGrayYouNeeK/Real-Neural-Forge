"""GQ voice: text stays free, audio spends characters, the Nous Plus subscription
includes a monthly allowance, owners are unlimited."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

import src.voice.service as voice_service
from src.api import app, load_model
from src.voice.store import SqlError, VoiceStore, audio_cache_key
from src.workspace.runtime import runtime
from tests.owner_session import owner_headers

WEBHOOK_SECRET = "whsec_test_voice"
STRIPE_SECRET = "sk_test_voice"


@pytest.fixture(scope="module")
def voice_client(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("voice")
    updates = {
        "NEURAL_FORGE_API_TOKEN": "owner-secret",
        "NEURAL_FORGE_WORKSPACE_DIR": str(root),
        "NEURAL_FORGE_QUEUE_RUNNER": "0",
        "NEURAL_FORGE_STORAGE": "local",
        "ELEVENLABS_API_KEY": "sk-test-elevenlabs",
        "NEURAL_FORGE_VOICE_FREE_CHARS": "100",
        "NEURAL_FORGE_VOICE_MAX_CHARS": "120",
        "NEURAL_FORGE_VOICE_DAILY_LIMIT": "1000000",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    load_model("config/config.yaml")
    try:
        with TestClient(app) as client:
            yield client, root
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def fake_speech(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake(text: str, cfg: object) -> bytes:
        del cfg
        calls.append(text)
        return b"GQ-AUDIO:" + text.encode("utf-8")

    monkeypatch.setattr(voice_service, "synthesize_speech", fake)
    return calls


def _signup(client: TestClient, email: str) -> str:
    client.cookies.clear()
    response = client.post(
        "/auth/signup",
        json={"email": email, "password": "password-a-ok"},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _login(client: TestClient, email: str) -> None:
    client.cookies.clear()
    response = client.post(
        "/auth/login",
        json={"email": email, "password": "password-a-ok"},
    )
    assert response.status_code == 200, response.text


def _speak(client: TestClient, text: str):
    return client.post(
        "/voice/speak",
        json={"text": text},
    )


def _stripe_signature(payload: bytes, secret: str) -> str:
    timestamp = str(int(time.time()))
    signed = f"{timestamp}.".encode() + payload
    tag = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={tag}"


def _unix(days: int) -> int:
    return int(time.time()) + days * 24 * 60 * 60


def _subscription(
    user_id: str,
    sub_id: str = "sub_test_1",
    *,
    status: str = "active",
    start_days: int = 0,
    end_days: int = 30,
    cancel_at_period_end: bool = False,
) -> dict:
    return {
        "id": sub_id,
        "object": "subscription",
        "customer": "cus_test_1",
        "status": status,
        "current_period_start": _unix(start_days),
        "current_period_end": _unix(end_days),
        "cancel_at_period_end": cancel_at_period_end,
        "metadata": {"nf_user_id": user_id, "nf_plan_id": "plus"},
    }


def _event(kind: str, obj: dict) -> bytes:
    return json.dumps(
        {"id": f"evt_{kind}_{obj['id']}", "type": kind, "data": {"object": obj}}
    ).encode()


def _post_event(client: TestClient, payload: bytes):
    return client.post(
        "/voice/stripe/webhook",
        content=payload,
        headers={"Stripe-Signature": _stripe_signature(payload, WEBHOOK_SECRET)},
    )


def _subscribe(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, user_id: str, sub_id: str
) -> None:
    """Deliver the subscription event Stripe sends after a paid checkout."""
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    response = _post_event(
        client,
        _event("customer.subscription.created", _subscription(user_id, sub_id)),
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "updated"


def _balance(client: TestClient) -> dict:
    response = client.get("/voice/status")
    assert response.status_code == 200, response.text
    return response.json()["balance"]


def test_signed_out_visitors_get_no_gq(voice_client):
    client, _root = voice_client
    client.cookies.clear()
    status = client.get("/voice/status")
    assert status.status_code == 200
    body = status.json()
    assert body["signed_in"] is False
    assert body["voice"] is True
    assert "free account" in body["message"]
    denied = _speak(client, "Hello from Nous.")
    assert denied.status_code == 401
    assert "free account" in denied.json()["detail"]


def test_owner_is_unlimited_and_bypasses_caps(
    voice_client, fake_speech, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    client.cookies.clear()
    monkeypatch.setenv("NEURAL_FORGE_VOICE_DAILY_LIMIT", "1")
    monkeypatch.setattr(voice_service, "utc_day", lambda: "2031-01-01-owner")
    owner = client.get("/voice/status", headers=owner_headers())
    assert owner.status_code == 200
    body = owner.json()
    assert body["owner_unlimited"] is True
    assert body["balance"] is None
    for index in range(3):
        spoken = client.post(
            "/voice/speak",
            json={"text": f"Owner briefing number {index} " + "x" * 90},
            headers=owner_headers(),
        )
        assert spoken.status_code == 200, spoken.text
    assert len(fake_speech) == 3


def test_metering_deducts_free_credit(voice_client, fake_speech):
    client, _root = voice_client
    _signup(client, "metered@example.com")
    spoken = _speak(client, "a" * 40)
    assert spoken.status_code == 200, spoken.text
    assert spoken.headers["x-neural-forge-voice-chars"] == "40"
    balance = _balance(client)
    assert balance["free"] == 60
    assert balance["spent"] == 40
    again = _speak(client, "b" * 40)
    assert again.status_code == 200
    balance = _balance(client)
    assert balance["free"] == 20
    assert balance["spent"] == 80


def test_cache_prevents_a_second_spend(voice_client, fake_speech):
    client, _root = voice_client
    _signup(client, "cached@example.com")
    text = "The forecast holds steady today."
    first = _speak(client, text)
    assert first.status_code == 200
    assert first.headers["x-neural-forge-voice-cache"] == "miss"
    assert len(fake_speech) == 1
    second = _speak(client, text)
    assert second.status_code == 200
    assert second.headers["x-neural-forge-voice-cache"] == "hit"
    assert len(fake_speech) == 1
    balance = _balance(client)
    assert balance["spent"] == len(text)
    assert balance["free"] == 100 - len(text)


def test_the_second_voice_shares_the_quota_and_gq_stays_the_default(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    used: list[str] = []

    def fake(text: str, cfg) -> bytes:
        used.append(cfg.voice_id)
        return b"AUDIO:" + cfg.voice_id.encode() + text.encode()

    monkeypatch.setattr(voice_service, "synthesize_speech", fake)
    _signup(client, "twovoices@example.com")
    status = client.get("/voice/status").json()
    assert status["voices"] == [
        {"key": "gq", "name": "GQ"},
        {"key": "female", "name": "Ivy"},
    ]
    text = "Hello from the other side."
    default = client.post("/voice/speak", json={"text": text})
    gq = client.post("/voice/speak", json={"text": text, "voice": "gq"})
    assert default.status_code == gq.status_code == 200
    assert gq.headers["x-neural-forge-voice-cache"] == "hit"
    ivy = client.post("/voice/speak", json={"text": text, "voice": "female"})
    assert ivy.status_code == 200
    assert ivy.headers["x-neural-forge-voice-cache"] == "miss"
    assert used == ["QMnwdN2oOwNXPcl9oXIl", "EXAVITQu4vr4xnSDxMaL"]
    assert ivy.content != default.content
    assert _balance(client)["spent"] == 2 * len(text)
    bad = client.post("/voice/speak", json={"text": text, "voice": "nope"})
    assert bad.status_code == 422


def test_balance_enforcement_blocks_empty_accounts(voice_client, fake_speech):
    client, _root = voice_client
    _signup(client, "short@example.com")
    first = _speak(client, "e" * 80)
    assert first.status_code == 200
    before = _balance(client)
    assert before["free"] == 20
    denied = _speak(client, "f" * 40)
    assert denied.status_code == 402
    assert "Subscribe" in denied.json()["detail"]
    assert "pack" not in denied.json()["detail"].lower()
    assert fake_speech == ["e" * 80]
    assert _balance(client) == before
    too_long = _speak(client, "g" * 130)
    # Longer than the 120 character cap, so it is cut on a word boundary first.
    # 120 characters still costs more than the 20 characters left.
    assert too_long.status_code == 402


def test_subscription_includes_voice_with_a_monthly_allowance(
    voice_client, fake_speech, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "split@example.com")
    monkeypatch.setenv("NEURAL_FORGE_VOICE_PLAN_CHARS", "200")
    _speak(client, "h" * 60)
    assert _balance(client)["free"] == 40
    before = client.get("/voice/status").json()
    assert before["subscribed"] is False
    _subscribe(client, monkeypatch, user_id, "sub_split")
    status = client.get("/voice/status").json()
    assert status["subscribed"] is True
    assert status["subscription"]["allowance_total"] == 200
    assert status["subscription"]["allowance_remaining"] == 200
    assert status["balance"]["plan"] == 200
    assert status["balance"]["total"] == 240
    # The monthly allowance is spent first, then the free allowance.
    second = _speak(client, "i" * 60)
    assert second.status_code == 200
    balance = _balance(client)
    assert balance["plan"] == 140
    assert balance["free"] == 40
    assert balance["spent"] == 60
    third = _speak(client, "j" * 100 + " " + "k" * 19)
    assert third.status_code == 200, third.text
    balance = _balance(client)
    assert balance["plan"] == 20
    assert balance["free"] == 40
    # The last 20 plan characters plus 40 free ones cover 60 more.
    fourth = _speak(client, "l" * 60)
    assert fourth.status_code == 200, fourth.text
    balance = _balance(client)
    assert balance["plan"] == 0
    assert balance["free"] == 0
    exhausted = _speak(client, "m" * 30)
    assert exhausted.status_code == 402
    detail = exhausted.json()["detail"]
    assert "this month" in detail
    assert "renews" in detail


def test_each_period_grants_the_allowance_exactly_once(
    voice_client, fake_speech, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    monkeypatch.setenv("NEURAL_FORGE_VOICE_PLAN_CHARS", "100")
    monkeypatch.setenv("NEURAL_FORGE_VOICE_FREE_CHARS", "0")
    user_id = _signup(client, "renews@example.com")
    first = _subscription(user_id, "sub_renew", start_days=0, end_days=30)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    for kind in ("customer.subscription.created", "customer.subscription.updated"):
        assert _post_event(client, _event(kind, first)).status_code == 200
    assert _balance(client)["plan"] == 100
    assert _speak(client, "n" * 70).status_code == 200
    assert _balance(client)["plan"] == 30
    # The same period delivered again does not top the allowance back up.
    assert _post_event(client, _event("customer.subscription.updated", first))
    assert _balance(client)["plan"] == 30
    # A renewal is a new period, so the allowance resets (no rollover).
    renewed = _subscription(user_id, "sub_renew", start_days=30, end_days=60)
    assert _post_event(client, _event("customer.subscription.updated", renewed))
    assert _balance(client)["plan"] == 100


def test_cancelled_subscription_stops_voice_but_text_stays_free(
    voice_client, fake_speech, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    monkeypatch.setenv("NEURAL_FORGE_VOICE_FREE_CHARS", "0")
    user_id = _signup(client, "cancels@example.com")
    _subscribe(client, monkeypatch, user_id, "sub_cancel")
    pending = _subscription(user_id, "sub_cancel", cancel_at_period_end=True)
    assert _post_event(client, _event("customer.subscription.updated", pending))
    status = client.get("/voice/status").json()
    assert status["subscribed"] is True
    assert status["subscription"]["cancel_at_period_end"] is True
    ended = _subscription(user_id, "sub_cancel", status="canceled")
    assert _post_event(client, _event("customer.subscription.deleted", ended))
    status = client.get("/voice/status").json()
    assert status["subscribed"] is False
    assert status["balance"]["plan"] == 0
    assert _speak(client, "Voice is over.").status_code == 402
    assert client.get("/series/recent", params={"points": 16}).status_code == 200


def test_subscribers_are_not_held_back_by_the_site_daily_cap(
    voice_client, fake_speech, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "paidcap@example.com")
    _subscribe(client, monkeypatch, user_id, "sub_cap")
    monkeypatch.setenv("NEURAL_FORGE_VOICE_DAILY_LIMIT", "10")
    monkeypatch.setattr(voice_service, "utc_day", lambda: "2031-01-03-paid")
    assert _speak(client, "p" * 40).status_code == 200
    _signup(client, "freecap@example.com")
    assert _speak(client, "q" * 40).status_code == 429


def test_an_expired_period_without_a_renewal_stops_voice(
    voice_client, fake_speech, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.setenv("NEURAL_FORGE_VOICE_FREE_CHARS", "0")
    user_id = _signup(client, "lapsed@example.com")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    stale = _subscription(user_id, "sub_lapsed", start_days=-40, end_days=-10)
    assert _post_event(client, _event("customer.subscription.updated", stale))
    assert client.get("/voice/status").json()["subscribed"] is False
    assert _speak(client, "Lapsed.").status_code == 402


def test_daily_cap_stops_the_site_bill(voice_client, fake_speech, monkeypatch):
    client, _root = voice_client
    _signup(client, "capped@example.com")
    monkeypatch.setenv("NEURAL_FORGE_VOICE_DAILY_LIMIT", "50")
    monkeypatch.setattr(voice_service, "utc_day", lambda: "2031-01-02-cap")
    first = _speak(client, "j" * 40)
    assert first.status_code == 200
    blocked = _speak(client, "k" * 40)
    assert blocked.status_code == 429
    assert "daily" in blocked.json()["detail"]
    assert fake_speech == ["j" * 40]
    assert _balance(client)["free"] == 60


def test_text_is_free_with_zero_voice_credit(voice_client, monkeypatch):
    client, _root = voice_client
    monkeypatch.setenv("NEURAL_FORGE_VOICE_FREE_CHARS", "0")
    _signup(client, "nocredit@example.com")
    assert _balance(client)["total"] == 0
    predicted = client.post(
        "/predict",
        json={"sequence": [[0.1], [0.2], [0.3]]},
    )
    assert predicted.status_code == 200, predicted.text
    recent = client.get("/series/recent", params={"points": 16})
    assert recent.status_code == 200
    denied = _speak(client, "No credit left at all.")
    assert denied.status_code == 402


def test_checkout_degrades_gracefully_without_stripe(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    _signup(client, "soon@example.com")
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    response = client.post("/voice/checkout")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is False
    assert "coming soon" in body["detail"]
    assert client.get("/health").status_code == 200


def test_checkout_opens_a_stripe_session(voice_client, monkeypatch: pytest.MonkeyPatch):
    client, _root = voice_client
    user_id = _signup(client, "buying@example.com")
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)
    captured: dict = {}

    def fake_session(plan, actor, secret, base_url):
        captured["plan_id"] = plan.id
        captured["price"] = plan.price_usd
        captured["monthly_chars"] = plan.monthly_chars
        captured["user_id"] = actor.user_id
        captured["secret"] = secret
        captured["base_url"] = base_url
        return {"id": "cs_test_1", "url": "https://checkout.stripe.com/c/pay/cs_test_1"}

    monkeypatch.setattr(voice_service, "create_checkout_session", fake_session)
    response = client.post("/voice/checkout")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is True
    assert body["url"].startswith("https://checkout.stripe.com/")
    assert captured["plan_id"] == "plus"
    assert captured["price"] == 9.0
    assert captured["monthly_chars"] == 40000
    assert captured["user_id"] == user_id
    assert captured["secret"] == STRIPE_SECRET
    assert captured["base_url"].startswith("https://")
    # There is nothing else to buy: packs are gone from the API.
    legacy = client.post("/voice/checkout", json={"pack_id": "starter"})
    assert legacy.status_code == 200
    assert "packs" not in client.get("/voice/status").json()


def test_checkout_uses_forwarded_https_host(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    _signup(client, "forwarded@example.com")
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)
    monkeypatch.delenv("NEURAL_FORGE_PUBLIC_URL", raising=False)
    captured: dict = {}

    def fake_session(plan, actor, secret, base_url):
        del plan, actor, secret
        captured["base_url"] = base_url
        return {
            "id": "cs_test_fwd",
            "url": "https://checkout.stripe.com/c/pay/cs_test_fwd",
        }

    monkeypatch.setattr(voice_service, "create_checkout_session", fake_session)
    response = client.post(
        "/voice/checkout",
        headers={
            "X-Forwarded-Proto": "http",
            "X-Forwarded-Host": "forge.youneek.dev",
        },
    )
    assert response.status_code == 200, response.text
    assert captured["base_url"] == "https://forge.youneek.dev"


def test_checkout_prefers_configured_public_url_when_forwarded_is_internal(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    _signup(client, "public-url@example.com")
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)
    monkeypatch.setenv("NEURAL_FORGE_PUBLIC_URL", "https://forge.youneek.dev")
    captured: dict = {}

    def fake_session(plan, actor, secret, base_url):
        del plan, actor, secret
        captured["base_url"] = base_url
        return {
            "id": "cs_test_env",
            "url": "https://checkout.stripe.com/c/pay/cs_test_env",
        }

    monkeypatch.setattr(voice_service, "create_checkout_session", fake_session)
    response = client.post(
        "/voice/checkout",
        headers={
            "X-Forwarded-Proto": "http",
            "X-Forwarded-Host": "0.0.0.0:8000",
        },
    )
    assert response.status_code == 200, response.text
    assert captured["base_url"] == "https://forge.youneek.dev"


def test_checkout_maps_stripe_errors_to_json(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    _signup(client, "stripe-error@example.com")
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)
    monkeypatch.setenv("NEURAL_FORGE_PUBLIC_URL", "https://forge.youneek.dev")

    def failing_create(**kwargs):
        del kwargs
        raise voice_service.stripe.InvalidRequestError(
            message="Not a valid URL",
            param="success_url",
        )

    monkeypatch.setattr(voice_service.stripe.checkout.Session, "create", failing_create)
    response = client.post("/voice/checkout")
    assert response.status_code == 502, response.text
    detail = response.json()["detail"]
    assert "Not a valid URL" in str(detail)
    assert "Internal Server Error" not in response.text
    assert "sk_" not in response.text


def test_webhook_verifies_the_signature_before_subscribing(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "paying@example.com")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    payload = _event("customer.subscription.created", _subscription(user_id, "sub_sig"))

    unsigned = client.post("/voice/stripe/webhook", content=payload)
    assert unsigned.status_code == 400

    forged = client.post(
        "/voice/stripe/webhook",
        content=payload,
        headers={"Stripe-Signature": "t=1,v1=" + "0" * 64},
    )
    assert forged.status_code == 400
    assert client.get("/voice/status").json()["subscribed"] is False

    good = _post_event(client, payload)
    assert good.status_code == 200, good.text
    assert good.json()["status"] == "updated"
    assert client.get("/voice/status").json()["subscribed"] is True

    other = _event("charge.refunded", {"id": "ch_1"})
    ignored = _post_event(client, other)
    assert ignored.status_code == 200
    assert ignored.json()["status"] == "ignored"


def test_checkout_completed_and_invoice_paid_read_the_subscription_from_stripe(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "stripe-read@example.com")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)
    retrieved: list[str] = []

    def fake_retrieve(sub_id: str, **kwargs):
        del kwargs
        retrieved.append(sub_id)
        return _subscription(user_id, sub_id)

    monkeypatch.setattr(voice_service.stripe.Subscription, "retrieve", fake_retrieve)
    session = {
        "id": "cs_sub_1",
        "object": "checkout.session",
        "mode": "subscription",
        "subscription": "sub_from_checkout",
        "metadata": {"nf_user_id": user_id},
    }
    response = _post_event(client, _event("checkout.session.completed", session))
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "updated"
    assert retrieved == ["sub_from_checkout"]
    assert client.get("/voice/status").json()["subscribed"] is True
    # A one-time (old pack) checkout session credits nothing now.
    pack_session = {
        "id": "cs_old",
        "mode": "payment",
        "metadata": {"nf_user_id": user_id},
    }
    ignored = _post_event(client, _event("checkout.session.completed", pack_session))
    assert ignored.json()["status"] == "ignored"
    invoice = {
        "id": "in_1",
        "object": "invoice",
        "parent": {"subscription_details": {"subscription": "sub_from_checkout"}},
    }
    paid = _post_event(client, _event("invoice.paid", invoice))
    assert paid.json()["status"] == "updated"
    assert retrieved[-1] == "sub_from_checkout"


def test_sync_turns_voice_on_for_the_buyers_own_completed_session(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "sync@example.com")
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)

    def make_session(owner: str, **extra):
        data = {
            "id": "cs_sync_1",
            "mode": "subscription",
            "status": "complete",
            "subscription": "sub_sync",
            "metadata": {"nf_user_id": owner},
        }
        data.update(extra)
        return data

    sessions = {"cs_sync_1": make_session(user_id)}

    def fake_session(session_id: str, **kwargs):
        del kwargs
        return sessions[session_id]

    monkeypatch.setattr(voice_service.stripe.checkout.Session, "retrieve", fake_session)
    monkeypatch.setattr(
        voice_service.stripe.Subscription,
        "retrieve",
        lambda sub_id, **kwargs: _subscription(user_id, sub_id),
    )
    sessions["cs_other"] = make_session("someone_else", id="cs_other")
    stolen = client.post("/voice/sync", json={"session_id": "cs_other"})
    assert stolen.status_code == 200
    assert stolen.json()["subscribed"] is False
    sessions["cs_open"] = make_session(user_id, id="cs_open", status="open")
    unfinished = client.post("/voice/sync", json={"session_id": "cs_open"})
    assert unfinished.json()["subscribed"] is False
    done = client.post("/voice/sync", json={"session_id": "cs_sync_1"})
    assert done.status_code == 200, done.text
    assert done.json()["subscribed"] is True
    assert done.json()["balance"]["plan"] == 40000
    again = client.post("/voice/sync", json={"session_id": "cs_sync_1"})
    assert again.json()["balance"]["plan"] == 40000


def test_subscribed_users_cannot_open_a_second_checkout(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "twice@example.com")
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)
    _subscribe(client, monkeypatch, user_id, "sub_twice")
    response = client.post("/voice/checkout")
    assert response.status_code == 409
    assert "already subscribed" in response.json()["detail"]


def test_portal_needs_a_subscription_and_opens_stripes_portal(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "portal@example.com")
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)
    monkeypatch.setenv("NEURAL_FORGE_PUBLIC_URL", "https://nous.youneek.dev")
    assert client.post("/voice/portal").status_code == 404
    _subscribe(client, monkeypatch, user_id, "sub_portal")
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return {"url": "https://billing.stripe.com/p/session/test"}

    monkeypatch.setattr(
        voice_service.stripe.billing_portal.Session, "create", fake_create
    )
    response = client.post("/voice/portal")
    assert response.status_code == 200, response.text
    assert response.json()["url"].startswith("https://billing.stripe.com/")
    assert captured["customer"] == "cus_test_1"
    assert captured["return_url"].startswith("https://")


def test_webhook_needs_a_configured_secret(voice_client, monkeypatch):
    client, _root = voice_client
    _signup(client, "nosecret@example.com")
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    payload = _event(
        "customer.subscription.created", _subscription("usr_anyone", "sub_nosecret")
    )
    response = client.post(
        "/voice/stripe/webhook",
        content=payload,
        headers={"Stripe-Signature": _stripe_signature(payload, WEBHOOK_SECRET)},
    )
    assert response.status_code == 503


def test_member_status_reports_balance_and_the_plan(voice_client, monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    client, _root = voice_client
    _signup(client, "status@example.com")
    body = client.get("/voice/status").json()
    assert body["signed_in"] is True
    assert body["role"] == "member"
    assert body["balance"]["free"] == 100
    assert "packs" not in body
    assert body["plan"]["name"] == "Nous Plus"
    assert body["plan"]["price_usd"] == 9.0
    assert body["plan"]["monthly_chars"] == 40000
    assert body["plan"]["interval"] == "month"
    assert body["subscribed"] is False
    assert body["checkout_available"] is False
    assert body["checkout_test_mode"] is False


def test_checkout_test_mode_is_reported_without_exposing_the_secret(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    client.cookies.clear()
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_not_a_real_secret")
    body = client.get("/voice/status").json()
    assert body["checkout_available"] is True
    assert body["checkout_test_mode"] is True
    dumped = str(body)
    assert "sk_test_not_a_real_secret" not in dumped
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_not_a_real_secret")
    live = client.get("/voice/status").json()
    assert live["checkout_available"] is True
    assert live["checkout_test_mode"] is False
    assert "sk_live_not_a_real_secret" not in str(live)


def test_voice_store_cache_keys_are_safe(voice_client):
    _client, _root = voice_client
    store = runtime.store
    assert store is not None
    voice = VoiceStore(store.sql, store.blobs)
    digest = hashlib.sha256(b"anything").hexdigest()
    assert audio_cache_key(digest).startswith("voice/cache/")
    voice.cache_put(digest, b"audio-bytes")
    assert voice.cache_get(digest) == b"audio-bytes"
    assert voice.cache_get("0" * 64) is None
    with pytest.raises(SqlError):
        voice.cache_get("../escape")


def test_cache_store_failure_is_json_503(voice_client, monkeypatch: pytest.MonkeyPatch):
    client, _root = voice_client
    from src.workspace.bindings import StorageError

    def boom(self, digest):
        raise StorageError("R2 get failed (400): Invalid object key.")

    monkeypatch.setattr(VoiceStore, "cache_get", boom)
    spoken = client.post(
        "/voice/speak",
        json={"text": "Hello from Nous."},
        headers=owner_headers(),
    )
    assert spoken.status_code == 503
    assert "application/json" in spoken.headers["content-type"]
    assert "temporarily unavailable" in spoken.json()["detail"]
    assert spoken.text != "Internal Server Error"


def test_resolve_checkout_base_url_upgrades_http_and_rejects_internal(
    monkeypatch: pytest.MonkeyPatch,
):
    from src.voice.config import normalize_public_url, resolve_checkout_base_url

    monkeypatch.delenv("NEURAL_FORGE_PUBLIC_URL", raising=False)
    assert (
        normalize_public_url("http://forge.youneek.dev") == "https://forge.youneek.dev"
    )
    assert normalize_public_url("http://localhost:8000") == "http://localhost:8000"
    assert normalize_public_url("http://0.0.0.0:8000") is None
    assert normalize_public_url("http://bindings.neuralforge") is None
    assert (
        resolve_checkout_base_url(
            forwarded_proto="https",
            forwarded_host="forge.youneek.dev",
            request_base_url="http://0.0.0.0:8000",
        )
        == "https://forge.youneek.dev"
    )
    monkeypatch.setenv("NEURAL_FORGE_PUBLIC_URL", "https://forge.youneek.dev")
    assert (
        resolve_checkout_base_url(
            forwarded_proto="http",
            forwarded_host="0.0.0.0:8000",
            request_base_url="http://0.0.0.0:8000",
        )
        == "https://forge.youneek.dev"
    )


def test_checkout_session_disables_managed_payments(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    _signup(client, "managed-payments@example.com")
    monkeypatch.setenv("STRIPE_SECRET_KEY", STRIPE_SECRET)
    monkeypatch.setenv("NEURAL_FORGE_PUBLIC_URL", "https://forge.youneek.dev")
    captured: dict = {}

    class FakeSession:
        id = "cs_test_managed"
        url = "https://checkout.stripe.com/c/pay/cs_test_managed"

        @classmethod
        def create(cls, **kwargs):
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(
        voice_service.stripe.checkout.Session, "create", FakeSession.create
    )
    response = client.post("/voice/checkout")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["url"].startswith("https://checkout.stripe.com/")
    assert captured["mode"] == "subscription"
    item = captured["line_items"][0]
    assert item["price_data"]["unit_amount"] == 900
    assert item["price_data"]["recurring"] == {"interval": "month"}
    assert captured["managed_payments"] == {"enabled": False}
    assert captured["customer_email"] == "managed-payments@example.com"
    assert captured["subscription_data"]["metadata"]["nf_user_id"]
    assert captured["success_url"].startswith("https://forge.youneek.dev/")
    assert "{CHECKOUT_SESSION_ID}" in captured["success_url"]
