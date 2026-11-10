"""GQ voice metering: text stays free, audio spends credit, owners are unlimited."""

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


def _checkout_event(user_id: str, pack_id: str, session_id: str) -> bytes:
    return json.dumps(
        {
            "id": f"evt_{session_id}",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": session_id,
                    "object": "checkout.session",
                    "metadata": {"nf_user_id": user_id, "nf_pack_id": pack_id},
                }
            },
        }
    ).encode()


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
    denied = _speak(client, "Hello from Neural Forge.")
    assert denied.status_code == 401
    assert "free account" in denied.json()["detail"]


def test_owner_is_unlimited_and_bypasses_caps(
    voice_client, fake_speech, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    client.cookies.clear()
    monkeypatch.setenv("NEURAL_FORGE_VOICE_DAILY_LIMIT", "1")
    monkeypatch.setattr(voice_service, "utc_day", lambda: "2031-01-01-owner")
    owner = client.get(
        "/voice/status", headers={"Authorization": "Bearer owner-secret"}
    )
    assert owner.status_code == 200
    body = owner.json()
    assert body["owner_unlimited"] is True
    assert body["balance"] is None
    for index in range(3):
        spoken = client.post(
            "/voice/speak",
            json={"text": f"Owner briefing number {index} " + "x" * 90},
            headers={"Authorization": "Bearer owner-secret"},
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


def test_balance_enforcement_blocks_empty_accounts(voice_client, fake_speech):
    client, _root = voice_client
    _signup(client, "short@example.com")
    first = _speak(client, "e" * 80)
    assert first.status_code == 200
    before = _balance(client)
    assert before["free"] == 20
    denied = _speak(client, "f" * 40)
    assert denied.status_code == 402
    assert "credit pack" in denied.json()["detail"]
    assert fake_speech == ["e" * 80]
    assert _balance(client) == before
    too_long = _speak(client, "g" * 130)
    assert too_long.status_code == 422


def test_spend_draws_on_purchased_credit_after_free(
    voice_client, fake_speech, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "split@example.com")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    _speak(client, "h" * 60)
    assert _balance(client)["free"] == 40
    payload = _checkout_event(user_id, "starter", "cs_split_1")
    credited = client.post(
        "/voice/stripe/webhook",
        content=payload,
        headers={
            "Stripe-Signature": _stripe_signature(payload, WEBHOOK_SECRET),
        },
    )
    assert credited.status_code == 200, credited.text
    assert credited.json()["status"] == "credited"
    assert _balance(client)["purchased"] == 25000
    second = _speak(client, "i" * 60)
    assert second.status_code == 200
    balance = _balance(client)
    assert balance["free"] == 0
    assert balance["purchased"] == 24980
    assert balance["spent"] == 120


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
    response = client.post("/voice/checkout", json={"pack_id": "starter"})
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

    def fake_session(pack, actor, secret, base_url):
        captured["pack_id"] = pack.id
        captured["price"] = pack.price_usd
        captured["user_id"] = actor.user_id
        captured["secret"] = secret
        return {"id": "cs_test_1", "url": "https://checkout.stripe.com/c/pay/cs_test_1"}

    monkeypatch.setattr(voice_service, "create_checkout_session", fake_session)
    response = client.post("/voice/checkout", json={"pack_id": "standard"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is True
    assert body["url"].startswith("https://checkout.stripe.com/")
    assert captured["pack_id"] == "standard"
    assert captured["user_id"] == user_id
    assert captured["secret"] == STRIPE_SECRET
    unknown = client.post("/voice/checkout", json={"pack_id": "nope"})
    assert unknown.status_code == 404


def test_webhook_verifies_the_signature_before_crediting(
    voice_client, monkeypatch: pytest.MonkeyPatch
):
    client, _root = voice_client
    user_id = _signup(client, "paying@example.com")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    payload = _checkout_event(user_id, "starter", "cs_sign_1")

    unsigned = client.post("/voice/stripe/webhook", content=payload)
    assert unsigned.status_code == 400

    forged = client.post(
        "/voice/stripe/webhook",
        content=payload,
        headers={"Stripe-Signature": "t=1,v1=" + "0" * 64},
    )
    assert forged.status_code == 400
    assert _balance(client)["purchased"] == 0

    good = client.post(
        "/voice/stripe/webhook",
        content=payload,
        headers={"Stripe-Signature": _stripe_signature(payload, WEBHOOK_SECRET)},
    )
    assert good.status_code == 200, good.text
    assert good.json()["status"] == "credited"
    assert _balance(client)["purchased"] == 25000

    replay = client.post(
        "/voice/stripe/webhook",
        content=payload,
        headers={"Stripe-Signature": _stripe_signature(payload, WEBHOOK_SECRET)},
    )
    assert replay.status_code == 200
    assert replay.json()["status"] == "already_processed"
    assert _balance(client)["purchased"] == 25000

    other_payload = _checkout_event(user_id, "nope", "cs_sign_2")
    other_type = client.post(
        "/voice/stripe/webhook",
        content=other_payload,
        headers={"Stripe-Signature": _stripe_signature(other_payload, WEBHOOK_SECRET)},
    )
    assert other_type.status_code == 200
    assert other_type.json()["status"] == "ignored"
    assert _balance(client)["purchased"] == 25000


def test_webhook_needs_a_configured_secret(voice_client, monkeypatch):
    client, _root = voice_client
    _signup(client, "nosecret@example.com")
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    payload = _checkout_event("usr_anyone", "starter", "cs_nosecret")
    response = client.post(
        "/voice/stripe/webhook",
        content=payload,
        headers={"Stripe-Signature": _stripe_signature(payload, WEBHOOK_SECRET)},
    )
    assert response.status_code == 503


def test_member_status_reports_balance_and_packs(voice_client):
    client, _root = voice_client
    _signup(client, "status@example.com")
    body = client.get("/voice/status").json()
    assert body["signed_in"] is True
    assert body["role"] == "member"
    assert body["balance"]["free"] == 100
    pack_ids = {pack["id"] for pack in body["packs"]}
    assert pack_ids == {"starter", "standard", "studio"}
    starter = next(pack for pack in body["packs"] if pack["id"] == "starter")
    assert starter["chars"] == 25000
    assert starter["price_usd"] == 6.0
    assert body["checkout_available"] is False


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
        json={"text": "Hello from Neural Forge."},
        headers={"Authorization": "Bearer owner-secret"},
    )
    assert spoken.status_code == 503
    assert "application/json" in spoken.headers["content-type"]
    assert "temporarily unavailable" in spoken.json()["detail"]
    assert spoken.text != "Internal Server Error"
