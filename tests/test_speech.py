"""Speech normalization, chunking, and the ElevenLabs request body."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.utils.config import load_config
from src.voice.config import load_voice_config
from src.voice.service import _cache_digest, concat_mp3, synthesize_speech
from src.voice.text import speech_text, split_for_speech, truncate_speech


def test_money_percent_temperature_date_and_units() -> None:
    assert speech_text("The quote is $189.42 today.") == (
        "The quote is 189 dollars and 42 cents today."
    )
    assert speech_text("$1") == "1 dollar"
    assert speech_text("Up 1.23% in the session.") == "Up 1.23 percent in the session."
    assert speech_text("High 72°F and low 4°C.") == (
        "High 72 degrees Fahrenheit and low 4 degrees Celsius."
    )
    assert speech_text("On 2024-11-10 the close was 12 km away.") == (
        "On November 10, 2024 the close was 12 kilometers away."
    )


def test_scientific_notation_becomes_a_plain_number() -> None:
    assert speech_text("About 1.23e+4 points.") == "About 12300 points."
    assert "e+" not in speech_text("1.5E-1")
    assert speech_text("1.5E-1") == "0.15"


def test_tickers_markdown_emoji_and_decorative_marks() -> None:
    raw = "**AAPL** rose — “wow”… • [note](https://example.test) 🎉"
    spoken = speech_text(raw)
    assert "Apple" in spoken
    assert "**" not in spoken
    assert "—" not in spoken
    assert "wow" in spoken
    assert "example.test" not in spoken
    assert "note" in spoken
    assert "http" not in spoken


def test_unknown_ticker_is_spelled_and_abbreviations_stay() -> None:
    assert speech_text("Watch TSLA and the API.") == "Watch Tesla and the API."
    assert speech_text("Ticker ZZZZ today.") == "Ticker Z Z Z Z today."


def test_truncate_keeps_a_whole_sentence() -> None:
    text = "First sentence is short. Second sentence should not be cut in half at all."
    cut = truncate_speech(text, 40)
    assert cut.endswith(".")
    assert "half" not in cut
    assert not cut.endswith(" ")


def test_chunks_follow_sentence_boundaries() -> None:
    sentence = "This is a complete sentence. "
    text = sentence * 40
    chunks = split_for_speech(speech_text(text))
    assert len(chunks) > 1
    assert all(len(chunk) <= 800 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)
    assert " ".join(chunks) == speech_text(text).strip()


def test_cache_key_changes_when_settings_or_model_change() -> None:
    cfg = load_voice_config({"voice": {}})
    original = _cache_digest(cfg, "Hello.")
    other = load_voice_config({"voice": {"stability": 0.2}})
    assert _cache_digest(other, "Hello.") != original
    renamed = load_voice_config({"voice": {"model_id": "eleven_turbo_v2"}})
    assert _cache_digest(renamed, "Hello.") != original


def test_config_yaml_uses_the_stable_voice_settings() -> None:
    cfg = load_voice_config(load_config("config/config.yaml"))
    assert cfg.model_id == "eleven_multilingual_v2"
    assert cfg.apply_text_normalization == "on"
    assert cfg.settings.stability == 0.7
    assert cfg.settings.similarity_boost == 0.75
    assert cfg.settings.style == 0.0
    assert cfg.settings.use_speaker_boost is True
    assert cfg.settings.speed == 1.0


def test_synthesis_sends_settings_and_joins_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    class _Response:
        def __init__(self, request_id: str, audio: bytes) -> None:
            self.headers = {"request-id": request_id}
            self._audio = audio

        def read(self) -> bytes:
            return self._audio

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def fake_open(request: Any, timeout: float) -> _Response:
        del timeout
        body = json.loads(request.data.decode())
        captured.append(body)
        tag = (
            b"ID3\x03\x00\x00\x00\x00\x00\x00"
            if len(captured) == 1
            else b"ID3\x03\x00\x00\x00\x00\x00\x00"
        )
        return _Response(f"req-{len(captured)}", tag + f"clip-{len(captured)}".encode())

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setattr("src.voice.service.urllib.request.urlopen", fake_open)
    text = "Alpha sentence here. " * 80
    audio = synthesize_speech(text, load_voice_config({"voice": {}}))
    assert captured
    first = captured[0]
    assert first["model_id"] == "eleven_multilingual_v2"
    assert first["apply_text_normalization"] == "on"
    assert first["voice_settings"] == {
        "stability": 0.7,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
        "speed": 1.0,
    }
    assert "previous_request_ids" not in first
    assert len(captured) > 1
    assert captured[1]["previous_text"]
    assert captured[1]["previous_request_ids"] == ["req-1"]
    assert audio.startswith(b"ID3")
    assert b"clip-1" in audio and b"clip-2" in audio
    assert audio.count(b"ID3") == 1


def test_concat_mp3_drops_later_id3_tags() -> None:
    first = b"ID3\x03\x00\x00\x00\x00\x00\x00one"
    second = b"ID3\x03\x00\x00\x00\x00\x00\x00two"
    assert concat_mp3([first, second]) == first + b"two"
