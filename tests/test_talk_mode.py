"""Talk mode frontend contracts — keep the listen/answer/speak loop reliable."""

from __future__ import annotations

from src.utils.paths import REPO_ROOT

APP_JS = REPO_ROOT / "frontend" / "app.js"


def _talk_section(source: str) -> str:
    start = source.index("/* --------------------------------------------------------------- talk mode */")
    end = source.index("/* ------------------------------------------------------------------ voice */", start)
    return source[start:end]


def test_talk_mode_keeps_playback_from_hanging() -> None:
    section = _talk_section(APP_JS.read_text(encoding="utf-8"))
    assert "function waitForTalkPlayback" in section
    assert 'addEventListener("ended", onEnded)' in section
    assert 'addEventListener("error", onError)' in section
    assert "TALK_PLAYBACK_PAD_MS" in section
    assert "TALK_PLAYBACK_MAX_MS" in section
    assert 'finish("timeout")' in section
    assert 'addEventListener("pause"' not in section
    assert "clearTalkObjectUrl" in section


def test_talk_mode_unlocks_audio_during_the_talk_tap() -> None:
    section = _talk_section(APP_JS.read_text(encoding="utf-8"))
    assert "function unlockTalkAudio" in section
    assert "function ensureTalkPlayer" in section
    assert "TALK_SILENT_WAV" in section
    assert "data:audio/wav;base64," in section
    assert "unlockTalkAudio();" in section
    assert "ensureTalkPlayer()" in section
    assert "TALK_PLAYBACK_BLOCKED" in section
    assert "NotAllowedError" in section


def test_talk_mode_starts_recognition_inside_the_gesture() -> None:
    section = _talk_section(APP_JS.read_text(encoding="utf-8"))
    assert "function abortTalkRecognition" in section
    assert "abortTalkRecognition();" in section
    assert "startListening({ immediate: true })" in section
    assert "options.immediate" in section
    assert "TALK_RESTART_MS" in section
    assert "TALK_AFTER_SPEECH_MS = 550" in section


def test_talk_mode_surfaces_auth_and_voice_failures() -> None:
    section = _talk_section(APP_JS.read_text(encoding="utf-8"))
    assert "TALK_SIGN_IN" in section
    assert "TALK_NO_CREDIT" in section
    assert "TALK_VOICE_FAILED" in section
    assert "response.status === 401 || response.status === 402" in section
    assert "tellTalkVoiceIssue(TALK_VOICE_FAILED)" in section
    assert "tellTalkVoiceIssue(TALK_PLAYBACK_BLOCKED)" in section


def test_talk_mode_speaks_only_the_final_answer() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "One moment — checking…" in source
    section = _talk_section(source)
    assert "await speakTalkTurn(answerText)" in section
    assert "await handleQuestion(question)" in section
    assert "One moment — checking…" not in section


def test_unhinged_mode_wiring_is_unchanged_by_talk_fixes() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = (REPO_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="unhinged-toggle"' in html
    assert "Unhinged" in html
    assert "neuralForgeChatMode" in source
    assert "Unhinged is on. Ask anything. I'll roast the forecast and keep every number exact." in source
    assert "payload.facts = facts" in source
    assert 'mode === "unhinged"' in source
