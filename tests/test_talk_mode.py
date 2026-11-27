"""Talk mode frontend contracts — keep the listen/answer/speak loop reliable."""

from __future__ import annotations

from src.utils.paths import REPO_ROOT

APP_JS = REPO_ROOT / "frontend" / "app.js"


def _talk_section(source: str) -> str:
    start = source.index(
        "/* --------------------------------------------------------------- talk mode */"
    )
    end = source.index(
        "/* ------------------------------------------------------------------ voice */",
        start,
    )
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
    assert (
        "response.status === 401 || response.status === 402 || response.status === 429"
        in section
    )
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
    assert (
        "Unhinged is on. Ask anything. I'll roast the forecast and keep every number exact."
        in source
    )
    assert "payload.facts = facts" in source
    assert 'mode === "unhinged"' in source


def test_talk_mode_defaults_on_for_phones_and_stays_manual_on_desktop() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    section = _talk_section(source)
    html = (REPO_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    css = (REPO_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    assert "function isPhoneSurface" in section
    assert "function isAppWrapper" in section
    assert "(pointer: coarse)" in section
    assert "(max-width: 900px)" in section
    assert "userAgent" in section
    assert "window.Capacitor" in section
    assert "neuralForgeTalkPref" in source
    assert "function shouldAutoArmTalk" in section
    assert 'isPhoneSurface() && talkPreference() !== "off"' in section
    assert "function armTalkForPhone" in section
    assert "function startTalkFromGesture" in section
    assert "function talkGestureRequired" not in section
    assert "event.isTrusted !== true" in section
    assert "userActivation" not in section
    assert "function disarmTalk" in section
    assert "armTalkForPhone();" in source
    assert 'id="talk-start-overlay"' in html
    assert "Tap to start talking" in html
    assert 'id="talk-start-dismiss"' in html
    assert "Not now" in html
    assert 'id="talk-status"' in html
    assert "Talk is on. Tap anywhere to start listening." in section
    assert "pointerdown" in section
    assert "onTalkAutoStartFocus" not in section
    assert 'data-state="armed"' in css or '.mode-pill[data-state="armed"]' in css
    # Desktop keeps the manual Talk click path and does not auto-arm.
    assert "function onTalkClick" in section
    assert "Prime audio synchronously on the click path before any await." in section
    assert "shouldAutoArmTalk" in section


def test_talk_mode_remembers_phone_off_switch_and_keeps_unhinged_alone() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    section = _talk_section(source)
    assert "disarmTalk({ persist: true })" in section
    assert 'setTalkPreference("off")' in section
    assert 'setTalkPreference("on")' in section
    assert "talk.armed && !talk.active" in section
    # Voice / credit / model behavior stays in the existing Talk helpers.
    assert "function speakTalkTurn" in section
    assert "TALK_NO_CREDIT" in section
    assert "TALK_SIGN_IN" in section
    assert "lottery" not in source.lower()


def test_there_is_no_listen_button_and_gq_speaks_on_its_own() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = (REPO_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    assert "speak-btn" not in source and "speak-btn" not in css
    assert "Listen to this reply" not in source
    assert "speakMessage" not in source
    section = _talk_section(source)
    # Replies typed while Talk is on are spoken too.
    assert "speakTalkTurn(replyText)" in source
    # Audio is unlocked during the Talk tap.
    assert "unlockTalkAudio();" in section


def test_talk_mode_hears_through_the_mic_not_only_browser_recognition() -> None:
    section = _talk_section(APP_JS.read_text(encoding="utf-8"))
    assert "function talkMicSupported" in section
    assert "function openTalkMic" in section
    assert "function closeTalkMic" in section
    assert "/agent/transcribe" in section
    assert "audio/wav" in section
    assert "createScriptProcessor" in section
    assert "echoCancellation: true" in section
    # The AudioContext is created before any await so iPhone allows it.
    open_mic = section[section.index("async function openTalkMic") :]
    assert open_mic.index("new AudioCtx()") < open_mic.index("await navigator")
    # The mic is closed whenever Talk stops.
    stop = section[section.index("function stopTalk()") :]
    assert "closeTalkMic();" in stop[:400]
    # Phones arm Talk when the mic works, even without built-in recognition.
    assert "if (!talkCanHear()) return;" in section


def test_talk_mode_falls_back_to_text_when_voice_runs_out() -> None:
    section = _talk_section(APP_JS.read_text(encoding="utf-8"))
    assert "TALK_VOICE_CAPPED" in section
    assert "response.status === 429" in section
    assert "talk.voiceOff = true" in section
    assert "VOICE_RESTING_MESSAGE" in section
