"""Spoken text drops markdown, emoji, and decorative symbols."""

from src.voice.text import speech_text


def test_speech_text_keeps_words_numbers_and_sentence_marks() -> None:
    assert speech_text("It's 12.5, really!") == "It's 12.5, really!"


def test_speech_text_strips_markdown_emoji_and_symbols() -> None:
    raw = "**Hello** `world` [site](https://example.test) 🎉 ###"
    assert speech_text(raw) == "Hello world site"


def test_speech_text_is_empty_when_nothing_speakable_remains() -> None:
    assert speech_text("### ***") == ""
    assert speech_text("") == ""
