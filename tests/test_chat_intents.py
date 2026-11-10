"""Chat intent routing must answer the question that was asked."""

from __future__ import annotations

import re

import pytest

from src.utils.paths import REPO_ROOT

APP_JS = REPO_ROOT / "frontend" / "app.js"

INTENT_RE = re.compile(
    r"name:\s*\"(?P<name>[^\"]+)\",\s*test:\s*/(?P<pattern>.+?)/(?P<flags>[a-z]*)",
    re.MULTILINE | re.DOTALL,
)
TRAIN_START_RE = re.compile(
    r"function isTrainStartRequest\(question\) \{\s*return /(.+)/([a-z]*)\.test",
    re.MULTILINE,
)


def _flags(text: str) -> int:
    value = 0
    if "i" in text:
        value |= re.IGNORECASE
    if "m" in text:
        value |= re.MULTILINE
    return value


def _js_regex(pattern: str, flags: str) -> re.Pattern[str]:
    # Intent patterns in app.js are written for JavaScript. The features we
    # use here (word boundaries, ?., simple groups) are valid in Python too.
    return re.compile(pattern, _flags(flags))


def load_intents() -> list[tuple[str, re.Pattern[str]]]:
    source = APP_JS.read_text(encoding="utf-8")
    block_start = source.index("const INTENTS = [")
    block_end = source.index("\nfunction matchIntent", block_start)
    block = source[block_start:block_end]
    intents = [
        (match.group("name"), _js_regex(match.group("pattern"), match.group("flags")))
        for match in INTENT_RE.finditer(block)
    ]
    names = [name for name, _pattern in intents]
    assert names == [
        "greet",
        "help",
        "lottery",
        "moon",
        "space",
        "solar",
        "crypto",
        "markets",
        "weather",
        "voice",
        "train",
    ], names
    return intents


def match_intent(question: str) -> str | None:
    for name, pattern in load_intents():
        if pattern.search(question):
            return name
    return None


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Hi there", "greet"),
        ("What can you do?", "help"),
        ("What will the weather be tomorrow?", "weather"),
        ("Weather in Dallas this weekend?", "weather"),
        ("How hot will it be at Central Park?", "weather"),
        ("Is it going to rain?", "weather"),
        ("How are the markets doing?", "markets"),
        ("How is AAPL doing?", "markets"),
        ("What about AAPL stock?", "markets"),
        ("How is Bitcoin doing?", "crypto"),
        ("What's Bitcoin at?", "crypto"),
        ("What are the next lottery numbers?", "lottery"),
        ("What were the last Powerball numbers?", "lottery"),
        ("Has 21 come up in Powerball?", "lottery"),
        ("What's the moon phase tonight?", "moon"),
        ("What's coming close to Earth?", "space"),
        ("What's happening in the solar system?", "solar"),
        ("How much GQ voice credit do I have?", "voice"),
        ("How does training work?", "train"),
        ("Start training", "train"),
        ("Train a model on my data", "train"),
        ("Train on my CSV", "train"),
    ],
)
def test_questions_route_to_the_right_topic(question: str, expected: str) -> None:
    assert match_intent(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Predict the next value",
        "See you tomorrow",
        "What will happen tomorrow?",
        "Can you predict my future?",
        "What's 2 + 2?",
        "Who was the first president?",
        "Who won the World Cup?",
        "Tell me a joke",
        "Give me some space in this box",
    ],
)
def test_unrelated_questions_are_out_of_scope(question: str) -> None:
    assert match_intent(question) is None


def test_generic_training_questions_do_not_start_a_run() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    match = TRAIN_START_RE.search(source)
    assert match, "isTrainStartRequest regex was not found in app.js"
    start = _js_regex(match.group(1), match.group(2))
    assert start.search("Start training")
    assert start.search("Train a model on my data")
    assert start.search("Please retrain")
    assert not start.search("How does training work?")
    assert not start.search("Tell me about training")
    assert not start.search("Train on my CSV")


def test_train_on_my_csv_opens_the_upload_flow() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "function isCsvTrainRequest" in source
    assert "/workspace/datasets/" in source
    assert "target_column" in source
    assert "Create a free account" in source
    assert "so this shared server stays on a small plan." in source


def test_live_endpoints_are_wired_in_the_chat() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "/live/weather" in source
    assert "/live/quote" in source
    assert "/live/lottery" in source
    assert "/live/ask" in source
    assert "askLive(question)" in source
