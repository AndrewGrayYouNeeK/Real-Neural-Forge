"""Turn chat text into speech that ElevenLabs can say cleanly.

The same string is the cache key, so this stays deterministic. Decorative
markdown, emoji, and symbols come out. Money, percents, temperatures, dates,
units, and tickers are spoken as words. Scientific notation becomes a plain
number so a chart label like ``1.23e+4`` is not read as letters.
"""

from __future__ import annotations

import re
from calendar import month_name

NORMALIZER_VERSION = "2"
_CHUNK_TARGET = 650
_CHUNK_LIMIT = 800

_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\ufe0f\u200d]+",
    flags=re.UNICODE,
)
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONEY = re.compile(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DEGREES = re.compile(r"(\d+(?:\.\d+)?)\s*°\s*([FfCc])\b")
_SCIENCE = re.compile(
    r"(?<![A-Za-z0-9.])([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![A-Za-z0-9])"
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_UNIT = re.compile(
    r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*"
    r"(km/h|mph|km|cm|mm|kg|lb|ft|mi|AU|GB|MB|KB)\b",
    flags=re.IGNORECASE,
)
_TICKER = re.compile(r"\b[A-Z]{2,10}\b")
_MARKDOWN = re.compile(r"[*_#~`|>]+")
_SPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_UNITS = {
    "km/h": "kilometers per hour",
    "mph": "miles per hour",
    "km": "kilometers",
    "cm": "centimeters",
    "mm": "millimeters",
    "kg": "kilograms",
    "lb": "pounds",
    "ft": "feet",
    "mi": "miles",
    "au": "astronomical units",
    "gb": "gigabytes",
    "mb": "megabytes",
    "kb": "kilobytes",
}
_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "META": "Meta",
    "NFLX": "Netflix",
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
}
# Ordinary abbreviations. These are not tickers, so they are not spelled out.
_NOT_TICKERS = {
    "AI",
    "API",
    "CSV",
    "GQ",
    "UTC",
    "US",
    "UK",
    "OK",
    "ID",
    "PM",
    "AM",
    "NOAA",
    "JPL",
    "NASA",
    "FAQ",
    "URL",
    "TTS",
    "MP3",
    "NEO",
    "MAE",
    "RMSE",
    "MAPE",
    "ISO",
}


def speech_text(raw: str) -> str:
    """Normalize one reply for synthesis. Empty when nothing speakable remains."""
    if not isinstance(raw, str):
        return ""
    text = raw.replace("\u2014", ", ").replace("\u2013", ", ").replace("\u2026", ", ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2022", " ")
    text = _LINK.sub(r"\1", text)
    text = _EMOJI.sub(" ", text)
    text = _DATE.sub(_date, text)
    text = _MONEY.sub(_money, text)
    text = _PERCENT.sub(r"\1 percent", text)
    text = _DEGREES.sub(_degrees, text)
    text = _SCIENCE.sub(_science, text)
    text = _THOUSANDS.sub("", text)
    text = _UNIT.sub(_unit, text)
    text = _TICKER.sub(_ticker, text)
    text = _MARKDOWN.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def truncate_speech(text: str, limit: int) -> str:
    """Cut at a sentence boundary near ``limit``, otherwise at a word."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    if len(text) <= limit:
        return text
    window = text[:limit]
    ends = [match.end() for match in re.finditer(r"[.!?]", window)]
    if ends and ends[-1] >= min(40, limit // 2):
        return window[: ends[-1]].strip()
    space = window.rfind(" ")
    if space > 0:
        return window[:space].strip()
    return window.strip()


def split_for_speech(
    text: str, *, target: int = _CHUNK_TARGET, limit: int = _CHUNK_LIMIT
) -> list[str]:
    """Split on sentence boundaries into chunks of about 500 to 800 characters."""
    if target < 1 or limit < target:
        raise ValueError("speech chunks need 1 <= target <= limit.")
    if not text:
        return []
    parts: list[str] = []
    current = ""
    for sentence in _sentences(text):
        piece = sentence.strip()
        if not piece:
            continue
        if len(piece) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_split_long(piece, limit))
            continue
        if not current:
            current = piece
            continue
        joined = f"{current} {piece}"
        if len(joined) <= limit and (len(current) < target or len(joined) <= target):
            current = joined
            continue
        parts.append(current)
        current = piece
    if current:
        parts.append(current)
    return parts


def _sentences(text: str) -> list[str]:
    return _SENTENCE.split(text)


def _split_long(sentence: str, limit: int) -> list[str]:
    words = sentence.split()
    parts: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
            continue
        if len(current) + 1 + len(word) > limit:
            parts.append(current)
            current = word
            continue
        current = f"{current} {word}"
    if current:
        parts.append(current)
    return parts


def _date(match: re.Match[str]) -> str:
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return match.group(0)
    return f"{month_name[month]} {day}, {year}"


def _money(match: re.Match[str]) -> str:
    dollars = int(match.group(1).replace(",", ""))
    cents_raw = match.group(2)
    dollar_word = "dollar" if dollars == 1 else "dollars"
    if cents_raw is None:
        return f"{dollars} {dollar_word}"
    cents = int(cents_raw.ljust(2, "0"))
    cent_word = "cent" if cents == 1 else "cents"
    return f"{dollars} {dollar_word} and {cents} {cent_word}"


def _degrees(match: re.Match[str]) -> str:
    scale = "Fahrenheit" if match.group(2).lower() == "f" else "Celsius"
    return f"{match.group(1)} degrees {scale}"


def _science(match: re.Match[str]) -> str:
    number = float(f"{match.group(1)}e{match.group(2)}")
    plain = format(number, "f")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    return plain or "0"


def _unit(match: re.Match[str]) -> str:
    name = _UNITS[match.group(2).lower()]
    return f"{match.group(1)} {name}"


def _ticker(match: re.Match[str]) -> str:
    token = match.group(0)
    if token in _NOT_TICKERS:
        return token
    named = _NAMES.get(token)
    if named:
        return named
    if len(token) > 5:
        return token
    return " ".join(token)
