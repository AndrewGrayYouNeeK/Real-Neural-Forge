"""GQ voice settings. Environment overrides config ``voice:`` which overrides defaults."""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import Any

# Andrew's professional voice clone, "GQ", on ElevenLabs.
DEFAULT_VOICE_ID = "QMnwdN2oOwNXPcl9oXIl"

VOICE_ID_ENV = "NEURAL_FORGE_VOICE_ID"
FREE_CHARS_ENV = "NEURAL_FORGE_VOICE_FREE_CHARS"
MAX_CHARS_ENV = "NEURAL_FORGE_VOICE_MAX_CHARS"
DAILY_LIMIT_ENV = "NEURAL_FORGE_VOICE_DAILY_LIMIT"
ENABLED_ENV = "NEURAL_FORGE_VOICE_ENABLED"

_API_KEY_ENV = "ELEVENLABS_API_KEY"
_STRIPE_SECRET_ENV = "STRIPE_SECRET_KEY"
_STRIPE_WEBHOOK_SECRET_ENV = "STRIPE_WEBHOOK_SECRET"
_PUBLIC_URL_ENV = "NEURAL_FORGE_PUBLIC_URL"


# One-time packs. Prices sit well above the ElevenLabs API cost of about
# $0.08 per 1,000 characters so the account stays comfortably profitable;
# every value is configurable in config.yaml ``voice:`` or by environment.
@dataclass(frozen=True)
class VoicePack:
    """A one-time bundle of GQ voice characters."""

    id: str
    name: str
    chars: int
    price_usd: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "chars": self.chars,
            "price_usd": self.price_usd,
        }


DEFAULT_PACKS = (
    VoicePack(id="starter", name="Starter pack", chars=25_000, price_usd=6.0),
    VoicePack(id="standard", name="Standard pack", chars=100_000, price_usd=18.0),
    VoicePack(id="studio", name="Studio pack", chars=500_000, price_usd=75.0),
)


@dataclass(frozen=True)
class VoiceSettings:
    """ElevenLabs delivery settings. Higher stability and no style reduce stutter."""

    stability: float = 0.7
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    speed: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
            "speed": self.speed,
        }

    def cache_token(self) -> str:
        boost = "1" if self.use_speaker_boost else "0"
        return (
            f"{self.stability}:{self.similarity_boost}:{self.style}:"
            f"{boost}:{self.speed}"
        )


@dataclass(frozen=True)
class VoiceConfig:
    """Resolved GQ voice limits and catalogue."""

    voice_id: str = DEFAULT_VOICE_ID
    model_id: str = "eleven_multilingual_v2"
    free_chars: int = 5_000
    max_chars_per_request: int = 2_000
    daily_char_limit: int = 300_000
    packs: tuple[VoicePack, ...] = DEFAULT_PACKS
    settings: VoiceSettings = VoiceSettings()
    apply_text_normalization: str = "on"

    def as_dict(self) -> dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "free_chars": self.free_chars,
            "max_chars_per_request": self.max_chars_per_request,
            "daily_char_limit": self.daily_char_limit,
            "packs": [pack.as_dict() for pack in self.packs],
        }


def api_key() -> str | None:
    """The ElevenLabs key. Never hard-coded; only from the environment."""
    raw = os.environ.get(_API_KEY_ENV, "").strip()
    return raw or None


def stripe_secret() -> str | None:
    raw = os.environ.get(_STRIPE_SECRET_ENV, "").strip()
    return raw or None


def stripe_is_test_mode() -> bool:
    """True only when a Stripe test secret is configured. Never logs the secret."""
    secret = stripe_secret()
    return secret is not None and secret.startswith("sk_test_")


def webhook_secret() -> str | None:
    raw = os.environ.get(_STRIPE_WEBHOOK_SECRET_ENV, "").strip()
    return raw or None


# Hosts that must not be sent to Stripe as success/cancel URLs.
_INTERNAL_HOSTS = {"0.0.0.0", "bindings.neuralforge", "::", "[::]"}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def public_base_url() -> str | None:
    raw = os.environ.get(_PUBLIC_URL_ENV, "").strip()
    return raw.rstrip("/") or None


def resolve_checkout_base_url(
    *,
    forwarded_proto: str | None = None,
    forwarded_host: str | None = None,
    request_base_url: str | None = None,
) -> str | None:
    """Public site origin for Stripe success/cancel URLs.

    The FastAPI process inside the Cloudflare container is plain HTTP, and
    uvicorn is not started with --proxy-headers unless the image opts in.
    ``request.base_url`` is therefore often ``http://forge.youneek.dev``.
    Stripe live mode rejects non-HTTPS URLs, which surfaces as an unhandled
    500 from Session.create. Prefer the Worker's forwarded host, then the
    configured public URL, then the request base, and upgrade http to https
    for non-loopback hosts.
    """
    host = (forwarded_host or "").split(",")[0].strip()
    proto = (forwarded_proto or "").split(",")[0].strip().lower()
    candidates = []
    if host and proto:
        candidates.append(f"{proto}://{host}")
    configured = public_base_url()
    if configured:
        candidates.append(configured)
    if request_base_url:
        candidates.append(request_base_url)
    for raw in candidates:
        resolved = normalize_public_url(raw)
        if resolved:
            return resolved
    return None


def normalize_public_url(raw: str) -> str | None:
    """Return a Stripe-safe origin, or None when the host is internal."""
    text = raw.strip()
    if not text:
        return None
    if "://" not in text:
        text = f"https://{text}"
    parsed = urllib.parse.urlparse(text)
    host = (parsed.hostname or "").lower().strip("[]")
    if not host or host in _INTERNAL_HOSTS:
        return None
    scheme = (parsed.scheme or "https").lower()
    if scheme not in {"http", "https"}:
        return None
    if scheme == "http" and host not in _LOOPBACK_HOSTS:
        scheme = "https"
    netloc = parsed.netloc
    # Keep userinfo/port if present, but swap the scheme only.
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def voice_enabled(config: dict[str, Any] | None = None) -> bool:
    """Voice is on when it is not switched off and an API key exists."""
    raw = os.environ.get(ENABLED_ENV, "").strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    return api_key() is not None


def _unit_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"voice.{name} must be a number from 0 to 1.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"voice.{name} must be a number from 0 to 1.") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"voice.{name} must be from 0 to 1, got {number}.")
    return number


def _speed(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"voice.{name} must be a number from 0.7 to 1.2.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"voice.{name} must be a number from 0.7 to 1.2.") from exc
    if not 0.7 <= number <= 1.2:
        raise ValueError(f"voice.{name} must be from 0.7 to 1.2, got {number}.")
    return number


def _bool_value(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"voice.{name} must be true or false.")


def _int_value(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"voice.{name} must be an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"voice.{name} must be an integer.") from exc
    env_name = {
        "free_chars": FREE_CHARS_ENV,
        "max_chars_per_request": MAX_CHARS_ENV,
        "daily_char_limit": DAILY_LIMIT_ENV,
    }.get(name)
    if env_name:
        override = os.environ.get(env_name, "").strip()
        if override:
            try:
                number = int(override)
            except ValueError as exc:
                raise ValueError(f"{env_name} must be an integer.") from exc
    if number < minimum:
        raise ValueError(f"voice.{name} must be at least {minimum}, got {number}.")
    return number


def _packs(value: Any) -> tuple[VoicePack, ...]:
    if value is None:
        return DEFAULT_PACKS
    if not isinstance(value, list) or not value:
        raise ValueError("voice.packs must be a non-empty list.")
    packs: list[VoicePack] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("voice.packs entries must be mappings.")
        pack_id = str(entry.get("id", "")).strip()
        name = str(entry.get("name", "")).strip()
        chars = entry.get("chars")
        price = entry.get("price_usd")
        if not pack_id or not name:
            raise ValueError("voice.packs entries need id and name.")
        if pack_id in seen:
            raise ValueError(f"voice.packs has a duplicate id {pack_id!r}.")
        seen.add(pack_id)
        if isinstance(chars, bool) or not isinstance(chars, int) or chars < 1:
            raise ValueError(
                f"voice.packs[{pack_id}].chars must be a positive integer."
            )
        if isinstance(price, bool) or not isinstance(price, int | float) or price <= 0:
            raise ValueError(f"voice.packs[{pack_id}].price_usd must be positive.")
        packs.append(
            VoicePack(id=pack_id, name=name, chars=chars, price_usd=float(price))
        )
    return tuple(packs)


def load_voice_config(config: dict[str, Any] | None = None) -> VoiceConfig:
    """Resolve voice settings from defaults, optional config, then environment."""
    section: dict[str, Any] = {}
    if isinstance(config, dict):
        raw = config.get("voice")
        if isinstance(raw, dict):
            section = raw
    voice_id = (
        str(section.get("voice_id", DEFAULT_VOICE_ID)).strip() or DEFAULT_VOICE_ID
    )
    env_voice_id = os.environ.get(VOICE_ID_ENV, "").strip()
    if env_voice_id:
        voice_id = env_voice_id
    model_id = str(section.get("model_id", "eleven_multilingual_v2")).strip()
    normalization = str(section.get("apply_text_normalization", "on")).strip().lower()
    if normalization not in {"on", "off", "auto"}:
        raise ValueError("voice.apply_text_normalization must be on, off, or auto.")
    return VoiceConfig(
        voice_id=voice_id,
        model_id=model_id or "eleven_multilingual_v2",
        free_chars=_int_value("free_chars", section.get("free_chars", 5_000), 0),
        max_chars_per_request=_int_value(
            "max_chars_per_request", section.get("max_chars_per_request", 2_000), 1
        ),
        daily_char_limit=_int_value(
            "daily_char_limit", section.get("daily_char_limit", 300_000), 0
        ),
        packs=_packs(section.get("packs")),
        settings=VoiceSettings(
            stability=_unit_float("stability", section.get("stability", 0.7)),
            similarity_boost=_unit_float(
                "similarity_boost", section.get("similarity_boost", 0.75)
            ),
            style=_unit_float("style", section.get("style", 0)),
            use_speaker_boost=_bool_value(
                "use_speaker_boost", section.get("use_speaker_boost", True)
            ),
            speed=_speed("speed", section.get("speed", 1.0)),
        ),
        apply_text_normalization=normalization,
    )


def pack_by_id(cfg: VoiceConfig, pack_id: str) -> VoicePack | None:
    for pack in cfg.packs:
        if pack.id == pack_id:
            return pack
    return None
