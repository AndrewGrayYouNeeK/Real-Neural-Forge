"""Password hashing that stays inside the Python standard library.

scrypt is memory-hard and available on the Python versions this project
tests (3.10 through 3.13). No extra package is required in the container.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_N = 2**14
_R = 8
_P = 1
_DKLEN = 32
_MAXMEM = 64 * 1024 * 1024
_MIN_CHARS = 10
_MAX_CHARS = 200


def validate_password(password: str, email: str) -> str | None:
    """Return an error message, or None when the password can be stored."""
    if not isinstance(password, str):
        return "Password must be a string."
    if len(password) < _MIN_CHARS:
        return f"Password must be at least {_MIN_CHARS} characters."
    if len(password) > _MAX_CHARS:
        return f"Password must be at most {_MAX_CHARS} characters."
    if password.casefold() == email.casefold():
        return "Password must be different from the email address."
    return None


def hash_password(password: str) -> str:
    """Return a stored scrypt hash. The plaintext is not retained."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_N,
        r=_R,
        p=_P,
        dklen=_DKLEN,
        maxmem=_MAXMEM,
    )
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    """Return True when ``password`` matches ``stored``."""
    if not stored or not isinstance(password, str):
        return False
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    try:
        n = int(parts[1])
        r = int(parts[2])
        p = int(parts[3])
        salt = bytes.fromhex(parts[4])
        expected = bytes.fromhex(parts[5])
    except ValueError:
        return False
    if n < 2 or n > 2**20 or r < 1 or p < 1 or not salt or not expected:
        return False
    try:
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=_MAXMEM,
        )
    except (ValueError, OSError):
        return False
    return hmac.compare_digest(digest, expected)
