"""Per-account GQ voice balances, usage, purchases, and the daily safety cap.

The tables live beside the workspace tables in the same SQL database (SQLite
locally, D1 in production), and cached audio bytes live in the same blob
store (files locally, R2 in production).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.workspace.keys import valid_object_key
from src.workspace.service_ids import new_id
from src.workspace.store import SqlError, utc_now

_SCHEMA = Path(__file__).with_name("schema.sql")


def sql_statements(script: str | None = None) -> list[str]:
    """Split the voice schema into individual statements."""
    text = _SCHEMA.read_text(encoding="utf-8") if script is None else script
    uncommented = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("--")
    )
    statements: list[str] = []
    for chunk in uncommented.split(";"):
        statement = chunk.strip()
        if statement:
            statements.append(statement)
    return statements


def audio_cache_key(digest: str) -> str:
    return f"voice/cache/{digest}.mp3"


def migrate_voice(store: Any) -> None:
    """Apply the voice schema to an open workspace store."""
    VoiceStore(store.sql, store.blobs).migrate()


class VoiceStore:
    """Metering on top of the workspace SQL and blob transports."""

    def __init__(self, sql: Any, blobs: Any) -> None:
        self.sql = sql
        self.blobs = blobs

    def migrate(self) -> None:
        self.sql.script(sql_statements())

    def account(self, user_id: str) -> dict[str, int]:
        rows = self.sql.query(
            "SELECT * FROM voice_balances WHERE user_id = ?",
            (user_id,),
        )
        if not rows:
            return {
                "free_remaining": 0,
                "purchased_remaining": 0,
                "spent": 0,
            }
        row = rows[0]
        return {
            "free_remaining": int(row["free_remaining"]),
            "purchased_remaining": int(row["purchased_remaining"]),
            "spent": int(row["spent"]),
        }

    def ensure_account(self, user_id: str, free_chars: int) -> dict[str, int]:
        """Create the balance row once. The free allowance is granted a single time."""
        self.sql.query(
            """
            INSERT OR IGNORE INTO voice_balances
            (user_id, free_remaining, purchased_remaining, spent, updated_at)
            VALUES (?, ?, 0, 0, ?)
            """,
            (user_id, free_chars, utc_now()),
        )
        return self.account(user_id)

    def spend(self, user_id: str, chars: int) -> dict[str, int] | None:
        """Deduct free credit first, then purchased. None means insufficient."""
        rows = self.sql.query(
            """
            UPDATE voice_balances
            SET free_remaining = MAX(free_remaining - ?, 0),
                purchased_remaining = purchased_remaining - MAX(? - free_remaining, 0),
                spent = spent + ?,
                updated_at = ?
            WHERE user_id = ? AND free_remaining + purchased_remaining >= ?
            RETURNING free_remaining, purchased_remaining, spent
            """,
            (chars, chars, chars, utc_now(), user_id, chars),
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "free_remaining": int(row["free_remaining"]),
            "purchased_remaining": int(row["purchased_remaining"]),
            "spent": int(row["spent"]),
        }

    def credit_purchase(
        self, user_id: str, pack_id: str, chars: int, session_id: str
    ) -> bool:
        """Add purchased credit exactly once per Stripe session."""
        try:
            self.sql.query(
                """
                INSERT INTO voice_purchases
                (id, user_id, pack_id, chars, session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (f"pur_{session_id}", user_id, pack_id, chars, session_id, utc_now()),
            )
        except SqlError:
            return False
        self.sql.query(
            """
            UPDATE voice_balances
            SET purchased_remaining = purchased_remaining + ?, updated_at = ?
            WHERE user_id = ?
            """,
            (chars, utc_now(), user_id),
        )
        return True

    def record_usage(
        self,
        user_id: str,
        kind: str,
        chars: int,
        *,
        cached: bool,
        balance: dict[str, int] | None,
    ) -> None:
        self.sql.query(
            """
            INSERT INTO voice_usage
            (id, user_id, kind, chars, cached, free_after, purchased_after, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("voc_"),
                user_id,
                kind,
                chars,
                1 if cached else 0,
                balance.get("free_remaining") if balance else None,
                balance.get("purchased_remaining") if balance else None,
                utc_now(),
            ),
        )

    def usage_history(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = self.sql.query(
            """
            SELECT kind, chars, cached, created_at FROM voice_usage
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, limit),
        )
        return rows

    def daily_chars(self, day: str) -> int:
        rows = self.sql.query(
            "SELECT chars FROM voice_daily WHERE day = ?",
            (day,),
        )
        return int(rows[0]["chars"]) if rows else 0

    def add_daily_chars(self, day: str, chars: int) -> None:
        self.sql.query(
            """
            INSERT INTO voice_daily (day, chars) VALUES (?, ?)
            ON CONFLICT(day) DO UPDATE SET chars = chars + excluded.chars
            """,
            (day, chars),
        )

    def cache_get(self, digest: str) -> bytes | None:
        key = audio_cache_key(digest)
        if not valid_object_key(key):
            raise SqlError("Unsafe voice cache key.")
        audio: bytes | None = self.blobs.get(key)
        return audio

    def cache_put(self, digest: str, audio: bytes) -> None:
        key = audio_cache_key(digest)
        if not valid_object_key(key):
            raise SqlError("Unsafe voice cache key.")
        self.blobs.put(key, audio)


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
