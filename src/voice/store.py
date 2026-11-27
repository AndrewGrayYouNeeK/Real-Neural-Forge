"""Per-account GQ voice balances, usage, purchases, and the daily safety cap.

The tables live beside the workspace tables in the same SQL database (SQLite
locally, D1 in production), and cached audio bytes live in the same blob
store (files locally, R2 in production).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.workspace.keys import valid_object_key
from src.workspace.service_ids import new_id
from src.workspace.store import SqlError, utc_now

_SCHEMA = Path(__file__).with_name("schema.sql")
_PLAN_SCHEMA = Path(__file__).with_name("plan_schema.sql")

# A paid period that Stripe has not renewed keeps working this long, so a late
# webhook never cuts a subscriber off in the middle of a conversation.
PERIOD_GRACE = timedelta(days=2)
ACTIVE_STATUSES = frozenset({"active", "trialing"})


def sql_statements(script: str | None = None, *, plan: bool = False) -> list[str]:
    """Split the voice schema (or the Henad Plus schema) into statements."""
    if script is None:
        text = (_PLAN_SCHEMA if plan else _SCHEMA).read_text(encoding="utf-8")
    else:
        text = script
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
        self.sql.script(sql_statements(plan=True))

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

    def plan(self, user_id: str) -> dict[str, Any] | None:
        """The subscriber row, with ``active`` worked out from status and period."""
        rows = self.sql.query(
            "SELECT * FROM voice_plans WHERE user_id = ?",
            (user_id,),
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["cancel_at_period_end"] = bool(row.get("cancel_at_period_end"))
        row["allowance_remaining"] = int(row.get("allowance_remaining") or 0)
        row["allowance_total"] = int(row.get("allowance_total") or 0)
        row["active"] = plan_is_active(row)
        return row

    def plan_by_subscription(self, subscription_id: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            "SELECT user_id FROM voice_plans WHERE stripe_subscription_id = ?",
            (subscription_id,),
        )
        return self.plan(str(rows[0]["user_id"])) if rows else None

    def apply_subscription(
        self,
        user_id: str,
        *,
        customer_id: str | None,
        subscription_id: str,
        status: str,
        period_start: str,
        period_end: str,
        cancel_at_period_end: bool,
        monthly_chars: int,
    ) -> bool:
        """Record the subscription. Returns True when a new period's allowance landed.

        The allowance is granted once per (subscription, period start), so
        replayed webhooks, the checkout sync, and renewals can arrive in any
        order without ever granting twice.
        """
        now = utc_now()
        self.sql.query(
            """
            INSERT INTO voice_plans
            (user_id, stripe_customer_id, stripe_subscription_id, status,
             period_start, period_end, cancel_at_period_end,
             allowance_total, allowance_remaining, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                stripe_customer_id = COALESCE(excluded.stripe_customer_id,
                                              voice_plans.stripe_customer_id),
                stripe_subscription_id = excluded.stripe_subscription_id,
                status = excluded.status,
                period_start = excluded.period_start,
                period_end = excluded.period_end,
                cancel_at_period_end = excluded.cancel_at_period_end,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                customer_id,
                subscription_id,
                status,
                period_start,
                period_end,
                1 if cancel_at_period_end else 0,
                now,
            ),
        )
        if status not in ACTIVE_STATUSES:
            return False
        ref = f"{subscription_id}:{period_start}"
        try:
            self.sql.query(
                "INSERT INTO voice_grants (ref, user_id, chars, created_at)"
                " VALUES (?, ?, ?, ?)",
                (ref, user_id, monthly_chars, now),
            )
        except SqlError:
            return False
        self.sql.query(
            """
            UPDATE voice_plans
            SET allowance_total = ?, allowance_remaining = ?, updated_at = ?
            WHERE user_id = ?
            """,
            (monthly_chars, monthly_chars, now, user_id),
        )
        return True

    def touch_plan(self, user_id: str) -> None:
        self.sql.query(
            "UPDATE voice_plans SET updated_at = ? WHERE user_id = ?",
            (utc_now(), user_id),
        )

    def spend_plan(self, user_id: str, chars: int) -> int:
        """Take up to ``chars`` from the monthly allowance. Returns how much it covered."""
        rows = self.sql.query(
            "SELECT allowance_remaining FROM voice_plans WHERE user_id = ?",
            (user_id,),
        )
        if not rows:
            return 0
        take = min(int(rows[0]["allowance_remaining"]), chars)
        if take <= 0:
            return 0
        updated = self.sql.query(
            """
            UPDATE voice_plans
            SET allowance_remaining = allowance_remaining - ?, updated_at = ?
            WHERE user_id = ? AND allowance_remaining >= ?
            RETURNING allowance_remaining
            """,
            (take, utc_now(), user_id, take),
        )
        return take if updated else 0

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


def plan_is_active(row: dict[str, Any]) -> bool:
    """A subscriber is active while Stripe says so and the paid period is not long over."""
    if str(row.get("status")) not in ACTIVE_STATUSES:
        return False
    end = parse_stamp(row.get("period_end"))
    if end is None:
        return False
    return datetime.now(timezone.utc) <= end + PERIOD_GRACE


def parse_stamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
