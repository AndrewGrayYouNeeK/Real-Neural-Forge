"""Record issued forecasts and score them when the actuals arrive.

Each lead is stored on its own row. Once an actual is known, rolling windows
keep the empirical 80% interval coverage and the p10 and p90 hit rates for
that model and lead. An 80% interval is the closed range from p10 to p90.
The p10 hit rate is the share of actuals at or above p10, and the p90 hit
rate is the share at or below p90. Those rates sit near 0.90 when the tails
are calibrated.

The SQL is ``deploy/cloudflare/migrations/0004_evaluation.sql``. Local tests
apply that file to SQLite. Production applies it to D1, and the API writes
through the same workspace SQL connection the rest of the account data uses.
"""

from __future__ import annotations

import math
import os
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.paths import REPO_ROOT
from src.workspace.runtime import runtime
from src.workspace.service_ids import new_id
from src.workspace.store import SqlError, SqliteSql, sql_statements, utc_now

MIGRATION_PATH = (
    REPO_ROOT / "deploy" / "cloudflare" / "migrations" / "0004_evaluation.sql"
)
WINDOW_SIZES = (30, 90, 365)
_CACHE: dict[int, OutcomeTracker] = {}


@dataclass(frozen=True)
class CoverageStats:
    """Empirical interval quality for one model, lead, and window."""

    model_id: str
    horizon: int
    window_size: int
    n: int
    coverage_80: float
    p10_hit_rate: float
    p90_hit_rate: float
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "horizon": self.horizon,
            "window_size": self.window_size,
            "n": self.n,
            "coverage_80": self.coverage_80,
            "p10_hit_rate": self.p10_hit_rate,
            "p90_hit_rate": self.p90_hit_rate,
            "updated_at": self.updated_at,
        }


class OutcomeTracker:
    """SQLite or D1-backed log of forecasts and their realized coverage."""

    def __init__(self, sql: Any) -> None:
        self.sql = sql
        self._ready = False

    def ensure(self) -> OutcomeTracker:
        if self._ready:
            return self
        script = MIGRATION_PATH.read_text(encoding="utf-8")
        self.sql.script(sql_statements(script))
        self._ready = True
        return self

    def record(
        self,
        model_id: str,
        steps: Sequence[Mapping[str, Any]],
        *,
        issued_at: str | None = None,
    ) -> list[str]:
        """Store one row per lead. Identical open forecasts are not repeated."""
        self.ensure()
        if not model_id or not str(model_id).strip():
            raise ValueError("model_id is required.")
        stamp = issued_at or utc_now()
        saved: list[str] = []
        for index, step in enumerate(steps, start=1):
            horizon = _positive_int(step.get("horizon"), default=index)
            p50 = _finite(step.get("p50"), "p50")
            p10 = _optional_finite(step.get("p10"))
            p90 = _optional_finite(step.get("p90"))
            target = _optional_text(step.get("target_time"))
            if _already_open(self.sql, model_id, horizon, target, p50):
                continue
            outcome_id = new_id("out_")
            self.sql.query(
                """
                INSERT INTO forecast_outcomes (
                    id, model_id, horizon, issued_at, target_time, p10, p50, p90
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (outcome_id, model_id, horizon, stamp, target, p10, p50, p90),
            )
            saved.append(outcome_id)
        return saved

    def score_actuals(self, model_id: str, actuals: Mapping[str, float]) -> int:
        """Fill unscored rows whose target time is in ``actuals``.

        Returns how many rows gained an actual. Coverage windows for the
        model are rebuilt when at least one row is scored.
        """
        self.ensure()
        if not actuals:
            return 0
        clean = {
            str(key): float(value)
            for key, value in actuals.items()
            if _optional_text(key) and _optional_finite(value) is not None
        }
        if not clean:
            return 0
        pending = self.sql.query(
            """
            SELECT id, target_time FROM forecast_outcomes
            WHERE model_id = ? AND actual IS NULL AND target_time IS NOT NULL
            """,
            (model_id,),
        )
        now = utc_now()
        updated = 0
        for row in pending:
            target = row.get("target_time")
            if target not in clean:
                continue
            self.sql.query(
                """
                UPDATE forecast_outcomes
                SET actual = ?, scored_at = ?
                WHERE id = ? AND actual IS NULL
                """,
                (clean[str(target)], now, row["id"]),
            )
            updated += 1
        if updated:
            self._recompute(model_id)
        return updated

    def coverage(
        self, model_id: str, *, horizon: int | None = None
    ) -> list[CoverageStats]:
        """Stored rolling windows, oldest lead and smallest window first."""
        self.ensure()
        if horizon is None:
            rows = self.sql.query(
                """
                SELECT model_id, horizon, window_size, n, coverage_80,
                       p10_hit_rate, p90_hit_rate, updated_at
                FROM coverage_windows
                WHERE model_id = ?
                ORDER BY horizon ASC, window_size ASC
                """,
                (model_id,),
            )
        else:
            rows = self.sql.query(
                """
                SELECT model_id, horizon, window_size, n, coverage_80,
                       p10_hit_rate, p90_hit_rate, updated_at
                FROM coverage_windows
                WHERE model_id = ? AND horizon = ?
                ORDER BY window_size ASC
                """,
                (model_id, int(horizon)),
            )
        return [_stats(row) for row in rows]

    def preferred_coverage(self, model_id: str, horizon: int) -> CoverageStats | None:
        """The widest window that already has 30 scored rows, else the largest n."""
        rows = self.coverage(model_id, horizon=horizon)
        if not rows:
            return None
        ready = [row for row in rows if row.n >= 30]
        pool = ready or rows
        return max(pool, key=lambda row: (row.n, row.window_size))

    def _recompute(self, model_id: str) -> None:
        scored = self.sql.query(
            """
            SELECT horizon, p10, p90, actual
            FROM forecast_outcomes
            WHERE model_id = ?
              AND actual IS NOT NULL
              AND p10 IS NOT NULL
              AND p90 IS NOT NULL
            ORDER BY COALESCE(target_time, scored_at) ASC, id ASC
            LIMIT 5000
            """,
            (model_id,),
        )
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            grouped[int(row["horizon"])].append(row)
        self.sql.query(
            "DELETE FROM coverage_windows WHERE model_id = ?",
            (model_id,),
        )
        now = utc_now()
        for horizon, items in grouped.items():
            for window in WINDOW_SIZES:
                if len(items) < window and window != WINDOW_SIZES[0]:
                    continue
                sample = items[-window:]
                summary = _summarize(sample)
                if summary is None:
                    continue
                coverage_80, p10_rate, p90_rate = summary
                self.sql.query(
                    """
                    INSERT INTO coverage_windows (
                        model_id, horizon, window_size, n, coverage_80,
                        p10_hit_rate, p90_hit_rate, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        horizon,
                        window,
                        len(sample),
                        coverage_80,
                        p10_rate,
                        p90_rate,
                        now,
                    ),
                )


def tracker_for(sql: Any) -> OutcomeTracker:
    """Return a schema-ready tracker for this SQL connection."""
    found = _CACHE.get(id(sql))
    if found is None:
        found = OutcomeTracker(sql).ensure()
        _CACHE[id(sql)] = found
    return found


def clear_tracker_cache() -> None:
    """Drop cached trackers. Tests call this between temporary databases."""
    _CACHE.clear()


def default_tracker() -> OutcomeTracker:
    """Use the live workspace database, or a local file when it is absent.

    ``NEURAL_FORGE_EVAL_DB`` forces the local file. Tests use that so a
    process that already has a workspace store still writes somewhere
    inspectable. Production leaves it unset and writes to D1.
    """
    override = os.environ.get("NEURAL_FORGE_EVAL_DB", "").strip()
    if override:
        return tracker_for(SqliteSql(Path(override)))
    store = runtime.store
    if store is not None:
        return tracker_for(store.sql)
    return tracker_for(SqliteSql(REPO_ROOT / "data" / "evaluation.db"))


def record_payload(model_id: str, payload: Mapping[str, Any]) -> None:
    """Record a forecast response. Failures stay out of the caller's result."""
    try:
        steps = steps_from_payload(payload)
        if steps:
            default_tracker().record(model_id, steps)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        SqlError,
        sqlite3.Error,
    ):
        # Tracking must not change the forecast the caller already computed.
        return


def steps_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read p10, p50, and p90 from a forecast payload without requiring them."""
    uncertainty = payload.get("uncertainty")
    if isinstance(uncertainty, dict) and isinstance(uncertainty.get("p50"), list):
        return _aligned_steps(
            uncertainty.get("p50"),
            uncertainty.get("p10"),
            uncertainty.get("p90"),
        )
    if isinstance(payload.get("p50"), list):
        return _aligned_steps(
            payload.get("p50"), payload.get("p10"), payload.get("p90")
        )
    prediction = payload.get("prediction")
    if isinstance(prediction, list) and prediction:
        value = _optional_finite(prediction[0])
        if value is None:
            return []
        return [{"horizon": 1, "p50": value, "p10": None, "p90": None}]
    forecast = payload.get("forecast")
    if isinstance(forecast, list) and forecast and isinstance(forecast[0], dict):
        steps: list[dict[str, Any]] = []
        for index, row in enumerate(forecast, start=1):
            if not isinstance(row, dict):
                continue
            p50 = _optional_finite(row.get("p50", row.get("value")))
            if p50 is None:
                continue
            steps.append(
                {
                    "horizon": _positive_int(row.get("step"), default=index),
                    "p50": p50,
                    "p10": _optional_finite(row.get("p10")),
                    "p90": _optional_finite(row.get("p90")),
                }
            )
        return steps
    return []


def _aligned_steps(p50: object, p10: object, p90: object) -> list[dict[str, Any]]:
    if not isinstance(p50, list):
        return []
    lows = p10 if isinstance(p10, list) else []
    highs = p90 if isinstance(p90, list) else []
    steps: list[dict[str, Any]] = []
    for index, value in enumerate(p50):
        point = _optional_finite(value)
        if point is None:
            continue
        steps.append(
            {
                "horizon": index + 1,
                "p50": point,
                "p10": _optional_finite(lows[index]) if index < len(lows) else None,
                "p90": _optional_finite(highs[index]) if index < len(highs) else None,
            }
        )
    return steps


def _already_open(
    sql: Any, model_id: str, horizon: int, target: str | None, p50: float
) -> bool:
    if target is None:
        return False
    rows = sql.query(
        """
        SELECT p50 FROM forecast_outcomes
        WHERE model_id = ? AND horizon = ? AND target_time = ? AND actual IS NULL
        """,
        (model_id, horizon, target),
    )
    return any(abs(float(row["p50"]) - p50) <= 1e-6 for row in rows)


def _summarize(
    sample: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float] | None:
    if not sample:
        return None
    covered = 0
    above_p10 = 0
    below_p90 = 0
    for row in sample:
        actual = float(row["actual"])
        low = float(row["p10"])
        high = float(row["p90"])
        if low <= actual <= high:
            covered += 1
        if actual >= low:
            above_p10 += 1
        if actual <= high:
            below_p90 += 1
    count = len(sample)
    return covered / count, above_p10 / count, below_p90 / count


def _stats(row: Mapping[str, Any]) -> CoverageStats:
    return CoverageStats(
        model_id=str(row["model_id"]),
        horizon=int(row["horizon"]),
        window_size=int(row["window_size"]),
        n=int(row["n"]),
        coverage_80=float(row["coverage_80"]),
        p10_hit_rate=float(row["p10_hit_rate"]),
        p90_hit_rate=float(row["p90_hit_rate"]),
        updated_at=str(row["updated_at"]),
    )


def _positive_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    if int(value) != value or int(value) < 1:
        return default
    return int(value)


def _finite(value: object, name: str) -> float:
    number = _optional_finite(value)
    if number is None:
        raise ValueError(f"{name} must be a finite number.")
    return number


def _optional_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
