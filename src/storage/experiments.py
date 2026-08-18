"""Native experiment storage (replaces Base44 entity persistence)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentStore:
    """SQLite-backed store for training runs and metrics."""

    def __init__(self, db_path: str | Path = "data/experiments.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    config_json TEXT NOT NULL,
                    best_loss REAL,
                    checkpoint_path TEXT,
                    metrics_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    epoch INTEGER,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );
                """
            )

    def create_experiment(
        self,
        name: str,
        config: dict[str, Any],
        model_name: str,
    ) -> int:
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments
                (name, model_name, status, config_json, created_at, updated_at)
                VALUES (?, ?, 'running', ?, ?, ?)
                """,
                (name, model_name, json.dumps(config), now, now),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to create experiment record.")
            return int(cursor.lastrowid)

    def log_metric(
        self,
        experiment_id: int,
        name: str,
        value: float,
        epoch: int | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (experiment_id, name, value, epoch, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (experiment_id, name, value, epoch, _utc_now()),
            )

    def complete_experiment(
        self,
        experiment_id: int,
        *,
        status: str,
        best_loss: float | None = None,
        checkpoint_path: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE experiments
                SET status = ?, best_loss = ?, checkpoint_path = ?,
                    metrics_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    best_loss,
                    checkpoint_path,
                    json.dumps(metrics) if metrics else None,
                    _utc_now(),
                    experiment_id,
                ),
            )

    def list_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, model_name, status, best_loss, checkpoint_path,
                       metrics_json, created_at, updated_at
                FROM experiments
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def get_experiment(self, experiment_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, model_name, status, best_loss, checkpoint_path,
                       metrics_json, config_json, created_at, updated_at
                FROM experiments
                WHERE id = ?
                """,
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        experiment = self._row_to_experiment(row)
        experiment["config"] = json.loads(row["config_json"])
        return experiment

    def get_metrics(self, experiment_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT name, value, epoch, recorded_at
                FROM metrics
                WHERE experiment_id = ?
                ORDER BY id ASC
                """,
                (experiment_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> dict[str, Any]:
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        return {
            "id": row["id"],
            "name": row["name"],
            "model_name": row["model_name"],
            "status": row["status"],
            "best_loss": row["best_loss"],
            "checkpoint_path": row["checkpoint_path"],
            "metrics": metrics,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
