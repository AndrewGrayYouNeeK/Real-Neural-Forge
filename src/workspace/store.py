"""Workspace metadata and bytes.

Local development and tests use SQLite plus files under a workspace directory.
Production (``NEURAL_FORGE_STORAGE=cloudflare``) uses the Worker binding proxy,
which stores rows in D1 and bytes in R2. That mode never creates a local
database for visitor work.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import resolve_repo_path
from src.workspace.bindings import BindingTransport, StorageError
from src.workspace.keys import valid_object_key

_SCHEMA = Path(__file__).with_name("schema.sql")
_JOB_COLUMNS = (
    "id",
    "user_id",
    "status",
    "name",
    "dataset_id",
    "config_id",
    "starter_id",
    "base_config",
    "message",
    "error",
    "result_json",
    "created_at",
    "started_at",
    "finished_at",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sql_statements(script: str | None = None) -> list[str]:
    """Split the workspace schema into individual statements."""
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


class SqlError(Exception):
    """A metadata statement failed."""


class SqliteSql:
    """One short-lived connection per call."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                cursor = connection.execute(sql, tuple(params))
                if cursor.description is None:
                    return []
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.IntegrityError as exc:
            raise SqlError(str(exc)) from exc

    def script(self, statements: Sequence[str]) -> None:
        with self._connect() as connection:
            for statement in statements:
                connection.execute(statement)


class BindingSql:
    """D1 through the Worker proxy."""

    def __init__(self, transport: BindingTransport) -> None:
        self.transport = transport

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        try:
            results = self.transport.query([(sql, params)])
        except StorageError as exc:
            raise SqlError(str(exc)) from exc
        rows = results[0].get("rows") if results else []
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def script(self, statements: Sequence[str]) -> None:
        try:
            self.transport.query([(statement, ()) for statement in statements])
        except StorageError as exc:
            raise SqlError(str(exc)) from exc


class LocalBlobs:
    """Files under a root. Keys cannot escape the root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not valid_object_key(key):
            raise SqlError("Unsafe object key.")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise SqlError("Unsafe object key.")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink()


class BindingBlobs:
    """R2 through the Worker proxy."""

    def __init__(self, transport: BindingTransport) -> None:
        self.transport = transport

    def put(self, key: str, data: bytes) -> None:
        self.transport.put_object(key, data)

    def get(self, key: str) -> bytes | None:
        return self.transport.get_object(key)

    def delete(self, key: str) -> None:
        self.transport.delete_object(key)


class WorkspaceStore:
    """Metadata plus blobs. Every read that a visitor triggers is user-scoped."""

    def __init__(
        self, sql: SqliteSql | BindingSql, blobs: LocalBlobs | BindingBlobs
    ) -> None:
        self.sql = sql
        self.blobs = blobs

    def migrate(self) -> None:
        self.sql.script(sql_statements())

    def put_blob(self, key: str, data: bytes) -> None:
        self.blobs.put(key, data)

    def get_blob(self, key: str) -> bytes | None:
        return self.blobs.get(key)

    def delete_blob(self, key: str) -> None:
        if key.startswith("bases/"):
            return
        self.blobs.delete(key)

    def ensure_owner(self, email: str) -> None:
        self.sql.query(
            """
            INSERT INTO users (id, email, password_hash, role, created_at)
            VALUES ('owner', ?, NULL, 'owner', ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (email, utc_now()),
        )

    def create_user(
        self, user_id: str, email: str, password_hash: str
    ) -> dict[str, Any]:
        now = utc_now()
        self.sql.query(
            """
            INSERT INTO users (id, email, password_hash, role, created_at)
            VALUES (?, ?, ?, 'member', ?)
            """,
            (user_id, email, password_hash, now),
        )
        return {
            "id": user_id,
            "email": email,
            "role": "member",
            "created_at": now,
        }

    def delete_user(self, user_id: str) -> None:
        if user_id == "owner":
            return
        for table in (
            "predictions",
            "jobs",
            "models",
            "configs",
            "datasets",
            "sessions",
        ):
            self.sql.query(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
        self.sql.query("DELETE FROM users WHERE id = ? AND role != 'owner'", (user_id,))

    def user_by_email(self, email: str) -> dict[str, Any] | None:
        rows = self.sql.query("SELECT * FROM users WHERE email = ?", (email,))
        return rows[0] if rows else None

    def user_by_id(self, user_id: str) -> dict[str, Any] | None:
        rows = self.sql.query("SELECT * FROM users WHERE id = ?", (user_id,))
        return rows[0] if rows else None

    def create_session(self, token_hash: str, user_id: str, expires_at: str) -> None:
        self.sql.query(
            """
            INSERT INTO sessions (token_hash, user_id, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (token_hash, user_id, expires_at, utc_now()),
        )

    def user_for_session(self, token_hash: str, now: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            """
            SELECT users.id, users.email, users.role
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ? AND sessions.expires_at > ?
            """,
            (token_hash, now),
        )
        return rows[0] if rows else None

    def delete_session(self, token_hash: str) -> None:
        self.sql.query("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    def insert_dataset(self, row: dict[str, Any]) -> None:
        self.sql.query(
            """
            INSERT INTO datasets
            (id, user_id, name, bytes, object_key, shared, starter_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["user_id"],
                row["name"],
                row["bytes"],
                row["object_key"],
                row["shared"],
                row.get("starter_id"),
                row["created_at"],
                row["updated_at"],
            ),
        )

    def list_datasets(self, user_id: str) -> list[dict[str, Any]]:
        return self.sql.query(
            """
            SELECT * FROM datasets WHERE user_id = ? ORDER BY created_at ASC
            """,
            (user_id,),
        )

    def get_dataset(self, user_id: str, dataset_id: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            "SELECT * FROM datasets WHERE user_id = ? AND id = ?",
            (user_id, dataset_id),
        )
        return rows[0] if rows else None

    def dataset_for_starter(
        self, user_id: str, starter_id: str
    ) -> dict[str, Any] | None:
        rows = self.sql.query(
            """
            SELECT * FROM datasets
            WHERE user_id = ? AND starter_id = ?
            ORDER BY created_at ASC LIMIT 1
            """,
            (user_id, starter_id),
        )
        return rows[0] if rows else None

    def count_datasets(self, user_id: str) -> int:
        rows = self.sql.query(
            "SELECT COUNT(*) AS n FROM datasets WHERE user_id = ?",
            (user_id,),
        )
        return int(rows[0]["n"]) if rows else 0

    def update_dataset_bytes(
        self, user_id: str, dataset_id: str, *, object_key: str, size: int, shared: int
    ) -> None:
        self.sql.query(
            """
            UPDATE datasets
            SET object_key = ?, bytes = ?, shared = ?, updated_at = ?
            WHERE user_id = ? AND id = ?
            """,
            (object_key, size, shared, utc_now(), user_id, dataset_id),
        )

    def delete_dataset_row(self, user_id: str, dataset_id: str) -> None:
        self.sql.query(
            "DELETE FROM datasets WHERE user_id = ? AND id = ?",
            (user_id, dataset_id),
        )

    def insert_config(self, row: dict[str, Any]) -> None:
        self.sql.query(
            """
            INSERT INTO configs (id, user_id, name, starter_id, body, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["user_id"],
                row["name"],
                row.get("starter_id"),
                row.get("body"),
                row["created_at"],
                row["updated_at"],
            ),
        )

    def list_configs(self, user_id: str) -> list[dict[str, Any]]:
        return self.sql.query(
            "SELECT * FROM configs WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,),
        )

    def get_config(self, user_id: str, config_id: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            "SELECT * FROM configs WHERE user_id = ? AND id = ?",
            (user_id, config_id),
        )
        return rows[0] if rows else None

    def config_for_starter(
        self, user_id: str, starter_id: str
    ) -> dict[str, Any] | None:
        rows = self.sql.query(
            """
            SELECT * FROM configs
            WHERE user_id = ? AND starter_id = ?
            ORDER BY created_at ASC LIMIT 1
            """,
            (user_id, starter_id),
        )
        return rows[0] if rows else None

    def update_config_body(self, user_id: str, config_id: str, body: str) -> None:
        self.sql.query(
            """
            UPDATE configs SET body = ?, updated_at = ? WHERE user_id = ? AND id = ?
            """,
            (body, utc_now(), user_id, config_id),
        )

    def insert_model(self, row: dict[str, Any]) -> None:
        self.sql.query(
            """
            INSERT INTO models
            (id, user_id, name, object_key, shared, starter_id, config_json,
             metrics_json, job_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["user_id"],
                row["name"],
                row["object_key"],
                row["shared"],
                row.get("starter_id"),
                row["config_json"],
                row.get("metrics_json"),
                row.get("job_id"),
                row["created_at"],
            ),
        )

    def list_models(self, user_id: str) -> list[dict[str, Any]]:
        return self.sql.query(
            "SELECT * FROM models WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,),
        )

    def get_model(self, user_id: str, model_id: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            "SELECT * FROM models WHERE user_id = ? AND id = ?",
            (user_id, model_id),
        )
        return rows[0] if rows else None

    def model_for_starter(self, user_id: str, starter_id: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            """
            SELECT * FROM models
            WHERE user_id = ? AND starter_id = ? AND shared = 1
            ORDER BY created_at ASC LIMIT 1
            """,
            (user_id, starter_id),
        )
        return rows[0] if rows else None

    def count_models(self, user_id: str) -> int:
        rows = self.sql.query(
            "SELECT COUNT(*) AS n FROM models WHERE user_id = ?",
            (user_id,),
        )
        return int(rows[0]["n"]) if rows else 0

    def insert_job(self, row: dict[str, Any]) -> None:
        self.sql.query(
            f"""
            INSERT INTO jobs ({", ".join(_JOB_COLUMNS)})
            VALUES ({", ".join("?" for _ in _JOB_COLUMNS)})
            """,
            tuple(row.get(column) for column in _JOB_COLUMNS),
        )

    def list_jobs(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        return self.sql.query(
            """
            SELECT * FROM jobs WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, limit),
        )

    def get_job(self, user_id: str, job_id: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            "SELECT * FROM jobs WHERE user_id = ? AND id = ?",
            (user_id, job_id),
        )
        return rows[0] if rows else None

    def count_active_jobs(self, user_id: str) -> int:
        rows = self.sql.query(
            """
            SELECT COUNT(*) AS n FROM jobs
            WHERE user_id = ? AND status IN ('queued', 'running')
            """,
            (user_id,),
        )
        return int(rows[0]["n"]) if rows else 0

    def count_queued_jobs(self) -> int:
        rows = self.sql.query(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'",
        )
        return int(rows[0]["n"]) if rows else 0

    def has_pending_jobs(self) -> bool:
        rows = self.sql.query(
            """
            SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running')
            """,
        )
        return bool(rows and int(rows[0]["n"]) > 0)

    def claim_next_job(self, now: str) -> dict[str, Any] | None:
        columns = ", ".join(_JOB_COLUMNS)
        rows = self.sql.query(
            f"""
            UPDATE jobs
            SET status = 'running', started_at = ?
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
            )
            AND status = 'queued'
            RETURNING {columns}
            """,
            (now,),
        )
        return rows[0] if rows else None

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None,
        result_json: str | None,
        finished_at: str,
    ) -> None:
        self.sql.query(
            """
            UPDATE jobs
            SET status = ?, error = ?, result_json = ?, finished_at = ?
            WHERE id = ?
            """,
            (status, error, result_json, finished_at, job_id),
        )

    def cancel_queued_job(self, user_id: str, job_id: str, finished_at: str) -> int:
        rows = self.sql.query(
            """
            UPDATE jobs
            SET status = 'cancelled', finished_at = ?, message = 'Cancelled.'
            WHERE user_id = ? AND id = ? AND status = 'queued'
            RETURNING id
            """,
            (finished_at, user_id, job_id),
        )
        return len(rows)

    def latest_job(self, user_id: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            """
            SELECT * FROM jobs WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (user_id,),
        )
        return rows[0] if rows else None

    def insert_prediction(self, row: dict[str, Any]) -> None:
        self.sql.query(
            """
            INSERT INTO predictions
            (id, user_id, model_id, starter_id, request_json, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["user_id"],
                row.get("model_id"),
                row.get("starter_id"),
                row["request_json"],
                row["result_json"],
                row["created_at"],
            ),
        )

    def list_predictions(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        return self.sql.query(
            """
            SELECT * FROM predictions WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, limit),
        )

    def get_prediction(self, user_id: str, prediction_id: str) -> dict[str, Any] | None:
        rows = self.sql.query(
            "SELECT * FROM predictions WHERE user_id = ? AND id = ?",
            (user_id, prediction_id),
        )
        return rows[0] if rows else None

    def count_predictions(self, user_id: str) -> int:
        rows = self.sql.query(
            "SELECT COUNT(*) AS n FROM predictions WHERE user_id = ?",
            (user_id,),
        )
        return int(rows[0]["n"]) if rows else 0

    def jobs_using_dataset(self, user_id: str, dataset_id: str) -> int:
        rows = self.sql.query(
            """
            SELECT COUNT(*) AS n FROM jobs
            WHERE user_id = ? AND dataset_id = ? AND status IN ('queued', 'running')
            """,
            (user_id, dataset_id),
        )
        return int(rows[0]["n"]) if rows else 0


def open_workspace(
    directory: str | Path | None = None,
) -> tuple[WorkspaceStore | None, str | None]:
    """Open the configured store.

    Cloudflare mode returns ``(None, message)`` when the binding proxy is not
    configured, and does not create a local database. Local mode always uses
    SQLite under ``directory``.
    """
    mode = os.environ.get("NEURAL_FORGE_STORAGE", "local").strip().lower() or "local"
    if mode == "cloudflare":
        base_url = os.environ.get("NEURAL_FORGE_BINDINGS_URL", "").strip()
        secret = os.environ.get("NEURAL_FORGE_STORAGE_SECRET", "").strip()
        if not base_url or not secret:
            return None, (
                "Durable workspace storage is not configured. "
                "Set NEURAL_FORGE_BINDINGS_URL and NEURAL_FORGE_STORAGE_SECRET."
            )
        transport = BindingTransport(base_url, secret)
        store = WorkspaceStore(BindingSql(transport), BindingBlobs(transport))
        try:
            store.migrate()
        except (SqlError, StorageError, OSError) as exc:
            return None, f"Durable workspace storage is unavailable: {exc}"
        return store, None
    if mode != "local":
        return None, f"Unknown NEURAL_FORGE_STORAGE value {mode!r}."
    raw = (
        directory
        if directory is not None
        else os.environ.get("NEURAL_FORGE_WORKSPACE_DIR", "data/workspaces")
    )
    root = resolve_repo_path(raw)
    store = WorkspaceStore(
        SqliteSql(root / "workspaces.db"), LocalBlobs(root / "objects")
    )
    store.migrate()
    return store, None


def dump_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
