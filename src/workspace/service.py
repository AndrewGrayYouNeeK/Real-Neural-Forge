"""Sign-in, isolation, copy-on-write starters, and the training queue."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import re
import secrets
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import Request, Response

from src.accounts.actor import (
    SESSION_COOKIE,
    SESSION_SECONDS,
    Actor,
    owner_email,
)
from src.accounts.limits import Limits, load_limits
from src.accounts.passwords import hash_password, validate_password, verify_password
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.workspace.errors import WorkspaceError
from src.workspace.keys import user_dataset_key
from src.workspace.runner import execute_member_job
from src.workspace.runtime import runtime
from src.workspace.service_ids import new_id
from src.workspace.starters import Starter, load_starters, starter_by_id
from src.workspace.store import (
    SqlError,
    WorkspaceStore,
    dump_json,
    open_workspace,
    utc_now,
)

logger = logging.getLogger("uvicorn.error")

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_API_TOKEN_ENV = "NEURAL_FORGE_API_TOKEN"
_MIN_CSV_ROWS = 32
# Computed once so a missing account still pays the scrypt cost.
_DUMMY_HASH = hash_password("workspace-timing-pad")


def configured_api_token() -> str | None:
    raw = os.environ.get(_API_TOKEN_ENV, "").strip()
    return raw or None


def prepare_workspace(config: dict[str, Any] | None = None) -> None:
    """Open durable storage, publish shared starter bases, and record the owner."""
    del config
    directory = os.environ.get("NEURAL_FORGE_WORKSPACE_DIR", "").strip() or None
    store, error = open_workspace(directory)
    runtime.store = store
    runtime.error = error
    if store is None:
        if error:
            logger.warning("%s", error)
        return
    try:
        store.ensure_owner(owner_email())
        publish_starter_bases(store)
    except (OSError, SqlError, ValueError) as exc:
        runtime.store = None
        runtime.error = f"Workspace storage failed to start: {exc}"
        logger.exception("Workspace storage failed to start")


def publish_starter_bases(store: WorkspaceStore) -> None:
    """Upload starter files once. Existing base objects are left in place."""
    for starter in load_starters():
        if store.get_blob(starter.dataset_key()) is None:
            store.put_blob(starter.dataset_key(), starter.dataset_path.read_bytes())
        if store.get_blob(starter.model_key()) is None:
            if not starter.model_path.is_file():
                raise ValueError(f"Starter model is missing: {starter.model_path}")
            store.put_blob(starter.model_key(), starter.model_path.read_bytes())


def require_store() -> WorkspaceStore:
    if runtime.store is None:
        raise WorkspaceError(
            503,
            runtime.error or "Workspace storage is unavailable.",
        )
    return runtime.store


def limits_for(config: dict[str, Any] | None) -> Limits:
    return load_limits(config)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_email(email: str) -> str:
    if not isinstance(email, str):
        raise WorkspaceError(422, "Email must be a string.")
    cleaned = email.strip().lower()
    if len(cleaned) > 254 or not _EMAIL.fullmatch(cleaned):
        raise WorkspaceError(422, "Enter a valid email address.")
    return cleaned


def _secure_cookie(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "")
    if forwarded:
        return forwarded.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


def apply_session_cookie(response: Response, request: Request, token: str) -> None:
    """Host-only session cookie. No Domain attribute, so each public host is separate."""
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_SECONDS,
        httponly=True,
        secure=_secure_cookie(request),
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE,
        path="/",
        httponly=True,
        secure=_secure_cookie(request),
        samesite="lax",
    )


def _owner_actor(via: str) -> Actor:
    return Actor(user_id="owner", email=owner_email(), role="owner", via=via)


def resolve_actor(request: Request, authorization: str | None) -> Actor:
    """Return the caller. A wrong bearer token is rejected before any cookie."""
    expected = configured_api_token()
    if authorization:
        scheme, _, provided = authorization.partition(" ")
        if scheme.lower() == "bearer":
            token = provided.strip()
            if expected and token and secrets.compare_digest(token, expected):
                return _owner_actor("bearer")
            if expected:
                raise WorkspaceError(401, "Invalid or missing API token.")
    raw_cookie = request.cookies.get(SESSION_COOKIE)
    store = runtime.store
    if raw_cookie and store is not None:
        user = store.user_for_session(_token_hash(raw_cookie), utc_now())
        if user is not None:
            return Actor(
                user_id=str(user["id"]),
                email=str(user["email"]),
                role=str(user["role"]),
                via="session",
            )
        if expected:
            raise WorkspaceError(401, "Invalid or missing API token.")
    if expected is None:
        return _owner_actor("local")
    raise WorkspaceError(401, "Invalid or missing API token.")


def watch_authorized(host: str, header: str | None) -> bool:
    """True only for the container's own training watch on localhost."""
    expected = configured_api_token()
    if not expected or not header:
        return False
    hostname = host.split(",")[0].strip().split(":")[0].lower()
    if hostname not in {"localhost", "127.0.0.1"}:
        return False
    return secrets.compare_digest(header.strip(), expected)


def _issue_session(store: WorkspaceStore, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(seconds=SESSION_SECONDS)
    store.create_session(
        _token_hash(token),
        user_id,
        expires.replace(microsecond=0).isoformat(),
    )
    return token


def signup(email: str, password: str) -> tuple[Actor, str]:
    store = require_store()
    cleaned = _normalize_email(email)
    if cleaned == owner_email():
        raise WorkspaceError(409, "That email is reserved.")
    problem = validate_password(password, cleaned)
    if problem:
        raise WorkspaceError(422, problem)
    user_id = new_id("usr_")
    try:
        store.create_user(user_id, cleaned, hash_password(password))
    except SqlError as exc:
        raise WorkspaceError(409, "An account with that email already exists.") from exc
    try:
        seed_starters(store, user_id)
    except Exception:
        store.delete_user(user_id)
        logger.exception("Could not seed starters for %s", user_id)
        raise
    token = _issue_session(store, user_id)
    return Actor(user_id=user_id, email=cleaned, role="member", via="session"), token


def seed_starters(store: WorkspaceStore, user_id: str) -> None:
    """Point a new account at the shared bases. Bytes are not copied."""
    now = utc_now()
    for starter in load_starters():
        dataset = store.get_blob(starter.dataset_key())
        model = store.get_blob(starter.model_key())
        if dataset is None or model is None:
            raise WorkspaceError(503, f"Starter {starter.id} is not published.")
        store.insert_dataset(
            {
                "id": new_id("dst_"),
                "user_id": user_id,
                "name": starter.name,
                "bytes": len(dataset),
                "object_key": starter.dataset_key(),
                "shared": 1,
                "starter_id": starter.id,
                "created_at": now,
                "updated_at": now,
            }
        )
        store.insert_config(
            {
                "id": new_id("cfg_"),
                "user_id": user_id,
                "name": f"{starter.name} template",
                "starter_id": starter.id,
                "body": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        store.insert_model(
            {
                "id": new_id("mdl_"),
                "user_id": user_id,
                "name": f"{starter.name} starter",
                "object_key": starter.model_key(),
                "shared": 1,
                "starter_id": starter.id,
                "config_json": json.dumps(starter.config()),
                "metrics_json": None,
                "job_id": None,
                "created_at": now,
            }
        )


def login(email: str, password: str) -> tuple[Actor, str]:
    store = require_store()
    cleaned = _normalize_email(email)
    user = store.user_by_email(cleaned)
    stored_hash = (
        str(user["password_hash"])
        if user and user.get("password_hash")
        else _DUMMY_HASH
    )
    if (
        user is None
        or user.get("role") == "owner"
        or not verify_password(password, stored_hash)
    ):
        raise WorkspaceError(401, "Invalid email or password.")
    token = _issue_session(store, str(user["id"]))
    return (
        Actor(
            user_id=str(user["id"]),
            email=str(user["email"]),
            role=str(user["role"]),
            via="session",
        ),
        token,
    )


def login_owner() -> tuple[Actor, str]:
    """Turn the owner bearer token into a session cookie for the dashboard."""
    store = require_store()
    store.ensure_owner(owner_email())
    token = _issue_session(store, "owner")
    return _owner_actor("session"), token


def logout(request: Request) -> None:
    raw = request.cookies.get(SESSION_COOKIE)
    store = runtime.store
    if raw and store is not None:
        store.delete_session(_token_hash(raw))


def _csv_values(data: bytes) -> list[float]:
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise WorkspaceError(422, "Dataset must be UTF-8 CSV.") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "value" not in reader.fieldnames:
        raise WorkspaceError(422, "CSV must include a numeric value column.")
    values: list[float] = []
    for row in reader:
        raw = (row.get("value") or "").strip()
        try:
            number = float(raw)
        except ValueError as exc:
            raise WorkspaceError(422, "The value column must be numeric.") from exc
        if number != number or number in {float("inf"), float("-inf")}:
            raise WorkspaceError(422, "The value column must be finite.")
        values.append(number)
    return values


def _validate_upload(data: bytes, limits: Limits, *, minimum_rows: int) -> list[float]:
    if not data:
        raise WorkspaceError(422, "Dataset is empty.")
    if len(data) > limits.max_dataset_bytes:
        raise WorkspaceError(413, "Dataset is larger than the configured limit.")
    values = _csv_values(data)
    if len(values) < minimum_rows:
        raise WorkspaceError(422, f"CSV needs at least {minimum_rows} numeric rows.")
    return values


def _dataset_public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "bytes": row["bytes"],
        "shared": bool(row["shared"]),
        "starter_id": row.get("starter_id"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _model_public(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics_json")
    return {
        "id": row["id"],
        "name": row["name"],
        "shared": bool(row["shared"]),
        "starter_id": row.get("starter_id"),
        "metrics": json.loads(metrics)
        if isinstance(metrics, str) and metrics
        else None,
        "job_id": row.get("job_id"),
        "created_at": row["created_at"],
    }


def _config_public(row: dict[str, Any], body: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "starter_id": row.get("starter_id"),
        "shared": row.get("body") in (None, ""),
        "body": body,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def config_text(row: dict[str, Any]) -> str:
    body = row.get("body")
    if isinstance(body, str) and body.strip():
        return body
    starter_id = row.get("starter_id")
    if isinstance(starter_id, str):
        starter = starter_by_id(starter_id)
        if starter is not None:
            return starter.template_text()
    raise WorkspaceError(422, "Config template is missing.")


def list_datasets(actor: Actor) -> dict[str, Any]:
    store = require_store()
    return {
        "datasets": [_dataset_public(row) for row in store.list_datasets(actor.user_id)]
    }


def upload_dataset(
    actor: Actor, name: str, data: bytes, config: dict[str, Any] | None
) -> dict[str, Any]:
    store = require_store()
    limits = limits_for(config)
    if store.count_datasets(actor.user_id) >= limits.max_datasets_per_user:
        raise WorkspaceError(429, "This account has reached its dataset limit.")
    _validate_upload(data, limits, minimum_rows=_MIN_CSV_ROWS)
    dataset_id = new_id("dst_")
    key = user_dataset_key(actor.user_id, dataset_id)
    store.put_blob(key, data)
    now = utc_now()
    cleaned = _safe_name(name)
    row = {
        "id": dataset_id,
        "user_id": actor.user_id,
        "name": cleaned,
        "bytes": len(data),
        "object_key": key,
        "shared": 0,
        "starter_id": None,
        "created_at": now,
        "updated_at": now,
    }
    store.insert_dataset(row)
    return _dataset_public(row)


def _safe_name(name: str) -> str:
    base = Path(name or "dataset.csv").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "", base).strip()[:80]
    return cleaned or "dataset.csv"


def get_dataset(actor: Actor, dataset_id: str) -> dict[str, Any]:
    row = require_store().get_dataset(actor.user_id, dataset_id)
    if row is None:
        raise WorkspaceError(404, "Dataset not found.")
    return _dataset_public(row)


def replace_dataset(
    actor: Actor, dataset_id: str, data: bytes, config: dict[str, Any] | None
) -> dict[str, Any]:
    """Copy on write. A shared starter object is never overwritten."""
    store = require_store()
    row = store.get_dataset(actor.user_id, dataset_id)
    if row is None:
        raise WorkspaceError(404, "Dataset not found.")
    limits = limits_for(config)
    _validate_upload(data, limits, minimum_rows=_MIN_CSV_ROWS)
    if int(row["shared"]):
        key = user_dataset_key(actor.user_id, dataset_id)
    else:
        key = str(row["object_key"])
    store.put_blob(key, data)
    store.update_dataset_bytes(
        actor.user_id, dataset_id, object_key=key, size=len(data), shared=0
    )
    updated = store.get_dataset(actor.user_id, dataset_id)
    if updated is None:
        raise WorkspaceError(404, "Dataset not found.")
    return _dataset_public(updated)


def delete_dataset(actor: Actor, dataset_id: str) -> None:
    store = require_store()
    row = store.get_dataset(actor.user_id, dataset_id)
    if row is None:
        raise WorkspaceError(404, "Dataset not found.")
    if store.jobs_using_dataset(actor.user_id, dataset_id):
        raise WorkspaceError(409, "This dataset is used by a training job.")
    if not int(row["shared"]):
        store.delete_blob(str(row["object_key"]))
    store.delete_dataset_row(actor.user_id, dataset_id)


def list_configs(actor: Actor) -> dict[str, Any]:
    store = require_store()
    configs = []
    for row in store.list_configs(actor.user_id):
        configs.append(_config_public(row, config_text(row)))
    return {"configs": configs}


def update_config(
    actor: Actor, config_id: str, body: str, config: dict[str, Any] | None
) -> dict[str, Any]:
    """Store a private copy of the template. The shared file is not written."""
    store = require_store()
    row = store.get_config(actor.user_id, config_id)
    if row is None:
        raise WorkspaceError(404, "Config not found.")
    limits = limits_for(config)
    if len(body.encode("utf-8")) > limits.max_config_bytes:
        raise WorkspaceError(413, "Config is larger than the configured limit.")
    try:
        loaded = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        raise WorkspaceError(422, "Config is not valid YAML.") from exc
    if not isinstance(loaded, dict) or not isinstance(loaded.get("model"), dict):
        raise WorkspaceError(422, "Config must include a model mapping.")
    if not isinstance(loaded.get("training"), dict):
        raise WorkspaceError(422, "Config must include a training mapping.")
    store.update_config_body(actor.user_id, config_id, body)
    updated = store.get_config(actor.user_id, config_id)
    if updated is None:
        raise WorkspaceError(404, "Config not found.")
    return _config_public(updated, body)


def list_models(actor: Actor) -> dict[str, Any]:
    store = require_store()
    return {"models": [_model_public(row) for row in store.list_models(actor.user_id)]}


def list_starter_cards(actor: Actor) -> dict[str, Any]:
    """The caller's seeded kits. Another account's rows are not included."""
    store = require_store()
    cards = []
    for starter in load_starters():
        dataset = store.dataset_for_starter(actor.user_id, starter.id)
        model = store.model_for_starter(actor.user_id, starter.id)
        config_row = store.config_for_starter(actor.user_id, starter.id)
        if dataset is None or model is None or config_row is None:
            continue
        cards.append(
            {
                "id": starter.id,
                "name": starter.name,
                "summary": starter.summary,
                "source": starter.source,
                "dataset_id": dataset["id"],
                "model_id": model["id"],
                "config_id": config_row["id"],
                "shared": bool(dataset["shared"]) and bool(model["shared"]),
            }
        )
    return {"starters": cards}


def _owned_starter(
    actor: Actor, starter_id: str
) -> tuple[WorkspaceStore, Starter, dict[str, Any], dict[str, Any]]:
    store = require_store()
    starter = starter_by_id(starter_id)
    if starter is None:
        raise WorkspaceError(404, "Starter not found.")
    dataset = store.dataset_for_starter(actor.user_id, starter.id)
    model = store.model_for_starter(actor.user_id, starter.id)
    if dataset is None or model is None:
        raise WorkspaceError(404, "Starter not found.")
    return store, starter, dataset, model


def _window_from_dataset(data: bytes, seq_len: int) -> list[list[float]]:
    values = _csv_values(data)
    if len(values) < seq_len:
        raise WorkspaceError(422, "The sample is shorter than the model window.")
    return [[value] for value in values[-seq_len:]]


def _predict_with_model(
    store: WorkspaceStore,
    actor: Actor,
    model: dict[str, Any],
    sequence: list[list[float]],
    *,
    starter_id: str | None,
    limits: Limits,
) -> dict[str, Any]:
    if store.count_predictions(actor.user_id) >= limits.max_predictions_stored:
        raise WorkspaceError(
            429, "This account has reached its saved prediction limit."
        )
    if len(sequence) > limits.max_prediction_points:
        raise WorkspaceError(422, "Sequence is longer than the configured limit.")
    try:
        cfg = json.loads(str(model["config_json"]))
    except json.JSONDecodeError as exc:
        raise WorkspaceError(500, "Model config is unreadable.") from exc
    if not isinstance(cfg, dict):
        raise WorkspaceError(500, "Model config is unreadable.")
    input_dim = int(cfg.get("model", {}).get("input_dim", 1) or 1)
    for index, step in enumerate(sequence):
        if len(step) != input_dim:
            raise WorkspaceError(
                422,
                f"Each step must have {input_dim} feature(s); step {index} has {len(step)}.",
            )
        if any(
            value != value or value in {float("inf"), float("-inf")} for value in step
        ):
            raise WorkspaceError(422, "Sequence values must be finite.")
    blob = store.get_blob(str(model["object_key"]))
    if blob is None:
        raise WorkspaceError(404, "Model weights are missing.")
    with tempfile.TemporaryDirectory(prefix="nf-predict-") as temporary:
        path = Path(temporary) / "model.pt"
        path.write_bytes(blob)
        built = build_model(cfg, ensemble=False)
        predictor = Predictor.from_checkpoint(built, path, device="cpu")
        tensor = torch.tensor(sequence, dtype=torch.float32)
        forecast = predictor.forecast(tensor, horizon=1)
    point = forecast.point[0].detach().cpu()
    prediction = point[0].tolist()
    record_id = new_id("prd_")
    result = {"prediction": prediction}
    store.insert_prediction(
        {
            "id": record_id,
            "user_id": actor.user_id,
            "model_id": model["id"],
            "starter_id": starter_id,
            "request_json": dump_json({"sequence": sequence}),
            "result_json": dump_json(result),
            "created_at": utc_now(),
        }
    )
    return {
        "id": record_id,
        "model_id": model["id"],
        "starter_id": starter_id,
        **result,
    }


def predict_starter(
    actor: Actor, starter_id: str, config: dict[str, Any] | None
) -> dict[str, Any]:
    store, starter, dataset, model = _owned_starter(actor, starter_id)
    blob = store.get_blob(str(dataset["object_key"]))
    if blob is None:
        raise WorkspaceError(404, "Dataset bytes are missing.")
    seq_len = int(starter.config().get("data", {}).get("seq_len", 16) or 16)
    sequence = _window_from_dataset(blob, seq_len)
    return _predict_with_model(
        store,
        actor,
        model,
        sequence,
        starter_id=starter.id,
        limits=limits_for(config),
    )


def predict_model(
    actor: Actor,
    model_id: str,
    sequence: list[list[float]],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    store = require_store()
    model = store.get_model(actor.user_id, model_id)
    if model is None:
        raise WorkspaceError(404, "Model not found.")
    return _predict_with_model(
        store,
        actor,
        model,
        sequence,
        starter_id=model.get("starter_id")
        if isinstance(model.get("starter_id"), str)
        else None,
        limits=limits_for(config),
    )


def list_predictions(actor: Actor) -> dict[str, Any]:
    store = require_store()
    rows = []
    for row in store.list_predictions(actor.user_id, 50):
        result = json.loads(row["result_json"]) if row.get("result_json") else {}
        rows.append(
            {
                "id": row["id"],
                "model_id": row.get("model_id"),
                "starter_id": row.get("starter_id"),
                "result": result,
                "created_at": row["created_at"],
            }
        )
    return {"predictions": rows}


def get_prediction(actor: Actor, prediction_id: str) -> dict[str, Any]:
    row = require_store().get_prediction(actor.user_id, prediction_id)
    if row is None:
        raise WorkspaceError(404, "Prediction not found.")
    return {
        "id": row["id"],
        "model_id": row.get("model_id"),
        "starter_id": row.get("starter_id"),
        "result": json.loads(row["result_json"]),
        "created_at": row["created_at"],
    }


def _enqueue(
    actor: Actor,
    *,
    name: str,
    dataset_id: str | None,
    config_id: str | None,
    starter_id: str | None,
    base_config: str | None,
    limits: Limits,
) -> dict[str, Any]:
    store = require_store()
    if store.count_active_jobs(actor.user_id) >= limits.max_active_jobs_per_user:
        raise WorkspaceError(409, "You already have a training job.")
    if store.count_queued_jobs() >= limits.global_queue_limit:
        raise WorkspaceError(429, "The training queue is full. Try again later.")
    if dataset_id is not None and store.get_dataset(actor.user_id, dataset_id) is None:
        raise WorkspaceError(404, "Dataset not found.")
    if config_id is not None and store.get_config(actor.user_id, config_id) is None:
        raise WorkspaceError(404, "Config not found.")
    job_id = new_id("job_")
    now = utc_now()
    row = {
        "id": job_id,
        "user_id": actor.user_id,
        "status": "queued",
        "name": name,
        "dataset_id": dataset_id,
        "config_id": config_id,
        "starter_id": starter_id,
        "base_config": base_config,
        "message": "Queued.",
        "error": None,
        "result_json": None,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
    }
    store.insert_job(row)
    return {"id": job_id, "status": "queued", "name": name, "starter_id": starter_id}


def enqueue_starter(
    actor: Actor, starter_id: str, config: dict[str, Any] | None
) -> dict[str, Any]:
    _store, starter, dataset, _model = _owned_starter(actor, starter_id)
    config_row = _store.config_for_starter(actor.user_id, starter.id)
    if config_row is None:
        raise WorkspaceError(404, "Starter not found.")
    return _enqueue(
        actor,
        name=f"Train {starter.name}",
        dataset_id=str(dataset["id"]),
        config_id=str(config_row["id"]),
        starter_id=starter.id,
        base_config=None,
        limits=limits_for(config),
    )


def enqueue_member_config(
    actor: Actor,
    *,
    base_config: str,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Queue a run of a shipped config. Paths inside it are overridden later."""
    return _enqueue(
        actor,
        name=Path(base_config).name,
        dataset_id=None,
        config_id=None,
        starter_id=None,
        base_config=base_config,
        limits=limits_for(config),
    )


def training_active() -> bool:
    """True when the single global runner should stay awake."""
    if runtime.owner_busy() or runtime.member_job_id:
        return True
    store = runtime.store
    return bool(store is not None and store.has_pending_jobs())


def member_training_view(actor: Actor) -> dict[str, Any]:
    store = require_store()
    job = store.latest_job(actor.user_id)
    if job is None:
        return {"status": "idle", "result": None}
    result: Any = None
    if job.get("result_json"):
        result = json.loads(str(job["result_json"]))
    elif job.get("error"):
        result = {"error": job["error"]}
    return {"status": job["status"], "result": result, "job_id": job["id"]}


def list_jobs(actor: Actor) -> dict[str, Any]:
    store = require_store()
    jobs = []
    for row in store.list_jobs(actor.user_id, 50):
        result = json.loads(row["result_json"]) if row.get("result_json") else None
        jobs.append(
            {
                "id": row["id"],
                "name": row["name"],
                "status": row["status"],
                "starter_id": row.get("starter_id"),
                "dataset_id": row.get("dataset_id"),
                "error": row.get("error"),
                "result": result,
                "created_at": row["created_at"],
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"),
            }
        )
    return {"jobs": jobs}


def get_job(actor: Actor, job_id: str) -> dict[str, Any]:
    row = require_store().get_job(actor.user_id, job_id)
    if row is None:
        raise WorkspaceError(404, "Job not found.")
    result = json.loads(row["result_json"]) if row.get("result_json") else None
    return {
        "id": row["id"],
        "name": row["name"],
        "status": row["status"],
        "starter_id": row.get("starter_id"),
        "error": row.get("error"),
        "result": result,
        "created_at": row["created_at"],
        "finished_at": row.get("finished_at"),
    }


def cancel_job(actor: Actor, job_id: str) -> dict[str, Any]:
    store = require_store()
    if store.get_job(actor.user_id, job_id) is None:
        raise WorkspaceError(404, "Job not found.")
    changed = store.cancel_queued_job(actor.user_id, job_id, utc_now())
    if changed == 0:
        raise WorkspaceError(409, "Only a queued job can be cancelled.")
    return get_job(actor, job_id)


def job_as_experiment(row: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(row["result_json"]) if row.get("result_json") else {}
    if not isinstance(result, dict):
        result = {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else None
    return {
        "id": row["id"],
        "name": row["name"],
        "model_name": "time_series_transformer",
        "status": row["status"],
        "best_loss": result.get("best_loss"),
        "metrics": metrics,
        "created_at": row["created_at"],
        "updated_at": row.get("finished_at") or row["created_at"],
        "starter_id": row.get("starter_id"),
    }


def member_experiments(actor: Actor, limit: int) -> list[dict[str, Any]]:
    store = require_store()
    return [job_as_experiment(row) for row in store.list_jobs(actor.user_id, limit)]


def member_experiment_detail(actor: Actor, experiment_id: str) -> dict[str, Any] | None:
    store = require_store()
    row = store.get_job(actor.user_id, experiment_id)
    if row is None:
        return None
    detail = job_as_experiment(row)
    result = json.loads(row["result_json"]) if row.get("result_json") else {}
    history = result.get("history") if isinstance(result, dict) else None
    metric_history: list[dict[str, Any]] = []
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            epoch = item.get("epoch")
            for name in ("train_loss", "val_loss"):
                if name in item:
                    metric_history.append(
                        {"name": name, "epoch": epoch, "value": item[name]}
                    )
    detail["metric_history"] = metric_history
    return detail


async def run_queue(stop: asyncio.Event) -> None:
    """Pull the global queue one job at a time."""
    while not stop.is_set():
        try:
            await _queue_tick()
        except Exception:
            logger.exception("Workspace queue tick failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except TimeoutError:
            continue


async def _queue_tick() -> None:
    if runtime.owner_busy() or runtime.member_job_id:
        return
    store = runtime.store
    if store is None:
        return
    job = await asyncio.to_thread(store.claim_next_job, utc_now())
    if job is None:
        return
    runtime.member_job_id = str(job["id"])
    try:
        await asyncio.to_thread(execute_member_job, store, job)
    finally:
        runtime.member_job_id = None
