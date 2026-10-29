"""FastAPI REST endpoint for time-series inference."""

import asyncio
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import __version__
from src.dashboard_data import recent_series, validation_predictions
from src.data.lunar import lunar_now, lunar_on
from src.evaluation.lunar_lottery import read_lunar_analysis
from src.forecasting.spec import MAX_HORIZON, forecast_spec_from_config
from src.inference.predictor import ForecastBatch, Predictor
from src.models.registry import ModelRegistry, build_model
from src.serving.checkpoint import checkpoint_destination, prepare_runtime_checkpoint
from src.space.approaches import near_earth_report
from src.space.refresh import (
    ensure_space_snapshot,
    space_refresh_enabled,
    space_refresh_loop,
)
from src.space.solar_system import solar_system_report
from src.storage.experiments import ExperimentStore, db_path_from_config
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config
from src.utils.paths import REPO_ROOT, resolve_repo_path
from src.workspace.errors import WorkspaceError
from src.workspace.routes import router as workspace_router
from src.workspace.runtime import runtime
from src.workspace.service import (
    enqueue_member_config,
    member_experiment_detail,
    member_experiments,
    member_training_view,
    prepare_workspace,
    resolve_actor,
    run_queue,
    training_active,
    watch_authorized,
)
from src.youneek import parse_moment, snapshot

logger = logging.getLogger("uvicorn.error")

API_TOKEN_ENV = "NEURAL_FORGE_API_TOKEN"
CONFIG_ENV = "NEURAL_FORGE_CONFIG"

_state: dict[str, Any] = {}
_store: ExperimentStore | None = None
FRONTEND_DIR = REPO_ROOT / "frontend"
CONFIG_DIR = REPO_ROOT / "config"
TRAINING_STATUS_PATH = REPO_ROOT / "data" / "training_status.json"
_CONFIG_SUFFIXES = {".yaml", ".yml"}
_FINISHED_TRAINING = {"completed", "failed"}


def _checkpoint_file(cfg: dict[str, Any]) -> Path:
    return checkpoint_destination(cfg)


def _sequence_limit(model: Any, cfg: dict[str, Any]) -> int:
    limit = getattr(model, "max_seq_len", None)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        model_cfg = cfg.get("model")
        configured = (
            model_cfg.get("max_seq_len", 512) if isinstance(model_cfg, dict) else 512
        )
        limit = int(configured)
    return limit if limit >= 1 else 512


def load_model(config_path: str | Path = "config/config.yaml") -> None:
    """Load (or lazily initialise) the model into the global state dict."""
    global _store
    cfg = prepare_runtime_checkpoint(load_config(config_path))
    i_cfg = cfg["inference"]

    device_str = i_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    spec = forecast_spec_from_config(cfg)
    model = build_model(cfg)
    checkpoint_path = _checkpoint_file(cfg)

    if checkpoint_path.is_file():
        loaded_mtime_ns = checkpoint_path.stat().st_mtime_ns
        predictor = Predictor.from_checkpoint(
            model, checkpoint_path, device=device, spec=spec
        )
        logger.info("Loaded checkpoint from %s", checkpoint_path)
        _state["predictor"] = predictor
        _state["model"] = predictor.model
        _state["checkpoint_loaded"] = True
        _state["checkpoint_mtime_ns"] = loaded_mtime_ns
    else:
        logger.warning(
            "Checkpoint not found at '%s'. Using untrained model.", checkpoint_path
        )
        model.to(device)
        model.eval()
        _state["predictor"] = Predictor(model, device=device, spec=spec)
        _state["model"] = model
        _state["checkpoint_loaded"] = False
        _state["checkpoint_mtime_ns"] = None

    _state["checkpoint_path"] = checkpoint_path
    _state["device"] = device
    _state["config"] = cfg
    _state["training"] = {"status": "idle", "result": None}
    _store = ExperimentStore(db_path_from_config(cfg))
    runtime.config = cfg


def _json_ready(value: Any) -> Any:
    """Convert a training payload into plain JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = None
        else:
            if isinstance(converted, (bool, int, float, str)) or converted is None:
                return _json_ready(converted)
    raise TypeError(f"Training status is not JSON serializable: {type(value)!r}")


def _persist_training(payload: dict[str, Any]) -> None:
    try:
        encoded = json.dumps(_json_ready(payload), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError):
        logger.exception("Could not serialize training status")
        return
    path = TRAINING_STATUS_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        logger.exception("Could not persist training status to %s", path)


def _set_training(status: str, result: Any) -> None:
    payload = {"status": status, "result": result}
    _state["training"] = payload
    _persist_training(payload)


def _restore_training_status() -> None:
    """Reload the last finished run from disk.

    A ``running`` file means this process did not finish that job. Report it
    as failed so a restart does not claim a live run that no longer exists.
    """
    path = TRAINING_STATUS_PATH
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring unreadable training status at %s", path)
        return
    if not isinstance(payload, dict):
        return
    status = payload.get("status")
    result = payload.get("result")
    if status == "running":
        _set_training(
            "failed",
            {"error": "Training was interrupted before it finished."},
        )
        return
    if status in _FINISHED_TRAINING and (result is None or isinstance(result, dict)):
        _state["training"] = {"status": status, "result": result}


def _owner_training_busy() -> bool:
    training = _state.get("training")
    return isinstance(training, dict) and training.get("status") == "running"


def _actor_or_http(request: Request, authorization: str | None):
    try:
        return resolve_actor(request, authorization)
    except WorkspaceError as exc:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        raise HTTPException(
            status_code=exc.status, detail=exc.detail, headers=headers
        ) from exc


def _workspace_http(exc: WorkspaceError) -> HTTPException:
    headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
    return HTTPException(status_code=exc.status, detail=exc.detail, headers=headers)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    load_model(os.environ.get(CONFIG_ENV, "config/config.yaml"))
    _restore_training_status()
    prepare_workspace(
        _state.get("config") if isinstance(_state.get("config"), dict) else None
    )
    runtime.owner_busy = _owner_training_busy
    ensure_space_snapshot()
    stop = asyncio.Event()
    queue_stop = asyncio.Event()
    task: asyncio.Task[None] | None = None
    queue_task: asyncio.Task[None] | None = None
    if space_refresh_enabled():
        task = asyncio.create_task(space_refresh_loop(stop))
    if os.environ.get("NEURAL_FORGE_QUEUE_RUNNER", "1").strip() != "0":
        queue_task = asyncio.create_task(run_queue(queue_stop))
    try:
        yield
    finally:
        stop.set()
        queue_stop.set()
        for background in (task, queue_task):
            if background is None:
                continue
            background.cancel()
            try:
                await asyncio.wait_for(background, timeout=5)
            except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
                pass
        runtime.store = None
        runtime.member_job_id = None
        _state.clear()


app = FastAPI(
    title="Neural Forge",
    description="Production-ready transformer pipeline for time-series prediction.",
    version=__version__,
    lifespan=lifespan,
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

app.include_router(workspace_router)


class PredictRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description="Time-series input as a 2-D list of shape [seq_len, input_dim].",
        examples=[[[0.0], [0.1], [0.2]]],
    )
    horizon: int | None = Field(
        default=None,
        ge=1,
        le=MAX_HORIZON,
        description=(
            "Future steps to return. Omit it, or pass 1, to keep the historical "
            "one-step prediction list."
        ),
    )
    include_intervals: bool = Field(
        default=False,
        description="Include lower and upper bounds when this checkpoint has them.",
    )


class PredictResponse(BaseModel):
    prediction: list[float]
    forecast: list[list[float]] | None = None
    horizon: int | None = None
    strategy: str | None = None
    lower: list[list[float]] | None = None
    upper: list[list[float]] | None = None
    # First target channel, so the dashboard can plot a path and a band.
    # p50 is the point forecast. p10 and p90 are the served interval endpoints
    # (outer quantiles, or conformal bounds when the head has no quantiles).
    p10: list[float] | None = None
    p50: list[float] | None = None
    p90: list[float] | None = None
    quantiles: list[float] | None = None
    nominal_coverage: float | None = None
    validation_coverage: float | None = None
    interval_method: str | None = None


class ForecastResponse(PredictResponse):
    horizon: int
    strategy: str
    forecast: list[list[float]]


class TrainRequest(BaseModel):
    config_path: str = Field(
        default="config/config.yaml",
        description="Path to a .yaml or .yml file inside the config/ directory.",
    )


class TrainResponse(BaseModel):
    status: str
    message: str


def resolve_train_config(config_path: str) -> Path:
    """Resolve a training config and reject anything outside ``config/``.

    Relative paths are resolved from the repository root, not the process
    working directory, so a request cannot name arbitrary files on disk.
    """
    raw = config_path.strip()
    if not raw:
        raise HTTPException(status_code=422, detail="config_path must not be empty.")

    candidate = Path(raw)
    try:
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (REPO_ROOT / candidate).resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid config_path.") from exc

    config_root = CONFIG_DIR.resolve()
    try:
        relative = resolved.relative_to(config_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="config_path must point to a YAML file inside the config/ directory.",
        ) from exc
    if relative == Path("."):
        raise HTTPException(
            status_code=422,
            detail="config_path must point to a YAML file inside the config/ directory.",
        )

    if resolved.suffix.lower() not in _CONFIG_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail="config_path must be a .yaml or .yml file.",
        )
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Config file not found.")

    try:
        cfg = load_config(resolved)
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid YAML config: {exc}"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=422, detail="Could not read config file."
        ) from exc

    if (
        not isinstance(cfg, dict)
        or not isinstance(cfg.get("model"), dict)
        or not isinstance(cfg.get("training"), dict)
    ):
        raise HTTPException(
            status_code=422,
            detail="Config must include mapping sections 'model' and 'training'.",
        )
    return resolved


def _run_training(config_path: str) -> None:
    _set_training("running", None)
    try:
        result = TimeSeriesTrainer(config_path).train()
        # Persist before reloading the model. load_model resets the in-memory
        # status, and a crash during reload should still keep the finished run.
        _set_training("completed", result)
        load_model(config_path)
        _state["training"] = {"status": "completed", "result": result}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        _set_training("failed", {"error": str(exc)})


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return FileResponse(index)


@app.get("/health", tags=["monitoring"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info", tags=["model"])
def model_info() -> dict[str, Any]:
    model = _state.get("model")
    cfg = _state.get("config", {})
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    param_count = sum(p.numel() for p in model.parameters())
    stored_path = _state.get("checkpoint_path")
    checkpoint_path = (
        stored_path if isinstance(stored_path, Path) else _checkpoint_file(cfg)
    )
    loaded = bool(_state.get("checkpoint_loaded"))
    loaded_mtime_ns = _state.get("checkpoint_mtime_ns")
    stale = False
    if checkpoint_path.is_file():
        disk_mtime_ns = checkpoint_path.stat().st_mtime_ns
        if not loaded:
            stale = True
        elif isinstance(loaded_mtime_ns, int) and disk_mtime_ns > loaded_mtime_ns:
            stale = True
    return {
        "architecture": cfg.get("model", {}).get("name", "time_series_transformer"),
        "available_models": ModelRegistry.list_models(),
        "parameters": param_count,
        "device": str(_state.get("device")),
        "checkpoint_loaded": loaded,
        "checkpoint_stale": stale,
        "checkpoint_path": str(checkpoint_path),
        "input_dim": cfg.get("model", {}).get("input_dim"),
        "output_dim": cfg.get("model", {}).get("output_dim"),
        "horizon": _loaded_spec().horizon,
        "forecast_strategy": _loaded_spec().strategy,
        "quantiles": list(_loaded_spec().quantiles),
    }


def _require_store() -> ExperimentStore:
    if _store is None:
        raise HTTPException(
            status_code=503, detail="Experiment store is not available."
        )
    return _store


@app.get("/experiments", tags=["experiments"])
def list_experiments(
    request: Request,
    limit: int = Query(20, ge=1, le=500),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor_or_http(request, authorization)
    if actor.is_owner:
        return {"experiments": _require_store().list_experiments(limit=limit)}
    return {"experiments": member_experiments(actor, limit)}


POWERBALL_SCORECARD = "predictions/powerball/scorecard.json"


def read_powerball_scorecard(path: str | Path | None = None) -> dict[str, Any]:
    """Return the public scorecard, or an empty card when the file is absent."""
    destination = resolve_repo_path(POWERBALL_SCORECARD if path is None else path)
    if not destination.is_file():
        return {
            "available": False,
            "summary": "No Powerball scorecard has been published.",
        }
    loaded = json.loads(destination.read_text())
    if not isinstance(loaded, dict):
        raise HTTPException(
            status_code=500, detail="Powerball scorecard is not a JSON object."
        )
    payload = cast(dict[str, Any], loaded)
    payload["available"] = True
    return payload


@app.get("/space/near-earth", tags=["space"])
def space_near_earth(days: int = Query(730, ge=1, le=730)) -> dict[str, Any]:
    """Upcoming Earth and Moon approaches, reentries, and monthly forecasts.

    ``days`` is the inclusive offset from today (UTC) through which JPL
    close-approach dates are returned. The default, 730, is about 24 months.
    Model outlooks are always 24 months; the window only filters the catalog
    list and the JPL monthly counts drawn on the dashboard.
    """
    return near_earth_report(days=days)


@app.get("/space/solar-system", tags=["space"])
def space_solar_system(days: int = Query(730, ge=1, le=730)) -> dict[str, Any]:
    """Upcoming approaches, perihelia, and alignments, plus monthly forecasts.

    ``days`` is the inclusive offset from today (UTC). The default, 730, is
    about 24 months. Model outlooks stay 24 months; the window filters the
    catalog lists and the in-window counts drawn on the dashboard.
    """
    return solar_system_report(days=days)


@app.get("/powerball/scorecard", tags=["lottery"])
def powerball_scorecard() -> dict[str, Any]:
    """Logged tickets scored against published draws. A missing file is not an error."""
    return read_powerball_scorecard()


@app.get("/lunar", tags=["lunar"])
def lunar_calendar(when: str | None = Query(None, alias="date")) -> dict[str, Any]:
    """Moon phase and the Chinese lunisolar date.

    With no ``date``, the instant is the current UTC time. With ``date``
    (``YYYY-MM-DD``), the instant is 12:00 UTC on that civil date. Full, new,
    supermoon, and eclipse flags use the UTC date of the event. The series is
    computed offline.
    """
    try:
        if when is None or not when.strip():
            return lunar_now().as_dict()
        day = date.fromisoformat(when.strip()[:10])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date: {exc}") from exc
    return lunar_on(day).as_dict()


@app.get("/lottery/lunar-analysis", tags=["lottery"])
def lottery_lunar_analysis() -> dict[str, Any]:
    """Published check of lunar features against lottery draws.

    A missing file is not an error. The report is the output of
    ``python -m src.evaluation.lunar_lottery``.
    """
    try:
        return read_lunar_analysis()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/series/recent", tags=["dashboard"])
def series_recent(points: int = Query(96, ge=8, le=400)) -> dict[str, Any]:
    """Recent history for the loaded config, with a one-step backtest and forecast."""
    predictor: Predictor | None = _state.get("predictor")
    cfg = _state.get("config")
    if predictor is None or not isinstance(cfg, dict):
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        payload = recent_series(cfg, predictor, points=points)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=422, detail="Could not read the configured dataset."
        ) from exc
    if not _state.get("checkpoint_loaded"):
        extra = "The loaded model has no checkpoint, so this forecast is untrained."
        note = payload.get("note")
        payload["note"] = f"{note} {extra}".strip() if isinstance(note, str) else extra
    payload["checkpoint_loaded"] = bool(_state.get("checkpoint_loaded"))
    return payload


@app.get("/experiments/{experiment_id}/predictions", tags=["experiments"])
def experiment_predictions(
    experiment_id: str,
    request: Request,
    limit: int = Query(200, ge=1, le=2000),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Stored validation predictions, or a replay from the run's checkpoint.

    Integer ids are the owner's legacy runs. A visitor asking for one of those
    ids gets 404, the same as an unknown id.
    """
    actor = _actor_or_http(request, authorization)
    if experiment_id.isdigit():
        if not actor.is_owner:
            raise HTTPException(status_code=404, detail="Experiment not found.")
        payload = validation_predictions(
            _require_store(), int(experiment_id), limit=limit
        )
        if payload is None:
            raise HTTPException(status_code=404, detail="Experiment not found.")
        return payload
    detail = member_experiment_detail(actor, experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {
        "available": False,
        "note": "Workspace runs store metrics on the job.",
        "points": [],
        "experiment_id": experiment_id,
    }


def _comparison_models(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn ``persistence.mae`` keys into one row per model."""
    rows: dict[str, dict[str, Any]] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or "." not in key:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        name, metric = key.split(".", 1)
        row = rows.get(name)
        if row is None:
            row = {"name": name}
            rows[name] = row
        row[metric] = float(value)
    return list(rows.values())


@app.get("/comparison", tags=["evaluation"])
def latest_comparison(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Latest stored comparison for the owner. Visitors get an empty list."""
    actor = _actor_or_http(request, authorization)
    if not actor.is_owner:
        return {"models": []}
    experiments = _require_store().list_experiments(limit=200)
    for experiment in experiments:
        if experiment.get("model_name") != "comparison":
            continue
        metrics = experiment.get("metrics")
        if not isinstance(metrics, dict):
            continue
        models = _comparison_models(metrics)
        if not models:
            continue
        return {
            "experiment_id": experiment["id"],
            "name": experiment.get("name"),
            "models": models,
            "horizon": metrics.get("horizon"),
            "n_train": metrics.get("n_train"),
            "n_val": metrics.get("n_val"),
            "nominal_coverage": metrics.get("nominal_coverage"),
        }
    return {"models": []}


@app.get("/experiments/{experiment_id}", tags=["experiments"])
def get_experiment(
    experiment_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor_or_http(request, authorization)
    if experiment_id.isdigit():
        if not actor.is_owner:
            raise HTTPException(status_code=404, detail="Experiment not found.")
        store = _require_store()
        numeric_id = int(experiment_id)
        experiment = store.get_experiment(numeric_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found.")
        experiment["metric_history"] = store.get_metrics(numeric_id)
        return experiment
    detail = member_experiment_detail(actor, experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return detail


@app.get("/training/status", tags=["training"])
def training_status(
    request: Request,
    authorization: str | None = Header(default=None),
    x_neural_forge_watch: str | None = Header(default=None),
) -> dict[str, Any]:
    # The container Durable Object polls this URL on localhost to stay awake.
    # The public Worker strips the watch header, so visitors cannot use it.
    if watch_authorized(request.headers.get("host", ""), x_neural_forge_watch):
        status = "running" if training_active() else "idle"
        return {"status": status, "result": None}
    actor = _actor_or_http(request, authorization)
    if actor.is_owner:
        default: dict[str, Any] = {"status": "idle", "result": None}
        training = _state.get("training")
        return training if training is not None else default
    return member_training_view(actor)


def _training_enabled() -> bool:
    cfg = _state.get("config")
    if not isinstance(cfg, dict):
        return True
    api_cfg = cfg.get("api")
    if not isinstance(api_cfg, dict) or "training_enabled" not in api_cfg:
        return True
    enabled = api_cfg["training_enabled"]
    if not isinstance(enabled, bool):
        return True
    return enabled


@app.post("/train", response_model=TrainResponse, tags=["training"])
def start_training(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    authorization: str | None = Header(default=None),
) -> TrainResponse:
    actor = _actor_or_http(request, authorization)
    if not _training_enabled():
        raise HTTPException(
            status_code=403,
            detail="Training is disabled. Set api.training_enabled to true to allow it.",
        )
    config_path = resolve_train_config(body.config_path)
    if not actor.is_owner:
        # Visitors never enter the owner's checkpoint path. The queue writes a
        # private checkpoint and leaves checkpoints/best_model.pt alone.
        try:
            queued = enqueue_member_config(
                actor,
                base_config=str(config_path),
                config=_state.get("config")
                if isinstance(_state.get("config"), dict)
                else None,
            )
        except WorkspaceError as exc:
            raise _workspace_http(exc) from exc
        return TrainResponse(
            status=str(queued.get("status", "queued")),
            message="Training queued for this account.",
        )
    if _owner_training_busy() or (
        runtime.store is not None and runtime.store.has_pending_jobs()
    ):
        raise HTTPException(status_code=409, detail="Training already in progress.")
    if runtime.member_job_id:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    background_tasks.add_task(_run_training, str(config_path))
    _set_training("running", None)
    return TrainResponse(status="running", message="Training started in background.")


@app.get("/youneek/now", tags=["youneek"])
def youneek_now() -> dict[str, Any]:
    return snapshot(parse_moment(None)).as_dict()


@app.get("/youneek/convert", tags=["youneek"])
def youneek_convert(at: str) -> dict[str, Any]:
    try:
        moment = parse_moment(at)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid timestamp: {exc}"
        ) from exc
    return snapshot(moment).as_dict()


@app.get("/youneek/forecast/next-minute", tags=["youneek"])
def youneek_forecast_next_minute(at: str | None = None) -> dict[str, Any]:
    try:
        moment = parse_moment(at)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid timestamp: {exc}"
        ) from exc
    snap = snapshot(moment)
    return {"utc": snap.as_dict()["utc"], **snap.next_minute}


@app.get("/youneek/calendar", tags=["youneek"])
def youneek_calendar(at: str | None = None) -> dict[str, Any]:
    try:
        moment = parse_moment(at)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid timestamp: {exc}"
        ) from exc
    snap = snapshot(moment)
    return {"utc": snap.as_dict()["utc"], **snap.calendar}


@app.get("/youneek/lunar", tags=["youneek"])
def youneek_lunar(at: str | None = None) -> dict[str, Any]:
    try:
        moment = parse_moment(at)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid timestamp: {exc}"
        ) from exc
    snap = snapshot(moment)
    return {"utc": snap.as_dict()["utc"], **snap.lunar}


def _loaded_spec():
    predictor: Predictor | None = _state.get("predictor")
    if predictor is None:
        return forecast_spec_from_config(_state.get("config", {}))
    return predictor.spec


def _require_predictor() -> Predictor:
    predictor: Predictor | None = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predictor


def _validated_sequence(body: PredictRequest, predictor: Predictor) -> torch.Tensor:
    cfg: dict = _state["config"]
    expected_input_dim: int = cfg["model"]["input_dim"]
    if not body.sequence:
        raise HTTPException(status_code=422, detail="sequence must not be empty.")
    for index, step in enumerate(body.sequence):
        if any(not math.isfinite(value) for value in step):
            raise HTTPException(
                status_code=422,
                detail=f"sequence step {index} must contain only finite numbers.",
            )
    limit = _sequence_limit(predictor.model, cfg)
    if len(body.sequence) > limit:
        raise HTTPException(
            status_code=422,
            detail=(
                f"sequence length {len(body.sequence)} exceeds the maximum of {limit}."
            ),
        )
    for index, step in enumerate(body.sequence):
        if len(step) != expected_input_dim:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Each time step must have {expected_input_dim} feature(s); "
                    f"step {index} has {len(step)} feature(s)."
                ),
            )
    return torch.tensor(body.sequence, dtype=torch.float32)


def _first_channel(rows: list[list[float]]) -> list[float]:
    return [float(row[0]) for row in rows]


def _forecast_payload(
    result: ForecastBatch,
    *,
    include_path: bool,
    include_intervals: bool,
) -> dict[str, Any]:
    point = result.point[0].detach().cpu()
    payload: dict[str, Any] = {"prediction": point[0].tolist()}
    # A one-step /predict call stays {"prediction": [...]} unless intervals
    # were requested. /forecast always returns the full path.
    if include_path or result.horizon > 1:
        payload["forecast"] = point.tolist()
        payload["horizon"] = result.horizon
        payload["strategy"] = result.strategy
        payload["p50"] = _first_channel(payload["forecast"])
    if include_intervals and result.lower is not None and result.upper is not None:
        payload["lower"] = result.lower[0].detach().cpu().tolist()
        payload["upper"] = result.upper[0].detach().cpu().tolist()
        payload["p10"] = _first_channel(payload["lower"])
        payload["p90"] = _first_channel(payload["upper"])
        if "p50" not in payload:
            payload["p50"] = [float(payload["prediction"][0])]
        if result.quantiles:
            payload["quantiles"] = list(result.quantiles)
        if result.nominal_coverage is not None:
            payload["nominal_coverage"] = result.nominal_coverage
        if result.validation_coverage is not None:
            payload["validation_coverage"] = result.validation_coverage
        if result.interval_method is not None:
            payload["interval_method"] = result.interval_method
    return payload


def _run_forecast(body: PredictRequest, *, horizon: int) -> ForecastBatch:
    predictor = _require_predictor()
    sequence = _validated_sequence(body, predictor)
    try:
        return predictor.forecast(sequence, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/predict",
    response_model=PredictResponse,
    response_model_exclude_none=True,
    tags=["inference"],
)
def predict(body: PredictRequest) -> PredictResponse:
    steps = 1 if body.horizon is None else body.horizon
    result = _run_forecast(body, horizon=steps)
    return PredictResponse(
        **_forecast_payload(
            result,
            include_path=False,
            include_intervals=body.include_intervals or steps > 1,
        )
    )


@app.post(
    "/forecast",
    response_model=ForecastResponse,
    response_model_exclude_none=True,
    tags=["inference"],
)
def forecast(body: PredictRequest) -> ForecastResponse:
    steps = _loaded_spec().horizon if body.horizon is None else body.horizon
    result = _run_forecast(body, horizon=steps)
    return ForecastResponse(
        **_forecast_payload(result, include_path=True, include_intervals=True)
    )
