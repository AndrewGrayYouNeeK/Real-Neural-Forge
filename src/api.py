"""FastAPI REST endpoint for time-series inference."""

import logging
import math
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import __version__
from src.inference.predictor import Predictor
from src.models.registry import ModelRegistry, build_model
from src.storage.experiments import ExperimentStore, db_path_from_config
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config
from src.utils.paths import REPO_ROOT, resolve_repo_path
from src.youneek import parse_moment, snapshot

logger = logging.getLogger("uvicorn.error")

API_TOKEN_ENV = "NEURAL_FORGE_API_TOKEN"

_state: dict[str, Any] = {}
_store: ExperimentStore | None = None
FRONTEND_DIR = REPO_ROOT / "frontend"
CONFIG_DIR = REPO_ROOT / "config"
_CONFIG_SUFFIXES = {".yaml", ".yml"}


def _checkpoint_file(cfg: dict[str, Any]) -> Path:
    inference = cfg.get("inference")
    raw = inference.get("checkpoint_path") if isinstance(inference, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        raw = "checkpoints/best_model.pt"
    return resolve_repo_path(raw)


def _sequence_limit(model: Any, cfg: dict[str, Any]) -> int:
    limit = getattr(model, "max_seq_len", None)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        model_cfg = cfg.get("model")
        configured = model_cfg.get("max_seq_len", 512) if isinstance(model_cfg, dict) else 512
        limit = int(configured)
    return limit if limit >= 1 else 512


def load_model(config_path: str | Path = "config/config.yaml") -> None:
    """Load (or lazily initialise) the model into the global state dict."""
    global _store
    cfg = load_config(config_path)
    i_cfg = cfg["inference"]

    device_str = i_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    model = build_model(cfg)
    checkpoint_path = _checkpoint_file(cfg)

    if checkpoint_path.is_file():
        loaded_mtime_ns = checkpoint_path.stat().st_mtime_ns
        predictor = Predictor.from_checkpoint(model, checkpoint_path, device=device)
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
        _state["predictor"] = Predictor(model, device=device)
        _state["model"] = model
        _state["checkpoint_loaded"] = False
        _state["checkpoint_mtime_ns"] = None

    _state["checkpoint_path"] = checkpoint_path
    _state["device"] = device
    _state["config"] = cfg
    _state["training"] = {"status": "idle", "result": None}
    _store = ExperimentStore(db_path_from_config(cfg))


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    load_model()
    yield
    _state.clear()


app = FastAPI(
    title="Neural Forge",
    description="Production-ready transformer pipeline for time-series prediction.",
    version=__version__,
    lifespan=lifespan,
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


class PredictRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description="Time-series input as a 2-D list of shape [seq_len, input_dim].",
        examples=[[[0.0], [0.1], [0.2]]],
    )


class PredictResponse(BaseModel):
    prediction: list[float]


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
    _state["training"] = {"status": "running", "result": None}
    try:
        result = TimeSeriesTrainer(config_path).train()
        load_model(config_path)
        _state["training"] = {"status": "completed", "result": result}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        _state["training"] = {"status": "failed", "result": {"error": str(exc)}}


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
    checkpoint_path = stored_path if isinstance(stored_path, Path) else _checkpoint_file(cfg)
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
    }


def _require_store() -> ExperimentStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Experiment store is not available.")
    return _store


@app.get("/experiments", tags=["experiments"])
def list_experiments(limit: int = Query(20, ge=1, le=500)) -> dict[str, Any]:
    return {"experiments": _require_store().list_experiments(limit=limit)}


@app.get("/experiments/{experiment_id}", tags=["experiments"])
def get_experiment(experiment_id: int) -> dict[str, Any]:
    store = _require_store()
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    experiment["metric_history"] = store.get_metrics(experiment_id)
    return experiment


@app.get("/training/status", tags=["training"])
def training_status() -> dict[str, Any]:
    default: dict[str, Any] = {"status": "idle", "result": None}
    training = _state.get("training")
    return training if training is not None else default


def _configured_api_token() -> str | None:
    raw = os.environ.get(API_TOKEN_ENV, "").strip()
    return raw or None


def _authorize_training(authorization: str | None) -> None:
    expected = _configured_api_token()
    if expected is None:
        return
    scheme, _, provided = (authorization or "").partition(" ")
    token = provided.strip()
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


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
    authorization: str | None = Header(default=None),
) -> TrainResponse:
    _authorize_training(authorization)
    if not _training_enabled():
        raise HTTPException(
            status_code=403,
            detail="Training is disabled. Set api.training_enabled to true to allow it.",
        )
    config_path = resolve_train_config(body.config_path)
    if _state.get("training", {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="Training already in progress.")

    background_tasks.add_task(_run_training, str(config_path))
    _state["training"] = {"status": "running", "result": None}
    return TrainResponse(status="running", message="Training started in background.")


@app.get("/youneek/now", tags=["youneek"])
def youneek_now() -> dict[str, Any]:
    return snapshot(parse_moment(None)).as_dict()


@app.get("/youneek/convert", tags=["youneek"])
def youneek_convert(at: str) -> dict[str, Any]:
    try:
        moment = parse_moment(at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid timestamp: {exc}") from exc
    return snapshot(moment).as_dict()


@app.get("/youneek/forecast/next-minute", tags=["youneek"])
def youneek_forecast_next_minute(at: str | None = None) -> dict[str, Any]:
    try:
        moment = parse_moment(at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid timestamp: {exc}") from exc
    snap = snapshot(moment)
    return {"utc": snap.as_dict()["utc"], **snap.next_minute}


@app.get("/youneek/calendar", tags=["youneek"])
def youneek_calendar(at: str | None = None) -> dict[str, Any]:
    try:
        moment = parse_moment(at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid timestamp: {exc}") from exc
    snap = snapshot(moment)
    return {"utc": snap.as_dict()["utc"], **snap.calendar}


@app.get("/youneek/lunar", tags=["youneek"])
def youneek_lunar(at: str | None = None) -> dict[str, Any]:
    try:
        moment = parse_moment(at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid timestamp: {exc}") from exc
    snap = snapshot(moment)
    return {"utc": snap.as_dict()["utc"], **snap.lunar}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(body: PredictRequest) -> PredictResponse:
    predictor: Predictor | None = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

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

    x = torch.tensor(body.sequence, dtype=torch.float32)
    output = predictor.predict(x)
    return PredictResponse(prediction=output.squeeze(0).tolist())
