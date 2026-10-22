"""FastAPI REST endpoint for time-series inference."""

import json
import logging
import math
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import __version__
from src.dashboard_data import recent_series, validation_predictions
from src.forecasting.spec import MAX_HORIZON, forecast_spec_from_config
from src.inference.predictor import ForecastBatch, Predictor
from src.models.registry import ModelRegistry, build_model
from src.serving.checkpoint import checkpoint_destination, prepare_runtime_checkpoint
from src.space.approaches import near_earth_report
from src.storage.experiments import ExperimentStore, db_path_from_config
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config
from src.utils.paths import REPO_ROOT, resolve_repo_path
from src.youneek import parse_moment, snapshot

logger = logging.getLogger("uvicorn.error")

API_TOKEN_ENV = "NEURAL_FORGE_API_TOKEN"
CONFIG_ENV = "NEURAL_FORGE_CONFIG"

_state: dict[str, Any] = {}
_store: ExperimentStore | None = None
FRONTEND_DIR = REPO_ROOT / "frontend"
CONFIG_DIR = REPO_ROOT / "config"
_CONFIG_SUFFIXES = {".yaml", ".yml"}


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


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    load_model(os.environ.get(CONFIG_ENV, "config/config.yaml"))
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
def list_experiments(limit: int = Query(20, ge=1, le=500)) -> dict[str, Any]:
    return {"experiments": _require_store().list_experiments(limit=limit)}


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


@app.get("/powerball/scorecard", tags=["lottery"])
def powerball_scorecard() -> dict[str, Any]:
    """Logged tickets scored against published draws. A missing file is not an error."""
    return read_powerball_scorecard()


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
    experiment_id: int,
    limit: int = Query(200, ge=1, le=2000),
) -> dict[str, Any]:
    """Stored validation predictions, or a replay from the run's checkpoint."""
    payload = validation_predictions(_require_store(), experiment_id, limit=limit)
    if payload is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return payload


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
def latest_comparison() -> dict[str, Any]:
    """Latest ``python -m src.compare`` result already stored for this config.

    This does not fit a new model. The dashboard reads ``models`` as rows with
    ``name`` and ``mae`` / ``rmse`` / ``mse``.
    """
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
