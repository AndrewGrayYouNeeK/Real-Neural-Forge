"""FastAPI REST endpoint for time-series inference."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.inference.predictor import Predictor
from src.models.registry import ModelRegistry, build_model
from src.storage.experiments import ExperimentStore
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config

logger = logging.getLogger("uvicorn.error")

_state: dict[str, Any] = {}
_store = ExperimentStore()
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def load_model(config_path: str = "config/config.yaml") -> None:
    """Load (or lazily initialise) the model into the global state dict."""
    cfg = load_config(config_path)
    i_cfg = cfg["inference"]

    device_str = i_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    model = build_model(cfg)
    checkpoint_path = Path(i_cfg.get("checkpoint_path", ""))

    if checkpoint_path.exists():
        predictor = Predictor.from_checkpoint(model, checkpoint_path, device=device)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
        _state["predictor"] = predictor
        _state["model"] = predictor.model
    else:
        logger.warning(
            "Checkpoint not found at '%s'. Using untrained model.", checkpoint_path
        )
        model.to(device)
        model.eval()
        _state["predictor"] = Predictor(model, device=device)
        _state["model"] = model

    _state["device"] = device
    _state["config"] = cfg
    _state["training"] = {"status": "idle", "result": None}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    load_model()
    yield
    _state.clear()


app = FastAPI(
    title="Neural Forge",
    description="Production-ready transformer pipeline for time-series prediction.",
    version="2.0.0",
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
    config_path: str = "config/config.yaml"


class TrainResponse(BaseModel):
    status: str
    message: str


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
    checkpoint_path = Path(cfg.get("inference", {}).get("checkpoint_path", ""))
    return {
        "architecture": cfg.get("model", {}).get("name", "time_series_transformer"),
        "available_models": ModelRegistry.list_models(),
        "parameters": param_count,
        "device": str(_state.get("device")),
        "checkpoint_loaded": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path),
        "input_dim": cfg.get("model", {}).get("input_dim"),
        "output_dim": cfg.get("model", {}).get("output_dim"),
    }


@app.get("/experiments", tags=["experiments"])
def list_experiments(limit: int = 20) -> dict[str, Any]:
    return {"experiments": _store.list_experiments(limit=limit)}


@app.get("/experiments/{experiment_id}", tags=["experiments"])
def get_experiment(experiment_id: int) -> dict[str, Any]:
    experiment = _store.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    experiment["metric_history"] = _store.get_metrics(experiment_id)
    return experiment


@app.get("/training/status", tags=["training"])
def training_status() -> dict[str, Any]:
    default: dict[str, Any] = {"status": "idle", "result": None}
    training = _state.get("training")
    return training if training is not None else default


@app.post("/train", response_model=TrainResponse, tags=["training"])
def start_training(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TrainResponse:
    if _state.get("training", {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="Training already in progress.")

    background_tasks.add_task(_run_training, body.config_path)
    _state["training"] = {"status": "running", "result": None}
    return TrainResponse(status="running", message="Training started in background.")


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(body: PredictRequest) -> PredictResponse:
    predictor: Predictor | None = _state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    cfg: dict = _state["config"]
    expected_input_dim: int = cfg["model"]["input_dim"]

    if not body.sequence:
        raise HTTPException(status_code=422, detail="sequence must not be empty.")
    if any(len(step) != expected_input_dim for step in body.sequence):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Each time step must have {expected_input_dim} feature(s); "
                f"got a step with {len(body.sequence[0])} feature(s)."
            ),
        )

    x = torch.tensor(body.sequence, dtype=torch.float32)
    output = predictor.predict(x)
    return PredictResponse(prediction=output.squeeze(0).tolist())
