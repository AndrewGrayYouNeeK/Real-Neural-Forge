"""Run one visitor training job without touching the owner's checkpoint."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import yaml

from src.accounts.limits import Limits, load_limits
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config, merge_config
from src.workspace.keys import user_model_key
from src.workspace.service_ids import new_id
from src.workspace.starters import starter_by_id
from src.workspace.store import WorkspaceStore, dump_json, utc_now

logger = logging.getLogger("uvicorn.error")


def _config_text(store: WorkspaceStore, job: dict[str, Any]) -> str:
    config_id = job.get("config_id")
    if isinstance(config_id, str) and config_id:
        row = store.get_config(str(job["user_id"]), config_id)
        if row is None:
            raise ValueError("The training config is not in this workspace.")
        body = row.get("body")
        if isinstance(body, str) and body.strip():
            return body
        starter_id = row.get("starter_id")
        if isinstance(starter_id, str):
            starter = starter_by_id(starter_id)
            if starter is not None:
                return starter.template_text()
        raise ValueError("The training config has no template.")
    base = job.get("base_config")
    if isinstance(base, str) and base:
        return Path(base).read_text(encoding="utf-8")
    raise ValueError("This job has no config.")


def _constrain(
    cfg: dict[str, Any], limits: Limits, *, csv_path: Path | None
) -> dict[str, Any]:
    """Force visitor runs onto a private temp dir and the configured caps."""
    raw_data = cfg.get("data")
    raw_training = cfg.get("training")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    training: dict[str, Any] = raw_training if isinstance(raw_training, dict) else {}
    epochs = int(training.get("epochs", 1) or 1)
    seq_len = int(data.get("seq_len", 16) or 16)
    batch = int(training.get("batch_size", 32) or 32)
    samples = int(data.get("n_samples", limits.max_samples) or limits.max_samples)
    overrides: dict[str, Any] = {
        "training": {
            "epochs": min(max(epochs, 1), limits.max_epochs),
            "device": "cpu",
            "batch_size": min(max(batch, 1), limits.max_batch_size),
        },
        "inference": {"device": "cpu"},
        "data": {"seq_len": min(max(seq_len, 1), limits.max_seq_len)},
    }
    if csv_path is None:
        overrides["data"]["source"] = "synthetic"
        overrides["data"]["n_samples"] = min(max(samples, 8), limits.max_samples)
    else:
        overrides["data"]["source"] = "csv"
        overrides["data"]["path"] = str(csv_path)
    return merge_config(cfg, overrides)


def execute_member_job(
    store: WorkspaceStore,
    job: dict[str, Any],
    *,
    trainer_factory: Callable[[str], TimeSeriesTrainer] = TimeSeriesTrainer,
    limits: Limits | None = None,
) -> None:
    """Train in a temp directory and store a private checkpoint.

    Shared starter objects and the owner's ``checkpoints/`` tree are not written.
    """
    resolved_limits = limits if limits is not None else load_limits()
    job_id = str(job["id"])
    try:
        result, model_id = _train(
            store, job, trainer_factory=trainer_factory, limits=resolved_limits
        )
        store.finish_job(
            job_id,
            status="completed",
            error=None,
            result_json=dump_json({"model_id": model_id, **result}),
            finished_at=utc_now(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Visitor training failed for job %s", job_id)
        message = str(exc).strip() or "Training failed."
        store.finish_job(
            job_id,
            status="failed",
            error=message[:500],
            result_json=None,
            finished_at=utc_now(),
        )


def _private_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Drop temp paths so a stored config cannot point at this run's disk."""
    cleaned = cast(dict[str, Any], json.loads(json.dumps(cfg)))
    if isinstance(cleaned.get("data"), dict):
        cleaned["data"].pop("path", None)
    if isinstance(cleaned.get("training"), dict):
        cleaned["training"].pop("checkpoint_dir", None)
    if isinstance(cleaned.get("storage"), dict):
        cleaned["storage"].pop("db_path", None)
    if isinstance(cleaned.get("inference"), dict):
        cleaned["inference"].pop("checkpoint_path", None)
    return cleaned


def _train(
    store: WorkspaceStore,
    job: dict[str, Any],
    *,
    trainer_factory: Callable[[str], Any],
    limits: Limits,
) -> tuple[dict[str, Any], str]:
    user_id = str(job["user_id"])
    dataset_id = job.get("dataset_id")
    csv_bytes: bytes | None = None
    if isinstance(dataset_id, str) and dataset_id:
        dataset = store.get_dataset(user_id, dataset_id)
        if dataset is None:
            raise ValueError("The dataset is not in this workspace.")
        csv_bytes = store.get_blob(str(dataset["object_key"]))
        if csv_bytes is None:
            raise ValueError("The dataset bytes are missing.")
    text = _config_text(store, job)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a mapping.")
    with tempfile.TemporaryDirectory(prefix="nf-job-") as temporary:
        root = Path(temporary)
        csv_path: Path | None = None
        if csv_bytes is not None:
            csv_path = root / "series.csv"
            csv_path.write_bytes(csv_bytes)
        cfg = _constrain(loaded, limits, csv_path=csv_path)
        checkpoint_dir = root / "checkpoints"
        cfg = merge_config(
            cfg,
            {
                "training": {"checkpoint_dir": str(checkpoint_dir)},
                "storage": {"db_path": str(root / "experiments.db")},
                "inference": {"checkpoint_path": str(checkpoint_dir / "best_model.pt")},
            },
        )
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        started = time.monotonic()

        def on_epoch(epoch: int, loss: float) -> None:
            del epoch, loss
            if time.monotonic() - started > limits.max_run_seconds:
                raise TimeoutError(
                    f"Stopped because the run reached {limits.max_run_seconds} seconds."
                )

        trained = trainer_factory(str(config_path))
        result = trained.train(on_epoch)
        checkpoint = checkpoint_dir / "best_model.pt"
        if not checkpoint.is_file():
            raise RuntimeError("Training finished without a checkpoint.")
        payload = checkpoint.read_bytes()
        if len(payload) > limits.max_checkpoint_bytes:
            raise RuntimeError("The checkpoint is larger than the configured limit.")
        model_id = new_id("mdl_")
        key = user_model_key(user_id, model_id)
        store.put_blob(key, payload)
        metrics = result.get("metrics") if isinstance(result, dict) else None
        history = result.get("history") if isinstance(result, dict) else None
        public = {
            "best_loss": result.get("best_loss") if isinstance(result, dict) else None,
            "metrics": metrics if isinstance(metrics, dict) else None,
            "history": history if isinstance(history, list) else None,
            "elapsed_seconds": (
                result.get("elapsed_seconds") if isinstance(result, dict) else None
            ),
            "best_epoch": result.get("best_epoch")
            if isinstance(result, dict)
            else None,
        }
        saved = _private_config(load_config(config_path))
        store.insert_model(
            {
                "id": model_id,
                "user_id": user_id,
                "name": str(job.get("name") or "trained model"),
                "object_key": key,
                "shared": 0,
                "starter_id": job.get("starter_id"),
                "config_json": json.dumps(saved),
                "metrics_json": dump_json(public["metrics"])
                if isinstance(public["metrics"], dict)
                else None,
                "job_id": job["id"],
                "created_at": utc_now(),
            }
        )
        return public, model_id
