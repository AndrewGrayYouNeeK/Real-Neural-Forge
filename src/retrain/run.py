"""Train one dataset and promote the checkpoint only when it wins."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.prepare import load_series, split_indices
from src.retrain.artifacts import (
    SCHEMA_VERSION,
    append_history,
    champion_path,
    export_release,
    forecast_path,
    import_release,
    promote_checkpoint,
    utc_now,
    write_index,
    write_json,
)
from src.retrain.baselines import (
    baseline_names,
    decide_promotion,
    promotion_metric,
    promotion_val_fraction,
    score_baselines,
)
from src.retrain.discover import DatasetSpec, FetchPlan
from src.retrain.scoring import (
    ScoreError,
    forecast_values,
    forward_window,
    score_checkpoint,
)
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """The dataset's fetch command failed."""


@dataclass
class RunOptions:
    """Paths and limits shared by every dataset in one invocation."""

    champions_dir: Path
    forecasts_dir: Path
    release_dir: Path | None = None
    max_epochs: int | None = None
    metric: str | None = None
    dataset_timeout: float = 720
    dry_run: bool = False
    in_process: bool = False


@dataclass
class DatasetResult:
    """The outcome recorded for one dataset."""

    dataset_id: str
    config_path: str
    promoted: bool
    decision: str
    error: str | None = None
    forecast_path: Path | None = None


def run_fetch(plan: FetchPlan, *, cwd: Path) -> None:
    """Run a fetch argv in ``cwd``. No shell is used."""
    logger.info("Fetching with %s", plan.argv)
    env = os.environ.copy()
    root = str(cwd)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not previous else os.pathsep.join((root, previous))
    try:
        completed = subprocess.run(
            plan.argv,
            cwd=cwd,
            env=env,
            timeout=plan.timeout,
            check=False,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"Fetch timed out after {plan.timeout:g}s.") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise FetchError(f"Fetch exited {completed.returncode}: {tail}")


def child_argv(spec: DatasetSpec, options: RunOptions) -> list[str]:
    """Return the in-process command that trains one dataset."""
    argv = [
        sys.executable,
        "-m",
        "src.retrain",
        "--config",
        str(spec.path),
        "--in-process",
        "--champions-dir",
        str(options.champions_dir),
        "--forecasts-dir",
        str(options.forecasts_dir),
        "--dataset-timeout",
        str(options.dataset_timeout),
    ]
    if options.max_epochs is not None:
        argv.extend(["--max-epochs", str(options.max_epochs)])
    if options.metric:
        argv.extend(["--metric", options.metric])
    return argv


def supervise(spec: DatasetSpec, options: RunOptions) -> DatasetResult:
    """Train ``spec`` in a child process so a timeout cannot wedge the others."""
    try:
        completed = subprocess.run(
            child_argv(spec, options),
            timeout=options.dataset_timeout,
            check=False,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        message = f"Timed out after {options.dataset_timeout:g}s."
        return _finish_error(spec, options, message)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        path = forecast_path(options.forecasts_dir, spec.id)
        if path.is_file():
            return _result_from_forecast(spec, path)
        return _finish_error(
            spec, options, tail or f"Retrain exited {completed.returncode}."
        )
    path = forecast_path(options.forecasts_dir, spec.id)
    if not path.is_file():
        return _finish_error(spec, options, "Retrain finished without a forecast file.")
    return _result_from_forecast(spec, path)


def run_many(specs: list[DatasetSpec], options: RunOptions) -> list[DatasetResult]:
    """Train each dataset, then write the index and release directory."""
    if options.dry_run:
        return [_dry_result(spec) for spec in specs]
    options.champions_dir.mkdir(parents=True, exist_ok=True)
    options.forecasts_dir.mkdir(parents=True, exist_ok=True)
    if options.release_dir is not None:
        import_release(options.release_dir, options.champions_dir)
    # --in-process trains here. Otherwise each dataset is a child process so
    # one timeout or crash still leaves the other forecasts writable.
    if options.in_process:
        results = [run_dataset(spec, options) for spec in specs]
    else:
        results = [supervise(spec, options) for spec in specs]
    entries = [_index_entry(result) for result in results]
    write_index(options.forecasts_dir, entries, release_dir=options.release_dir)
    if options.release_dir is not None:
        export_release(
            options.champions_dir, options.forecasts_dir, options.release_dir
        )
    return results


def _dry_result(spec: DatasetSpec) -> DatasetResult:
    return DatasetResult(
        dataset_id=spec.id,
        config_path=spec.config_label(),
        promoted=False,
        decision="dry_run",
    )


def run_dataset(spec: DatasetSpec, options: RunOptions) -> DatasetResult:
    """Fetch, train, score, and maybe promote one dataset."""
    generated_at = utc_now()
    try:
        if spec.fetch is not None:
            run_fetch(spec.fetch, cwd=spec.root)
        cfg = load_config(spec.path)
        metric = promotion_metric(cfg, options.metric)
        fraction = promotion_val_fraction(cfg)
        names = baseline_names(cfg)
        with tempfile.TemporaryDirectory(prefix=f"retrain-{spec.id}-") as temporary:
            candidate = Path(temporary) / "candidate.pt"
            _train_candidate(spec, cfg, options, fraction, Path(temporary))
            inputs, targets = load_series(cfg)
            train_idx, val_idx = split_indices(len(inputs), fraction, "temporal", 0)
            device = _device(cfg)
            batch_size = int(cfg["training"].get("batch_size", 32))
            candidate_metrics = score_checkpoint(
                candidate,
                inputs,
                targets,
                val_idx,
                batch_size=batch_size,
                device=device,
            )
            baselines = score_baselines(names, inputs, targets, train_idx, val_idx, cfg)
            champion = champion_path(options.champions_dir, spec.id)
            champion_metrics, champion_error = _score_champion(
                champion, inputs, targets, val_idx, batch_size=batch_size, device=device
            )
            decision, lost_to = decide_promotion(
                candidate=candidate_metrics,
                champion=champion_metrics,
                baselines=baselines,
                metric=metric,
                champion_error=champion_error,
            )
            if decision == "promoted":
                promote_checkpoint(candidate, champion)
            serving, serving_metrics, forecast = _serving_forecast(
                decision,
                champion,
                candidate,
                cfg,
                device,
                champion_metrics,
                candidate_metrics,
                champion_error,
            )
        document = _document(
            spec,
            generated_at=generated_at,
            promoted=decision == "promoted",
            decision=decision,
            metric=metric,
            fraction=fraction,
            n_train=len(train_idx),
            n_val=len(val_idx),
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
            baselines=baselines,
            lost_to=lost_to,
            serving=serving,
            serving_metrics=serving_metrics,
            forecast=forecast,
            model_cfg=_model_section(cfg),
            champion_error=champion_error,
        )
    except (
        FetchError,
        ScoreError,
        ValueError,
        OSError,
        RuntimeError,
        KeyError,
    ) as exc:
        logger.exception("Retrain failed for %s", spec.id)
        return _finish_error(spec, options, str(exc), generated_at=generated_at)
    path = forecast_path(options.forecasts_dir, spec.id)
    write_json(path, document)
    append_history(
        options.champions_dir / "metrics-history.json", _history_record(document)
    )
    logger.info("%s decision=%s promoted=%s", spec.id, decision, decision == "promoted")
    return DatasetResult(
        dataset_id=spec.id,
        config_path=spec.config_label(),
        promoted=decision == "promoted",
        decision=decision,
        forecast_path=path,
    )


def _model_section(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        return model_cfg
    return {}


def _train_candidate(
    spec: DatasetSpec,
    cfg: dict[str, Any],
    options: RunOptions,
    fraction: float,
    directory: Path,
) -> None:
    training = cfg.get("training")
    if not isinstance(training, dict):
        raise ValueError("training config must be a mapping.")
    configured = int(training.get("epochs", 1))
    epochs = (
        configured
        if options.max_epochs is None
        else min(configured, options.max_epochs)
    )
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    overrides: dict[str, Any] = {
        "data": {"split_mode": "temporal", "val_split": fraction},
        "training": {"epochs": epochs, "checkpoint_dir": str(directory)},
    }
    trainer = TimeSeriesTrainer(str(spec.path), overrides=overrides)
    trainer.train(checkpoint_name="candidate.pt")


def _device(cfg: dict[str, Any]) -> torch.device:
    training = cfg.get("training")
    requested = "cpu"
    if isinstance(training, dict):
        value = training.get("device", "cpu")
        if isinstance(value, str) and value.strip():
            requested = value.strip()
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _score_champion(
    path: Path,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_idx: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return (
            score_checkpoint(
                path, inputs, targets, val_idx, batch_size=batch_size, device=device
            ),
            None,
        )
    except ScoreError as exc:
        return None, str(exc)


def _serving_forecast(
    decision: str,
    champion: Path,
    candidate: Path,
    cfg: dict[str, Any],
    device: torch.device,
    champion_metrics: dict[str, float] | None,
    candidate_metrics: dict[str, float],
    champion_error: str | None,
) -> tuple[str, dict[str, float] | None, dict[str, Any] | None]:
    if decision == "promoted" or (
        decision == "kept_champion" and champion.is_file() and champion_error is None
    ):
        serving_path, serving, metrics = champion, "champion", champion_metrics
        if decision == "promoted":
            metrics = candidate_metrics
    elif decision == "rejected_baseline":
        serving_path, serving, metrics = candidate, "candidate", candidate_metrics
    elif champion.is_file() and champion_error is None:
        serving_path, serving, metrics = champion, "champion", champion_metrics
    else:
        return "none", None, None
    try:
        window, as_of = forward_window(cfg)
        values = forecast_values(serving_path, window, device=device)
    except (ScoreError, ValueError, OSError) as exc:
        logger.warning("Forecast failed for %s: %s", serving, exc)
        return serving, metrics, None
    return serving, metrics, {"horizon": len(values), "values": values, "as_of": as_of}


def _document(
    spec: DatasetSpec,
    *,
    generated_at: str,
    promoted: bool,
    decision: str,
    metric: str,
    fraction: float,
    n_train: int,
    n_val: int,
    candidate_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    baselines: dict[str, dict[str, float]],
    lost_to: list[str],
    serving: str,
    serving_metrics: dict[str, float] | None,
    forecast: dict[str, Any] | None,
    model_cfg: dict[str, Any],
    champion_error: str | None,
) -> dict[str, Any]:
    serving_value = None if serving_metrics is None else serving_metrics.get(metric)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": spec.id,
        "config_path": spec.config_label(),
        "generated_at": generated_at,
        "promoted": promoted,
        "serving": serving,
        "decision": decision,
        "metric": metric,
        "lower_is_better": True,
        "lost_to": lost_to,
        "serving_metric": serving_value,
        "held_out": {
            "split": "temporal",
            "val_fraction": fraction,
            "n_train": n_train,
            "n_val": n_val,
        },
        "candidate": candidate_metrics,
        "champion": champion_metrics,
        "champion_error": champion_error,
        "baselines": baselines,
        "forecast": forecast,
        "model": {
            "name": model_cfg.get("name", "time_series_transformer"),
            "input_dim": model_cfg.get("input_dim"),
            "output_dim": model_cfg.get("output_dim"),
        },
    }


def _history_record(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": document["dataset_id"],
        "generated_at": document["generated_at"],
        "promoted": document["promoted"],
        "decision": document["decision"],
        "metric": document.get("metric"),
        "candidate": document.get("candidate"),
        "champion": document.get("champion"),
        "baselines": document.get("baselines"),
        "lost_to": document.get("lost_to"),
        "serving": document.get("serving"),
        "serving_metric": document.get("serving_metric"),
        "error": document.get("error"),
    }


def _finish_error(
    spec: DatasetSpec,
    options: RunOptions,
    message: str,
    *,
    generated_at: str | None = None,
) -> DatasetResult:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": spec.id,
        "config_path": spec.config_label(),
        "generated_at": generated_at or utc_now(),
        "promoted": False,
        "serving": "champion"
        if champion_path(options.champions_dir, spec.id).is_file()
        else "none",
        "decision": "error",
        "error": message,
        "forecast": None,
    }
    path = forecast_path(options.forecasts_dir, spec.id)
    write_json(path, document)
    append_history(
        options.champions_dir / "metrics-history.json", _history_record(document)
    )
    return DatasetResult(
        dataset_id=spec.id,
        config_path=spec.config_label(),
        promoted=False,
        decision="error",
        error=message,
        forecast_path=path,
    )


def _result_from_forecast(spec: DatasetSpec, path: Path) -> DatasetResult:
    document = json.loads(path.read_text(encoding="utf-8"))
    error = document.get("error") if isinstance(document, dict) else None
    decision = document.get("decision") if isinstance(document, dict) else "error"
    promoted = bool(document.get("promoted")) if isinstance(document, dict) else False
    return DatasetResult(
        dataset_id=spec.id,
        config_path=spec.config_label(),
        promoted=promoted,
        decision=str(decision or "error"),
        error=str(error) if error else None,
        forecast_path=path,
    )


def _index_entry(result: DatasetResult) -> dict[str, Any]:
    return {
        "id": result.dataset_id,
        "config": result.config_path,
        "forecast": f"forecast-{result.dataset_id}.json",
        "champion": f"champion-{result.dataset_id}.pt",
        "promoted": result.promoted,
        "decision": result.decision,
        "error": result.error,
    }


def plan_document(specs: list[DatasetSpec], errors: list[str]) -> dict[str, Any]:
    """Return the machine-readable dry-run description."""
    return {
        "schema_version": SCHEMA_VERSION,
        "datasets": [
            {
                "id": spec.id,
                "config": spec.config_label(),
                "fetch": None
                if spec.fetch is None
                else {"argv": spec.fetch.argv, "timeout_seconds": spec.fetch.timeout},
            }
            for spec in specs
        ],
        "errors": errors,
    }
