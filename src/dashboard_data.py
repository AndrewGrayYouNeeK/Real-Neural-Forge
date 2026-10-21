"""Read-only series and validation data for the forecast dashboard.

The shapes here are what the dashboard plots today. Richer forecast payloads
(multi-step horizons, p10/p50/p90 intervals, model comparison) are left for the
client to merge in when those API responses exist.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from src.data.timeseries import load_csv_timeseries, make_sine_dataset
from src.evaluation.metrics import mae, mse, rmse
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.storage.experiments import ExperimentStore
from src.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)

_TIME_CANDIDATES = ("timestamp", "time", "date", "datetime")
_STORED_POINT_KEYS = (
    "validation_predictions",
    "predictions",
    "val_predictions",
)
_COMPARISON_KEYS = ("comparison", "model_comparison")


def _section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    return value if isinstance(value, dict) else {}


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _uses_table(data_cfg: dict[str, Any]) -> bool:
    """True when config points at a file the CSV loader can read.

    ``source: synthetic`` always stays on the sine generator. Any other source
    with ``data.path`` is treated as a table so file-based datasets (weather,
    prices, and similar) can be charted without a new loader.
    """
    path = data_cfg.get("path")
    if not isinstance(path, str) or not path.strip():
        return False
    return str(data_cfg.get("source", "synthetic")) != "synthetic"


def _dataset_label(data_cfg: dict[str, Any], source: str) -> str:
    for key in ("name", "dataset", "dataset_name", "id"):
        value = data_cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:80]
    return source


def _paired_metrics(
    actual: list[float], predicted: list[float]
) -> dict[str, float | int] | None:
    if not actual or len(actual) != len(predicted):
        return None
    actual_t = torch.tensor(actual, dtype=torch.float32)
    predicted_t = torch.tensor(predicted, dtype=torch.float32)
    return {
        "mse": mse(predicted_t, actual_t),
        "rmse": rmse(predicted_t, actual_t),
        "mae": mae(predicted_t, actual_t),
        "n": len(actual),
    }


def _metrics_document(
    model: dict[str, float | int] | None,
    baselines: dict[str, dict[str, float | int]],
) -> dict[str, Any] | None:
    if model is None and not baselines:
        return None
    return {"model": model, "baselines": baselines}


def _predict_first_output(predictor: Any, windows: torch.Tensor) -> list[float | None]:
    predicted: list[float | None] = []
    for start in range(0, int(windows.size(0)), 256):
        batch = windows[start : start + 256]
        raw = predictor.predict(batch)
        if not isinstance(raw, torch.Tensor):
            raise TypeError("predictor.predict must return a tensor.")
        column = raw.reshape(raw.size(0), -1)[:, 0]
        predicted.extend(_as_float(value) for value in column.detach().cpu().tolist())
    return predicted


def _time_label(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return None
    if isinstance(value, pd.Timestamp):
        text = str(value.isoformat())
    else:
        text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text[:80]


def _feature_columns(data_cfg: dict[str, Any]) -> list[str]:
    raw = data_cfg.get("feature_columns", ["value"])
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(col, str) for col in raw)
    ):
        raise ValueError("data.feature_columns must be a list of column names.")
    return list(raw)


def _target_column(data_cfg: dict[str, Any]) -> str:
    raw = data_cfg.get("target_column", "value")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("data.target_column must be a column name.")
    return raw


def _seq_len(data_cfg: dict[str, Any]) -> int:
    try:
        seq_len = int(data_cfg.get("seq_len", 64))
    except (TypeError, ValueError) as exc:
        raise ValueError("data.seq_len must be an integer.") from exc
    if seq_len < 1:
        raise ValueError("data.seq_len must be at least 1.")
    return seq_len


def _read_table(
    data_cfg: dict[str, Any], seq_len: int, points: int
) -> tuple[str, str, np.ndarray, np.ndarray, list[int], list[str | None]]:
    raw_path = data_cfg.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Tabular data config is missing data.path.")
    path = resolve_repo_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {raw_path}")

    feature_columns = _feature_columns(data_cfg)
    target_column = _target_column(data_cfg)
    header = pd.read_csv(path, nrows=0)
    columns = [str(column) for column in header.columns]
    configured_time = data_cfg.get("time_column")
    time_column: str | None
    if isinstance(configured_time, str) and configured_time.strip():
        if configured_time not in columns:
            raise ValueError(f"Time column {configured_time!r} is not in the dataset.")
        time_column = configured_time
    else:
        time_column = next(
            (
                candidate
                for candidate in _TIME_CANDIDATES
                if candidate in columns
                and candidate not in feature_columns
                and candidate != target_column
            ),
            None,
        )

    missing = [
        column for column in [*feature_columns, target_column] if column not in columns
    ]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    usecols = list(dict.fromkeys([*feature_columns, target_column]))
    if time_column is not None:
        usecols.append(time_column)
    frame = pd.read_csv(path, usecols=usecols)
    if frame.empty:
        raise ValueError("Dataset has no rows.")

    keep = points + seq_len
    total = len(frame)
    if total > keep:
        frame = frame.iloc[-keep:].copy()
    start_index = total - len(frame)
    features = cast(
        np.ndarray,
        frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64),
    )
    actual = cast(
        np.ndarray,
        pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.float64),
    )
    times: list[str | None]
    if time_column is not None:
        times = [_time_label(value) for value in frame[time_column].tolist()]
    else:
        times = [None] * len(frame)
    indices = list(range(start_index, start_index + len(frame)))
    source = str(data_cfg.get("source") or "csv")
    return source, target_column, actual, features, indices, times


def _synthetic_frame(
    data_cfg: dict[str, Any], seq_len: int, points: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    length = seq_len + points
    try:
        noise = float(data_cfg.get("noise", 0.05))
    except (TypeError, ValueError) as exc:
        raise ValueError("data.noise must be a number.") from exc
    seed = data_cfg.get("seed")
    generator = torch.Generator()
    if isinstance(seed, int) and not isinstance(seed, bool):
        generator.manual_seed(seed)
    else:
        generator.manual_seed(0)
    step = 4 * math.pi / seq_len
    offset = float(torch.rand(1, generator=generator).item()) * 2 * math.pi
    timeline = torch.arange(length, dtype=torch.float32) * step
    series = torch.sin(timeline + offset)
    if noise:
        series = series + torch.randn(length, generator=generator) * noise
    values = series.numpy().astype(np.float64)
    notes: list[str] = [
        "Synthetic history is one seeded series from the loaded data config, "
        "not the full training set."
    ]
    if not (isinstance(seed, int) and not isinstance(seed, bool)):
        notes.append(
            "data.seed is unset; the chart uses a fixed display seed so it stays still."
        )
    return values, values.reshape(-1, 1), notes


def _load_windows(data_cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = _seq_len(data_cfg)
    if _uses_table(data_cfg):
        path = data_cfg.get("path")
        if not isinstance(path, str):
            raise ValueError("Tabular data config is missing data.path.")
        return load_csv_timeseries(
            resolve_repo_path(path),
            feature_columns=_feature_columns(data_cfg),
            target_column=_target_column(data_cfg),
            seq_len=seq_len,
        )
    seed = data_cfg.get("seed")
    return make_sine_dataset(
        n_samples=int(data_cfg.get("n_samples", 1024)),
        seq_len=seq_len,
        noise=float(data_cfg.get("noise", 0.05)),
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
    )


def _evaluation_indices(n: int, data_cfg: dict[str, Any]) -> tuple[list[int], str]:
    """Mirror ``TimeSeriesTrainer``'s validation split.

    File-based sources other than ``csv`` still follow the trainer's current
    default (random unless ``split_mode`` or ``source: csv`` says otherwise).
    """
    try:
        val_split = float(data_cfg.get("val_split", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("data.val_split must be a number.") from exc
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"data.val_split must be in [0, 1), got {val_split}")
    n_val = int(n * val_split)
    if val_split > 0 and (n_val == 0 or n_val == n):
        raise ValueError("data.val_split leaves an empty train or validation set.")
    if n_val == 0:
        return list(range(n)), "training"

    split_mode = data_cfg.get(
        "split_mode", "temporal" if data_cfg.get("source") == "csv" else "random"
    )
    if split_mode not in ("random", "temporal"):
        raise ValueError(
            f"data.split_mode must be 'random' or 'temporal', got {split_mode!r}"
        )
    if split_mode == "temporal":
        return list(range(n - n_val, n)), "validation"

    generator = torch.Generator().manual_seed(int(data_cfg.get("split_seed", 0)))
    # Same permutation ``random_split`` uses: the second subset is the tail.
    permutation = torch.randperm(n, generator=generator).tolist()
    return sorted(permutation[n - n_val :]), "validation"


def recent_series(
    cfg: dict[str, Any], predictor: Any, *, points: int
) -> dict[str, Any]:
    """Return recent target history, one-step backtest, forecast, and baselines."""
    points = max(1, min(int(points), 400))
    data_cfg = _section(cfg, "data")
    model_cfg = _section(cfg, "model")
    try:
        input_dim = int(model_cfg.get("input_dim", 1))
        output_dim = int(model_cfg.get("output_dim", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "model.input_dim and model.output_dim must be integers."
        ) from exc
    if input_dim < 1 or output_dim < 1:
        raise ValueError("model.input_dim and model.output_dim must be at least 1.")
    seq_len = _seq_len(data_cfg)

    notes: list[str] = []
    if _uses_table(data_cfg):
        source, target, actual, features, indices, times = _read_table(
            data_cfg, seq_len, points
        )
    elif data_cfg.get("source", "synthetic") not in (None, "synthetic"):
        source_name = data_cfg.get("source")
        raise ValueError(
            f"No tabular file is configured for data.source={source_name!r}, "
            "so recent history is unavailable."
        )
    else:
        if input_dim != 1:
            raise ValueError(
                "The synthetic series has 1 feature, but the loaded model expects "
                f"{input_dim}."
            )
        source = "synthetic"
        target = "value"
        actual, features, synthetic_notes = _synthetic_frame(data_cfg, seq_len, points)
        notes.extend(synthetic_notes)
        indices = list(range(len(actual)))
        times = [None] * len(actual)

    if features.shape[1] != input_dim:
        raise ValueError(
            f"Loaded model expects {input_dim} feature(s) but the dataset has "
            f"{features.shape[1]}."
        )

    length = int(actual.shape[0])
    predicted_by_pos: dict[int, float] = {}
    if length > seq_len:
        windows: list[np.ndarray] = []
        positions: list[int] = []
        for position in range(seq_len, length):
            window = features[position - seq_len : position]
            if np.isfinite(window).all() and math.isfinite(float(actual[position])):
                windows.append(window)
                positions.append(position)
        if windows:
            batch = torch.tensor(np.stack(windows), dtype=torch.float32)
            for position, value in zip(
                positions, _predict_first_output(predictor, batch), strict=True
            ):
                if value is not None:
                    predicted_by_pos[position] = value

    start_row = max(0, length - points)
    history: list[dict[str, Any]] = []
    model_actual: list[float] = []
    model_predicted: list[float] = []
    persistence_actual: list[float] = []
    persistence_predicted: list[float] = []
    mean_actual: list[float] = []
    mean_predicted: list[float] = []
    finite_before: list[float] = [
        float(actual[index])
        for index in range(start_row)
        if math.isfinite(float(actual[index]))
    ]

    for position in range(start_row, length):
        actual_value = _as_float(float(actual[position]))
        predicted = predicted_by_pos.get(position)
        history.append(
            {
                "index": int(indices[position]),
                "time": times[position],
                "actual": actual_value,
                "predicted": predicted,
            }
        )
        if actual_value is None:
            continue
        if predicted is not None:
            model_actual.append(actual_value)
            model_predicted.append(predicted)
            if position > 0 and math.isfinite(float(actual[position - 1])):
                persistence_actual.append(actual_value)
                persistence_predicted.append(float(actual[position - 1]))
            if finite_before:
                mean_actual.append(actual_value)
                mean_predicted.append(sum(finite_before) / len(finite_before))
        finite_before.append(actual_value)

    forecast: list[dict[str, Any]] = []
    predict_window: list[list[float]] = []
    if length >= seq_len:
        window = features[-seq_len:]
        if np.isfinite(window).all():
            predict_window = [
                [float(value) for value in row] for row in window.tolist()
            ]
            raw = predictor.predict(torch.tensor(window, dtype=torch.float32))
            if not isinstance(raw, torch.Tensor):
                raise TypeError("predictor.predict must return a tensor.")
            flat = raw.reshape(-1).detach().cpu()
            value = _as_float(float(flat[0])) if int(flat.numel()) else None
            if value is not None:
                next_index = int(indices[-1]) + 1 if indices else length
                forecast.append(
                    {
                        "step": 1,
                        "index": next_index,
                        "value": value,
                        "p10": None,
                        "p50": None,
                        "p90": None,
                    }
                )
    else:
        notes.append("The series is shorter than seq_len, so there is no forecast yet.")

    if output_dim > 1 and forecast:
        notes.append(
            f"The model output has {output_dim} values; the chart shows the first "
            "as the next target."
        )

    baselines: dict[str, dict[str, float | int]] = {}
    persistence = _paired_metrics(persistence_actual, persistence_predicted)
    mean_baseline = _paired_metrics(mean_actual, mean_predicted)
    if persistence is not None:
        baselines["persistence"] = persistence
    if mean_baseline is not None:
        baselines["mean"] = mean_baseline

    return {
        "source": source,
        "dataset": _dataset_label(data_cfg, source),
        "target": target,
        "seq_len": seq_len,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "history": history,
        "forecast": forecast,
        "predict_window": predict_window,
        "metrics": _metrics_document(
            _paired_metrics(model_actual, model_predicted), baselines
        ),
        "note": " ".join(notes) or None,
    }


def _extract_comparison(metrics: object) -> Any | None:
    if not isinstance(metrics, dict):
        return None
    for key in _COMPARISON_KEYS:
        value = metrics.get(key)
        if isinstance(value, dict | list):
            return value
    return None


def _parallel_points(
    actual: object,
    predicted: object,
    p10: object,
    p50: object,
    p90: object,
) -> list[dict[str, Any]] | None:
    if not isinstance(actual, list) and not isinstance(predicted, list):
        return None
    actual_list = actual if isinstance(actual, list) else []
    predicted_list = predicted if isinstance(predicted, list) else []
    p10_list = p10 if isinstance(p10, list) else []
    p50_list = p50 if isinstance(p50, list) else []
    p90_list = p90 if isinstance(p90, list) else []
    count = max(len(actual_list), len(predicted_list), len(p50_list))
    if count == 0:
        return None
    points: list[dict[str, Any]] = []
    for index in range(count):
        point = {
            "index": index,
            "actual": _as_float(actual_list[index])
            if index < len(actual_list)
            else None,
            "predicted": (
                _as_float(predicted_list[index])
                if index < len(predicted_list)
                else None
            ),
            "p10": _as_float(p10_list[index]) if index < len(p10_list) else None,
            "p50": _as_float(p50_list[index]) if index < len(p50_list) else None,
            "p90": _as_float(p90_list[index]) if index < len(p90_list) else None,
        }
        if point["predicted"] is None and point["p50"] is not None:
            point["predicted"] = point["p50"]
        if point["actual"] is None and point["predicted"] is None:
            continue
        points.append(point)
    return points or None


def _row_points(rows: list[object]) -> list[dict[str, Any]] | None:
    points: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        predicted = _as_float(
            row.get("predicted", row.get("prediction", row.get("yhat", row.get("p50"))))
        )
        point = {
            "index": index,
            "actual": _as_float(row.get("actual", row.get("target", row.get("y")))),
            "predicted": predicted,
            "p10": _as_float(row.get("p10", row.get("lower"))),
            "p50": _as_float(row.get("p50", row.get("median"))),
            "p90": _as_float(row.get("p90", row.get("upper"))),
        }
        if point["predicted"] is None and point["p50"] is not None:
            point["predicted"] = point["p50"]
        if point["actual"] is None and point["predicted"] is None:
            continue
        points.append(point)
    return points or None


def _extract_stored_points(metrics: object) -> list[dict[str, Any]] | None:
    if not isinstance(metrics, dict):
        return None
    for key in _STORED_POINT_KEYS:
        block = metrics.get(key)
        parsed = _parse_point_block(block)
        if parsed:
            return parsed
    return _parse_point_block(metrics)


def _parse_point_block(block: object) -> list[dict[str, Any]] | None:
    if isinstance(block, list):
        return _row_points(block)
    if not isinstance(block, dict):
        return None
    for key in ("points", "predictions", "samples"):
        rows = block.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return _row_points(rows)
    return _parallel_points(
        block.get("actual", block.get("actuals", block.get("targets"))),
        block.get("predicted", block.get("preds", block.get("yhat"))),
        block.get("p10"),
        block.get("p50"),
        block.get("p90"),
    )


def _baselines_for_points(
    points: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    actuals = [point["actual"] for point in points if point["actual"] is not None]
    baselines: dict[str, dict[str, float | int]] = {}
    if actuals:
        center = sum(actuals) / len(actuals)
        mean_metrics = _paired_metrics(actuals, [center] * len(actuals))
        if mean_metrics is not None:
            baselines["mean"] = mean_metrics
    persistence_actual: list[float] = []
    persistence_predicted: list[float] = []
    for previous, point in zip(points, points[1:], strict=False):
        if point["index"] != previous["index"] + 1:
            continue
        if previous["actual"] is None or point["actual"] is None:
            continue
        persistence_actual.append(point["actual"])
        persistence_predicted.append(previous["actual"])
    persistence = _paired_metrics(persistence_actual, persistence_predicted)
    if persistence is not None:
        baselines["persistence"] = persistence
    return baselines


def _point_payload(
    experiment_id: int,
    points: list[dict[str, Any]],
    *,
    source: str,
    split: str | None,
    note: str,
    comparison: Any | None,
    limit: int,
    truncated: bool = False,
) -> dict[str, Any]:
    if len(points) > limit:
        points = points[-limit:]
        truncated = True
    model_actual = [
        point["actual"]
        for point in points
        if point["actual"] is not None and point["predicted"] is not None
    ]
    model_predicted = [
        point["predicted"]
        for point in points
        if point["actual"] is not None and point["predicted"] is not None
    ]
    return {
        "experiment_id": experiment_id,
        "available": True,
        "source": source,
        "split": split,
        "truncated": truncated,
        "note": note,
        "points": points,
        "metrics": _metrics_document(
            _paired_metrics(model_actual, model_predicted),
            _baselines_for_points(points),
        ),
        "comparison": comparison,
    }


def _unavailable(
    experiment_id: int, comparison: Any | None, note: str
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "available": False,
        "source": None,
        "split": None,
        "truncated": False,
        "note": note,
        "points": [],
        "metrics": None,
        "baselines": None,
        "comparison": comparison,
    }


def _replay(
    config: dict[str, Any], checkpoint: Any, limit: int
) -> tuple[list[dict[str, Any]], str, bool]:
    data_cfg = _section(config, "data")
    inputs, targets = _load_windows(data_cfg)
    indices, split = _evaluation_indices(int(inputs.size(0)), data_cfg)
    if not indices:
        raise ValueError("No evaluation windows.")
    truncated = len(indices) > limit
    chosen = indices[-limit:] if truncated else indices
    windows = inputs[chosen]
    actual = targets[chosen].reshape(-1).detach().cpu().tolist()
    model = build_model(config)
    predictor = Predictor.from_checkpoint(model, checkpoint, device="cpu")
    predicted = _predict_first_output(predictor, windows)
    points: list[dict[str, Any]] = []
    for index, actual_value, pred in zip(chosen, actual, predicted, strict=True):
        points.append(
            {
                "index": int(index),
                "actual": _as_float(actual_value),
                "predicted": pred,
                "p10": None,
                "p50": None,
                "p90": None,
            }
        )
    return points, split, truncated


def validation_predictions(
    store: ExperimentStore, experiment_id: int, *, limit: int
) -> dict[str, Any] | None:
    """Return stored validation points, or replay them from the run's checkpoint."""
    limit = max(1, min(int(limit), 2000))
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        return None

    metrics = experiment.get("metrics")
    comparison = _extract_comparison(metrics)
    stored = _extract_stored_points(metrics)
    if stored:
        return _point_payload(
            experiment_id,
            stored,
            source="stored",
            split=None,
            note="Validation predictions stored with this run.",
            comparison=comparison,
            limit=limit,
        )

    checkpoint = experiment.get("checkpoint_path")
    config = experiment.get("config")
    missing = (
        "This run has no stored validation predictions, and its checkpoint "
        "is not on disk."
    )
    if (
        not isinstance(checkpoint, str)
        or not checkpoint.strip()
        or not isinstance(config, dict)
    ):
        return _unavailable(experiment_id, comparison, missing)
    path = resolve_repo_path(checkpoint)
    if not path.is_file():
        return _unavailable(experiment_id, comparison, missing)

    try:
        points, split, truncated = _replay(config, path, limit)
    except Exception:  # noqa: BLE001 - a bad checkpoint should degrade, not 500
        logger.exception(
            "Could not replay validation predictions for %s", experiment_id
        )
        return _unavailable(
            experiment_id,
            comparison,
            "Stored predictions are missing and the checkpoint could not be replayed.",
        )
    return _point_payload(
        experiment_id,
        points,
        source="replay",
        split=split,
        note=(
            f"Replayed on the {split} windows from the run's checkpoint and saved "
            "config. A shared checkpoint file may have been overwritten by a later run."
        ),
        comparison=comparison,
        limit=limit,
        truncated=truncated,
    )
