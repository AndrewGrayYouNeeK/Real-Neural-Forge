"""Train every space-series config and write scores plus a few charts.

Run after ``python -m scripts.fetch_space``:

    python -m scripts.train_space

Checkpoints stay under ``checkpoints/space/`` and are not committed. Metrics
are in original units on the temporal validation split.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.space.charts import write_forecast_svg
from src.space.evaluate import extra_baseline_scores, validation_frame
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config
from src.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)

_COLORS = {
    "actual": "#0f172a",
    "model": "#c2410c",
    "persistence": "#0369a1",
    "seasonal": "#15803d",
    "drift": "#7c3aed",
}


def config_paths(directory: str | Path = "config/space") -> list[Path]:
    """YAML configs in ``directory``, in filename order."""
    root = resolve_repo_path(directory)
    paths = sorted(root.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No space configs in {root}")
    return paths


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, np.floating | float):
        return _finite(float(value))
    if isinstance(value, np.integer | int) and not isinstance(value, bool):
        return int(value)
    return value


def _best_epoch(history: list[dict[str, float]]) -> int | None:
    if not history:
        return None
    best = min(history, key=lambda row: row.get("val_loss", row["train_loss"]))
    return int(best["epoch"])


def _chart(
    cfg: dict[str, Any],
    trace: dict[str, Any],
    extra: dict[str, Any],
    destination: Path,
) -> None:
    space_cfg = cfg.get("space") or {}
    rows = trace["rows"]
    dates = trace["dates"]
    start = str(dates[int(rows[0])]) if dates is not None else ""
    end = str(dates[int(rows[-1])]) if dates is not None else ""
    title = f"{space_cfg.get('title', cfg['training']['experiment_name'])} ({start} to {end})"
    series: list[tuple[str, str, list[float]]] = [
        ("actual", _COLORS["actual"], [float(value) for value in trace["targets"]]),
        ("model", _COLORS["model"], [float(value) for value in trace["predictions"]]),
        (
            "persistence",
            _COLORS["persistence"],
            [float(value) for value in extra["persistence_trace"]],
        ),
    ]
    lag_traces: dict[str, np.ndarray] = extra["lag_traces"]
    if lag_traces:
        first_key = sorted(lag_traces, key=lambda key: int(key.split("_")[1]))[0]
        series.append(
            (
                first_key.replace("_", " "),
                _COLORS["seasonal"],
                [float(value) for value in lag_traces[first_key]],
            )
        )
    elif extra["drift_trace"] is not None:
        series.append(
            (
                "drift",
                _COLORS["drift"],
                [float(value) for value in extra["drift_trace"]],
            )
        )
    threshold = space_cfg.get("storm_threshold")
    write_forecast_svg(
        destination,
        title=title,
        series=series,
        hline=float(threshold) if threshold is not None else None,
        hline_label="Kp 5" if threshold is not None else None,
    )


def train_configs(
    paths: list[Path],
    *,
    chart_dir: str | Path = "data/space/charts",
    results_path: str | Path = "data/space/results.json",
) -> list[dict[str, Any]]:
    """Train each config and return JSON-ready score records."""
    charts = resolve_repo_path(chart_dir)
    records: list[dict[str, Any]] = []
    for path in paths:
        cfg = load_config(path)
        space_cfg = cfg.get("space") or {}
        logger.info("Training %s", path)
        trained = TimeSeriesTrainer(str(path)).train()
        trace = validation_frame(cfg)
        extra = extra_baseline_scores(cfg, trace)
        trainer_mae = float(trained["metrics"]["mae"])
        scored_mae = float(extra["model"]["mae"])
        if abs(trainer_mae - scored_mae) > 1e-3:
            raise RuntimeError(
                f"{path} checkpoint MAE {scored_mae} disagrees with the trainer MAE {trainer_mae}"
            )
        if space_cfg.get("chart"):
            _chart(cfg, trace, extra, charts / f"{space_cfg.get('id', path.stem)}.svg")
        dates = trace["dates"]
        rows = trace["rows"]
        record = {
            "id": space_cfg.get("id", path.stem),
            "config": str(path),
            "units": space_cfg.get("units", ""),
            "title": space_cfg.get("title", path.stem),
            "rows": int(len(trace["values"])),
            "val_rows": int(len(rows)),
            "val_start": None if dates is None else str(dates[int(rows[0])]),
            "val_end": None if dates is None else str(dates[int(rows[-1])]),
            "epochs_ran": len(trained["history"]),
            "best_epoch": _best_epoch(trained["history"]),
            "train_seconds": round(float(trained["elapsed_seconds"]), 1),
            "param_count": int(trained["param_count"]),
            "model": extra["model"],
            "trainer_baselines": trained["baselines"],
            "lags": extra["lags"],
            "drift": extra["drift"],
            "storms": extra["storms"],
        }
        records.append(_clean(record))
        _print_record(records[-1])
    destination = resolve_repo_path(results_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def _metric(block: dict[str, Any] | None, name: str) -> str:
    if not block or block.get(name) is None:
        return ""
    return f"{float(block[name]):.4f}"


def _print_record(record: dict[str, Any]) -> None:
    print(
        f"{record['id']}: val {record['val_start']}..{record['val_end']} "
        f"({record['val_rows']} steps), best epoch {record['best_epoch']}/"
        f"{record['epochs_ran']}, {record['train_seconds']}s"
    )
    baselines = record.get("trainer_baselines") or {}
    print(
        "  model MAE/RMSE "
        f"{_metric(record['model'], 'mae')} / {_metric(record['model'], 'rmse')}  "
        "persistence "
        f"{_metric(baselines.get('persistence'), 'mae')} / "
        f"{_metric(baselines.get('persistence'), 'rmse')}  "
        "calendar "
        f"{_metric(baselines.get('seasonal_naive'), 'mae')} / "
        f"{_metric(baselines.get('seasonal_naive'), 'rmse')}"
    )
    for name, scores in (record.get("lags") or {}).items():
        print(f"  {name} MAE/RMSE {_metric(scores, 'mae')} / {_metric(scores, 'rmse')}")
    if record.get("drift"):
        print(
            "  drift MAE/RMSE "
            f"{_metric(record['drift'], 'mae')} / {_metric(record['drift'], 'rmse')}"
        )
    storms = record.get("storms") or {}
    for name, scores in storms.items():
        print(
            f"  storm {name}: recall {_metric(scores, 'recall')} "
            f"precision {_metric(scores, 'precision')} "
            f"f1 {_metric(scores, 'f1')} "
            f"alarms {_metric(scores, 'alarms')} "
            f"storm_days {_metric(scores, 'storm_days')}"
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. See ``python -m src.space.train --help``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train the space-series configs.")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated config ids (the space.id field or the filename stem).",
    )
    args = parser.parse_args(argv)
    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    paths = config_paths()
    if selected:
        paths = [
            path
            for path in paths
            if path.stem in selected
            or str(load_config(path).get("space", {}).get("id")) in selected
        ]
        if not paths:
            raise SystemExit(f"No configs matched --only {sorted(selected)}")
    train_configs(paths)


if __name__ == "__main__":
    main()
