"""Train one small transformer per market series and score the holdout.

The catalogs in ``config/stocks.yaml``, ``config/fx.yaml``, ``config/crypto.yaml``,
and ``config/commodities.yaml`` supply the architecture, the temporal split, the
seed, and early stopping. Each series becomes an ordinary trainer config pointed
at that series' return CSV. The reported metrics come from the best checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from src.markets.evaluate import score_checkpoint
from src.markets.report import (
    DISCLAIMER,
    GROUP_ORDER,
    group_lead,
    readme_section,
    summarize_group,
)
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config
from src.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)

CONFIG_FILES = {
    "stocks": "config/stocks.yaml",
    "fx": "config/fx.yaml",
    "crypto": "config/crypto.yaml",
    "commodities": "config/commodities.yaml",
}

PROTOCOL_KEYS = (
    "d_model",
    "layers",
    "seq_len",
    "val_split",
    "epochs",
    "patience",
    "seed",
    "batch_size",
    "learning_rate",
)


def protocol_view(cfg: dict[str, Any]) -> dict[str, Any]:
    model = cfg["model"]
    data = cfg["data"]
    training = cfg["training"]
    return {
        "d_model": int(model["d_model"]),
        "layers": int(model["num_encoder_layers"]),
        "nhead": int(model["nhead"]),
        "seq_len": int(data["seq_len"]),
        "val_split": float(data["val_split"]),
        "split_mode": str(data["split_mode"]),
        "normalize": bool(data.get("normalize", True)),
        "epochs": int(training["epochs"]),
        "patience": int(training["patience"]),
        "seed": int(training["seed"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "features": list(data["feature_columns"]),
        "target": str(data["target_column"]),
    }


def load_catalogs() -> dict[str, dict[str, Any]]:
    catalogs = {group: load_config(path) for group, path in CONFIG_FILES.items()}
    views = {group: protocol_view(cfg) for group, cfg in catalogs.items()}
    first = next(iter(views.values()))
    drifted = [group for group, view in views.items() if view != first]
    if drifted:
        raise ValueError(
            "market configs must share one training protocol; drifted: "
            + ", ".join(drifted)
        )
    return catalogs


def materialize(
    catalog: dict[str, Any],
    series_id: str,
    feature_path: str,
    checkpoint_dir: str,
    db_path: str,
) -> dict[str, Any]:
    """Build the trainer config for one series. Paths stay repository-relative."""
    data = dict(catalog["data"])
    data["source"] = "csv"
    data["path"] = feature_path
    training = dict(catalog["training"])
    training["experiment_name"] = f"market-{series_id}"
    training["checkpoint_dir"] = checkpoint_dir
    model = dict(catalog["model"])
    features = list(data["feature_columns"])
    if int(model["input_dim"]) != len(features):
        raise ValueError(
            f"{series_id}: input_dim {model['input_dim']} != {len(features)} features"
        )
    return {
        "model": model,
        "data": data,
        "training": training,
        "storage": {"db_path": db_path},
    }


def train_one(
    catalog: dict[str, Any],
    coverage: dict[str, Any],
    work_dir: Path,
    *,
    checkpoint_dir: str | None = None,
    db_path: str = "data/markets/experiments.db",
) -> dict[str, Any]:
    """Train one series and score the best checkpoint on the recent holdout."""
    series_id = str(coverage["id"])
    feature_path = str(coverage["feature_path"])
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/markets/{series_id}"
    cfg = materialize(
        catalog,
        series_id,
        feature_path,
        checkpoint_dir,
        db_path,
    )
    config_path = work_dir / f"{series_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    started = time.perf_counter()
    trained = TimeSeriesTrainer(str(config_path)).train()
    elapsed = time.perf_counter() - started
    checkpoint = resolve_repo_path(checkpoint_dir) / "best_model.pt"
    scored = score_checkpoint(resolve_repo_path(feature_path), checkpoint, cfg)
    digest = hashlib.sha256(resolve_repo_path(feature_path).read_bytes()).hexdigest()
    scored["epochs_ran"] = len(trained["history"])
    scored["train_seconds"] = elapsed
    scored["experiment_id"] = trained["experiment_id"]
    scored["feature_sha256"] = digest
    return scored


def _coverage_rows(
    payload: dict[str, Any], groups: set[str], only: set[str] | None
) -> list[dict[str, Any]]:
    rows = []
    for row in payload["series"]:
        if str(row["group"]) not in groups:
            continue
        if only is not None and str(row["id"]) not in only:
            continue
        if row.get("train"):
            rows.append(row)
    return rows


def run_training(
    coverage_path: Path,
    results_path: Path,
    *,
    groups: set[str] | None = None,
    only: set[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train every selected series and write the results JSON."""
    coverage = json.loads(coverage_path.read_text())
    catalogs = load_catalogs()
    protocol = protocol_view(next(iter(catalogs.values())))
    selected_groups = set(groups or GROUP_ORDER)
    todo = _coverage_rows(coverage, selected_groups, only)
    done: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    todo_ids = {str(row["id"]) for row in todo}
    if resume and results_path.exists():
        previous = json.loads(results_path.read_text())
        for row in previous.get("series") or []:
            if "model_rmse" in row:
                done[str(row["id"])] = row
        failures.extend(
            item
            for item in previous.get("failures") or []
            if item["id"] not in todo_ids
        )
    work_dir = resolve_repo_path("data/markets/runs")
    started = time.perf_counter()
    for index, row in enumerate(todo, start=1):
        series_id = str(row["id"])
        if series_id in done:
            continue
        catalog = catalogs[str(row["group"])]
        print(f"[{index}/{len(todo)}] training {series_id}", flush=True)
        try:
            metrics = train_one(catalog, row, work_dir)
        except Exception as exc:
            logger.exception("training failed for %s", series_id)
            failures.append({"id": series_id, "error": str(exc)})
            _write_partial(results_path, coverage, protocol, done, failures, started)
            continue
        merged = {**row, **metrics}
        done[series_id] = merged
        ratio = float(metrics["rmse_ratio_vs_persistence"])
        direction = 100.0 * float(metrics["directional_accuracy"])
        print(
            f"  {series_id} {metrics['train_seconds']:.1f}s "
            f"rmse {metrics['model_rmse']:.5f} "
            f"persistence {metrics['persistence_rmse']:.5f} "
            f"ratio {ratio:.3f} dir {direction:.1f}%",
            flush=True,
        )
        _write_partial(results_path, coverage, protocol, done, failures, started)
    return _write_partial(results_path, coverage, protocol, done, failures, started)


def _write_partial(
    results_path: Path,
    coverage: dict[str, Any],
    protocol: dict[str, Any],
    done: dict[str, dict[str, Any]],
    failures: list[dict[str, str]],
    started: float,
) -> dict[str, Any]:
    series = [done[key] for key in sorted(done)]
    by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUP_ORDER}
    for row in series:
        by_group.setdefault(str(row["group"]), []).append(row)
    summaries = {
        group: summarize_group(rows) for group, rows in by_group.items() if rows
    }
    narratives = {
        group: group_lead(group, summary) for group, summary in summaries.items()
    }
    payload: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "protocol": protocol,
        "train_seconds": time.perf_counter() - started,
        "summaries": summaries,
        "narratives": narratives,
        "failures": failures,
        "series": series,
    }
    # Preserve the cumulative training time already stored when resuming.
    if series:
        payload["train_seconds"] = float(
            sum(float(row.get("train_seconds") or 0) for row in series)
        )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2) + "\n")
    section_path = results_path.with_name("market_section.md")
    section_path.write_text(readme_section(payload, coverage))
    coverage_copy = results_path.with_name("market_coverage.json")
    coverage_copy.write_text(json.dumps(coverage, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score market return models")
    parser.add_argument("--coverage", default="data/markets/coverage.json")
    parser.add_argument("--results", default="results/market_forecasts.json")
    parser.add_argument("--group", action="append", choices=list(CONFIG_FILES))
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger("src.training.trainer").setLevel(logging.WARNING)
    payload = run_training(
        resolve_repo_path(args.coverage),
        resolve_repo_path(args.results),
        groups=set(args.group) if args.group else None,
        only=set(args.ids) if args.ids else None,
        resume=bool(args.resume),
    )
    failures = payload.get("failures") or []
    print(
        f"trained {len(payload['series'])} series in {payload['train_seconds']:.0f}s; "
        f"failures {len(failures)}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
