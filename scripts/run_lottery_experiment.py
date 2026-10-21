#!/usr/bin/env python3
"""Train the lottery config on a few seeds and compare it with chance.

Run from the repository root, after ``python scripts/fetch_powerball.py``
(the fetch also runs automatically when the CSV is missing):

    python scripts/run_lottery_experiment.py --seeds 0 1 2

Checkpoints are written under ``checkpoints/lottery/seed-<n>/`` and are not
committed. The default JSON report is ``results/lottery_sanity.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.lottery import fetch_and_write  # noqa: E402
from src.evaluation.lottery import config_for_seed, run_seed  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import resolve_repo_path  # noqa: E402


def _model_record(scored: dict[str, Any]) -> dict[str, Any]:
    training = scored["training"]
    return {
        "seed": training["seed"],
        "epochs_ran": training["epochs_ran"],
        "best_scaled_val_loss": training["best_scaled_val_loss"],
        "metrics": scored["model"],
        "exact_match_z": scored["exact_match_z"],
        "skill_vs_historical_mean_mse": scored["skill_vs_historical_mean_mse"],
    }


def _same_baselines(reports: list[dict[str, Any]]) -> None:
    signatures = [json.dumps(report["baselines"], sort_keys=True) for report in reports]
    if len(set(signatures)) != 1:
        raise RuntimeError(
            "baselines differed across seeds; the holdout and uniform seed are fixed"
        )


def build_report(reports: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Combine per-seed scores into one document. Baselines are seed-invariant."""
    if not reports:
        raise ValueError("no seed reports")
    _same_baselines(reports)
    first = reports[0]
    training = cfg["training"]
    model = cfg["model"]
    return {
        "schema_version": 1,
        "source": first["source"],
        "series": first["series"],
        "split": first["split"],
        "chance_exact_match_rate": first["chance_exact_match_rate"],
        "exact_match_null_se": first["exact_match_null_se"],
        "historical_mean_value": first["historical_mean_value"],
        "uniform_seed": first["uniform_seed"],
        "baselines": first["baselines"],
        "models": [_model_record(report) for report in reports],
        "hyperparameters": {
            "d_model": model["d_model"],
            "nhead": model["nhead"],
            "num_encoder_layers": model["num_encoder_layers"],
            "dim_feedforward": model["dim_feedforward"],
            "dropout": model["dropout"],
            "seq_len": cfg["data"]["seq_len"],
            "val_split": cfg["data"]["val_split"],
            "split_mode": cfg["data"]["split_mode"],
            "normalize": cfg["data"].get("normalize", True),
            "batch_size": training["batch_size"],
            "learning_rate": training["learning_rate"],
            "epochs": training["epochs"],
            "patience": training.get("patience", 0),
            "max_grad_norm": training.get("max_grad_norm", 0),
        },
    }


def _format_rate(metrics: dict[str, Any]) -> str:
    matches = metrics["exact_matches"]
    if isinstance(matches, float):
        shown = f"{matches:.1f}"
    else:
        shown = str(matches)
    return f"{shown}/{metrics['n']} ({float(metrics['exact_match_rate']) * 100:.2f}%)"


def format_table(report: dict[str, Any]) -> str:
    """Markdown table of model seeds and baselines."""
    lines = [
        "| Predictor | MAE | RMSE | Exact match | z vs chance |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    rows: list[tuple[str, dict[str, Any], str]] = []
    for model in report["models"]:
        rows.append(
            (
                f"Transformer seed {model['seed']}",
                model["metrics"],
                f"{model['exact_match_z']:.2f}",
            )
        )
    baselines = report["baselines"]
    rows.append(("Historical mean", baselines["historical_mean"], ""))
    rows.append(("Persistence (last draw)", baselines["persistence"], ""))
    rows.append(("Uniform expected", baselines["uniform_expected"], "0.00"))
    rows.append(
        (
            f"Uniform sample (seed {report['uniform_seed']})",
            baselines["uniform_sample"],
            "",
        )
    )
    for name, metrics, z_value in rows:
        z_cell = z_value if z_value else f"{_z(metrics, report):.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{float(metrics['mae']):.3f}",
                    f"{float(metrics['rmse']):.3f}",
                    _format_rate(metrics),
                    z_cell,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _z(metrics: dict[str, Any], report: dict[str, Any]) -> float:
    chance = float(report["chance_exact_match_rate"])
    se = float(report["exact_match_null_se"])
    return (float(metrics["exact_match_rate"]) - chance) / se


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/lottery.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--uniform-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/lottery_sanity.json",
        help="JSON report path (default: results/lottery_sanity.json)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    csv_path = resolve_repo_path(str(cfg["data"]["path"]))
    if not csv_path.is_file():
        print(f"{csv_path} is missing; fetching NY Open Data Powerball draws.")
        fetch_and_write(csv_path)

    reports: list[dict[str, Any]] = []
    for seed in args.seeds:
        checkpoint_dir = f"checkpoints/lottery/seed-{seed}"
        seeded = config_for_seed(cfg, seed, checkpoint_dir)
        print(f"Training seed {seed} -> {checkpoint_dir}")
        reports.append(run_seed(seeded, uniform_seed=args.uniform_seed))

    report = build_report(reports, cfg)
    destination = resolve_repo_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print()
    print(format_table(report))
    print()
    series = report["series"]
    print(
        f"Draws: {series['n_draws']} ({series['start']} to {series['end']}), "
        f"mean {series['mean']:.3f} (uniform {series['theoretical_mean']:.3f}), "
        f"std {series['std']:.3f} (uniform {series['theoretical_std']:.3f}), "
        f"lag-1 {series['lag1_correlation']}, "
        f"consecutive repeats {series['consecutive_repeat_rate']:.4f}."
    )
    print(
        "Chance exact match is "
        f"{report['chance_exact_match_rate']:.6f}; null SE on the holdout is "
        f"{report['exact_match_null_se']:.4f}. "
        "Seeds repeat the same holdout, so the z-scores are not independent tests."
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
