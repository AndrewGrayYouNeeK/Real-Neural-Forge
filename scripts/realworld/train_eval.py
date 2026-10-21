"""Train each real-world config and score it against the simple baselines.

Checkpoints stay in ``checkpoints/realworld/`` (gitignored). The metrics JSON
is the small artifact worth keeping.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from scripts.realworld.plots import write_mae_bars
from src.realworld.charts import svg_bars
from src.realworld.evaluate import compare_checkpoint
from src.training.trainer import TimeSeriesTrainer
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

CONFIG_DIR = REPO_ROOT / "config" / "realworld"
REPORT_PATH = REPO_ROOT / "reports" / "realworld" / "forecast_results.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "realworld" / "conflict_summary.json"
GAP_CHART_DIR = Path("/opt/cursor/artifacts/conflict-spells")


def main() -> list[dict[str, Any]]:
    """Train every config in filename order and write ``forecast_results.json``."""
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"no configs in {CONFIG_DIR}")
    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for path in configs:
        logger.info("training %s", path.name)
        train_started = time.perf_counter()
        TimeSeriesTrainer(str(path)).train()
        elapsed = time.perf_counter() - train_started
        row = compare_checkpoint(path)
        row["train_seconds"] = elapsed
        runs.append(row)
        model_mae = row["model"]["mae"]
        baseline = row["best_simple_baseline"]
        baseline_mae = row["baselines"][baseline]["mae"]
        logger.info(
            "%s MAE model %.4g %s %s %.4g in %.1fs",
            row["id"],
            model_mae,
            row["units"],
            baseline,
            baseline_mae,
            elapsed,
        )
    payload = {
        "total_train_seconds": time.perf_counter() - started,
        "runs": runs,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    _write_gap_forecast_chart(runs)
    logger.info(
        "wrote %s (%.1f minutes)", REPORT_PATH, payload["total_train_seconds"] / 60
    )
    return runs


def _write_gap_forecast_chart(runs: list[dict[str, Any]]) -> None:
    """Holdout MAE for the onset-gap series, including the Poisson baseline."""
    row = next((item for item in runs if item.get("id") == "ucdp_onset_gaps"), None)
    if row is None:
        return
    pairs: list[tuple[str, float]] = [
        ("transformer", float(row["model"]["mae"])),
        ("mean gap", float(row["baselines"]["mean"]["mae"])),
        ("last gap", float(row["baselines"]["persistence"]["mae"])),
        ("seasonal naive", float(row["baselines"]["seasonal_naive"]["mae"])),
    ]
    trend = row["baselines"].get("exponential_trend")
    if isinstance(trend, dict) and "mae" in trend:
        pairs.append(("exponential trend", float(trend["mae"])))
    if SUMMARY_PATH.is_file():
        summary = json.loads(SUMMARY_PATH.read_text())
        poisson = (
            summary.get("onset_gaps", {})
            .get("forecast_baselines", {})
            .get("baselines", {})
            .get("poisson_trend")
        )
        if isinstance(poisson, dict) and "mae" in poisson:
            pairs.append(("poisson trend", float(poisson["mae"])))
    pairs.sort(key=lambda item: item[1])
    labels = [label for label, _value in pairs]
    values = [value for _label, value in pairs]
    GAP_CHART_DIR.mkdir(parents=True, exist_ok=True)
    title = "Holdout MAE for days between new conflict onsets"
    (GAP_CHART_DIR / "gap_forecast_mae.svg").write_text(
        svg_bars(labels, values, title, "MAE (days)")
    )
    write_mae_bars(
        GAP_CHART_DIR / "gap_forecast_mae.png", labels, values, title, "MAE (days)"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
