"""Score a saved checkpoint on one chronological split.

Training does not compute test metrics. Call this after the model is chosen,
and only once for the test split: those days are not a place to compare ideas.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.data.windows import prepare_csv_series
from src.evaluation.baselines import persistence_forecast, regression_scores
from src.evaluation.classical import covariate_frame, score_classical_baselines
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.paths import resolve_repo_path


def score_checkpoint(
    config_path: str = "config/weather.yaml",
    *,
    split: str = "validation",
    device: str = "cpu",
) -> dict[str, Any]:
    """Return RMSE and MAE for ``split`` (``validation`` or ``test``)."""
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    cfg = load_config(config_path)
    prepared = prepare_csv_series(cfg.get("data", {}))
    indexes = prepared.val_index if split == "validation" else prepared.test_index
    if len(indexes) == 0:
        raise ValueError(f"The {split} split is empty for {config_path}.")
    checkpoint = resolve_repo_path(cfg["inference"]["checkpoint_path"])
    model = build_model(cfg)
    predictor = Predictor.from_checkpoint(model, checkpoint, device=device)
    inputs = prepared.inputs[indexes]
    predictions = predictor.predict(inputs).cpu().numpy().reshape(-1)
    label_rows = indexes + int(cfg["data"]["seq_len"])
    predictions = _blend(
        predictions, cfg, prepared.target_values, prepared.series_dates, label_rows
    )
    targets = prepared.targets[indexes].numpy().reshape(-1)
    report: dict[str, Any] = dict(regression_scores(predictions, targets))
    if prepared.series_dates is not None:
        persistence = persistence_forecast(prepared.target_values, label_rows)
        report["persistence_rmse"] = regression_scores(persistence, targets)["rmse"]
        report["start"] = str(prepared.series_dates[int(label_rows[0])])
        report["end"] = str(prepared.series_dates[int(label_rows[-1])])
    report["n"] = int(len(indexes))
    report["predictions"] = predictions
    report["targets"] = targets
    return report


def _blend(
    predictions: Any,
    cfg: dict[str, Any],
    values: Any,
    dates: Any,
    label_rows: Any,
) -> Any:
    data_cfg = cfg.get("data", {})
    blend_with = data_cfg.get("blend_with") or None
    if not blend_with:
        return predictions
    if dates is None:
        raise ValueError("data.blend_with requires data.date_column.")
    frame = pd.read_csv(resolve_repo_path(data_cfg["path"]))
    target_column = data_cfg.get("target_column", "value")
    columns = [
        column
        for column in data_cfg.get("feature_columns", [])
        if column != target_column
        and column in frame.columns
        and not str(column).endswith(("_sin", "_cos"))
    ]
    classical = score_classical_baselines(
        values,
        label_rows,
        dates=dates,
        covariates=covariate_frame(frame, columns, int(label_rows.min())),
        seed=int(cfg.get("training", {}).get("seed", 0) or 0),
        boosting_params=data_cfg.get("boosting_params"),
    )
    if blend_with not in classical:
        raise ValueError(f"Unknown blend partner {blend_with!r}.")
    weight = float(data_cfg.get("blend_weight", 0.5))
    return weight * predictions + (1.0 - weight) * classical[blend_with]


def main(argv: list[str] | None = None) -> None:
    """Print one split's scores. ``--split test`` is the held-out report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/weather.yaml")
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    scores = score_checkpoint(args.config, split=args.split, device=args.device)
    print(
        f"{args.split} n={scores['n']} RMSE={scores['rmse']:.4f} "
        f"MAE={scores['mae']:.4f}"
    )
    if "persistence_rmse" in scores:
        print(
            f"persistence RMSE={scores['persistence_rmse']:.4f} "
            f"({scores['start']} through {scores['end']})"
        )


if __name__ == "__main__":
    main()
