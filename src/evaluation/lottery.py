"""Baselines for the Powerball sanity check.

The transformer is scored in original Powerball units on a temporal holdout.
Exact matches use the rounded prediction. Chance for a uniform draw on
``low..high`` is ``1 / (high - low + 1)``, which is 1/26 for the current
Powerball.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.lottery import (
    POWERBALL_HIGH,
    POWERBALL_LOW,
    draw_date_span,
    powerball_series,
    source_metadata,
)
from src.data.timeseries import load_csv_timeseries
from src.evaluation.metrics import mae, mse, rmse
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.training.trainer import TimeSeriesTrainer
from src.utils.paths import resolve_repo_path

Score = dict[str, float | int]


def chance_exact_match_rate(low: int, high: int) -> float:
    """Probability a uniform guess on ``low..high`` hits one draw."""
    span = _span(low, high)
    return 1.0 / span


def exact_match_null_se(n: int, low: int, high: int) -> float:
    """Binomial standard error of the chance exact-match rate."""
    if n < 1:
        raise ValueError("n must be positive")
    p = chance_exact_match_rate(low, high)
    return float(np.sqrt(p * (1.0 - p) / n))


def exact_match_z(rate: float, n: int, low: int, high: int) -> float:
    """How many null standard errors ``rate`` sits from chance."""
    se = exact_match_null_se(n, low, high)
    return (rate - chance_exact_match_rate(low, high)) / se


def temporal_window_split(
    n_rows: int, seq_len: int, val_split: float
) -> dict[str, int]:
    """Match ``TimeSeriesTrainer``'s temporal holdout.

    ``n_val = int(n_windows * val_split)`` and the validation windows are the
    most recent ones. ``first_val_target`` is the series index of the first
    held-out draw.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if n_rows <= seq_len:
        raise ValueError("need more rows than seq_len")
    if not 0.0 < float(val_split) < 1.0:
        raise ValueError("val_split must be in (0, 1)")
    n_windows = n_rows - seq_len
    n_val = int(n_windows * float(val_split))
    n_fit = n_windows - n_val
    if n_val == 0 or n_fit == 0:
        raise ValueError("val_split leaves an empty train or validation set")
    return {
        "n_windows": n_windows,
        "n_fit": n_fit,
        "n_val": n_val,
        "first_val_target": seq_len + n_fit,
    }


def round_to_range(preds: np.ndarray, low: int, high: int) -> np.ndarray:
    """Round to nearest integer (half to even) and clip to ``low..high``."""
    _span(low, high)
    values = np.asarray(preds, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("no predictions")
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    rounded = np.rint(values)
    return np.clip(rounded, low, high).astype(np.int64)


def _span(low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"invalid range {low}..{high}")
    return high - low + 1


def _as_int_targets(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("no targets")
    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        raise ValueError("targets must be integer lottery numbers")
    return rounded.astype(np.int64)


def score_continuous(
    preds: np.ndarray, targets: np.ndarray, low: int, high: int
) -> Score:
    """MSE, RMSE, and MAE in original units, plus rounded exact matches."""
    target_ints = _as_int_targets(targets)
    pred_values = np.asarray(preds, dtype=np.float64).reshape(-1)
    if pred_values.shape != target_ints.shape:
        raise ValueError(
            f"got {pred_values.size} predictions for {target_ints.size} targets"
        )
    rounded = round_to_range(pred_values, low, high)
    torch_preds = torch.tensor(pred_values, dtype=torch.float32).unsqueeze(-1)
    torch_targets = torch.tensor(target_ints, dtype=torch.float32).unsqueeze(-1)
    matches = int(np.sum(rounded == target_ints))
    return {
        "mse": mse(torch_preds, torch_targets),
        "rmse": rmse(torch_preds, torch_targets),
        "mae": mae(torch_preds, torch_targets),
        "exact_match_rate": matches / int(target_ints.size),
        "exact_matches": matches,
        "n": int(target_ints.size),
    }


def expected_uniform_score(targets: np.ndarray, low: int, high: int) -> Score:
    """Expected error of an independent uniform guess on ``low..high``.

    The expectation is over the guess. Targets stay fixed, so this is the
    average a random-guess baseline would reach on this holdout.
    """
    target_ints = _as_int_targets(targets)
    grid = np.arange(low, high + 1, dtype=np.float64)
    diff = target_ints.astype(np.float64)[:, None] - grid[None, :]
    mse_value = float(np.mean(diff**2))
    mae_value = float(np.mean(np.abs(diff)))
    n = int(target_ints.size)
    rate = chance_exact_match_rate(low, high)
    return {
        "mse": mse_value,
        "rmse": float(np.sqrt(mse_value)),
        "mae": mae_value,
        "exact_match_rate": rate,
        "exact_matches": n * rate,
        "n": n,
    }


def uniform_sample(n: int, low: int, high: int, seed: int) -> np.ndarray:
    """Seeded uniform integers on ``low..high`` (inclusive)."""
    if n < 1:
        raise ValueError("n must be positive")
    _span(low, high)
    return np.random.default_rng(seed).integers(low, high + 1, size=n)


def lag1_correlation(series: np.ndarray) -> float | None:
    """Pearson correlation of each draw with the previous one.

    ``None`` when either side has no variance.
    """
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if values.size < 3:
        raise ValueError("need at least 3 draws")
    left, right = values[:-1], values[1:]
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def consecutive_repeat_rate(series: np.ndarray) -> float:
    """Fraction of draws that equal the immediately previous draw."""
    values = _as_int_targets(series)
    if values.size < 2:
        raise ValueError("need at least 2 draws")
    return float(np.mean(values[1:] == values[:-1]))


def series_diagnostics(series: np.ndarray, low: int, high: int) -> dict[str, Any]:
    """Marginal summary of a draw series, compared with a discrete uniform."""
    values = _as_int_targets(series)
    if values.min() < low or values.max() > high:
        raise ValueError(f"series values fall outside {low}..{high}")
    grid = np.arange(low, high + 1, dtype=np.float64)
    return {
        "n_draws": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "theoretical_mean": float(grid.mean()),
        "theoretical_std": float(grid.std(ddof=0)),
        "lag1_correlation": lag1_correlation(values),
        "consecutive_repeat_rate": consecutive_repeat_rate(values),
        "chance_exact_match_rate": chance_exact_match_rate(low, high),
    }


def baseline_report(
    series: np.ndarray,
    *,
    seq_len: int,
    val_split: float,
    low: int = POWERBALL_LOW,
    high: int = POWERBALL_HIGH,
    uniform_seed: int = 0,
) -> dict[str, Any]:
    """Score mean, persistence, and uniform baselines on the temporal holdout.

    The historical mean is the mean of every draw strictly before the first
    held-out target. Persistence predicts the previous draw. Neither uses a
    future number.
    """
    values = _as_int_targets(series).astype(np.float64)
    split = temporal_window_split(int(values.size), seq_len, val_split)
    start = split["first_val_target"]
    history = values[:start]
    targets = values[start:]
    if int(targets.size) != split["n_val"]:
        raise ValueError("holdout length does not match the temporal split")
    persistence = values[start - 1 : start - 1 + split["n_val"]]
    mean_value = float(history.mean())
    mean_preds = np.full(split["n_val"], mean_value, dtype=np.float64)
    sampled = uniform_sample(split["n_val"], low, high, uniform_seed).astype(np.float64)
    return {
        "split": split,
        "historical_mean_value": mean_value,
        "baselines": {
            "historical_mean": score_continuous(mean_preds, targets, low, high),
            "persistence": score_continuous(persistence, targets, low, high),
            "uniform_sample": score_continuous(sampled, targets, low, high),
            "uniform_expected": expected_uniform_score(targets, low, high),
        },
    }


def _lottery_data_config(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("config is missing data")
    features = data_cfg.get("feature_columns")
    target = data_cfg.get("target_column")
    if features != ["powerball"] or target != "powerball":
        raise ValueError(
            "lottery evaluation expects feature_columns ['powerball'] "
            "and target_column powerball"
        )
    if data_cfg.get("split_mode") != "temporal":
        raise ValueError("lottery evaluation requires data.split_mode: temporal")
    if data_cfg.get("source") != "csv":
        raise ValueError("lottery evaluation requires data.source: csv")
    return data_cfg


def evaluate_checkpoint(
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    uniform_seed: int = 0,
    low: int = POWERBALL_LOW,
    high: int = POWERBALL_HIGH,
) -> dict[str, Any]:
    """Score a trained checkpoint and the baselines on the same holdout."""
    data_cfg = _lottery_data_config(cfg)
    csv_path = resolve_repo_path(str(data_cfg["path"]))
    series = powerball_series(csv_path)
    start_date, end_date = draw_date_span(csv_path)
    seq_len = int(data_cfg.get("seq_len", 64))
    val_split = float(data_cfg.get("val_split", 0.2))
    report = baseline_report(
        series,
        seq_len=seq_len,
        val_split=val_split,
        low=low,
        high=high,
        uniform_seed=uniform_seed,
    )
    split = report["split"]
    inputs, targets = load_csv_timeseries(
        csv_path,
        feature_columns=["powerball"],
        target_column="powerball",
        seq_len=seq_len,
    )
    if int(inputs.shape[0]) != int(split["n_windows"]):
        raise ValueError("CSV windows do not match the temporal split")
    n_fit = int(split["n_fit"])
    val_inputs = inputs[n_fit:]
    val_targets = targets[n_fit:].reshape(-1).numpy()
    expected_targets = series[int(split["first_val_target"]) :]
    if not np.allclose(val_targets, expected_targets):
        raise ValueError("validation targets do not match the Powerball series")

    model = build_model(cfg)
    predictor = Predictor.from_checkpoint(model, checkpoint_path, device="cpu")
    predictions = predictor.predict(val_inputs).detach().cpu().reshape(-1).numpy()
    model_score = score_continuous(predictions, val_targets, low, high)
    n_val = int(model_score["n"])
    diagnostics = series_diagnostics(series, low, high)
    diagnostics["start"] = start_date
    diagnostics["end"] = end_date
    mean_mse = float(report["baselines"]["historical_mean"]["mse"])
    model_mse = float(model_score["mse"])
    skill = None if mean_mse == 0.0 else 1.0 - model_mse / mean_mse
    return {
        "source": source_metadata(),
        "series": diagnostics,
        "split": {
            **split,
            "seq_len": seq_len,
            "val_split": val_split,
            "split_mode": "temporal",
        },
        "chance_exact_match_rate": chance_exact_match_rate(low, high),
        "exact_match_null_se": exact_match_null_se(n_val, low, high),
        "historical_mean_value": report["historical_mean_value"],
        "baselines": report["baselines"],
        "model": model_score,
        "exact_match_z": exact_match_z(
            float(model_score["exact_match_rate"]), n_val, low, high
        ),
        "skill_vs_historical_mean_mse": skill,
        "uniform_seed": uniform_seed,
    }


def config_for_seed(
    cfg: dict[str, Any], seed: int, checkpoint_dir: str
) -> dict[str, Any]:
    """Copy ``cfg`` with a fixed training seed and an isolated checkpoint path."""
    seeded = copy.deepcopy(cfg)
    training = seeded.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training config must be a mapping")
    base_name = str(training.get("experiment_name", "lottery"))
    training["seed"] = int(seed)
    training["experiment_name"] = f"{base_name}-seed-{int(seed)}"
    training["checkpoint_dir"] = checkpoint_dir
    inference = seeded.setdefault("inference", {})
    if not isinstance(inference, dict):
        raise ValueError("inference config must be a mapping")
    inference["checkpoint_path"] = str(Path(checkpoint_dir) / "best_model.pt")
    inference["device"] = training.get("device", "cpu")
    return seeded


def run_seed(cfg: dict[str, Any], *, uniform_seed: int = 0) -> dict[str, Any]:
    """Train one config and score its checkpoint.

    Checkpoints and the experiment database are written where ``cfg`` points.
    """
    checkpoint = resolve_repo_path(str(cfg["inference"]["checkpoint_path"]))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(cfg, handle)
        config_path = handle.name
    try:
        trained = TimeSeriesTrainer(config_path).train()
    finally:
        Path(config_path).unlink(missing_ok=True)
    scored = evaluate_checkpoint(cfg, checkpoint, uniform_seed=uniform_seed)
    scored["training"] = {
        "seed": cfg["training"].get("seed"),
        "epochs_ran": len(trained["history"]),
        "best_scaled_val_loss": float(trained["best_loss"]),
        "experiment_id": trained["experiment_id"],
    }
    return scored
