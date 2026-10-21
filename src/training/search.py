"""Reproducible hyperparameter search for one forecasting config.

The objective is the temporal validation RMSE of a config built from the base
YAML. Pass a base config whose ``val_start`` is an inner slice and whose
``test_start`` is the decision-validation boundary when the decision split must
stay untouched: this module scores whatever split the config calls validation,
and it never reads a test split.

Budget, neural family (the default): 12 trials, at most 20 epochs, patience 5.
RevIN, patching, and the residual link are not searched: a paired ablation on
the decision split kept the residual link and dropped the other two. The
search tunes width, depth, dropout, weight decay, learning rate, and context
length inside that architecture. Each trial trains on CPU. The sampler seed is
0. A trial worse than the running median after 6 scored epochs is pruned.

Budget, boosting family: 20 trials of ``HistGradientBoostingRegressor`` on the
same classical design matrix. Each trial is a few seconds on the Central Park
series.

Results are written to ``<checkpoint_dir>/search/trials.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml

from src.data.windows import prepare_csv_series
from src.evaluation.classical import covariate_frame, score_classical_baselines
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config
from src.utils.paths import resolve_repo_path

NEURAL_TRIALS = 12
NEURAL_EPOCHS = 20
NEURAL_PATIENCE = 5
BOOSTING_TRIALS = 20


def search_neural(
    config_path: str, *, n_trials: int = NEURAL_TRIALS, seed: int = 0
) -> dict[str, Any]:
    """Search width, depth, dropout, weight decay, learning rate, and context length."""
    base = load_config(config_path)
    output_dir = (
        resolve_repo_path(base.get("training", {}).get("checkpoint_dir", "checkpoints"))
        / "search"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_dir / 'neural.db'}"

    def objective(trial: optuna.Trial) -> float:
        cfg = _neural_config(base, trial, output_dir)
        trial_path = output_dir / f"trial_{trial.number}.yaml"
        trial_path.write_text(yaml.safe_dump(cfg))
        trainer = TimeSeriesTrainer(str(trial_path))

        def report(epoch: int, value: float) -> None:
            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            result = trainer.train(epoch_callback=report)
        except optuna.TrialPruned:
            raise
        return float(result["metrics"]["rmse"])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=6),
        study_name="neural",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError, ValueError))
    _write_trials(output_dir / "neural_trials.csv", study)
    return {
        "best_rmse": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": n_trials,
    }


def search_boosting(
    config_path: str, *, n_trials: int = BOOSTING_TRIALS, seed: int = 0
) -> dict[str, Any]:
    """Search gradient-boosting depth and shrinkage on the validation split."""
    base = load_config(config_path)
    prepared_path = base["data"]["path"]
    prepared = prepare_csv_series(base["data"])
    if prepared.series_dates is None:
        raise ValueError("Boosting search requires data.date_column.")
    target_rows = prepared.val_index + int(base["data"]["seq_len"])
    frame = pd.read_csv(resolve_repo_path(prepared_path))
    target_column = base["data"].get("target_column", "value")
    columns = [
        column
        for column in base["data"].get("feature_columns", [])
        if column != target_column
        and column in frame.columns
        and not str(column).endswith(("_sin", "_cos"))
    ]
    covariates = covariate_frame(frame, columns, int(target_rows.min()))
    output_dir = (
        resolve_repo_path(base.get("training", {}).get("checkpoint_dir", "checkpoints"))
        / "search"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_iter": trial.suggest_categorical("max_iter", [150, 300, 500]),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", [10, 20, 40]
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 10.0, log=True
            ),
        }
        forecasts = score_classical_baselines(
            prepared.target_values,
            target_rows,
            dates=prepared.series_dates,
            covariates=covariates,
            seed=seed,
            boosting_params=params,
        )
        residual = forecasts["gradient_boosting"] - prepared.target_values[target_rows]
        return float((residual**2).mean() ** 0.5)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="boosting",
        storage=f"sqlite:///{output_dir / 'boosting.db'}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    _write_trials(output_dir / "boosting_trials.csv", study)
    return {
        "best_rmse": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": n_trials,
    }


def _neural_config(
    base: dict[str, Any], trial: optuna.Trial, output_dir: Path
) -> dict[str, Any]:
    loaded = yaml.safe_load(yaml.safe_dump(base))
    if not isinstance(loaded, dict):
        raise TypeError("Config round-trip did not produce a mapping.")
    cfg: dict[str, Any] = loaded
    model = cfg["model"]
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    if d_model % nhead != 0:
        raise optuna.TrialPruned()
    model["d_model"] = d_model
    model["nhead"] = nhead
    model["num_encoder_layers"] = trial.suggest_int("num_encoder_layers", 1, 3)
    model["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [128, 256])
    model["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    # Residual forecasting won the ablation. RevIN and patching did not.
    model["patch_len"] = 1
    model["use_revin"] = False
    model["residual"] = True
    model["ensemble_size"] = 1
    cfg["data"]["seq_len"] = trial.suggest_categorical("seq_len", [14, 28, 56])
    cfg["training"].pop("ensemble_seeds", None)
    training = cfg["training"]
    training["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-3, log=True
    )
    training["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    training["epochs"] = min(int(training.get("epochs", NEURAL_EPOCHS)), NEURAL_EPOCHS)
    training["patience"] = NEURAL_PATIENCE
    training["checkpoint_dir"] = str(output_dir / f"trial_{trial.number}")
    training["log_interval"] = training["epochs"]
    cfg.setdefault("storage", {})["db_path"] = str(
        output_dir / f"trial_{trial.number}.db"
    )
    return cfg


def _write_trials(path: Path, study: optuna.Study) -> None:
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "value": "" if trial.value is None else f"{trial.value:.6f}",
        }
        for key, value in trial.params.items():
            row[key] = value
        rows.append(row)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/weather.yaml")
    parser.add_argument("--family", choices=("neural", "boosting"), default="neural")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args(argv)
    if args.family == "neural":
        result = search_neural(args.config, n_trials=args.trials or NEURAL_TRIALS)
    else:
        result = search_boosting(args.config, n_trials=args.trials or BOOSTING_TRIALS)
    print(
        f"{args.family} best validation RMSE {result['best_rmse']:.4f} "
        f"over {result['n_trials']} trials"
    )
    print(result["best_params"])


if __name__ == "__main__":
    main()
