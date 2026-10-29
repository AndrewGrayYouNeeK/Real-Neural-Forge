"""Training loop for time-series models."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from src.data.prepare import load_series, resolve_split_mode, split_indices
from src.data.scaling import StandardScaler
from src.data.windows import prepare_csv_series
from src.evaluation.baselines import (
    compare_to_baselines,
    format_score_table,
    load_baseline_series,
    regression_scores,
    temporal_validation_target_rows,
)
from src.evaluation.classical import covariate_frame, score_classical_baselines
from src.evaluation.metrics import mae, mse, rmse
from src.forecasting.intervals import calibrate_intervals, pinball_loss
from src.forecasting.output import horizon_targets, training_targets
from src.forecasting.spec import ForecastSpec, forecast_spec_from_config
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.storage.experiments import ExperimentStore, db_path_from_config
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config, merge_config
from src.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)

EpochCallback = Callable[[int, float], None]


def ensemble_seeds(cfg: dict[str, Any]) -> list[int | None]:
    """Seeds for one run, or one seed per ensemble member."""
    size = int(cfg.get("model", {}).get("ensemble_size", 1) or 1)
    training = cfg.get("training", {})
    configured = training.get("ensemble_seeds")
    if size <= 1:
        if configured:
            raise ValueError(
                "training.ensemble_seeds requires model.ensemble_size greater than 1."
            )
        seed = training.get("seed")
        return [None if seed is None else int(seed)]
    if configured is None:
        base = int(training.get("seed", 0) or 0)
        return [base + offset for offset in range(size)]
    seeds: list[int | None] = [int(seed) for seed in configured]
    if len(seeds) != size:
        raise ValueError(
            f"model.ensemble_size is {size} but {len(seeds)} ensemble seeds were given."
        )
    return seeds


class TimeSeriesTrainer:
    """End-to-end trainer for the time-series transformer."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        *,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = config_path
        self.cfg = load_config(config_path)
        if overrides:
            self.cfg = merge_config(self.cfg, overrides)
        self.store = ExperimentStore(db_path_from_config(self.cfg))

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available – falling back to CPU.")
            device_str = "cpu"
        return torch.device(device_str)

    def train(
        self,
        epoch_callback: EpochCallback | None = None,
        *,
        checkpoint_name: str = "best_model.pt",
    ) -> dict[str, Any]:
        cfg = self.cfg
        t_cfg = cfg["training"]
        device = self._resolve_device(t_cfg.get("device", "cpu"))
        logger.info("Using device: %s", device)
        started = time.perf_counter()

        spec = forecast_spec_from_config(cfg)
        seeds = ensemble_seeds(cfg)
        if len(seeds) == 1 and seeds[0] is not None:
            torch.manual_seed(int(seeds[0]))
        if len(seeds) > 1 and (spec.model_horizon != 1 or spec.quantiles):
            raise ValueError("Seed ensembles train one-step point models.")

        # Members of a seed ensemble are averaged after independent fits. The
        # training loop below fits one network; ensemble_size 1 is the default.
        model = build_model(cfg, ensemble=False).to(device)
        if len(seeds) > 1:
            raise ValueError(
                "model.ensemble_size above 1 is available on the model, but this "
                "training loop fits one seed. Set ensemble_size to 1."
            )
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model parameters: %d", param_count)

        weight_decay = float(t_cfg.get("weight_decay", 0) or 0)
        if weight_decay < 0:
            raise ValueError("training.weight_decay must be >= 0")
        if weight_decay > 0:
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=t_cfg["learning_rate"],
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
        mse_loss = nn.MSELoss()
        quantiles = spec.quantiles

        def criterion(preds: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
            if quantiles:
                return pinball_loss(preds, observed, quantiles)
            return cast(torch.Tensor, mse_loss(preds, observed))

        data_cfg = cfg.get("data", {})
        seq_len = int(data_cfg.get("seq_len", 64))
        prepared = None
        uses_dated_windows = bool(
            data_cfg.get("source") == "csv"
            and data_cfg.get("path")
            and (
                data_cfg.get("val_start")
                or data_cfg.get("test_start")
                or data_cfg.get("missing_indicators")
                or data_cfg.get("calendar_features")
                or data_cfg.get("future_calendar")
                or data_cfg.get("lunar_features")
                or data_cfg.get("future_lunar")
                or float(data_cfg.get("test_split", 0) or 0) > 0
            )
        )
        if uses_dated_windows:
            if spec.horizon != 1 or spec.quantiles:
                raise ValueError(
                    "Date splits and missing indicators score one-step point targets."
                )
            prepared = prepare_csv_series(data_cfg)
            inputs, targets = prepared.inputs, prepared.targets
            feature_count = int(inputs.shape[-1])
            configured = cfg.get("model", {}).get("input_dim")
            if configured is not None and int(configured) != feature_count:
                raise ValueError(
                    f"model.input_dim is {configured} but the series windows have "
                    f"{feature_count} features ({', '.join(prepared.feature_names)})."
                )
        else:
            inputs, targets = load_series(cfg)
        fit_targets = training_targets(targets, spec)

        full_dataset = TensorDataset(inputs, fit_targets)
        # CSV windows overlap and are ordered in time, so a random split leaks
        # near-duplicate windows into validation; hold out the most recent ones.
        split_mode = resolve_split_mode(data_cfg)
        val_split = float(data_cfg.get("val_split", 0.0))
        if prepared is not None:
            if split_mode != "temporal":
                raise ValueError(
                    "val_start, test_start, test_split, and missing_indicators "
                    "require a temporal split."
                )
            train_idx = prepared.train_index.tolist()
            val_idx = prepared.val_index.tolist()
        else:
            train_idx, val_idx = split_indices(
                len(full_dataset),
                val_split,
                split_mode,
                int(data_cfg.get("split_seed", 0)),
            )
        # Load before any checkpoint is written so a missing date or target
        # column fails before epochs run. The same arrays are scored at the end.
        baseline_frame = None
        if data_cfg.get("report_baselines"):
            if data_cfg.get("source") != "csv" or not data_cfg.get("path"):
                raise ValueError(
                    "data.report_baselines requires a CSV source with a path."
                )
            if split_mode != "temporal" or not val_idx:
                raise ValueError(
                    "data.report_baselines requires a non-empty temporal validation split."
                )
            if spec.horizon != 1:
                raise ValueError(
                    "data.report_baselines scores one-step labels. "
                    f"forecast.horizon is {spec.horizon}; use horizon 1, or "
                    "python -m src.compare for a multi-step table."
                )
            baseline_frame = load_baseline_series(
                resolve_repo_path(data_cfg["path"]),
                target_column=data_cfg.get("target_column", "value"),
                date_column=data_cfg.get("date_column") or None,
            )
        if (
            data_cfg.get("blend_with")
            and not data_cfg.get("report_baselines")
            and val_idx
        ):
            raise ValueError(
                "data.blend_with requires data.report_baselines "
                "when validation is scored."
            )
        dataset: Dataset[tuple[torch.Tensor, ...]]
        val_dataset: Dataset[tuple[torch.Tensor, ...]] | None
        val_indices: list[int] | None
        if val_idx:
            dataset = Subset(full_dataset, train_idx)
            val_dataset = Subset(full_dataset, val_idx)
            val_indices = list(val_idx)
        else:
            dataset, val_dataset = full_dataset, None
            val_indices = None
        n_train = len(train_idx)

        # Standardise features/targets (default on for CSV data). Statistics come
        # from the training split only so validation data never leaks into them.
        scaler: StandardScaler | None = None
        if data_cfg.get("normalize", data_cfg.get("source") == "csv"):
            scaler = StandardScaler.fit(inputs[train_idx], fit_targets[train_idx])
            full_dataset.tensors = (
                scaler.transform_inputs(inputs),
                scaler.transform_targets(fit_targets),
            )
        scaler_state = scaler.state_dict() if scaler is not None else None

        loader = DataLoader(dataset, batch_size=t_cfg["batch_size"], shuffle=True)

        def evaluate(
            ds: Dataset[tuple[torch.Tensor, ...]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            model.eval()
            preds_l: list[torch.Tensor] = []
            targets_l: list[torch.Tensor] = []
            with torch.no_grad():
                for bx, by in DataLoader(ds, batch_size=t_cfg["batch_size"]):
                    preds_l.append(model(bx.to(device)).cpu())
                    targets_l.append(by)
            return torch.cat(preds_l), torch.cat(targets_l)

        def _mean_loss(ds: Dataset[tuple[torch.Tensor, ...]]) -> float:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for bx, by in DataLoader(ds, batch_size=t_cfg["batch_size"]):
                    loss = criterion(model(bx.to(device)), by.to(device))
                    total += loss.item() * bx.size(0)
                    count += bx.size(0)
            return total / count

        checkpoint_dir = resolve_repo_path(t_cfg.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        best_epoch = 0
        log_interval = t_cfg.get("log_interval", 10)
        history: list[dict[str, float]] = []
        patience = int(t_cfg.get("patience", 0) or 0)
        max_grad_norm = float(t_cfg.get("max_grad_norm", 0) or 0)
        forecast_meta = _forecast_checkpoint(spec)
        rule = str(t_cfg.get("checkpoint_rule", "best"))
        if rule not in ("best", "last"):
            raise ValueError(
                f"training.checkpoint_rule must be 'best' or 'last', got {rule!r}"
            )
        if patience < 0 or max_grad_norm < 0:
            raise ValueError(
                "training.patience and training.max_grad_norm must be >= 0"
            )
        epochs_without_improvement = 0

        experiment_id = self.store.create_experiment(
            name=t_cfg.get("experiment_name", "default"),
            config=cfg,
            model_name=cfg["model"].get("name", "time_series_transformer"),
        )

        try:
            for epoch in range(1, t_cfg["epochs"] + 1):
                model.train()
                running_loss = 0.0
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    preds = model(batch_x)
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                    optimizer.step()
                    running_loss += loss.item() * batch_x.size(0)

                epoch_loss = running_loss / n_train
                record: dict[str, float] = {"epoch": epoch, "train_loss": epoch_loss}
                self.store.log_metric(experiment_id, "train_loss", epoch_loss, epoch)
                # Select the best checkpoint on held-out loss when available.
                select_loss = epoch_loss
                if val_dataset is not None:
                    if quantiles:
                        select_loss = _mean_loss(val_dataset)
                    else:
                        vp, vt = evaluate(val_dataset)
                        select_loss = mse(vp, vt)  # in scaled units, like train_loss
                    record["val_loss"] = select_loss
                    self.store.log_metric(experiment_id, "val_loss", select_loss, epoch)
                history.append(record)

                if epoch % log_interval == 0 or epoch == 1:
                    logger.info(
                        "Epoch %d/%d  loss=%.6f", epoch, t_cfg["epochs"], epoch_loss
                    )
                    if (
                        not quantiles
                        and data_cfg.get("report_baselines")
                        and scaler is not None
                        and val_dataset is not None
                    ):
                        # Target z-score RMSE converts by the training-split std.
                        # Pinball loss is not an MSE, so this line is point models only.
                        original_rmse = (select_loss**0.5) * float(
                            scaler.y_std.reshape(-1)[0]
                        )
                        logger.info(
                            "Epoch %d validation RMSE (original units): %.4f",
                            epoch,
                            original_rmse,
                        )

                improved = select_loss < best_loss
                if improved:
                    epochs_without_improvement = 0
                    best_loss = select_loss
                    best_epoch = epoch
                else:
                    epochs_without_improvement += 1

                save_this_epoch = improved or (
                    rule == "last" and epoch == t_cfg["epochs"]
                )
                if save_this_epoch:
                    if rule == "last" and epoch == t_cfg["epochs"]:
                        best_epoch = epoch
                        best_loss = select_loss
                    checkpoint_path = checkpoint_dir / checkpoint_name
                    save_checkpoint(
                        model,
                        optimizer,
                        best_epoch,
                        best_loss,
                        cfg,
                        checkpoint_path,
                        scaler=scaler_state,
                        forecast=dict(forecast_meta),
                    )
                    self.store.complete_experiment(
                        experiment_id,
                        status="running",
                        best_loss=best_loss,
                        checkpoint_path=str(checkpoint_path),
                    )

                reported = select_loss**0.5
                if scaler is not None and not quantiles:
                    reported *= float(scaler.y_std.reshape(-1)[0])
                if epoch_callback is not None:
                    epoch_callback(epoch, reported)

                if (
                    rule == "best"
                    and patience
                    and epochs_without_improvement >= patience
                ):
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch,
                        patience,
                    )
                    break

            # The checkpoint is the best epoch. Score that model, not whatever
            # weights the loop stopped on after the patience window.
            best_path = checkpoint_dir / checkpoint_name
            if best_path.is_file():
                saved = torch.load(best_path, map_location=device, weights_only=True)
                model.load_state_dict(saved["model_state_dict"])

            metrics, point, observed = _served_metrics(
                model,
                inputs,
                targets,
                val_indices,
                spec=spec,
                scaler=scaler,
                device=device,
                forecast_meta=forecast_meta,
            )
            if best_path.is_file():
                saved = torch.load(best_path, map_location="cpu", weights_only=True)
                saved["forecast"] = forecast_meta
                torch.save(saved, best_path)

            baseline_scores: dict[str, dict[str, float]] | None = None
            if baseline_frame is not None:
                series_values, series_dates = baseline_frame
                if prepared is not None:
                    target_rows = prepared.val_index + seq_len
                else:
                    target_rows = temporal_validation_target_rows(
                        len(series_values), seq_len, val_split
                    )
                classical = None
                if series_dates is not None and data_cfg.get("blend_with"):
                    frame = pd.read_csv(resolve_repo_path(data_cfg["path"]))
                    classical = _classical_forecasts(
                        frame,
                        data_cfg,
                        series_values,
                        series_dates,
                        target_rows,
                        seed=int(t_cfg.get("seed", 0) or 0),
                    )
                    flat = point.reshape(-1)
                    blended = _apply_blend(flat, classical, data_cfg)
                    point = blended.reshape(point.shape)
                    metrics["mse"] = mse(point, observed)
                    metrics["rmse"] = rmse(point, observed)
                    metrics["mae"] = mae(point, observed)
                baseline_scores = compare_to_baselines(
                    series_values,
                    target_rows,
                    point.detach().cpu().numpy(),
                    dates=series_dates,
                    model_targets=observed.detach().cpu().numpy(),
                )
                if classical is not None:
                    reference = series_values[np.asarray(target_rows)]
                    for name, forecast in classical.items():
                        baseline_scores[name] = regression_scores(forecast, reference)
                logger.info(
                    "Validation comparison (original units):\n%s",
                    format_score_table(baseline_scores),
                )
            for name, value in metrics.items():
                self.store.log_metric(experiment_id, name, value)
            if baseline_scores is not None:
                for name, scores in baseline_scores.items():
                    if name == "model":
                        continue
                    for metric_name, value in scores.items():
                        metrics[f"baseline_{name}_{metric_name}"] = value
                        self.store.log_metric(
                            experiment_id, f"baseline_{name}_{metric_name}", value
                        )

        except Exception:
            self.store.complete_experiment(
                experiment_id,
                status="failed",
                best_loss=best_loss if best_loss != float("inf") else None,
            )
            raise

        self.store.complete_experiment(
            experiment_id,
            status="completed",
            best_loss=best_loss,
            checkpoint_path=str(checkpoint_dir / checkpoint_name),
            metrics=metrics,
        )

        elapsed_seconds = time.perf_counter() - started
        logger.info(
            "Training complete. Best loss: %.6f (%.1f seconds)",
            best_loss,
            elapsed_seconds,
        )
        return {
            "experiment_id": experiment_id,
            "best_loss": best_loss,
            "metrics": metrics,
            "history": history,
            "param_count": param_count,
            "elapsed_seconds": elapsed_seconds,
            "baselines": baseline_scores,
            "best_epoch": best_epoch,
        }


def _forecast_checkpoint(spec: ForecastSpec) -> dict[str, Any]:
    meta = spec.as_dict()
    meta.update(
        {
            "interval_method": None,
            "conformal_qhat": None,
            "validation_coverage": None,
            "raw_coverage": None,
            "interval_width": None,
            "coverage_horizon": spec.horizon,
        }
    )
    return meta


def _served_metrics(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_indices: list[int] | None,
    *,
    spec: ForecastSpec,
    scaler: StandardScaler | None,
    device: torch.device,
    forecast_meta: dict[str, Any],
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Score the served horizon in original units and calibrate intervals.

    Point metrics use the validation rows when a split exists, otherwise the
    training rows. Interval offsets are fit only on a validation split, and
    the quoted coverage is the later half of that split. The returned point
    forecast and labels are the same tensors the metrics use.
    """
    eval_indices = val_indices if val_indices is not None else list(range(len(inputs)))
    predictor = Predictor(model, device=device, scaler=scaler, spec=spec)
    batch = predictor.forecast(inputs[eval_indices], horizon=spec.horizon)
    observed = horizon_targets(targets[eval_indices], spec.horizon)
    point = batch.point.detach().cpu()
    metrics = {
        "mse": mse(point, observed),
        "rmse": rmse(point, observed),
        "mae": mae(point, observed),
    }
    if val_indices is None:
        return metrics, point, observed

    ordered = sorted(val_indices)
    if ordered == list(eval_indices):
        cal_point = point
        cal_lower = None if batch.lower is None else batch.lower.detach().cpu()
        cal_upper = None if batch.upper is None else batch.upper.detach().cpu()
        cal_targets = observed
    else:
        cal_batch = predictor.forecast(inputs[ordered], horizon=spec.horizon)
        cal_point = cal_batch.point.detach().cpu()
        cal_lower = None if cal_batch.lower is None else cal_batch.lower.detach().cpu()
        cal_upper = None if cal_batch.upper is None else cal_batch.upper.detach().cpu()
        cal_targets = horizon_targets(targets[ordered], spec.horizon)

    method = "cqr" if spec.quantiles and cal_lower is not None else "residual"
    calibration = calibrate_intervals(
        point=cal_point,
        targets=cal_targets,
        nominal=spec.nominal_coverage,
        method=method,
        lower=cal_lower,
        upper=cal_upper,
    )
    forecast_meta["interval_method"] = calibration.method
    forecast_meta["conformal_qhat"] = list(calibration.qhat)
    forecast_meta["validation_coverage"] = calibration.validation_coverage
    forecast_meta["raw_coverage"] = calibration.raw_coverage
    forecast_meta["interval_width"] = calibration.interval_width
    forecast_meta["coverage_horizon"] = calibration.coverage_horizon
    if calibration.validation_coverage is not None:
        metrics["coverage"] = calibration.validation_coverage
    if calibration.raw_coverage is not None:
        metrics["raw_coverage"] = calibration.raw_coverage
    if calibration.interval_width is not None:
        metrics["interval_width"] = calibration.interval_width
    metrics["nominal_coverage"] = calibration.nominal_coverage
    return metrics, point, observed


def _classical_forecasts(
    frame: pd.DataFrame,
    data_cfg: dict[str, Any],
    series_values: Any,
    series_dates: Any,
    target_rows: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    target_column = data_cfg.get("target_column", "value")
    covariate_columns = [
        column
        for column in data_cfg.get("feature_columns", [])
        if column != target_column
        and column in frame.columns
        and not str(column).endswith(("_sin", "_cos"))
    ]
    return score_classical_baselines(
        series_values,
        target_rows,
        dates=series_dates,
        covariates=covariate_frame(frame, covariate_columns, int(np.min(target_rows))),
        seed=seed,
        ridge_alpha=_optional_float(data_cfg.get("ridge_alpha")),
        ar_order=_optional_int(data_cfg.get("ar_order")),
        boosting_params=data_cfg.get("boosting_params"),
    )


def _apply_blend(
    predictions: torch.Tensor,
    classical: dict[str, Any] | None,
    data_cfg: dict[str, Any],
) -> torch.Tensor:
    """Average the network with one classical forecast when configured.

    ``blend_weight`` is the network's share. The default is an equal average,
    which is fixed in advance rather than fit on the split being scored.
    """
    blend_with = data_cfg.get("blend_with") or None
    if not blend_with:
        return predictions
    if classical is None or blend_with not in classical:
        raise ValueError(
            f"data.blend_with is {blend_with!r}, which needs dated classical forecasts."
        )
    weight = float(data_cfg.get("blend_weight", 0.5))
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"data.blend_weight must be in [0, 1], got {weight}")
    other = torch.tensor(classical[blend_with], dtype=predictions.dtype).reshape(
        predictions.shape
    )
    return weight * predictions + (1.0 - weight) * other


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
