"""Training loop for time-series models."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from src.data.prepare import load_series, resolve_split_mode, split_indices
from src.data.scaling import StandardScaler
from src.evaluation.baselines import (
    compare_to_baselines,
    format_score_table,
    load_baseline_series,
    temporal_validation_target_rows,
)
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

    def train(self, *, checkpoint_name: str = "best_model.pt") -> dict[str, Any]:
        cfg = self.cfg
        t_cfg = cfg["training"]
        device = self._resolve_device(t_cfg.get("device", "cpu"))
        logger.info("Using device: %s", device)
        started = time.perf_counter()

        seed = t_cfg.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        spec = forecast_spec_from_config(cfg)
        model = build_model(cfg).to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model parameters: %d", param_count)

        optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
        mse_loss = nn.MSELoss()
        quantiles = spec.quantiles

        def criterion(preds: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
            if quantiles:
                return pinball_loss(preds, observed, quantiles)
            return cast(torch.Tensor, mse_loss(preds, observed))

        data_cfg = cfg.get("data", {})
        seq_len = int(data_cfg.get("seq_len", 64))
        inputs, targets = load_series(cfg)
        fit_targets = training_targets(targets, spec)

        full_dataset = TensorDataset(inputs, fit_targets)
        # CSV windows overlap and are ordered in time, so a random split leaks
        # near-duplicate windows into validation; hold out the most recent ones.
        split_mode = resolve_split_mode(data_cfg)
        val_split = float(data_cfg.get("val_split", 0.0))
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
            if not val_idx or split_mode != "temporal":
                raise ValueError(
                    "data.report_baselines requires a temporal validation split."
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
        log_interval = t_cfg.get("log_interval", 10)
        history: list[dict[str, float]] = []
        patience = int(t_cfg.get("patience", 0) or 0)
        max_grad_norm = float(t_cfg.get("max_grad_norm", 0) or 0)
        forecast_meta = _forecast_checkpoint(spec)
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

                if select_loss < best_loss:
                    epochs_without_improvement = 0
                    best_loss = select_loss
                    checkpoint_path = checkpoint_dir / checkpoint_name
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
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
                else:
                    epochs_without_improvement += 1
                    if patience and epochs_without_improvement >= patience:
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
            for name, value in metrics.items():
                self.store.log_metric(experiment_id, name, value)

            baseline_scores: dict[str, dict[str, float]] | None = None
            if baseline_frame is not None:
                series_values, series_dates = baseline_frame
                target_rows = temporal_validation_target_rows(
                    len(series_values), seq_len, val_split
                )
                baseline_scores = compare_to_baselines(
                    series_values,
                    target_rows,
                    point.detach().cpu().numpy(),
                    dates=series_dates,
                    model_targets=observed.detach().cpu().numpy(),
                )
                logger.info(
                    "Validation comparison (original units):\n%s",
                    format_score_table(baseline_scores),
                )
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
