"""Training loop for time-series models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.timeseries import make_sine_dataset
from src.evaluation.metrics import mae, mse, rmse
from src.models.registry import build_model
from src.storage.experiments import ExperimentStore
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class TimeSeriesTrainer:
    """End-to-end trainer for the time-series transformer."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.store = ExperimentStore()

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available – falling back to CPU.")
            device_str = "cpu"
        return torch.device(device_str)

    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        t_cfg = cfg["training"]
        device = self._resolve_device(t_cfg.get("device", "cpu"))
        logger.info("Using device: %s", device)

        model = build_model(cfg).to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model parameters: %d", param_count)

        optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
        criterion = nn.MSELoss()

        data_cfg = cfg.get("data", {})
        if data_cfg.get("source") == "csv" and data_cfg.get("path"):
            from src.data.timeseries import load_csv_timeseries

            inputs, targets = load_csv_timeseries(
                data_cfg["path"],
                feature_columns=data_cfg.get("feature_columns", ["value"]),
                target_column=data_cfg.get("target_column", "value"),
                seq_len=data_cfg.get("seq_len", 64),
            )
        else:
            inputs, targets = make_sine_dataset(
                n_samples=data_cfg.get("n_samples", 1024),
                seq_len=data_cfg.get("seq_len", 64),
                noise=data_cfg.get("noise", 0.05),
            )

        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(
            dataset, batch_size=t_cfg["batch_size"], shuffle=True
        )

        checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        log_interval = t_cfg.get("log_interval", 10)
        history: list[dict[str, float]] = []

        experiment_id = self.store.create_experiment(
            name=t_cfg.get("experiment_name", "default"),
            config=cfg,
            model_name=cfg["model"].get("name", "time_series_transformer"),
        )

        for epoch in range(1, t_cfg["epochs"] + 1):
            model.train()
            running_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            epoch_loss = running_loss / len(dataset)
            history.append({"epoch": epoch, "train_loss": epoch_loss})
            self.store.log_metric(experiment_id, "train_loss", epoch_loss, epoch)

            if epoch % log_interval == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d  loss=%.6f", epoch, t_cfg["epochs"], epoch_loss
                )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_loss,
                    cfg,
                    checkpoint_path,
                )
                self.store.complete_experiment(
                    experiment_id,
                    status="running",
                    best_loss=best_loss,
                    checkpoint_path=str(checkpoint_path),
                )

        model.eval()
        with torch.no_grad():
            sample_x, sample_y = next(iter(loader))
            sample_x, sample_y = sample_x.to(device), sample_y.to(device)
            preds = model(sample_x)
            metrics = {
                "mse": mse(preds, sample_y),
                "rmse": rmse(preds, sample_y),
                "mae": mae(preds, sample_y),
            }
            for name, value in metrics.items():
                self.store.log_metric(experiment_id, name, value)

        self.store.complete_experiment(
            experiment_id,
            status="completed",
            best_loss=best_loss,
            checkpoint_path=str(checkpoint_dir / "best_model.pt"),
            metrics=metrics,
        )

        logger.info("Training complete. Best loss: %.6f", best_loss)
        return {
            "experiment_id": experiment_id,
            "best_loss": best_loss,
            "metrics": metrics,
            "history": history,
            "param_count": param_count,
        }
