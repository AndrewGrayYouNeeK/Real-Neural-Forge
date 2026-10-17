"""Training loop for time-series models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

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

        seed = t_cfg.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

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
                seed=data_cfg.get("seed"),
            )

        full_dataset = TensorDataset(inputs, targets)
        val_split = float(data_cfg.get("val_split", 0.0))
        if not 0.0 <= val_split < 1.0:
            raise ValueError(f"data.val_split must be in [0, 1), got {val_split}")
        n_val = int(len(full_dataset) * val_split)
        if val_split > 0 and (n_val == 0 or n_val == len(full_dataset)):
            raise ValueError("data.val_split leaves an empty train or validation set.")
        dataset: Dataset[tuple[torch.Tensor, ...]]
        val_dataset: Dataset[tuple[torch.Tensor, ...]] | None
        if n_val:
            generator = torch.Generator().manual_seed(int(data_cfg.get("split_seed", 0)))
            dataset, val_dataset = random_split(
                full_dataset, [len(full_dataset) - n_val, n_val], generator=generator
            )
        else:
            dataset, val_dataset = full_dataset, None
        n_train = len(full_dataset) - n_val
        loader = DataLoader(
            dataset, batch_size=t_cfg["batch_size"], shuffle=True
        )

        def evaluate(ds: Dataset[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
            model.eval()
            preds_l: list[torch.Tensor] = []
            targets_l: list[torch.Tensor] = []
            with torch.no_grad():
                for bx, by in DataLoader(ds, batch_size=t_cfg["batch_size"]):
                    preds_l.append(model(bx.to(device)).cpu())
                    targets_l.append(by)
            return torch.cat(preds_l), torch.cat(targets_l)

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
                    optimizer.step()
                    running_loss += loss.item() * batch_x.size(0)

                epoch_loss = running_loss / n_train
                record: dict[str, float] = {"epoch": epoch, "train_loss": epoch_loss}
                self.store.log_metric(experiment_id, "train_loss", epoch_loss, epoch)
                # Select the best checkpoint on held-out loss when available.
                select_loss = epoch_loss
                if val_dataset is not None:
                    vp, vt = evaluate(val_dataset)
                    select_loss = mse(vp, vt)
                    record["val_loss"] = select_loss
                    self.store.log_metric(experiment_id, "val_loss", select_loss, epoch)
                history.append(record)

                if epoch % log_interval == 0 or epoch == 1:
                    logger.info(
                        "Epoch %d/%d  loss=%.6f", epoch, t_cfg["epochs"], epoch_loss
                    )

                if select_loss < best_loss:
                    best_loss = select_loss
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

            # Report metrics on the validation set when one exists, otherwise
            # on the full training set (deterministic, unshuffled).
            preds, targets_all = evaluate(val_dataset if val_dataset is not None else dataset)
            metrics = {
                "mse": mse(preds, targets_all),
                "rmse": rmse(preds, targets_all),
                "mae": mae(preds, targets_all),
            }
            for name, value in metrics.items():
                self.store.log_metric(experiment_id, name, value)

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
