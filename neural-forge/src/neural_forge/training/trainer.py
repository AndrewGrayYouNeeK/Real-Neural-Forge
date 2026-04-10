"""Core training loop (Trainer)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_forge.utils.checkpoint import save_checkpoint
from neural_forge.utils.logger import get_logger
from neural_forge.utils.seed import set_seed

logger = get_logger(__name__)


class Trainer:
    """Generic model trainer.

    Args:
        model:       PyTorch model to train.
        criterion:   Loss function.
        optimizer:   Optimiser instance.
        scheduler:   Optional LR scheduler.
        device:      Target device (``"cpu"`` or ``"cuda"``).
        max_epochs:  Number of training epochs.
        checkpoint_dir: Directory to save checkpoints.
        log_every:   Log training stats every *n* batches.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any = None,
        device: str | torch.device = "cpu",
        max_epochs: int = 10,
        checkpoint_dir: str | Path = "checkpoints",
        log_every: int = 50,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.max_epochs = max_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every

        self._global_step = 0
        self._best_val_loss = float("inf")

    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Train the model and optionally evaluate on *val_loader* each epoch.

        Returns:
            History dict with ``"train_loss"`` (and ``"val_loss"`` if applicable).
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        self.model.on_train_start()  # type: ignore[attr-defined]

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)

            val_loss: float | None = None
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                history["val_loss"].append(val_loss)

                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    ckpt_path = self.checkpoint_dir / "best.pt"
                    save_checkpoint(self.model, self.optimizer, epoch, ckpt_path)

            elapsed = time.time() - t0
            val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else ""
            logger.info(
                "Epoch %d/%d — train_loss=%.4f%s  (%.1fs)",
                epoch,
                self.max_epochs,
                train_loss,
                val_str,
                elapsed,
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()

        self.model.on_train_end()  # type: ignore[attr-defined]
        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for step, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False), 1):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self._global_step += 1
            if step % self.log_every == 0:
                logger.debug("  step=%d  loss=%.4f", self._global_step, loss.item())
        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item()
        return total_loss / len(loader)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    """Entry-point called by ``neural-forge-train`` or ``run_training.sh``."""
    parser = argparse.ArgumentParser(description="Neural Forge — training entry-point")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()

    cfg: DictConfig = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(overrides),
    )

    set_seed(cfg.get("seed", 42))
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Training complete (implement data/model wiring here).")


if __name__ == "__main__":
    main()
