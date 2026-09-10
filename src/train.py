"""Training script for the TimeSeriesTransformer."""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.common import build_model, load_config, resolve_device
from src.data import (
    load_splits,
    make_sine_dataset,  # noqa: F401 — re-exported for tests
    maybe_standardize,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Make training deterministic enough to reproduce a run."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mse, mae) over a loader."""
    model.eval()
    sse = 0.0
    sae = 0.0
    n = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = model(batch_x)
        err = preds - batch_y
        sse += (err * err).sum().item()
        sae += err.abs().sum().item()
        n += batch_y.numel()
    if n == 0:
        return float("nan"), float("nan")
    return sse / n, sae / n


def train(config_path: str = "config/config.yaml") -> None:
    """Run the full training loop."""
    cfg = load_config(config_path)
    t_cfg = cfg["training"]

    set_seed(int(t_cfg.get("seed", 42)))
    device = resolve_device(t_cfg.get("device", "cpu"))
    logger.info("Using device: %s", device)

    model = build_model(cfg, device)
    logger.info(
        "Model parameters: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
    criterion = nn.MSELoss()
    scheduler = None
    if t_cfg.get("scheduler") in ("plateau", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    splits = load_splits(cfg)
    train_x, train_y, val_x, val_y, scaler = maybe_standardize(
        cfg, splits.train_x, splits.train_y, splits.val_x, splits.val_y
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=t_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_x, val_y),
        batch_size=t_cfg["batch_size"],
        shuffle=False,
    )

    checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"

    start_epoch = 1
    best_val_mse = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    if t_cfg.get("resume") and best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_mse = float(
            checkpoint.get("val_mse", checkpoint.get("loss", float("inf")))
        )
        best_val_mae = float(checkpoint.get("val_mae", float("inf")))
        best_epoch = int(checkpoint.get("epoch", 0))
        logger.info("Resumed from %s at epoch %d", best_path, start_epoch)

    stale = 0
    patience = int(t_cfg.get("patience", 0) or 0)
    log_interval = t_cfg.get("log_interval", 10)
    grad_clip = float(t_cfg.get("grad_clip") or t_cfg.get("max_grad_norm") or 0)
    total_epochs = t_cfg["epochs"]

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        train_mse = running_loss / len(train_x)
        val_mse, val_mae = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_mse)

        if epoch % log_interval == 0 or epoch == start_epoch:
            logger.info(
                "Epoch %d/%d  train_mse=%.6f  val_mse=%.6f  val_mae=%.6f",
                epoch,
                total_epochs,
                train_mse,
                val_mse,
                val_mae,
            )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_epoch = epoch
            stale = 0
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_mse,
                "val_mse": best_val_mse,
                "val_mae": best_val_mae,
                "train_mse": train_mse,
                "source": splits.source,
                "scaler": scaler,
                "config": cfg,
            }
            torch.save(payload, best_path)
            (checkpoint_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_mse": best_val_mse,
                        "val_mae": best_val_mae,
                        "train_mse": train_mse,
                        "source": splits.source,
                        "n_train": int(train_x.size(0)),
                        "n_val": int(val_x.size(0)),
                        "n_test": int(splits.test_x.size(0)),
                        "scaler": scaler,
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d). Best epoch %d.",
                    epoch,
                    patience,
                    best_epoch,
                )
                break

    logger.info(
        "Training complete. Best val_mse=%.6f  val_mae=%.6f  (epoch %d)",
        best_val_mse,
        best_val_mae,
        best_epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TimeSeriesTransformer")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    train(args.config)
