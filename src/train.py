"""Training script for the TimeSeriesTransformer."""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data import load_train_val, maybe_standardize
from src.data import make_sine_dataset  # noqa: F401 — re-exported for tests
from src.model import TimeSeriesTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Make training deterministic enough to reproduce a run."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, device: torch.device) -> TimeSeriesTransformer:
    """Instantiate and return the transformer model."""
    m_cfg = cfg["model"]
    model = TimeSeriesTransformer(
        input_dim=m_cfg["input_dim"],
        output_dim=m_cfg["output_dim"],
        d_model=m_cfg["d_model"],
        nhead=m_cfg["nhead"],
        num_encoder_layers=m_cfg["num_encoder_layers"],
        dim_feedforward=m_cfg["dim_feedforward"],
        dropout=m_cfg["dropout"],
        max_seq_len=m_cfg["max_seq_len"],
    )
    return model.to(device)


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

    device_str = t_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available – falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    model = build_model(cfg, device)
    logger.info(
        "Model parameters: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
    criterion = nn.MSELoss()

    train_x, train_y, val_x, val_y, source = load_train_val(cfg)
    train_x, train_y, val_x, val_y, scaler = maybe_standardize(
        cfg, train_x, train_y, val_x, val_y
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

    best_val_mse = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    stale = 0
    patience = int(t_cfg.get("patience", 0) or 0)
    log_interval = t_cfg.get("log_interval", 10)

    for epoch in range(1, t_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        train_mse = running_loss / len(train_x)
        val_mse, val_mae = evaluate(model, val_loader, device)

        if epoch % log_interval == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d  train_mse=%.6f  val_mse=%.6f  val_mae=%.6f",
                epoch,
                t_cfg["epochs"],
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
                "source": source,
                "scaler": scaler,
                "config": cfg,
            }
            torch.save(payload, checkpoint_dir / "best_model.pt")
            (checkpoint_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_mse": best_val_mse,
                        "val_mae": best_val_mae,
                        "train_mse": train_mse,
                        "source": source,
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
