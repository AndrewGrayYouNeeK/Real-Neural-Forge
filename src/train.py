"""Training script for the TimeSeriesTransformer."""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml

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


def make_sine_dataset(
    n_samples: int,
    seq_len: int,
    noise: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic sine-wave dataset for demonstration.

    Each sample is a sequence of length seq_len drawn from sin(t + offset)
    with added Gaussian noise.  The target is the value one step ahead.

    Returns:
        inputs: (n_samples, seq_len, 1)
        targets: (n_samples, 1)
    """
    t = torch.linspace(0, 4 * torch.pi, seq_len + 1)
    offsets = torch.rand(n_samples) * 2 * torch.pi
    # (n_samples, seq_len+1)
    series = torch.sin(t.unsqueeze(0) + offsets.unsqueeze(1))
    series += torch.randn_like(series) * noise
    inputs = series[:, :-1].unsqueeze(-1)   # (n_samples, seq_len, 1)
    targets = series[:, -1].unsqueeze(-1)   # (n_samples, 1)
    return inputs, targets


def train(config_path: str = "config/config.yaml") -> None:
    """Run the full training loop."""
    cfg = load_config(config_path)
    t_cfg = cfg["training"]

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

    # Synthetic dataset
    inputs, targets = make_sine_dataset(n_samples=1024, seq_len=64)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=t_cfg["batch_size"], shuffle=True
    )

    checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    log_interval = t_cfg.get("log_interval", 10)

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
        if epoch % log_interval == 0 or epoch == 1:
            logger.info("Epoch %d/%d  loss=%.6f", epoch, t_cfg["epochs"], epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": cfg,
                },
                checkpoint_dir / "best_model.pt",
            )

    logger.info("Training complete. Best loss: %.6f", best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TimeSeriesTransformer")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    train(args.config)
