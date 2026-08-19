"""Training script for the TimeSeriesTransformer."""

import argparse
import logging

from src.data.timeseries import make_sine_dataset
from src.models.registry import build_model as build_model_from_config
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config

__all__ = ["build_model", "load_config", "make_sine_dataset", "train"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(cfg: dict, device):
    """Instantiate and return the transformer model."""
    return build_model_from_config(cfg).to(device)


def train(config_path: str = "config/config.yaml") -> None:
    """Run the full training loop."""
    TimeSeriesTrainer(config_path).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TimeSeriesTransformer")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    train(args.config)
