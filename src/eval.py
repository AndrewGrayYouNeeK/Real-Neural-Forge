"""Evaluate a saved checkpoint on the configured holdout split."""

from __future__ import annotations

import argparse
import json
import logging

import torch

from src.common import (
    apply_checkpoint_config,
    build_model,
    checkpoint_path_from_cfg,
    load_config,
    resolve_device,
)
from src.data import apply_scaler, load_splits
from src.train import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_checkpoint(config_path: str = "config/config.yaml") -> dict:
    """Score the best checkpoint on test windows, falling back to val."""
    cfg = load_config(config_path)
    device = resolve_device(
        cfg.get("inference", {}).get("device") or cfg.get("training", {}).get("device")
    )
    ckpt_path = checkpoint_path_from_cfg(cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = apply_checkpoint_config(cfg, checkpoint)
    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits = load_splits(cfg)
    if splits.test_x.size(0) > 0:
        features, targets, split_name = splits.test_x, splits.test_y, "test"
    else:
        features, targets, split_name = splits.val_x, splits.val_y, "val"

    scaler = checkpoint.get("scaler")
    if scaler:
        features = apply_scaler(features, scaler)
        targets = apply_scaler(targets, scaler)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, targets),
        batch_size=int(cfg.get("training", {}).get("batch_size", 32)),
        shuffle=False,
    )
    mse, mae = evaluate(model, loader, device)
    result = {
        "split": split_name,
        "n": int(features.size(0)),
        "mse": mse,
        "mae": mae,
        "checkpoint": str(ckpt_path),
        "source": splits.source,
        "epoch": checkpoint.get("epoch"),
    }
    logger.info(
        "Eval %s  n=%d  mse=%.6f  mae=%.6f",
        split_name,
        result["n"],
        mse,
        mae,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Neural Forge checkpoint")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.config), indent=2))


if __name__ == "__main__":
    main()
