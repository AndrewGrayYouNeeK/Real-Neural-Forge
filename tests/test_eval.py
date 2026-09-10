"""Tests for checkpoint evaluation."""

import pytest
import yaml

from src.eval import evaluate_checkpoint
from src.train import load_config, train


def test_evaluate_checkpoint_missing(tmp_path):
    cfg = load_config("config/config.yaml")
    cfg["inference"]["checkpoint_path"] = str(tmp_path / "missing.pt")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        evaluate_checkpoint(str(cfg_path))


def test_evaluate_checkpoint_on_holdout(tmp_path):
    values = [f"{i},{0.1 * i:.4f}" for i in range(80)]
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("t,value\n" + "\n".join(values) + "\n")

    cfg = load_config("config/config.yaml")
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 8
    cfg["training"]["device"] = "cpu"
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
    cfg["training"]["scheduler"] = None
    cfg["data"]["csv_path"] = str(csv_path)
    cfg["data"]["seq_len"] = 8
    cfg["data"]["val_split"] = 0.2
    cfg["data"]["test_split"] = 0.2
    cfg["inference"]["checkpoint_path"] = str(tmp_path / "ckpts" / "best_model.pt")

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    train(str(cfg_path))
    result = evaluate_checkpoint(str(cfg_path))
    assert result["split"] == "test"
    assert result["n"] > 0
    assert result["mse"] >= 0
    assert result["mae"] >= 0
