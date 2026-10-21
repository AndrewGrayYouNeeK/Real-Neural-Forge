"""A tiny CSV run checks that checkpoint scores line up with the baselines."""

from datetime import date, timedelta

import yaml

from src.realworld.evaluate import compare_checkpoint
from src.train import train


def test_ramp_persistence_mae_is_one(tmp_path):
    rows = ["date,value"]
    start = date(2020, 1, 1)
    for index in range(48):
        rows.append(f"{(start + timedelta(days=index)).isoformat()},{index}")
    csv_path = tmp_path / "ramp.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    cfg = {
        "model": {
            "name": "time_series_transformer",
            "input_dim": 1,
            "output_dim": 1,
            "d_model": 8,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "max_seq_len": 16,
        },
        "data": {
            "source": "csv",
            "path": str(csv_path),
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 4,
            "val_split": 0.2,
            "split_mode": "temporal",
            "normalize": True,
            "seasonal_period": 7,
        },
        "training": {
            "experiment_name": "ramp",
            "batch_size": 8,
            "learning_rate": 0.01,
            "epochs": 2,
            "patience": 2,
            "max_grad_norm": 1.0,
            "device": "cpu",
            "seed": 0,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_interval": 1,
        },
        "storage": {"db_path": str(tmp_path / "experiments.db")},
        "realworld": {
            "id": "ramp",
            "title": "ramp",
            "units": "count",
            "licence": "test",
        },
    }
    config_path = tmp_path / "ramp.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    train(str(config_path))
    result = compare_checkpoint(config_path)
    assert abs(result["baselines"]["persistence"]["mae"] - 1.0) < 1e-5
    assert result["n_val"] == int((48 - 4) * 0.2)
    assert result["model"]["n"] == result["n_val"]
    assert "skill_vs_persistence" in result
