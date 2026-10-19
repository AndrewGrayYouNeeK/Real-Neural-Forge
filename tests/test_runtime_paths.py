"""Paths used by training and config loading follow the repository root."""

import yaml

from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config
from src.utils.paths import REPO_ROOT, resolve_repo_path


def test_resolve_repo_path_relative_and_absolute(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.REPO_ROOT", tmp_path)
    assert resolve_repo_path("a/b") == (tmp_path / "a" / "b").resolve()
    absolute = tmp_path / "abs.txt"
    assert resolve_repo_path(absolute) == absolute.resolve()


def test_load_config_ignores_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config/config.yaml")
    assert cfg["model"]["name"] == "time_series_transformer"
    assert cfg["storage"]["db_path"] == "data/experiments.db"


def _small_cfg() -> dict:
    cfg = load_config("config/config.yaml")
    cfg["training"].update(
        epochs=1,
        batch_size=8,
        device="cpu",
        log_interval=1,
        checkpoint_dir="rel_ckpt",
    )
    cfg["data"].update(n_samples=16, seq_len=8, val_split=0.0)
    cfg["model"].update(d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32)
    cfg["storage"] = {"db_path": "runs/experiments.db"}
    return cfg


def test_trainer_db_and_checkpoint_follow_repo_root(tmp_path, monkeypatch):
    cfg = _small_cfg()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    repo = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr("src.utils.paths.REPO_ROOT", repo)

    trainer = TimeSeriesTrainer(str(config_path))
    assert trainer.store.db_path == (repo / "runs" / "experiments.db").resolve()
    trainer.train()

    assert (repo / "rel_ckpt" / "best_model.pt").is_file()
    assert (repo / "runs" / "experiments.db").is_file()
    assert not (elsewhere / "rel_ckpt").exists()
    assert not (elsewhere / "runs").exists()
    assert REPO_ROOT != repo


def test_trainer_resolves_relative_csv_from_repo_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    series = repo / "series.csv"
    series.parent.mkdir(parents=True)
    series.write_text(
        "value\n" + "\n".join(str(i) for i in range(30)) + "\n", encoding="utf-8"
    )

    cfg = load_config("config/config.yaml")
    cfg["data"] = {
        "source": "csv",
        "path": "series.csv",
        "feature_columns": ["value"],
        "target_column": "value",
        "seq_len": 4,
        "val_split": 0.0,
    }
    cfg["training"].update(
        epochs=1,
        batch_size=8,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    cfg["model"].update(d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32)
    cfg["storage"] = {"db_path": str(tmp_path / "exp.db")}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr("src.utils.paths.REPO_ROOT", repo)

    TimeSeriesTrainer(str(config_path)).train()
    assert (tmp_path / "ckpt" / "best_model.pt").is_file()
    assert not (elsewhere / "series.csv").exists()
