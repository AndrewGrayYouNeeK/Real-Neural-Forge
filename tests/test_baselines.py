"""Tests for temporal-split forecast baselines."""

import numpy as np
import pandas as pd
import pytest

from src.data.timeseries import resolve_val_count
from src.evaluation.baselines import (
    compare_to_baselines,
    format_score_table,
    persistence_forecast,
    regression_scores,
    seasonal_naive_calendar,
    temporal_validation_target_rows,
)
from src.models.registry import build_model
from src.storage.experiments import ExperimentStore
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config


def test_resolve_val_count_matches_trainer_rule():
    assert resolve_val_count(100, 0.2) == 20
    assert resolve_val_count(10, 0.0) == 0
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        resolve_val_count(10, 1.0)
    with pytest.raises(ValueError, match="empty"):
        resolve_val_count(1, 0.5)


def test_temporal_validation_rows_are_the_latest_labels():
    # 10 rows, seq_len 4 -> 6 windows. val_split 0.5 holds out int(6 * 0.5) = 3.
    rows = temporal_validation_target_rows(10, seq_len=4, val_split=0.5)
    assert rows.tolist() == [7, 8, 9]


def test_persistence_is_the_previous_observation():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    preds = persistence_forecast(values, np.array([2, 3]))
    assert preds.tolist() == [2.0, 3.0]


def test_seasonal_naive_uses_the_same_calendar_day():
    dates = np.array(["2019-06-01", "2020-06-01"], dtype="datetime64[D]")
    values = np.array([10.0, 20.0])
    preds = seasonal_naive_calendar(values, dates, np.array([1]))
    assert preds.tolist() == [10.0]


def test_seasonal_naive_maps_leap_day_to_february_28():
    dates = np.array(["2019-02-28", "2020-02-29"], dtype="datetime64[D]")
    values = np.array([1.5, 9.0])
    preds = seasonal_naive_calendar(values, dates, np.array([1]))
    assert preds.tolist() == [1.5]


def test_seasonal_naive_rejects_duplicate_dates():
    dates = np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="unique"):
        seasonal_naive_calendar(np.array([1.0, 2.0]), dates, np.array([1]))


def test_compare_rejects_misaligned_model_targets():
    values = np.arange(10, dtype=np.float64)
    rows = temporal_validation_target_rows(10, seq_len=4, val_split=0.5)
    with pytest.raises(ValueError, match="do not match"):
        compare_to_baselines(
            values,
            rows,
            model_preds=values[rows],
            model_targets=np.zeros(len(rows)),
        )


def test_compare_scores_model_and_persistence():
    values = np.array([0.0, 0.0, 10.0, 13.0])
    rows = np.array([2, 3])
    scores = compare_to_baselines(values, rows, model_preds=np.array([10.0, 10.0]))
    assert scores["persistence"]["mae"] == pytest.approx(6.5)
    assert scores["model"]["mae"] == pytest.approx(1.5)
    assert "seasonal_naive" not in scores
    table = format_score_table(scores)
    assert "persistence" in table and "RMSE" in table


def test_regression_scores_match_metric_helpers():
    scores = regression_scores(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    assert scores["mae"] == pytest.approx(1.0)
    assert scores["mse"] == pytest.approx(2.0)
    assert scores["rmse"] == pytest.approx(2.0**0.5)


def _trainer(tmp_path, cfg):
    trainer = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
    trainer.cfg = cfg
    trainer.store = ExperimentStore(str(tmp_path / "exp.db"))
    return trainer


def _tiny_model(cfg):
    cfg["training"].update(
        epochs=1,
        batch_size=16,
        device="cpu",
        seed=0,
        log_interval=1,
    )
    cfg["model"].update(
        input_dim=1,
        output_dim=1,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=32,
    )
    return cfg


class TestBaselineReporting:
    def test_weather_config_requests_a_temporal_baseline_run(self):
        cfg = load_config("config/weather.yaml")
        data = cfg["data"]
        assert data["source"] == "csv"
        assert data["path"] == "data/weather_central_park.csv"
        assert data["split_mode"] == "temporal"
        assert data["normalize"] is True
        assert data["report_baselines"] is True
        assert data["val_split"] == 0.2
        assert data["val_start"] == "2009-10-31"
        assert data["test_start"] == "2017-01-07"
        assert data["missing_indicators"] is True
        assert data["blend_with"] == "gradient_boosting"
        assert data["blend_weight"] == 0.5
        assert data["feature_columns"] == [
            "temperature_c",
            "tmin_c",
            "prcp_mm",
            "snow_mm",
            "snwd_mm",
            "awnd_ms",
            "doy_sin",
            "doy_cos",
        ]
        # Two missingness indicators are appended to those eight columns.
        assert cfg["model"]["input_dim"] == 10
        assert cfg["model"]["residual"] is True
        assert cfg["model"]["use_revin"] is False
        assert cfg["training"]["seed"] == 0
        assert cfg["training"]["patience"] > 0
        assert cfg["training"]["checkpoint_dir"] == "checkpoints/weather"

    def test_csv_run_reports_persistence_and_seasonal_naive(self, tmp_path):
        n = 500
        dates = pd.date_range("2019-01-01", periods=n, freq="D")
        temperature = 10 + 8 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        path = tmp_path / "series.csv"
        pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "temperature_c": temperature,
            }
        ).to_csv(path, index=False)

        cfg = _tiny_model(load_config("config/config.yaml"))
        cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        cfg["data"] = {
            "source": "csv",
            "path": str(path),
            "date_column": "date",
            "feature_columns": ["temperature_c"],
            "target_column": "temperature_c",
            "seq_len": 8,
            "val_split": 0.2,
            "split_mode": "temporal",
            "normalize": True,
            "report_baselines": True,
        }
        result = _trainer(tmp_path, cfg).train()
        rows = temporal_validation_target_rows(n, seq_len=8, val_split=0.2)
        expected = regression_scores(
            persistence_forecast(temperature, rows), temperature[rows]
        )
        assert result["baselines"]["persistence"]["rmse"] == pytest.approx(
            expected["rmse"]
        )
        assert "seasonal_naive" in result["baselines"]
        assert result["baselines"]["model"]["rmse"] == pytest.approx(
            result["metrics"]["rmse"], rel=1e-5
        )
        assert "baseline_persistence_rmse" in result["metrics"]
        store = ExperimentStore(str(tmp_path / "exp.db"))
        names = {row["name"] for row in store.get_metrics(result["experiment_id"])}
        assert "baseline_persistence_rmse" in names
        assert "baseline_seasonal_naive_mae" in names

    def test_flag_off_leaves_baselines_empty(self, tmp_path):
        cfg = _tiny_model(load_config("config/config.yaml"))
        cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        cfg["data"].update(n_samples=16, seq_len=8, val_split=0.25)
        result = _trainer(tmp_path, cfg).train()
        assert result["baselines"] is None
        assert "baseline_persistence_rmse" not in result["metrics"]

    def test_synthetic_source_rejected(self, tmp_path):
        cfg = _tiny_model(load_config("config/config.yaml"))
        cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        cfg["data"].update(n_samples=16, seq_len=8, val_split=0.25)
        cfg["data"]["report_baselines"] = True
        with pytest.raises(ValueError, match="CSV source"):
            _trainer(tmp_path, cfg).train()

    def test_missing_date_column_rejected_before_training(self, tmp_path):
        path = tmp_path / "series.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=40, freq="D").strftime(
                    "%Y-%m-%d"
                ),
                "value": np.arange(40, dtype=float),
            }
        ).to_csv(path, index=False)
        cfg = _tiny_model(load_config("config/config.yaml"))
        cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        cfg["data"] = {
            "source": "csv",
            "path": str(path),
            "date_column": "when",
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 4,
            "val_split": 0.25,
            "split_mode": "temporal",
            "report_baselines": True,
        }
        with pytest.raises(ValueError, match="date column"):
            _trainer(tmp_path, cfg).train()
        assert not (tmp_path / "ckpt").exists()

    def test_random_split_rejected(self, tmp_path):
        path = tmp_path / "series.csv"
        pd.DataFrame({"value": np.arange(40, dtype=float)}).to_csv(path, index=False)
        cfg = _tiny_model(load_config("config/config.yaml"))
        cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        cfg["data"] = {
            "source": "csv",
            "path": str(path),
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 4,
            "val_split": 0.25,
            "split_mode": "random",
            "report_baselines": True,
        }
        with pytest.raises(ValueError, match="temporal validation"):
            _trainer(tmp_path, cfg).train()

    def test_multi_step_horizon_rejected_before_training(self, tmp_path):
        path = tmp_path / "series.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=40, freq="D").strftime(
                    "%Y-%m-%d"
                ),
                "value": np.arange(40, dtype=float),
            }
        ).to_csv(path, index=False)
        cfg = _tiny_model(load_config("config/config.yaml"))
        cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        cfg["forecast"] = {"horizon": 7, "strategy": "direct"}
        cfg["data"] = {
            "source": "csv",
            "path": str(path),
            "date_column": "date",
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 4,
            "val_split": 0.25,
            "split_mode": "temporal",
            "report_baselines": True,
        }
        with pytest.raises(ValueError, match="forecast.horizon"):
            _trainer(tmp_path, cfg).train()
        assert not (tmp_path / "ckpt").exists()


def test_zero_blend_weight_reports_the_classical_forecast(tmp_path):
    n = 500
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    temperature = 10 + 8 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temperature_c": temperature,
        }
    ).to_csv(path, index=False)
    cfg = _tiny_model(load_config("config/config.yaml"))
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["data"] = {
        "source": "csv",
        "path": str(path),
        "date_column": "date",
        "feature_columns": ["temperature_c"],
        "target_column": "temperature_c",
        "seq_len": 8,
        "val_split": 0.2,
        "split_mode": "temporal",
        "normalize": True,
        "report_baselines": True,
        "blend_with": "gradient_boosting",
        "blend_weight": 0.0,
    }
    result = _trainer(tmp_path, cfg).train()
    assert result["metrics"]["rmse"] == pytest.approx(
        result["baselines"]["gradient_boosting"]["rmse"]
    )


def test_blend_without_baselines_rejected_when_validation_is_scored(tmp_path):
    n = 80
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2019-01-01", periods=n, freq="D").strftime(
                "%Y-%m-%d"
            ),
            "temperature_c": np.linspace(0.0, 1.0, n),
        }
    ).to_csv(path, index=False)
    cfg = _tiny_model(load_config("config/config.yaml"))
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["data"] = {
        "source": "csv",
        "path": str(path),
        "date_column": "date",
        "feature_columns": ["temperature_c"],
        "target_column": "temperature_c",
        "seq_len": 8,
        "val_split": 0.2,
        "split_mode": "temporal",
        "blend_with": "gradient_boosting",
    }
    with pytest.raises(ValueError, match="report_baselines"):
        _trainer(tmp_path, cfg).train()
    assert not (tmp_path / "ckpt").exists()


def test_empty_validation_refit_accepts_a_holdout_blend(tmp_path):
    n = 80
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2019-01-01", periods=n, freq="D").strftime(
                "%Y-%m-%d"
            ),
            "temperature_c": np.linspace(0.0, 1.0, n),
        }
    ).to_csv(path, index=False)
    cfg = _tiny_model(load_config("config/config.yaml"))
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["training"]["checkpoint_rule"] = "last"
    cfg["data"] = {
        "source": "csv",
        "path": str(path),
        "date_column": "date",
        "feature_columns": ["temperature_c"],
        "target_column": "temperature_c",
        "seq_len": 8,
        "val_split": 0.0,
        "test_start": "2019-03-01",
        "split_mode": "temporal",
        "blend_with": "gradient_boosting",
        "blend_weight": 0.5,
    }
    result = _trainer(tmp_path, cfg).train()
    assert result["baselines"] is None
    assert result["best_epoch"] == 1
    assert (tmp_path / "ckpt" / "best_model.pt").is_file()


def test_weather_refit_uses_the_decision_epoch_and_does_not_score_validation():
    cfg = load_config("config/weather_refit.yaml")
    data = cfg["data"]
    assert data["test_start"] == "2017-01-07"
    assert not data.get("val_start")
    assert data["report_baselines"] is False
    assert data["blend_with"] == "gradient_boosting"
    assert data["blend_weight"] == 0.5
    assert data["seq_len"] == 28
    assert cfg["model"]["input_dim"] == 10
    assert cfg["model"]["residual"] is True
    assert cfg["training"]["epochs"] == 17
    assert cfg["training"]["patience"] == 0
    assert cfg["training"]["checkpoint_rule"] == "last"


def test_build_model_accepts_weather_width():
    cfg = load_config("config/weather.yaml")
    model = build_model(cfg)
    assert model.input_dim == 10
    assert model.residual is True
