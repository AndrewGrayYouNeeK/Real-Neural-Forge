"""Tests for the temporal model-comparison command."""

import numpy as np
import pandas as pd
import pytest

from src.data.timeseries import load_windows_from_config
from src.evaluation.baselines import (
    regression_scores,
    seasonal_naive_calendar,
    temporal_validation_target_rows,
)
from src.evaluation.compare import compare_models
from src.storage.experiments import ExperimentStore
from src.utils.config import load_config


def _cfg(tmp_path, **forecast):
    cfg = load_config("config/config.yaml")
    cfg["data"] = {
        "source": "synthetic",
        "n_samples": 32,
        "seq_len": 6,
        "noise": 0.01,
        "val_split": 0.25,
        "seed": 0,
        "split_mode": "random",
    }
    cfg["training"].update(
        epochs=1,
        batch_size=8,
        learning_rate=0.01,
        device="cpu",
        seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    cfg["model"].update(
        d_model=8, nhead=2, num_encoder_layers=1, dim_feedforward=16, dropout=0.0
    )
    cfg["storage"] = {"db_path": str(tmp_path / "exp.db")}
    cfg["forecast"] = {
        "horizon": 2,
        "strategy": "direct",
        "season_length": 2,
        **forecast,
    }
    cfg["evaluation"] = {"lstm_hidden_size": 4, "lstm_layers": 1}
    return cfg


class TestCompareModels:
    def test_scores_every_model_on_the_temporal_holdout(self, tmp_path):
        cfg = _cfg(tmp_path)
        report = compare_models(cfg, epochs=1)
        names = [row.name for row in report.rows]
        assert names == [
            "persistence",
            "seasonal_naive",
            "linear_lags",
            "lstm",
            "time_series_transformer",
        ]
        inputs, _ = load_windows_from_config(cfg["data"], horizon=2)
        assert report.n_train + report.n_val == len(inputs)
        assert report.n_val == int(len(inputs) * 0.25)
        assert "persistence" in report.table
        assert "time_series_transformer" in report.table
        for row in report.rows:
            assert row.metrics["mse"] >= 0.0
            assert abs(row.metrics["rmse"] ** 2 - row.metrics["mse"]) < 1e-5
        neural = report.rows[-1]
        assert neural.interval_method == "residual"
        assert 0.0 <= neural.metrics["coverage"] <= 1.0

        store = ExperimentStore(tmp_path / "exp.db")
        experiment = store.get_experiment(report.experiment_id)
        assert experiment is not None
        assert experiment["status"] == "completed"
        assert experiment["model_name"] == "comparison"
        assert (
            experiment["config"]["evaluation"]["comparison"]["split_mode"] == "temporal"
        )
        assert "persistence.mse" in experiment["metrics"]
        assert experiment["metrics"]["persistence.mse"] == pytest.approx(
            report.rows[0].metrics["mse"]
        )
        # The configured split_mode was random; the stored comparison still says temporal.
        assert cfg["data"]["split_mode"] == "random"

    def test_quantile_model_reports_raw_and_calibrated_coverage(self, tmp_path):
        cfg = _cfg(tmp_path, quantiles=[0.1, 0.5, 0.9])
        report = compare_models(cfg, epochs=1)
        transformer = report.rows[-1]
        assert transformer.interval_method == "cqr"
        assert "raw_coverage" in transformer.metrics
        assert 0.0 <= transformer.metrics["raw_coverage"] <= 1.0
        assert 0.0 <= transformer.metrics["coverage"] <= 1.0

    def test_skips_seasonal_naive_without_a_period(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["forecast"].pop("season_length")
        report = compare_models(cfg, epochs=1)
        assert "seasonal_naive" not in [row.name for row in report.rows]
        assert any("seasonal_naive" in note for note in report.notes)

    def test_long_season_without_dates_is_skipped(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["forecast"]["season_length"] = 100
        report = compare_models(cfg, epochs=1)
        assert "seasonal_naive" not in [row.name for row in report.rows]
        assert any("longer than the input window" in note for note in report.notes)

    def test_dated_csv_uses_calendar_seasonal_naive(self, tmp_path):
        n = 800
        dates = pd.date_range("2018-01-01", periods=n, freq="D")
        values = 10 + np.sin(2 * np.pi * np.arange(n) / 365.25)
        path = tmp_path / "series.csv"
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "value": values}).to_csv(
            path, index=False
        )
        cfg = _cfg(tmp_path)
        cfg["forecast"] = {"horizon": 1, "strategy": "direct"}
        cfg["data"] = {
            "source": "csv",
            "path": str(path),
            "date_column": "date",
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 8,
            "val_split": 0.25,
            "normalize": False,
        }
        report = compare_models(cfg, epochs=1)
        rows = temporal_validation_target_rows(n, seq_len=8, val_split=0.25)
        expected = regression_scores(
            seasonal_naive_calendar(
                values, dates.to_numpy(dtype="datetime64[D]"), rows
            ),
            values[rows],
        )
        seasonal = next(row for row in report.rows if row.name == "seasonal_naive")
        assert seasonal.metrics["rmse"] == pytest.approx(expected["rmse"])
        assert any("calendar day" in note for note in report.notes)

    def test_lstm_config_is_not_duplicated(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["model"]["name"] = "lstm"
        cfg["model"]["hidden_size"] = 4
        report = compare_models(cfg, epochs=1)
        assert [row.name for row in report.rows].count("lstm") == 1
        assert "time_series_transformer" not in [row.name for row in report.rows]
