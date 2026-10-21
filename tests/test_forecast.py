"""Tests for multi-step forecasts, quantile heads, and calibrated intervals."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.data.calendar import calendar_features
from src.data.splits import (
    fraction_window_indices,
    split_by_label_dates,
    split_counts,
)
from src.data.timeseries import load_csv_timeseries, make_sine_dataset
from src.data.windows import prepare_csv_series
from src.evaluation.bootstrap import block_bootstrap_improvement
from src.evaluation.classical import score_classical_baselines
from src.forecasting.head import ForecastHead
from src.forecasting.intervals import (
    apply_qhat,
    calibrate_intervals,
    conformal_quantile,
    interval_coverage,
    pinball_loss,
)
from src.forecasting.output import point_forecast, training_targets
from src.forecasting.recursive import advance_window
from src.forecasting.spec import ForecastSpec, forecast_spec_from_config
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.models.revin import RevIN
from src.models.transformer import TimeSeriesTransformer
from src.storage.experiments import ExperimentStore
from src.training.trainer import TimeSeriesTrainer
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config


def _tiny_transformer(**kwargs: object) -> TimeSeriesTransformer:
    params: dict[str, object] = {
        "input_dim": 1,
        "output_dim": 1,
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
    }
    params.update(kwargs)
    return TimeSeriesTransformer(**params)  # type: ignore[arg-type]


class TestLegacyHead:
    def test_one_step_point_head_is_linear(self):
        model = _tiny_transformer()
        assert isinstance(model.output_projection, nn.Linear)
        assert "output_projection.weight" in model.state_dict()
        assert "output_projection.projection.weight" not in model.state_dict()

    def test_config_without_quantiles_loads_legacy_state_dict(self):
        legacy = _tiny_transformer()
        cfg = {
            "model": {
                "input_dim": 1,
                "output_dim": 1,
                "d_model": 8,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
            }
        }
        built = build_model(cfg)
        built.load_state_dict(legacy.state_dict())
        x = torch.randn(2, 5, 1)
        with torch.no_grad():
            assert torch.allclose(built(x), legacy(x))

    def test_checkpoint_without_forecast_meta_predicts_one_step(self, tmp_path):
        model = _tiny_transformer()
        path = tmp_path / "legacy.pt"
        save_checkpoint(model, torch.optim.Adam(model.parameters()), 1, 0.1, {}, path)
        loaded = Predictor.from_checkpoint(_tiny_transformer(), path)
        out = loaded.predict(torch.zeros(4, 1))
        assert out.shape == (1, 1)
        forecast = loaded.forecast(torch.zeros(4, 1), horizon=1)
        assert forecast.point.shape == (1, 1, 1)
        assert forecast.lower is None


class TestForecastHead:
    def test_direct_horizon_shape(self):
        model = _tiny_transformer(horizon=7)
        out = model(torch.randn(3, 6, 1))
        assert out.shape == (3, 7, 1)

    def test_quantiles_do_not_cross(self):
        model = _tiny_transformer(horizon=4, quantiles=(0.1, 0.5, 0.9))
        out = model(torch.randn(5, 6, 1))
        assert out.shape == (5, 4, 3, 1)
        assert torch.all(out[:, :, 0, :] <= out[:, :, 1, :])
        assert torch.all(out[:, :, 1, :] <= out[:, :, 2, :])
        point = point_forecast(out, horizon=4, quantiles=(0.1, 0.5, 0.9))
        assert point.shape == (5, 4, 1)
        assert torch.allclose(point, out[:, :, 1, :])

    def test_one_step_quantiles_shape(self):
        head = ForecastHead(8, 1, 1, (0.1, 0.9))
        out = head(torch.randn(2, 8))
        assert out.shape == (2, 2, 1)

    def test_gradient_flows_through_quantile_head(self):
        model = _tiny_transformer(horizon=3, quantiles=(0.1, 0.5, 0.9))
        loss = model(torch.randn(2, 4, 1)).sum()
        loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )


class TestIntervals:
    def test_pinball_zero_at_the_target(self):
        loss = pinball_loss(
            torch.zeros(1, 2, 1),
            torch.zeros(1, 1),
            (0.1, 0.9),
        )
        assert loss.item() == 0.0

    def test_pinball_known_value(self):
        preds = torch.zeros(1, 1, 1)
        target = torch.ones(1, 1)
        assert pinball_loss(preds, target, (0.9,)).item() == pytest.approx(0.9)

    def test_coverage_counts_hits(self):
        lower = torch.tensor([[0.0], [0.0]])
        upper = torch.tensor([[1.0], [1.0]])
        assert interval_coverage(lower, upper, torch.tensor([[0.5], [2.0]])) == 0.5

    def test_conformal_rank_is_finite_sample(self):
        scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
        # ceil((4 + 1) * 0.8) = 4 -> the largest score.
        assert conformal_quantile(scores, 0.8) == 3.0

    def test_cqr_holdout_is_not_the_fit_half(self):
        point = torch.zeros(8, 2, 1)
        lower = point - 0.1
        upper = point + 0.1
        targets = torch.zeros(8, 2, 1)
        targets[6:] = 5.0
        result = calibrate_intervals(
            point=point,
            targets=targets,
            nominal=0.8,
            method="cqr",
            lower=lower,
            upper=upper,
        )
        assert result.method == "cqr"
        assert len(result.qhat) == 2
        assert result.raw_coverage is not None
        assert result.validation_coverage is not None
        assert 0.0 <= result.validation_coverage <= 1.0
        # The later half is outside the raw bounds, so raw coverage is not perfect.
        assert result.raw_coverage < 1.0

    def test_negative_qhat_does_not_flip_bounds(self):
        lower = torch.zeros(1, 1, 1)
        upper = torch.ones(1, 1, 1)
        adjusted_lower, adjusted_upper = apply_qhat(lower, upper, (-2.0,))
        assert torch.all(adjusted_lower <= adjusted_upper)


class TestWindows:
    def test_horizon_one_sine_matches_historical_layout(self):
        for seed in (0, 3, 11):
            inputs, targets = make_sine_dataset(8, 6, noise=0.05, seed=seed, horizon=1)
            generator = torch.Generator().manual_seed(seed)
            t = torch.linspace(0, 4 * torch.pi, 7)
            offsets = torch.rand(8, generator=generator) * 2 * torch.pi
            series = torch.sin(t.unsqueeze(0) + offsets.unsqueeze(1))
            series += torch.randn(series.shape, generator=generator) * 0.05
            assert torch.equal(inputs, series[:, :-1].unsqueeze(-1))
            assert torch.equal(targets, series[:, -1].unsqueeze(-1))

    def test_multi_step_sine_shape(self):
        inputs, targets = make_sine_dataset(4, 5, horizon=3, seed=1)
        assert inputs.shape == (4, 5, 1)
        assert targets.shape == (4, 3, 1)

    def test_csv_horizon_stacks_future_rows(self, tmp_path):
        path = tmp_path / "series.csv"
        pd.DataFrame({"value": list(range(10))}).to_csv(path, index=False)
        one_x, one_y = load_csv_timeseries(
            path, ["value"], "value", seq_len=4, horizon=1
        )
        many_x, many_y = load_csv_timeseries(
            path, ["value"], "value", seq_len=4, horizon=3
        )
        assert one_y.shape == (6, 1)
        assert torch.equal(
            one_y.view(-1), torch.tensor([4, 5, 6, 7, 8, 9], dtype=torch.float32)
        )
        assert many_x.shape == (4, 4, 1)
        assert many_y.shape == (4, 3, 1)
        assert torch.equal(
            many_y[0, :, 0], torch.tensor([4, 5, 6], dtype=torch.float32)
        )

    def test_training_targets_keep_one_step_layout(self):
        targets = torch.arange(6, dtype=torch.float32).view(2, 3, 1)
        spec = ForecastSpec(horizon=3, strategy="recursive")
        assert training_targets(targets, spec).shape == (2, 1)
        direct = ForecastSpec(horizon=3, strategy="direct")
        assert training_targets(targets, direct).shape == (2, 3, 1)

    def test_recursive_advance_appends_prediction(self):
        window = torch.tensor([[[1.0], [2.0], [3.0]]])
        nxt = advance_window(window, torch.tensor([[[4.0], [5.0]]]), target_index=0)
        assert torch.equal(nxt, torch.tensor([[[3.0], [4.0], [5.0]]]))

    def test_recursive_requires_a_target_index_when_dims_differ(self):
        window = torch.ones(1, 3, 2)
        with pytest.raises(ValueError, match="target_feature_index"):
            advance_window(window, torch.ones(1, 1, 1), target_index=None)


class TestSpec:
    def test_missing_section_is_one_step(self):
        spec = forecast_spec_from_config({"model": {}, "data": {}})
        assert spec.horizon == 1
        assert spec.strategy == "direct"
        assert spec.quantiles == ()
        assert spec.model_horizon == 1

    def test_quantiles_set_nominal_coverage(self):
        spec = forecast_spec_from_config(
            {"forecast": {"horizon": 7, "quantiles": [0.1, 0.5, 0.9]}}
        )
        assert spec.model_horizon == 7
        assert spec.nominal_coverage == 0.8

    def test_recursive_head_stays_one_step(self):
        spec = forecast_spec_from_config(
            {"forecast": {"horizon": 30, "strategy": "recursive"}}
        )
        assert spec.horizon == 30
        assert spec.model_horizon == 1

    def test_explicit_null_target_index_is_not_inferred(self):
        spec = forecast_spec_from_config(
            {
                "model": {"input_dim": 2, "output_dim": 1},
                "data": {"source": "synthetic"},
                "forecast": {"target_feature_index": None},
            }
        )
        assert spec.target_feature_index is None

    def test_bad_strategy_rejected(self):
        with pytest.raises(ValueError, match="strategy"):
            forecast_spec_from_config({"forecast": {"strategy": "magic"}})


class TestTrainingIntervals:
    def test_quantile_run_records_holdout_coverage(self, tmp_path):
        cfg = load_config("config/config.yaml")
        cfg["training"].update(
            epochs=1,
            batch_size=8,
            device="cpu",
            seed=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cfg["data"].update(
            n_samples=32, seq_len=6, val_split=0.25, seed=0, split_mode="temporal"
        )
        cfg["model"].update(
            d_model=8, nhead=2, num_encoder_layers=1, dim_feedforward=16, dropout=0.0
        )
        cfg["forecast"] = {
            "horizon": 3,
            "strategy": "direct",
            "quantiles": [0.1, 0.5, 0.9],
        }
        trainer = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
        trainer.cfg = cfg
        trainer.store = ExperimentStore(tmp_path / "exp.db")
        result = trainer.train()
        assert 0.0 <= result["metrics"]["coverage"] <= 1.0
        assert 0.0 <= result["metrics"]["raw_coverage"] <= 1.0
        checkpoint = torch.load(tmp_path / "ckpt" / "best_model.pt", weights_only=True)
        meta = checkpoint["forecast"]
        assert meta["interval_method"] == "cqr"
        assert len(meta["conformal_qhat"]) == 3
        loaded = Predictor.from_checkpoint(
            build_model(cfg), tmp_path / "ckpt" / "best_model.pt"
        )
        forecast = loaded.forecast(torch.zeros(6, 1), horizon=3)
        assert forecast.point.shape == (1, 3, 1)
        assert forecast.lower is not None and forecast.upper is not None
        assert torch.all(forecast.lower <= forecast.upper)
        assert forecast.interval_method == "cqr"
        assert forecast.validation_coverage == result["metrics"]["coverage"]


class TestPredictorRollout:
    def test_recursive_horizon_extends_a_one_step_model(self):
        model = _tiny_transformer()
        predictor = Predictor(model, spec=ForecastSpec(horizon=1, strategy="direct"))
        one = predictor.forecast(torch.zeros(5, 1), horizon=1)
        many = predictor.forecast(torch.zeros(5, 1), horizon=4)
        assert many.point.shape == (1, 4, 1)
        assert torch.allclose(many.point[:, :1, :], one.point)
        assert many.strategy == "recursive"

    def test_direct_model_slices_a_shorter_request(self):
        model = _tiny_transformer(horizon=4)
        predictor = Predictor(
            model, spec=ForecastSpec(horizon=4, strategy="direct", quantiles=())
        )
        out = predictor.forecast(torch.randn(6, 1), horizon=2)
        assert out.point.shape == (1, 2, 1)
        assert out.strategy == "direct"

    def test_residual_interval_expands_the_point(self):
        model = _tiny_transformer()
        predictor = Predictor(
            model,
            spec=ForecastSpec(horizon=2, strategy="recursive"),
            forecast_meta={
                "interval_method": "residual",
                "conformal_qhat": [0.5, 1.5],
                "validation_coverage": 0.75,
                "coverage_horizon": 2,
            },
        )
        out = predictor.forecast(torch.linspace(0, 1, 4).unsqueeze(-1), horizon=2)
        assert out.interval_method == "residual"
        assert out.validation_coverage == 0.75
        assert torch.allclose(
            out.lower, out.point - torch.tensor([0.5, 1.5]).view(1, 2, 1)
        )
        assert torch.allclose(
            out.upper, out.point + torch.tensor([0.5, 1.5]).view(1, 2, 1)
        )
        shorter = predictor.forecast(torch.linspace(0, 1, 4).unsqueeze(-1), horizon=1)
        assert shorter.validation_coverage is None


def test_fraction_counts_match_the_historical_validation_rule():
    train, val, test = fraction_window_indices(10, val_split=0.2, test_split=0.0)
    assert train.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert val.tolist() == [8, 9]
    assert test.tolist() == []
    assert split_counts(10, 0.2, 0.3) == (6, 1, 3)


def test_test_start_alone_leaves_validation_empty():
    labels = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-08"))
    train, val, test = split_by_label_dates(labels, test_start="2020-01-05")
    assert train.tolist() == [0, 1, 2, 3]
    assert val.tolist() == []
    assert test.tolist() == [4, 5, 6]


def test_date_split_holds_out_the_named_days():
    labels = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-11"))
    train, val, test = split_by_label_dates(
        labels, val_start="2020-01-06", test_start="2020-01-09"
    )
    assert train.tolist() == [0, 1, 2, 3, 4]
    assert val.tolist() == [5, 6, 7]
    assert test.tolist() == [8, 9]


def test_prepare_without_extensions_matches_the_csv_loader(tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=40, freq="D").strftime(
                "%Y-%m-%d"
            ),
            "value": np.linspace(0.0, 4.0, 40),
        }
    )
    path = tmp_path / "series.csv"
    frame.to_csv(path, index=False)
    cfg = {
        "path": str(path),
        "feature_columns": ["value"],
        "target_column": "value",
        "seq_len": 5,
        "val_split": 0.2,
        "split_mode": "temporal",
        "date_column": "date",
    }
    prepared = prepare_csv_series(cfg)
    loaded_x, loaded_y = load_csv_timeseries(path, ["value"], "value", seq_len=5)
    assert torch.equal(prepared.inputs, loaded_x)
    assert torch.equal(prepared.targets, loaded_y)
    assert len(prepared.val_index) == int(35 * 0.2)


def test_missing_indicator_uses_only_rows_before_validation(tmp_path):
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    values = np.arange(30, dtype=float)
    covariate = values.copy()
    covariate[0] = np.nan
    covariate[25] = np.nan
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "value": values,
            "extra": covariate,
        }
    )
    path = tmp_path / "series.csv"
    frame.to_csv(path, index=False)
    prepared = prepare_csv_series(
        {
            "path": str(path),
            "feature_columns": ["value", "extra"],
            "target_column": "value",
            "seq_len": 4,
            "date_column": "date",
            "val_start": "2020-01-21",
            "missing_indicators": True,
        }
    )
    assert prepared.feature_names == ["value", "extra", "extra_missing"]
    # Row 0 is in the fitting block, so the indicator exists. The median of the
    # fitting rows (extra[1:20], since the label boundary is 2020-01-21 and
    # seq_len is 4) fills the gap. Row 25 is a validation day and must not
    # change that median.
    assert prepared.inputs[0, 0, 2].item() == 1.0
    assert np.isfinite(prepared.inputs.numpy()).all()


def test_calendar_day_of_year_matches_the_weather_encoding():
    frame = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"]})
    features = calendar_features(frame["date"], ["doy"])
    angle = 2.0 * math.pi * 1 / 365.25
    assert features.loc[0, "doy_sin"] == pytest.approx(math.sin(angle))
    assert features.loc[0, "doy_cos"] == pytest.approx(math.cos(angle))


def test_revin_round_trip_restores_the_window():
    torch.manual_seed(0)
    layer = RevIN(3, affine=True)
    window = torch.randn(4, 8, 3)
    normalized, stats = layer.normalize(window)
    restored = normalized.clone()
    restored = (restored - layer.bias) / layer.weight
    mean, stdev = stats
    restored = restored * stdev + mean
    assert torch.allclose(restored, window, atol=1e-5)


def test_zero_head_residual_is_persistence():
    model = TimeSeriesTransformer(
        input_dim=1,
        output_dim=1,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        residual=True,
    )
    torch.nn.init.zeros_(model.output_projection.weight)
    torch.nn.init.zeros_(model.output_projection.bias)
    window = torch.randn(4, 10, 1)
    output = model(window)
    assert torch.allclose(output, window[:, -1, :], atol=1e-5)


def test_patched_forward_keeps_the_latest_step():
    model = TimeSeriesTransformer(
        input_dim=2,
        output_dim=1,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        patch_len=4,
        patch_stride=4,
        use_revin=True,
        residual=True,
    )
    torch.nn.init.zeros_(model.output_projection.weight)
    torch.nn.init.zeros_(model.output_projection.bias)
    window = torch.randn(3, 12, 2)
    output = model(window)
    assert output.shape == (3, 1)
    assert torch.allclose(output, window[:, -1, :1], atol=1e-4)


def test_block_bootstrap_is_zero_for_identical_forecasts():
    target = np.linspace(0.0, 1.0, 40)
    stats = block_bootstrap_improvement(
        target, target + 1.0, target + 1.0, block=7, n_resamples=50, seed=0
    )
    assert stats["improvement"] == pytest.approx(0.0)
    assert stats["p05"] == pytest.approx(0.0)
    assert stats["p95"] == pytest.approx(0.0)


def test_classical_climatology_beats_persistence_on_a_seasonal_series():
    dates = pd.date_range("2000-01-01", periods=1200, freq="D")
    doy = dates.dayofyear.to_numpy()
    signal = 15.0 + 10.0 * np.sin(2.0 * np.pi * doy / 365.25)
    rows = np.arange(1000, 1200)
    forecasts = score_classical_baselines(
        signal, rows, dates=dates.to_numpy(dtype="datetime64[D]"), seed=0
    )
    actual = signal[rows]
    persistence = signal[rows - 1]
    clim_rmse = float(np.sqrt(np.mean((forecasts["climatology"] - actual) ** 2)))
    persist_rmse = float(np.sqrt(np.mean((persistence - actual) ** 2)))
    assert clim_rmse < persist_rmse
    assert forecasts["gradient_boosting"].shape == rows.shape
