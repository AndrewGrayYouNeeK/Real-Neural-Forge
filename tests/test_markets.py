"""Market return features, parsers, baselines, and the training glue."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.timeseries import load_csv_timeseries
from src.markets.evaluate import (
    directional_accuracy,
    label_date_bounds,
    positive_fraction,
    split_sizes,
)
from src.markets.parse import (
    max_relative_diff,
    parse_binance_klines,
    parse_ecb_hist_csv,
    parse_frankfurter_rates,
    parse_fred_csv,
    parse_stooq_csv,
    parse_yahoo_chart,
    per_usd_frames,
)
from src.markets.report import DISCLAIMER, group_lead, summarize_group
from src.markets.returns import (
    build_return_frame,
    one_step_log_returns,
    price_frame_from_pairs,
    rolling_population_std,
)
from src.markets.run import load_catalogs, protocol_view, train_one
from src.markets.universe import (
    classify_history,
    dedupe_ranked,
    is_stablecoin,
    is_wrapped,
    price_specs,
)
from src.retrain.discover import discover


class TestLogReturns:
    def test_one_step_log_return(self):
        returns = one_step_log_returns(np.array([1.0, math.e, math.e**3]))
        assert math.isnan(returns[0])
        assert returns[1] == pytest.approx(1.0)
        assert returns[2] == pytest.approx(2.0)

    def test_nonpositive_price_does_not_bridge_the_gap(self):
        returns = one_step_log_returns(np.array([10.0, 11.0, -5.0, 12.0]))
        assert returns[1] == pytest.approx(math.log(11 / 10))
        assert math.isnan(returns[2])
        assert math.isnan(returns[3])

    def test_rolling_std_is_causal_and_population(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        vol = rolling_population_std(values, window=2)
        assert math.isnan(vol[0])
        assert vol[1] == pytest.approx(0.5)
        changed = values.copy()
        changed[-1] = 100.0
        again = rolling_population_std(changed, window=2)
        assert again[1] == pytest.approx(vol[1])
        assert again[2] == pytest.approx(vol[2])

    def test_nan_inside_the_window_blocks_volatility(self):
        vol = rolling_population_std(np.array([np.nan, 1.0, 1.0, 1.0]), window=3)
        assert math.isnan(vol[2])
        assert vol[3] == pytest.approx(0.0)

    def test_feature_frame_drops_warmup(self):
        dates = pd.date_range("2020-01-01", periods=15, freq="D")
        prices = price_frame_from_pairs(dates.tolist(), np.linspace(10, 20, 15))
        frame = build_return_frame(prices, vol_window=10)
        assert list(frame.columns) == ["date", "log_return", "vol_10", "close"]
        assert len(frame) == 5
        assert frame["date"].iloc[0] == "2020-01-11"


class TestParsers:
    def test_fred_drops_missing_observations(self):
        text = "observation_date,SP500\n2020-01-02,3200.0\n2020-01-03,.\n2020-01-06,3210.0\n"
        frame = parse_fred_csv(text)
        assert list(frame["close"]) == [3200.0, 3210.0]
        assert frame["date"].iloc[-1] == pd.Timestamp("2020-01-06")

    def test_stooq_sorts_oldest_first(self):
        text = "Date,Open,High,Low,Close,Volume\n2020-01-03,1,1,1,3,1\n2020-01-02,1,1,1,2,1\n"
        frame = parse_stooq_csv(text)
        assert list(frame["close"]) == [2.0, 3.0]

    def test_yahoo_prefers_adjusted_close(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1_577_836_800, 1_577_923_200],
                        "indicators": {
                            "quote": [{"close": [100.0, 110.0]}],
                            "adjclose": [{"adjclose": [50.0, 55.0]}],
                        },
                    }
                ],
                "error": None,
            }
        }
        frame, field = parse_yahoo_chart(payload)
        assert field == "adjclose"
        assert list(frame["close"]) == [50.0, 55.0]
        assert frame["date"].iloc[0] == pd.Timestamp("2020-01-01")

    def test_binance_uses_the_close(self):
        rows = [[1_577_836_800_000, "1", "3", "1", "2.5", "9"]]
        frame = parse_binance_klines(rows)
        assert frame["close"].iloc[0] == pytest.approx(2.5)
        assert frame["date"].iloc[0] == pd.Timestamp("2020-01-01")

    def test_ecb_rates_become_currency_per_usd_and_are_sorted(self):
        text = (
            "Date,USD,JPY,GBP\n"
            "2020-01-06,1.2000,120.0,0.8400\n"
            "1999-01-04,1.1789,133.73,0.7111\n"
            "1999-01-05,N/A,130.0,0.7000\n"
        )
        wide = parse_ecb_hist_csv(text)
        assert list(wide["Date"])[0] == pd.Timestamp("1999-01-04")
        frames = per_usd_frames(wide)
        assert frames["JPY"]["close"].iloc[0] == pytest.approx(133.73 / 1.1789)
        assert frames["EUR"]["close"].iloc[0] == pytest.approx(1 / 1.1789)
        assert frames["GBP"]["date"].iloc[0] == pd.Timestamp("1999-01-04")
        assert pd.Timestamp("1999-01-05") not in set(frames["JPY"]["date"])

    def test_frankfurter_relative_gap(self):
        day, rates = parse_frankfurter_rates(
            {"base": "USD", "date": "2020-01-06", "rates": {"JPY": 120.0, "EUR": 0.9}}
        )
        assert day == "2020-01-06"
        gap, compared = max_relative_diff({"JPY": 121.2, "EUR": 0.9}, rates)
        assert compared == 2
        assert gap == pytest.approx(0.01)


class TestEligibility:
    def test_stablecoins_and_wrapped_tokens_are_flagged(self):
        assert is_stablecoin("USDT", "Tether")
        assert is_stablecoin("OTHER", "Some USD Coin")
        assert is_wrapped("STETH", "Lido Staked Ether")
        assert not is_stablecoin("BTC", "Bitcoin")
        assert not is_wrapped("ETH", "Ethereum")

    def test_dedupe_keeps_the_better_rank(self):
        coins = dedupe_ranked(
            [
                {"symbol": "BTC", "rank": 4, "name": "other"},
                {"symbol": "ETH", "rank": 2, "name": "Ethereum"},
                {"symbol": "BTC", "rank": 1, "name": "Bitcoin"},
            ]
        )
        assert [coin["symbol"] for coin in coins] == ["BTC", "ETH"]
        assert coins[0]["name"] == "Bitcoin"

    def test_short_history_is_rejected(self):
        reason = classify_history(
            observations=100,
            span_days=100,
            feature_rows=80,
            seq_len=16,
            val_split=0.2,
        )
        assert reason is not None
        assert "shorter" in reason

    def test_two_year_sample_is_trainable(self):
        reason = classify_history(
            observations=800,
            span_days=900,
            feature_rows=780,
            seq_len=16,
            val_split=0.2,
        )
        assert reason is None

    def test_price_specs_have_unique_ids(self):
        ids = [spec.id for spec in price_specs()]
        assert len(ids) == len(set(ids))
        assert "cmdty-aluminum" in ids
        assert "cmdty-gold" in ids


class TestHoldout:
    def test_split_matches_the_trainer_formula(self):
        # resolve_val_count holds out int(n_windows * val_split) from the tail.
        n_windows = 100 - 16
        n_train, n_val = split_sizes(n_windows, 0.2)
        assert n_val == int(n_windows * 0.2)
        assert n_train + n_val == n_windows

    def test_holdout_is_the_most_recent_labels(self):
        dates = [f"2020-01-{day:02d}" for day in range(1, 31)]
        n_train, n_val = split_sizes(len(dates) - 4, 0.2)
        bounds = label_date_bounds(dates, seq_len=4, n_train=n_train, n_val=n_val)
        assert bounds["val_end"] == dates[-1]
        assert bounds["val_start"] > bounds["train_end"]
        assert bounds["train_start"] == dates[4]

    def test_window_does_not_contain_the_target_return(self, tmp_path):
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        frame = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "log_return": np.arange(30, dtype=float),
                "vol_10": np.ones(30),
            }
        )
        path = tmp_path / "features.csv"
        frame.to_csv(path, index=False)
        inputs, targets = load_csv_timeseries(
            path, ["log_return", "vol_10"], "log_return", seq_len=4
        )
        assert targets[0].item() == pytest.approx(4.0)
        assert inputs[0, -1, 0].item() == pytest.approx(3.0)
        assert inputs[0, 0, 0].item() == pytest.approx(0.0)

    def test_directional_accuracy_excludes_zero_moves(self):
        preds = torch.tensor([1.0, -1.0, 1.0, 0.0])
        targets = torch.tensor([1.0, -1.0, -1.0, 0.0])
        assert directional_accuracy(preds, targets) == pytest.approx(2 / 3)
        assert positive_fraction(targets) == pytest.approx(1 / 3)

    def test_empty_direction_is_rejected(self):
        with pytest.raises(ValueError):
            directional_accuracy(torch.zeros(2), torch.zeros(2))


class TestReport:
    def test_fx_summary_states_the_random_walk_result(self):
        rows = [
            _row(
                "fx-eur",
                "fx",
                model=0.005,
                persistence=0.004,
                mean=0.0041,
                direction=0.51,
            ),
            _row(
                "fx-jpy",
                "fx",
                model=0.006,
                persistence=0.0055,
                mean=0.0056,
                direction=0.49,
            ),
        ]
        summary = summarize_group(rows)
        assert summary["n_beat_persistence"] == 0
        text = group_lead("fx", summary)
        assert "random walk" in text
        assert DISCLAIMER.startswith("Not financial advice")

    def test_a_lower_rmse_counts_as_beating_persistence(self):
        rows = [
            _row(
                "crypto-btc",
                "crypto",
                model=0.02,
                persistence=0.03,
                mean=0.025,
                direction=0.54,
            )
        ]
        summary = summarize_group(rows)
        assert summary["n_beat_persistence"] == 1
        assert summary["n_beat_mean"] == 1
        assert summary["median_rmse_ratio_vs_persistence"] == pytest.approx(0.02 / 0.03)


class TestCommittedResults:
    def test_readme_contains_the_measured_summaries(self):
        results_path = Path("results/market_forecasts.json")
        readme = Path("README.md").read_text()
        results = json.loads(results_path.read_text())
        assert results["disclaimer"].startswith("Not financial advice")
        assert "Not financial advice" in readme
        for text in results["narratives"].values():
            assert text in readme
        assert "FX reference rates are close to a random walk." in readme
        groups = {row["group"] for row in results["series"]}
        assert groups == {"stocks", "fx", "crypto", "commodities"}
        assert sum(row["group"] == "crypto" for row in results["series"]) == 50
        assert sum(row["group"] == "fx" for row in results["series"]) >= 30


class TestConfigs:
    def test_market_configs_share_one_protocol(self):
        catalogs = load_catalogs()
        assert set(catalogs) == {"stocks", "fx", "crypto", "commodities"}
        view = protocol_view(catalogs["fx"])
        assert view["seed"] == 0
        assert view["split_mode"] == "temporal"
        assert view["normalize"] is True
        assert view["patience"] > 0
        assert view["target"] == "log_return"
        assert view["features"] == ["log_return", "vol_10"]
        assert view["seq_len"] == 16
        assert view["val_split"] == pytest.approx(0.2)

    def test_catalogs_stay_out_of_champion_retraining(self):
        specs, errors = discover(Path("config"))
        assert errors == []
        found = {spec.id for spec in specs}
        assert {"stocks", "fx", "crypto", "commodities"}.isdisjoint(found)
        assert "default" in found


class TestTrainAndScore:
    def test_checkpoint_scores_match_the_training_split(self, tmp_path):
        dates = pd.date_range("2018-01-01", periods=160, freq="D")
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0005, 0.01, size=len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        frame = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "log_return": returns,
                "vol_10": np.full(len(dates), 0.01),
                "close": close,
            }
        )
        feature_path = tmp_path / "walk.csv"
        frame.to_csv(feature_path, index=False)
        catalog = {
            "model": {
                "name": "time_series_transformer",
                "input_dim": 2,
                "output_dim": 1,
                "d_model": 8,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
                "max_seq_len": 32,
            },
            "data": {
                "feature_columns": ["log_return", "vol_10"],
                "target_column": "log_return",
                "seq_len": 8,
                "val_split": 0.2,
                "split_mode": "temporal",
                "normalize": True,
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 1,
                "patience": 2,
                "max_grad_norm": 1.0,
                "device": "cpu",
                "seed": 0,
                "log_interval": 100,
            },
        }
        scored = train_one(
            catalog,
            {"id": "toy", "feature_path": str(feature_path)},
            tmp_path / "runs",
            checkpoint_dir=str(tmp_path / "ckpt"),
            db_path=str(tmp_path / "experiments.db"),
        )
        assert scored["val_end"] == frame["date"].iloc[-1]
        assert scored["val_start"] > scored["train_end"]
        assert scored["model_rmse"] > 0
        assert 0.0 <= scored["directional_accuracy"] <= 1.0
        assert scored["scaler_y_mean"] == pytest.approx(scored["mean_return"], rel=1e-5)
        assert scored["persistence_rmse"] > 0
        assert scored["best_epoch"] == 1


def _row(
    series_id: str,
    group: str,
    model: float,
    persistence: float,
    mean: float,
    direction: float,
) -> dict[str, object]:
    return {
        "id": series_id,
        "group": group,
        "model_rmse": model,
        "persistence_rmse": persistence,
        "mean_rmse": mean,
        "directional_accuracy": direction,
        "mean_directional_accuracy": 0.55,
        "positive_fraction": 0.52,
        "beats_persistence_rmse": model < persistence,
        "beats_mean_rmse": model < mean,
        "rmse_ratio_vs_persistence": model / persistence,
        "rmse_ratio_vs_mean": model / mean,
        "train_seconds": 1.0,
    }
