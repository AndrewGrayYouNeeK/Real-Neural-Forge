"""Baselines, leak-free racing sequences, and a tiny series training run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.domain_forecasts.baselines import (
    doubling_time_years,
    flat_stake_profit,
    halving_time_years,
    iter_months,
    linear_fit,
    market_implied_probabilities,
    multiclass_brier,
    multiclass_log_loss,
    rank_best_first,
    uniform_race_expectations,
    window_split_bounds,
)
from src.domain_forecasts.racing import (
    DRAW_PCT,
    IS_TODAY,
    JOCKEY_RATE,
    LOG_ODDS,
    WON,
    build_form_sequences,
    score_race,
    split_masks,
)
from src.domain_forecasts.series import make_windows, train_series
from src.utils.config import load_config


def _runners() -> pd.DataFrame:
    rows = []
    for race_index, (day, winner) in enumerate(
        (("2010-01-01", "a"), ("2010-02-01", "b"), ("2010-03-15", "a"))
    ):
        for horse, odds, draw, finish in (
            ("a", 2.0, 1, 1 if winner == "a" else 2),
            ("b", 3.5, 2, 1 if winner == "b" else 2),
        ):
            rows.append(
                {
                    "race_date": pd.Timestamp(day),
                    "race_key": f"race-{race_index}",
                    "horse_id": horse,
                    "finish": finish,
                    "odds": odds,
                    "weight": 120.0 + draw,
                    "draw": draw,
                    "jockey_id": f"j-{horse}",
                    "trainer_id": "t-1",
                }
            )
    return pd.DataFrame(rows)


class TestBaselines:
    def test_months_and_split(self):
        assert iter_months("2012-11", "2013-01") == ["2012-11", "2012-12", "2013-01"]
        train_end, val_end = window_split_bounds(10, 0.2, 0.1)
        assert (train_end, val_end, 10) == (7, 8, 10)

    def test_doubling_and_halving(self):
        slope = np.log10(2.0) / 2.0
        assert doubling_time_years(slope, "log10") == 2.0
        assert doubling_time_years(0.0, "log10") is None
        assert halving_time_years(-slope, "log10") == 2.0
        intercept, fitted_slope = linear_fit(
            np.array([0.0, 2.0]), np.array([0.0, np.log10(2.0)])
        )
        assert abs(intercept) < 1e-9
        assert abs(fitted_slope - slope) < 1e-9

    def test_market_and_stake(self):
        probs, overround = market_implied_probabilities(np.array([2.0, 2.0]))
        assert np.allclose(probs, [0.5, 0.5])
        assert abs(overround - 0.0) < 1e-12
        assert flat_stake_profit(3.0, True) == 2.0
        assert flat_stake_profit(3.0, False) == -1.0
        assert abs(multiclass_log_loss(probs, 0) - np.log(2.0)) < 1e-12
        assert abs(multiclass_brier(np.array([0.5, 0.5]), 0) - 0.5) < 1e-12
        expectations = uniform_race_expectations(4)
        assert expectations["win_accuracy"] == 0.25
        assert expectations["top3_hit_rate"] == 0.75

    def test_favourite_tie_breaks_on_draw(self):
        order = rank_best_first(
            np.array([2.0, 2.0]), [np.array([5.0, 1.0]), np.array([0, 1])]
        )
        assert int(order[0]) == 1


class TestRacingSequences:
    def test_current_odds_and_result_are_hidden(self):
        dataset = build_form_sequences(_runners(), history_len=2)
        second_a = np.flatnonzero(
            (dataset.race_keys == "race-1") & (dataset.horse_codes == 0)
        )[0]
        sequence = dataset.inputs[second_a]
        today = sequence[-1]
        assert today[IS_TODAY] == 1.0
        assert today[LOG_ODDS] == 0.0
        assert today[WON] == 0.0
        past = sequence[-2]
        assert past[IS_TODAY] == 0.0
        assert past[WON] == 1.0
        assert abs(past[LOG_ODDS] - np.log(2.0)) < 1e-5
        # The March race must not appear in the February sequence.
        assert dataset.finish[second_a] == 2
        assert abs(float(dataset.targets[second_a, 0]) - 1.0) < 1e-6

    def test_jockey_rate_uses_only_earlier_races(self):
        dataset = build_form_sequences(_runners(), history_len=2)
        # Horse a's jockey won race 0, so the rate on race 1 is 1.
        second_a = np.flatnonzero(
            (dataset.race_keys == "race-1") & (dataset.horse_codes == 0)
        )[0]
        assert dataset.inputs[second_a, -1, JOCKEY_RATE] == 1.0
        first_a = np.flatnonzero(
            (dataset.race_keys == "race-0") & (dataset.horse_codes == 0)
        )[0]
        assert dataset.inputs[first_a, -1, JOCKEY_RATE] == 0.0
        assert dataset.padding_mask[first_a, 0]
        assert dataset.padding_mask[first_a, 1]
        assert not dataset.padding_mask[first_a, -1]
        assert dataset.inputs[first_a, -1, DRAW_PCT] == 0.5

    def test_later_races_are_the_test_split(self):
        dataset = build_form_sequences(_runners(), history_len=1)
        masks = split_masks(dataset, test_fraction=0.34, val_fraction=0.34)
        test_dates = dataset.race_dates[masks["test"]]
        train_dates = dataset.race_dates[masks["train"]]
        assert test_dates.min() > train_dates.max()
        assert masks["train"].sum() and masks["val"].sum() and masks["test"].sum()

    def test_score_race_picks_the_lower_score(self):
        scored = score_race(
            {
                "date": "2010-01-01",
                "scores": [0.9, 0.1],
                "odds": [1.5, 4.0],
                "finish": [1, 2],
                "draw": [1, 2],
                "horse_codes": [0, 1],
            },
            temperature=0.2,
        )
        assert scored["favourite_won"] is True
        assert scored["model_won"] is False
        assert scored["model_profit"] == -1.0
        assert abs(scored["favourite_profit"] - 0.5) < 1e-12
        assert scored["model_log_loss"] > scored["market_log_loss"]


class TestSeriesTraining:
    def test_windows_and_tiny_fit(self, tmp_path: Path):
        rows = 24
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=rows, freq="MS"),
                "value": np.linspace(1.0, 3.0, rows) + 0.01 * np.arange(rows),
            }
        )
        csv_path = tmp_path / "series.csv"
        frame.to_csv(csv_path, index=False)
        inputs, targets = make_windows(frame[["value"]].to_numpy(), seq_len=4)
        assert inputs.shape == (rows - 4, 4, 1)
        assert targets.shape == (rows - 4, 1)
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
                "path": str(csv_path),
                "date_column": "date",
                "feature_columns": ["value"],
                "seq_len": 4,
                "test_fraction": 0.2,
                "val_fraction": 0.2,
                "normalize": True,
                "transform": "none",
                "already_log10": False,
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.01,
                "epochs": 1,
                "patience": 0,
                "max_grad_norm": 1.0,
                "device": "cpu",
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "seed": 0,
            },
            "output": {"results": str(tmp_path / "results.json")},
        }
        config_path = tmp_path / "series.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        payload = train_series(str(config_path))
        assert payload["n_test_windows"] >= 1
        assert "mean_rmse" in payload["columns"]["value"]
        assert (tmp_path / "ckpt" / "best_model.pt").exists()


class TestConfigs:
    def test_domain_configs_fix_the_split_and_the_seed(self):
        names = [
            "horse_racing.yaml",
            "ai_frontier_compute.yaml",
            "ai_arxiv_counts.yaml",
            "ai_github_repos.yaml",
            "quantum_qubits.yaml",
            "quantum_qubits_rearranged.yaml",
            "quantum_gate_error.yaml",
            "quantum_arxiv.yaml",
            "quantum_random.yaml",
            "quantum_oulu_neutrons.yaml",
        ]
        for name in names:
            cfg = load_config(f"config/{name}")
            assert cfg["data"]["split_mode"] == "temporal"
            assert cfg["data"]["normalize"] is True
            assert cfg["training"]["seed"] == 0
            assert cfg["training"]["patience"] > 0
            assert "checkpoints/" in cfg["training"]["checkpoint_dir"]

    def test_filtered_qubit_record_is_filled_through_the_last_year(self):
        frame = pd.read_csv("datasets/quantum/qubit_record_annual.csv")
        assert frame["record_qubits_excluding_unrearranged"].notna().all()
        assert frame["log10_qubits_excluding_unrearranged"].notna().all()
        last = frame.iloc[-1]
        assert last["record_qubits"] == 11000
        assert last["record_qubits_excluding_unrearranged"] == 2400
