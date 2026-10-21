"""Tests for the Powerball fetch, holdout, and chance baselines."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.lottery import (
    MATRIX_START,
    POWERBALL_HIGH,
    PowerballDraw,
    draw_date_span,
    draw_from_api_row,
    fetch_powerball_rows,
    parse_winning_numbers,
    powerball_series,
    read_draw_table,
    select_current_matrix,
    source_metadata,
    write_draws_csv,
)
from src.data.timeseries import load_csv_timeseries
from src.evaluation.lottery import (
    baseline_report,
    chance_exact_match_rate,
    config_for_seed,
    consecutive_repeat_rate,
    evaluate_checkpoint,
    expected_uniform_score,
    lag1_correlation,
    round_to_range,
    run_seed,
    score_continuous,
    series_diagnostics,
    temporal_window_split,
    uniform_sample,
)
from src.models.registry import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config


def _row(draw_date: str, numbers: str) -> dict[str, str]:
    return {"draw_date": draw_date, "winning_numbers": numbers}


def _draws(n: int, *, start: date = MATRIX_START) -> list[PowerballDraw]:
    draws: list[PowerballDraw] = []
    for index in range(n):
        powerball = (index % POWERBALL_HIGH) + 1
        draws.append(
            PowerballDraw(
                draw_date=start + timedelta(days=index),
                white_balls=(1, 2, 3, 4, 5),
                powerball=powerball,
            )
        )
    return draws


def _write_draws(path: Path, n: int = 40) -> Path:
    return write_draws_csv(path, _draws(n))


class TestParse:
    def test_last_field_is_the_powerball_and_whites_are_sorted(self):
        whites, powerball = parse_winning_numbers("05 04 03 02 01 26")
        assert whites == (1, 2, 3, 4, 5)
        assert powerball == 26

    def test_api_timestamp_and_row(self):
        draw = draw_from_api_row(_row("2015-10-07T00:00:00.000", "18 30 40 48 52 09"))
        assert draw.draw_date == date(2015, 10, 7)
        assert draw.white_balls == (18, 30, 40, 48, 52)
        assert draw.powerball == 9

    @pytest.mark.parametrize(
        "text",
        ["1 2 3 4 5", "1 2 3 4 5 6 7", "1 1 2 3 4 5", "a 2 3 4 5 6", ""],
    )
    def test_rejects_malformed_numbers(self, text: str):
        with pytest.raises(ValueError):
            parse_winning_numbers(text)


class TestMatrixFilter:
    def test_keeps_current_matrix_and_skips_older_draws(self):
        rows = [
            _row("2015-10-03T00:00:00.000", "06 26 33 44 46 04"),
            _row("2015-09-30T00:00:00.000", "21 39 40 55 59 17"),
            _row("2015-10-07T00:00:00.000", "18 30 40 48 52 09"),
            _row("2015-10-10T00:00:00.000", "01 02 03 04 69 26"),
        ]
        parsed = select_current_matrix(rows)
        assert parsed.skipped_before_matrix == 2
        assert [draw.draw_date for draw in parsed.draws] == [
            date(2015, 10, 7),
            date(2015, 10, 10),
        ]

    def test_sorts_by_draw_date(self):
        rows = [
            _row("2015-10-10T00:00:00.000", "01 02 03 04 05 02"),
            _row("2015-10-07T00:00:00.000", "01 02 03 04 05 01"),
        ]
        parsed = select_current_matrix(rows)
        assert [draw.powerball for draw in parsed.draws] == [1, 2]

    def test_out_of_range_after_matrix_start_raises(self):
        rows = [_row("2015-10-07T00:00:00.000", "01 02 03 04 70 26")]
        with pytest.raises(ValueError, match="outside"):
            select_current_matrix(rows)

    def test_duplicate_dates_raise(self):
        rows = [
            _row("2015-10-07T00:00:00.000", "01 02 03 04 05 01"),
            _row("2015-10-07T00:00:00.000", "01 02 03 04 06 02"),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            select_current_matrix(rows)

    def test_source_metadata_records_licence(self):
        meta = source_metadata()
        assert meta["dataset_id"] == "d6yy-54nr"
        assert meta["licence"] == "OPEN-NY Terms of Use"
        assert meta["licence_url"].endswith(".pdf")
        assert "data.ny.gov" in meta["page"]


class TestFetchAndCsv:
    def test_pages_until_a_short_page(self):
        pages = [
            [_row("2010-02-03T00:00:00.000", "17 22 36 37 52 24")],
            [_row("2015-10-07T00:00:00.000", "18 30 40 48 52 09")],
            [],
        ]
        calls: list[str] = []

        class _Response:
            def __init__(self, payload: list[dict[str, str]]) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return json.dumps(self._payload).encode()

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

        def opener(request: object, timeout: float) -> _Response:
            assert timeout == 5
            url = request.full_url  # type: ignore[attr-defined]
            calls.append(url)
            return _Response(pages[len(calls) - 1])

        rows = fetch_powerball_rows(
            "https://example.test/resource.json",
            page_size=1,
            timeout=5,
            opener=opener,
        )
        assert len(rows) == 2
        assert "offset=0" in calls[0]
        assert "offset=1" in calls[1]
        assert len(calls) == 3

    def test_csv_roundtrip_preserves_order_and_span(self, tmp_path: Path):
        path = _write_draws(tmp_path / "powerball.csv", n=12)
        frame = read_draw_table(path)
        assert list(frame["powerball"]) == [(index % 26) + 1 for index in range(12)]
        assert bool(frame["draw_date"].is_monotonic_increasing)
        assert draw_date_span(path) == ("2015-10-07", "2015-10-18")
        series = powerball_series(path)
        assert series.dtype == np.float64
        assert series.tolist() == frame["powerball"].astype(float).tolist()


class TestHoldoutAndBaselines:
    def test_split_matches_trainer_truncation(self):
        # 1422 draws and seq_len 32 is the shape of the published demo.
        split = temporal_window_split(1422, 32, 0.2)
        assert split["n_windows"] == 1390
        assert split["n_val"] == int(1390 * 0.2)
        assert split["n_fit"] == 1390 - split["n_val"]
        assert split["first_val_target"] == 32 + split["n_fit"]

    def test_split_rejects_empty_sides(self):
        with pytest.raises(ValueError):
            temporal_window_split(10, 8, 0.05)
        with pytest.raises(ValueError):
            temporal_window_split(10, 4, 0.0)

    def test_round_half_even_and_clip(self):
        # numpy rint uses half-to-even: 0.5 -> 0, 1.5 -> 2, 13.5 -> 14.
        assert round_to_range(np.array([0.5, 1.5, 13.5]), 0, 26).tolist() == [
            0,
            2,
            14,
        ]
        # Values outside the legal range clip after rounding.
        assert round_to_range(np.array([-1.0, 0.4, 40.0]), 1, 26).tolist() == [
            1,
            1,
            26,
        ]

    def test_perfect_score_and_uniform_expectation(self):
        targets = np.array([1.0, 2.0, 3.0])
        perfect = score_continuous(targets, targets, 1, 3)
        assert perfect["mse"] == 0.0
        assert perfect["exact_match_rate"] == 1.0
        expected = expected_uniform_score(np.array([1.0]), 1, 3)
        assert expected["mae"] == pytest.approx(1.0)
        assert expected["mse"] == pytest.approx(5.0 / 3.0)
        assert expected["exact_match_rate"] == pytest.approx(1.0 / 3.0)
        assert chance_exact_match_rate(1, 26) == pytest.approx(1.0 / 26.0)

    def test_uniform_sample_is_seeded_and_inside_range(self):
        first = uniform_sample(100, 1, 26, seed=0)
        second = uniform_sample(100, 1, 26, seed=0)
        assert np.array_equal(first, second)
        assert first.min() >= 1
        assert first.max() <= 26

    def test_historical_mean_ignores_the_holdout(self):
        series = np.arange(1, 11, dtype=np.float64)
        report = baseline_report(series, seq_len=3, val_split=0.4, low=1, high=100)
        # 10 rows, 7 windows, n_val = int(7 * 0.4) = 2, first target index = 8.
        assert report["split"]["first_val_target"] == 8
        assert report["historical_mean_value"] == pytest.approx(series[:8].mean())
        changed = series.copy()
        changed[-1] = 100
        changed_report = baseline_report(
            changed, seq_len=3, val_split=0.4, low=1, high=100
        )
        assert (
            changed_report["historical_mean_value"] == report["historical_mean_value"]
        )
        persistence = changed_report["baselines"]["persistence"]
        # Targets are 9 and 100. Previous draws are 8 and 9. No exact hits.
        assert persistence["exact_matches"] == 0
        assert persistence["n"] == 2

    def test_loader_windows_match_the_holdout(self, tmp_path: Path):
        path = _write_draws(tmp_path / "powerball.csv", n=40)
        series = powerball_series(path)
        seq_len = 4
        split = temporal_window_split(len(series), seq_len, 0.2)
        inputs, targets = load_csv_timeseries(
            path,
            feature_columns=["powerball"],
            target_column="powerball",
            seq_len=seq_len,
        )
        n_fit = split["n_fit"]
        start = split["first_val_target"]
        np.testing.assert_allclose(targets[n_fit:].reshape(-1).numpy(), series[start:])
        np.testing.assert_allclose(
            inputs[n_fit:, -1, 0].numpy(),
            series[start - 1 : start - 1 + split["n_val"]],
        )

    def test_diagnostics_match_a_uniform_grid(self):
        series = np.arange(1, 27)
        stats = series_diagnostics(series, 1, 26)
        assert stats["mean"] == pytest.approx(13.5)
        assert stats["std"] == pytest.approx(stats["theoretical_std"])
        assert stats["theoretical_std"] == pytest.approx(7.5)
        assert lag1_correlation(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)
        assert lag1_correlation(np.ones(5)) is None
        # Pairs are (1, 1), (1, 2), (2, 2): two of the three steps repeat.
        assert consecutive_repeat_rate(np.array([1, 1, 2, 2])) == pytest.approx(2 / 3)

    def test_checkpoint_eval_uses_the_same_holdout(self, tmp_path: Path):
        path = _write_draws(tmp_path / "powerball.csv", n=48)
        cfg = load_config("config/lottery.yaml")
        cfg["data"]["path"] = str(path)
        cfg["data"]["seq_len"] = 8
        cfg["data"]["val_split"] = 0.25
        cfg["model"].update(
            {
                "d_model": 8,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 16,
                "max_seq_len": 16,
            }
        )
        checkpoint = tmp_path / "best_model.pt"
        model = build_model(cfg)
        save_checkpoint(
            model,
            torch.optim.Adam(model.parameters(), lr=0.001),
            epoch=0,
            loss=1.0,
            config=cfg,
            path=checkpoint,
            scaler=None,
        )
        scored = evaluate_checkpoint(cfg, checkpoint, uniform_seed=0)
        split = temporal_window_split(48, 8, 0.25)
        assert scored["model"]["n"] == split["n_val"]
        assert scored["chance_exact_match_rate"] == pytest.approx(1 / 26)
        assert scored["baselines"]["historical_mean"]["n"] == split["n_val"]
        assert 0.0 <= scored["model"]["exact_match_rate"] <= 1.0
        manual = baseline_report(
            powerball_series(path), seq_len=8, val_split=0.25, uniform_seed=0
        )
        assert scored["historical_mean_value"] == pytest.approx(
            manual["historical_mean_value"]
        )
        assert scored["baselines"]["persistence"] == manual["baselines"]["persistence"]


class TestTrainingSmoke:
    def test_run_seed_on_a_tiny_series(self, tmp_path: Path):
        path = _write_draws(tmp_path / "powerball.csv", n=48)
        cfg = load_config("config/lottery.yaml")
        cfg["data"]["path"] = str(path)
        cfg["data"]["seq_len"] = 8
        cfg["data"]["val_split"] = 0.25
        cfg["model"].update(
            {
                "d_model": 8,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 16,
                "max_seq_len": 16,
            }
        )
        cfg["training"].update(
            {
                "epochs": 1,
                "batch_size": 16,
                "device": "cpu",
                "log_interval": 1,
                "patience": 0,
            }
        )
        cfg["storage"]["db_path"] = str(tmp_path / "experiments.db")
        seeded = config_for_seed(cfg, seed=0, checkpoint_dir=str(tmp_path / "ckpt"))
        assert seeded["inference"]["checkpoint_path"].endswith("best_model.pt")
        assert "checkpoints/best_model.pt" not in seeded["inference"]["checkpoint_path"]
        scored = run_seed(seeded, uniform_seed=0)
        assert scored["training"]["seed"] == 0
        assert scored["training"]["epochs_ran"] == 1
        assert (tmp_path / "ckpt" / "best_model.pt").is_file()
        assert scored["model"]["n"] == temporal_window_split(48, 8, 0.25)["n_val"]


class TestCommittedReport:
    def test_snapshot_stays_on_the_chance_rate(self):
        report = json.loads(Path("results/lottery_sanity.json").read_text())
        chance = report["chance_exact_match_rate"]
        se = report["exact_match_null_se"]
        assert chance == pytest.approx(1 / 26)
        assert report["series"]["n_draws"] == 1422
        assert report["series"]["end"] == "2026-10-19"
        assert report["split"]["n_val"] == 278
        assert report["split"]["split_mode"] == "temporal"
        hits = [model["metrics"]["exact_matches"] for model in report["models"]]
        assert hits == [13, 9, 13]
        for model in report["models"]:
            rate = float(model["metrics"]["exact_match_rate"])
            assert abs(rate - chance) < 2 * se
            assert float(model["skill_vs_historical_mean_mse"]) < 0.02
        assert report["baselines"]["historical_mean"]["exact_matches"] == 17
        assert report["baselines"]["persistence"]["exact_matches"] == 6
        assert report["baselines"]["uniform_expected"][
            "exact_match_rate"
        ] == pytest.approx(chance)


class TestLotteryConfig:
    def test_committed_config_is_isolated_from_the_api_checkpoint(self):
        cfg = load_config("config/lottery.yaml")
        assert cfg["data"]["source"] == "csv"
        assert cfg["data"]["split_mode"] == "temporal"
        assert cfg["data"]["target_column"] == "powerball"
        assert cfg["data"]["feature_columns"] == ["powerball"]
        assert str(cfg["training"]["checkpoint_dir"]).startswith("checkpoints/lottery")
        assert cfg["storage"]["db_path"] != "data/experiments.db"
        assert cfg["inference"]["checkpoint_path"] != "checkpoints/best_model.pt"
        # The file is valid YAML with the sections the trainer requires.
        text = Path("config/lottery.yaml").read_text()
        assert isinstance(yaml.safe_load(text), dict)

    def test_draw_table_rejects_a_powerball_outside_the_matrix(self, tmp_path: Path):
        path = _write_draws(tmp_path / "powerball.csv", n=4)
        frame = pd.read_csv(path)
        frame.loc[0, "powerball"] = 27
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="powerball"):
            read_draw_table(path)
