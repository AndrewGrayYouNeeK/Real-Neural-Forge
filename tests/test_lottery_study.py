"""Tests for the Powerball dataset comparison."""

from __future__ import annotations

import math
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.lottery import PowerballDraw, write_draws_csv
from src.lottery_study.align import (
    PRIOR_CALENDAR_DAY,
    SAME_DAY_BEFORE_DRAW,
    align_observations,
    draw_cutoff,
)
from src.lottery_study.catalog import dataset_catalog
from src.lottery_study.closecall import (
    attach_to_scorecard_module,
    binomial_two_sided_p,
    chance_powerball_rate,
    close_call_scorecard,
    decode_ticket,
    frequency_ticket,
    lottery_target_frame,
    repair_whites,
    white_match_at_least,
    white_match_probability,
)
from src.lottery_study.modeling import build_windows
from src.lottery_study.registry import resolve_datasets
from src.lottery_study.report import conclusion_text, headline_table, render_chart
from src.lottery_study.run import _mark_dataset_signals, run_study
from src.lottery_study.stats import (
    benjamini_hochberg,
    bonferroni_adjust,
    expected_luck_hits,
    mutual_info_permutation,
    pearson_p_value,
    pearson_permutation_p,
    pearson_r,
    regularized_incomplete_beta,
    student_t_two_sided_p,
)

REQUIRED_DATASETS = {
    "weather",
    "stocks",
    "fx",
    "crypto",
    "commodities",
    "electricity",
    "air_quality",
    "earthquakes",
    "sunspots",
    "economy",
    "sports",
    "traffic",
    "ucdp",
    "space_f107",
    "space_flares",
    "space_kp_ap",
    "space_solar_wind",
    "space_neos",
    "space_launches",
    "space_exoplanets",
    "horse_racing",
    "ai_progress",
    "quantum_qubits",
    "arxiv_quant_ph",
    "qrng",
}


def _draws(n: int, start: date = date(2016, 1, 1)) -> list[PowerballDraw]:
    rng = np.random.default_rng(1)
    draws: list[PowerballDraw] = []
    day = start
    for _ in range(n):
        whites = tuple(
            sorted(
                int(value)
                for value in rng.choice(np.arange(1, 70), size=5, replace=False)
            )
        )
        draws.append(
            PowerballDraw(
                draw_date=day,
                white_balls=(whites[0], whites[1], whites[2], whites[3], whites[4]),
                powerball=int(rng.integers(1, 27)),
            )
        )
        day += timedelta(days=3)
    return draws


class TestCorrections:
    def test_incomplete_beta_and_student_t(self):
        assert regularized_incomplete_beta(1, 1, 0.5) == pytest.approx(0.5)
        assert student_t_two_sided_p(0.0, 10) == pytest.approx(1.0)
        # Two-sided 5% critical value of t with 10 degrees of freedom.
        assert student_t_two_sided_p(2.228138851986, 10) == pytest.approx(
            0.05, abs=1e-4
        )
        assert student_t_two_sided_p(1e6, 20) == pytest.approx(0.0, abs=1e-8)

    def test_pearson_p_is_one_when_uncorrelated_and_zero_when_perfect(self):
        assert pearson_p_value(0.0, 100) == pytest.approx(1.0)
        assert pearson_p_value(1.0, 30) == 0.0
        x = np.arange(40, dtype=float)
        assert pearson_r(x, 3 * x + 1) == pytest.approx(1.0)

    def test_benjamini_hochberg_and_bonferroni(self):
        raw = np.array([0.001, 0.04, 0.2])
        assert bonferroni_adjust(raw).tolist() == pytest.approx([0.003, 0.12, 0.6])
        adjusted = benjamini_hochberg(raw)
        assert adjusted.tolist() == pytest.approx([0.003, 0.06, 0.2])
        assert expected_luck_hits(100, 0.05) == 5

    def test_permutation_confirms_a_real_correlation_and_mutual_information(self):
        rng = np.random.default_rng(0)
        x = np.linspace(-2, 2, 80)
        y = x + rng.normal(0, 0.05, size=x.size)
        p_value, exceedances = pearson_permutation_p(x, y, n_perm=49, rng=rng)
        assert exceedances == 0
        assert p_value == pytest.approx(1 / 50)
        mi, mi_p, mi_exceedances = mutual_info_permutation(
            x, x**2, n_bins=6, n_perm=29, rng=np.random.default_rng(1)
        )
        assert mi > 0.2
        assert mi_exceedances == 0
        assert mi_p == pytest.approx(1 / 30)


class TestAlignment:
    def test_cutoff_uses_eastern_time_including_daylight_saving(self):
        summer = draw_cutoff(date(2026, 7, 15))
        winter = draw_cutoff(date(2026, 1, 15))
        assert summer.hour == 22 and summer.minute == 59
        assert summer.utcoffset() == timedelta(hours=-4)
        assert winter.utcoffset() == timedelta(hours=-5)
        draw_cutoff(date(2026, 3, 8))
        draw_cutoff(date(2026, 11, 1))

    def test_prior_day_excludes_the_draw_date_and_the_future(self):
        draws = [date(2026, 7, 15), date(2026, 7, 16)]
        observed = pd.Series(["2026-07-14", "2026-07-15", "2026-07-16"])
        values = np.array([1.0, 2.0, 3.0])
        aligned, rule = align_observations(
            draws, observed, values, availability=PRIOR_CALENDAR_DAY
        )
        assert rule == "calendar:prior_calendar_day"
        assert aligned[:, 0].tolist() == [1.0, 2.0]

    def test_same_day_keeps_the_draw_date(self):
        draws = [date(2026, 7, 15), date(2026, 7, 16)]
        observed = pd.Series(["2026-07-14", "2026-07-15", "2026-07-16"])
        values = np.array([1.0, 2.0, 3.0])
        aligned, rule = align_observations(
            draws, observed, values, availability=SAME_DAY_BEFORE_DRAW
        )
        assert rule == "calendar:same_day_before_draw"
        assert aligned[:, 0].tolist() == [2.0, 3.0]

    def test_clock_times_stop_at_1059_pm_eastern(self):
        observed = pd.Series(
            [
                "2026-07-15 16:00:00-04:00",
                "2026-07-15 23:30:00-04:00",
                "2026-07-16 23:30:00-04:00",
            ]
        )
        values = np.array([1.0, 2.0, 3.0])
        aligned, rule = align_observations(
            [date(2026, 7, 15), date(2026, 7, 16)],
            observed,
            values,
            availability=PRIOR_CALENDAR_DAY,
        )
        assert rule == "timestamp:at_or_before_draw"
        assert aligned[:, 0].tolist() == [1.0, 2.0]

    def test_midnight_timestamp_stays_on_the_calendar_rule(self):
        observed = pd.Series(["2026-07-14T00:00:00", "2026-07-15T00:00:00"])
        values = np.array([4.0, 9.0])
        aligned, rule = align_observations(
            [date(2026, 7, 15)],
            observed,
            values,
            availability=PRIOR_CALENDAR_DAY,
        )
        assert rule.startswith("calendar:")
        assert aligned[0, 0] == 4.0

    def test_missing_values_are_not_treated_as_known(self):
        observed = pd.Series(["2026-07-13", "2026-07-14", "2026-07-15"])
        values = np.array([4.0, np.nan, 9.0])
        aligned, _rule = align_observations(
            [date(2026, 7, 15)],
            observed,
            values,
            availability=PRIOR_CALENDAR_DAY,
        )
        assert aligned[0, 0] == 4.0


class TestTickets:
    def test_partial_match_probabilities_sum_to_one(self):
        pmf = [white_match_probability(k) for k in range(6)]
        assert sum(pmf) == pytest.approx(1.0)
        assert pmf[5] == pytest.approx(1 / math.comb(69, 5))
        assert white_match_at_least(5) == pytest.approx(pmf[5])
        assert (
            white_match_at_least(2) > white_match_at_least(3) > white_match_at_least(4)
        )
        assert chance_powerball_rate() == pytest.approx(1 / 26)
        assert binomial_two_sided_p(0, 10, 0.5) < 0.01

    def test_repair_and_close_calls(self):
        assert repair_whites(np.array([1, 1, 1, 1, 1])).tolist() == [1, 2, 3, 4, 5]
        ticket = decode_ticket(np.array([69, 69, 70, 0, 5, 26.4]))
        assert len(set(ticket[:5].tolist())) == 5
        assert ticket.min() >= 1
        assert ticket[5] == 26
        predicted = np.array([[1, 2, 3, 4, 5, 6]])
        actual = np.array([[1, 2, 3, 10, 11, 6]])
        card = close_call_scorecard(predicted, actual)
        assert card["mean_whites_matched"] == 3
        assert card["rate_at_least_2"] == 1
        assert card["rate_at_least_3"] == 1
        assert card["rate_at_least_4"] == 0
        assert card["mean_within_1"] == 4
        assert card["mean_within_2"] == 5
        assert card["mean_near_miss_1"] == 1
        assert card["mean_near_miss_2"] == 1
        assert card["white_sum_mae"] == 12
        assert card["powerball_match_rate"] == 1
        assert card["full_ticket_exact_rate"] == 0

    def test_near_repeat_and_neighbor_targets(self):
        whites = np.array(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 10, 11],
                [6, 8, 10, 12, 14],
                [20, 22, 24, 26, 28],
            ]
        )
        powerball = np.array([1, 2, 3, 4])
        targets = lottery_target_frame(whites, powerball)
        assert np.isnan(targets["near_repeat_3"][0])
        assert targets["near_repeat_3"][1] == 1
        assert targets["near_repeat_4"][1] == 0
        assert targets["neighbor_of_previous"][1] == 0
        assert targets["neighbor_of_previous"][2] == 1
        assert targets["consecutive_in_draw"][0] == 1
        assert targets["consecutive_in_draw"][3] == 0
        assert targets["white_sum"][0] == 15
        assert targets["odd_count"][1] == 3

    def test_frequency_ticket_breaks_ties_by_the_smaller_number(self):
        history = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]])
        powerball = np.array([8, 8])
        hot = frequency_ticket(history, powerball, cold=False)
        assert hot[:5].tolist() == [1, 2, 3, 4, 5]
        assert hot[5] == 8


class TestWindowsAndRegistry:
    def test_windows_use_the_covariate_known_before_the_target_draw(self):
        balls = np.arange(30, dtype=float).reshape(5, 6)
        covariates = np.zeros((5, 1))
        covariates[-1, 0] = 999
        inputs, targets, target_index = build_windows(balls, covariates, seq_len=2)
        assert target_index.tolist() == [2, 3, 4]
        assert inputs[-1, -1, -1] == 999
        assert np.all(inputs[:-1, :, -1] == 0)
        assert np.allclose(inputs[-1, -1, :6], balls[-2])
        for window in inputs:
            for step in window[:, :6]:
                assert not np.allclose(step, balls[-1])
        assert np.allclose(targets[-1], balls[-1])

    def test_catalog_hooks_cover_every_requested_dataset(self):
        assert REQUIRED_DATASETS <= {row.dataset_id for row in dataset_catalog()}

    def test_a_catalog_csv_is_not_discovered_under_a_second_name(self):
        loads = resolve_datasets(
            {
                "fetch_missing": False,
                "discover_unlisted": True,
                "only_datasets": ["weather"],
            }
        )
        available = [load.dataset_id for load in loads if load.available]
        assert available.count("weather") <= 1
        assert "weather_central_park" not in available

    def test_module_hook_and_missing_fetcher(self, monkeypatch: pytest.MonkeyPatch):
        module = types.ModuleType("src.data.stocks")
        module.LOTTERY_STUDY_HOOKS = (  # type: ignore[attr-defined]
            {
                "dataset_id": "stocks",
                "description": "Test prices.",
                "availability": "same_day_before_draw",
                "frame": pd.DataFrame(
                    {"date": ["2020-01-01", "2020-01-02"], "close": [1.0, 2.0]}
                ),
            },
        )
        monkeypatch.setitem(sys.modules, "src.data.stocks", module)
        loads = resolve_datasets(
            {
                "fetch_missing": False,
                "discover_unlisted": False,
                "only_datasets": ["stocks", "qrng"],
            }
        )
        by_id = {load.dataset_id: load for load in loads}
        assert by_id["stocks"].available
        assert by_id["stocks"].columns == ("close",)
        assert not by_id["qrng"].available
        assert "not on main yet" in by_id["qrng"].detail

    def test_scorecard_hook_registers_close_calls(self):
        registered: dict[str, object] = {}

        def register_metric(name: str, fn: object) -> None:
            registered[name] = fn

        module = types.SimpleNamespace(register_metric=register_metric)
        assert attach_to_scorecard_module(module)
        metric = registered["rate_at_least_3"]
        assert callable(metric)
        predicted = np.array([[1, 2, 3, 4, 5, 6]])
        actual = np.array([[1, 2, 3, 10, 11, 6]])
        assert metric(predicted, actual) == 1

    def test_dataset_signal_requires_both_references(self):
        scored = [
            {
                "name": "Weather",
                "kind": "dataset",
                "dataset_id": "weather",
                "dataset_signal": False,
            },
            {
                "name": "All datasets",
                "kind": "combined",
                "dataset_id": "combined",
                "alias_of": "weather",
                "dataset_signal": False,
            },
        ]
        base = {
            "family": "predictive",
            "kind": "paired",
            "series_id": "Weather",
            "survives_bh": True,
            "p_bh": 0.01,
            "effect": -1.0,
            "higher_is_better": False,
        }
        tests = [
            {**base, "target": "white_sum_abs_error_vs_historical_mean"},
            {**base, "target": "white_sum_abs_error_vs_plain"},
        ]
        _mark_dataset_signals(scored, tests, 0.05)
        assert scored[0]["dataset_signal"]
        assert scored[1]["dataset_signal"]
        tests[1] = {**tests[1], "effect": 2.0}
        scored[0]["dataset_signal"] = False
        scored[1]["dataset_signal"] = False
        _mark_dataset_signals(scored, tests, 0.05)
        assert not scored[0]["dataset_signal"]


class TestReport:
    def test_conclusion_says_when_nothing_survives(self):
        report = _fixture_report(signal=False, association_hit=False)
        text = conclusion_text(report)
        assert "about 5.0" in text
        assert "Nothing here beats luck" in text
        assert "stocks" in text
        table = headline_table(report)
        assert "Historical mean" in table
        assert "Plain transformer" in table
        assert render_chart(report).startswith("<svg")

    def test_conclusion_names_a_surviving_dataset_row(self):
        report = _fixture_report(signal=True, association_hit=True)
        text = conclusion_text(report)
        assert "Weather" in text
        assert "pre-registered" in text


def test_end_to_end_on_synthetic_draws(tmp_path: Path):
    draws_path = write_draws_csv(tmp_path / "powerball.csv", _draws(48))
    dates = pd.date_range("2015-12-01", periods=200, freq="D")
    weather = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temperature_c": np.linspace(-5, 30, len(dates)),
            "doy_sin": np.sin(np.arange(len(dates)) / 10),
            "doy_cos": np.cos(np.arange(len(dates)) / 10),
        }
    )
    weather_path = tmp_path / "weather.csv"
    weather.to_csv(weather_path, index=False)
    report = run_study(
        {
            "alpha": 0.05,
            "min_aligned": 20,
            "n_permutations": 19,
            "permutation_seed": 0,
            "mi_bins": 4,
            "chance_simulations": 30,
            "chance_seed": 0,
            "fetch_missing": False,
            "fetch_lottery_if_missing": False,
            "discover_unlisted": False,
            "lottery_csv": str(draws_path),
            "dataset_paths": {"weather": str(weather_path)},
            "verbose": False,
            "training": {
                "enabled": True,
                "seq_len": 4,
                "val_split": 0.2,
                "epochs": 1,
                "patience": 0,
                "batch_size": 16,
                "learning_rate": 0.01,
                "d_model": 8,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
                "max_seq_len": 16,
                "seed": 0,
                "device": "cpu",
            },
            "output": {
                "json": str(tmp_path / "report.json"),
                "markdown": str(tmp_path / "report.md"),
                "chart": str(tmp_path / "chart.svg"),
            },
        }
    )
    names = {row["name"] for row in report["predictors"]}
    assert {
        "Plain transformer",
        "Weather",
        "All datasets",
        "Historical mean",
        "Random tickets",
    } <= names
    weather_row = next(
        row for row in report["datasets"] if row["dataset_id"] == "weather"
    )
    assert weather_row["available"]
    assert weather_row["detail"].startswith("loaded")
    stocks = next(row for row in report["datasets"] if row["dataset_id"] == "stocks")
    assert not stocks["available"]
    combined = next(
        row for row in report["predictors"] if row["name"] == "All datasets"
    )
    assert combined["alias_of"] == "weather"
    assert report["scorecard_module"] is None
    assert "luck" in report["conclusion"].lower()
    assert (tmp_path / "chart.svg").read_text().startswith("<svg")
    assert "Benjamini" in (tmp_path / "report.md").read_text()
    for row in report["predictors"]:
        assert math.isfinite(float(row["white_sum_mae"]))
        assert 0.0 <= float(row["rate_at_least_3"]) <= 1.0


def _fixture_report(*, signal: bool, association_hit: bool) -> DictReport:
    association = {
        "family": "association",
        "kind": "pearson",
        "series_id": "weather.temperature_c",
        "target": "white_sum",
        "pearson_r": 0.02,
        "mutual_info_bits": 0.01,
        "p": 0.4,
        "p_bh": 0.9,
        "p_bonferroni": 1.0,
        "permutation_p": 0.5,
        "survives_bh": association_hit,
    }
    if association_hit:
        association["p_bh"] = 0.01
        association["pearson_r"] = 0.4
    predictors = [
        _predictor("Plain transformer", "plain"),
        _predictor("Weather", "dataset", signal=signal),
        _predictor("Historical mean", "historical_mean"),
        _predictor("Random tickets", "random"),
    ]
    return {
        "datasets": [
            {
                "dataset_id": "weather",
                "group": "environment",
                "description": "Central Park.",
                "available": True,
                "detail": "loaded",
                "columns": ["temperature_c"],
            },
            {
                "dataset_id": "stocks",
                "group": "markets",
                "description": "TODO",
                "available": False,
                "detail": "not on main yet",
                "columns": [],
            },
        ],
        "corrections": {
            "association": {
                "n_tests": 100,
                "alpha": 0.05,
                "expected_luck_hits": 5.0,
                "n_raw_below_alpha": 4,
                "n_bh_rejects": 0,
                "n_bonferroni_rejects": 0,
                "permutation_floor": 0.002,
            },
            "predictive": {
                "n_tests": 20,
                "alpha": 0.05,
                "expected_luck_hits": 1.0,
                "n_raw_below_alpha": 1,
                "n_bh_rejects": 0,
                "n_bonferroni_rejects": 0,
            },
            "combined": {
                "n_tests": 120,
                "alpha": 0.05,
                "expected_luck_hits": 6.0,
                "n_raw_below_alpha": 5,
                "n_bh_rejects": 0,
                "n_bonferroni_rejects": 0,
            },
        },
        "association_tests": [association],
        "predictors": predictors,
        "chance": {
            "powerball_exact_rate": 1 / 26,
            "rate_at_least_2": 0.03,
            "rate_at_least_3": 0.001,
            "rate_at_least_4": 0.00001,
            "mean_within_1": 1.2,
            "mean_within_2": 2.0,
            "white_sum_mae": 40.0,
        },
        "chart_path": "results/lottery_dataset_study.svg",
    }


class DictReport(dict):
    """Typed alias so the fixture reads as a report mapping."""


def _predictor(name: str, kind: str, *, signal: bool = False) -> dict[str, object]:
    return {
        "name": name,
        "kind": kind,
        "n": 10,
        "powerball_match_rate": 0.04,
        "powerball_exact_matches": 0,
        "powerball_mae": 7.0,
        "white_sum_mae": 30.0,
        "mean_whites_matched": 0.4,
        "rate_at_least_2": 0.02,
        "rate_at_least_3": 0.0,
        "rate_at_least_4": 0.0,
        "mean_within_1": 1.0,
        "mean_within_2": 1.8,
        "dataset_signal": signal,
    }
