"""Offline lunar features and the lottery association check."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.lunar import (
    LUNAR_MODEL_COLUMNS,
    chinese_date,
    join_lunar,
    julian_day,
    lottery_study_hooks,
    lunar_feature_frame,
    lunar_on,
)
from src.data.mega_millions import select_current_matrix
from src.data.windows import prepare_csv_series
from src.evaluation.lunar_lottery import (
    POWERBALL,
    analyze_game,
    association_tests,
    high_ball_count,
)
from src.lottery_study.catalog import dataset_catalog
from src.retrain.discover import discover
from src.utils.paths import REPO_ROOT


def test_julian_day_round_trips_j2000():
    moment = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    assert julian_day(moment) == 2451545.0


def test_named_events_and_chinese_new_year():
    eclipse = lunar_on(date(2024, 4, 8))
    assert eclipse.is_new_moon
    assert eclipse.eclipse_kind == "solar"
    assert eclipse.chinese.year == 2024
    assert eclipse.chinese.zodiac == "Dragon"

    full = lunar_on(date(2024, 10, 17))
    assert full.is_full_moon
    assert full.is_supermoon
    assert full.eclipse_kind is None
    assert full.named_phase == "full"
    assert full.illumination > 0.98
    assert abs(full.moon_phase - 0.5) < 0.03

    quiet = lunar_on(date(2024, 5, 8))
    assert quiet.is_new_moon
    assert quiet.eclipse_kind is None

    lunar_eclipse = lunar_on(date(2024, 9, 18))
    assert lunar_eclipse.is_full_moon
    assert lunar_eclipse.eclipse_kind == "lunar"
    assert lunar_eclipse.is_supermoon

    assert (
        chinese_date(date(2024, 2, 10))
        .text()
        .startswith("Dragon year 2024, month 1 day 1")
    )
    assert chinese_date(date(2024, 2, 9)).year == 2023
    assert chinese_date(date(2024, 9, 17)).month == 8
    assert chinese_date(date(2024, 9, 17)).day == 15
    assert chinese_date(date(2023, 3, 22)).leap
    assert chinese_date(date(2023, 3, 22)).month == 2
    assert chinese_date(date(2020, 5, 23)).leap
    assert chinese_date(date(2020, 5, 23)).month == 4
    assert (
        chinese_date(date(2025, 1, 29))
        .text()
        .startswith("Snake year 2025, month 1 day 1")
    )
    assert chinese_date(date(2025, 7, 25)).leap
    assert (
        chinese_date(date(2000, 2, 5))
        .text()
        .startswith("Dragon year 2000, month 1 day 1")
    )


def test_vectorized_frame_matches_single_days():
    days = [date(2020, 1, 1) + timedelta(days=17 * index) for index in range(12)]
    frame = lunar_feature_frame(days)
    assert list(frame.columns) == list(LUNAR_MODEL_COLUMNS)
    for index, day in enumerate(days):
        single = lunar_on(day).as_features()
        for name in LUNAR_MODEL_COLUMNS:
            assert frame.iloc[index][name] == pytest.approx(single[name], abs=1e-6)


def test_join_lunar_appends_columns_and_refuses_overlaps():
    frame = pd.DataFrame(
        {"draw_date": ["2024-04-08", "2024-10-17"], "powerball": [1, 2]}
    )
    joined = join_lunar(frame, "draw_date")
    assert set(LUNAR_MODEL_COLUMNS) <= set(joined.columns)
    assert list(joined["powerball"]) == [1, 2]
    try:
        join_lunar(joined, "draw_date")
    except ValueError as exc:
        assert "already present" in str(exc)
    else:
        raise AssertionError("overlapping lunar columns should be rejected")


def test_future_lunar_widens_the_training_windows(tmp_path):
    rows = []
    day = date(2024, 1, 1)
    for index in range(12):
        rows.append(
            {
                "draw_date": (day + timedelta(days=index)).isoformat(),
                "powerball": index + 1,
            }
        )
    path = tmp_path / "draws.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    prepared = prepare_csv_series(
        {
            "path": str(path),
            "feature_columns": ["powerball"],
            "target_column": "powerball",
            "date_column": "draw_date",
            "seq_len": 4,
            "val_split": 0.25,
            "future_lunar": True,
        }
    )
    assert prepared.inputs.shape[-1] == 1 + len(LUNAR_MODEL_COLUMNS)
    assert prepared.feature_names[0] == "powerball"
    assert prepared.feature_names[-1] == f"future_{LUNAR_MODEL_COLUMNS[-1]}"


def test_lunar_config_is_optional_and_matches_the_column_count():
    loaded = yaml.safe_load((REPO_ROOT / "config" / "lottery_lunar.yaml").read_text())
    assert loaded["retrain"]["enabled"] is False
    assert loaded["data"]["future_lunar"] is True
    assert loaded["model"]["input_dim"] == 1 + len(LUNAR_MODEL_COLUMNS)
    specs, errors = discover(REPO_ROOT / "config")
    assert errors == []
    assert "lottery_lunar" not in {spec.id for spec in specs}
    assert "lunar" in {row.dataset_id for row in dataset_catalog()}


def test_study_hook_is_a_daily_frame():
    hook = lottery_study_hooks()[0]
    assert hook["dataset_id"] == "lunar"
    assert hook["availability"] == "same_day_before_draw"
    frame = hook["frame"]
    assert "2015-10-07" in set(frame["date"])
    assert set(LUNAR_MODEL_COLUMNS) <= set(frame.columns)


def test_mega_millions_keeps_the_current_matrix_only():
    parsed = select_current_matrix(
        [
            {
                "draw_date": "2017-10-27T00:00:00.000",
                "winning_numbers": "1 2 3 4 5",
                "mega_ball": "6",
            },
            {
                "draw_date": "2017-10-31T00:00:00.000",
                "winning_numbers": "10 20 30 40 50",
                "mega_ball": "12",
            },
        ]
    )
    assert parsed.skipped_before_matrix == 1
    assert len(parsed.draws) == 1
    assert parsed.draws[0].white_balls == (10, 20, 30, 40, 50)
    assert parsed.draws[0].mega_ball == 12


def test_phase_separation_is_detected_and_a_null_draw_stays_a_probability():
    phase = np.concatenate([np.zeros(40), np.full(40, 0.5)])
    whites = np.vstack(
        [
            np.tile(np.array([1, 2, 3, 4, 5]), (40, 1)),
            np.tile(np.array([10, 20, 30, 40, 50]), (40, 1)),
        ]
    )
    bonus = np.concatenate([np.ones(40, dtype=int), np.full(40, 10, dtype=int)])
    tests = association_tests(phase, whites, bonus, POWERBALL, n_perm=99, seed=0)
    by_id = {test["id"]: test for test in tests}
    assert by_id["white_numbers_by_phase"]["p_permutation"] <= 0.05
    assert by_id["white_numbers_by_phase"]["effect"] > 0.2

    rng = np.random.default_rng(1)
    null_phase = rng.random(60)
    null_whites = np.vstack(
        [
            np.sort(rng.choice(np.arange(1, 70), size=5, replace=False))
            for _ in range(60)
        ]
    )
    null_bonus = rng.integers(1, 27, size=60)
    null_tests = association_tests(
        null_phase, null_whites, null_bonus, POWERBALL, n_perm=49, seed=2
    )
    for test in null_tests:
        assert 0.0 <= test["p_permutation"] <= 1.0
        assert np.isfinite(test["effect"])


def test_model_comparison_scores_holdout_against_chance():
    rng = np.random.default_rng(3)
    n = 36
    day = date(2023, 1, 1)
    rows = []
    whites = []
    bonus = []
    for index in range(n):
        balls = np.sort(rng.choice(np.arange(1, 70), size=5, replace=False))
        mega = int(rng.integers(1, 27))
        whites.append(balls)
        bonus.append(mega)
        rows.append(
            {
                "draw_date": day + timedelta(days=3 * index),
                "white_1": int(balls[0]),
                "white_2": int(balls[1]),
                "white_3": int(balls[2]),
                "white_4": int(balls[3]),
                "white_5": int(balls[4]),
                "powerball": mega,
            }
        )
    frame = pd.DataFrame(rows)
    report = analyze_game(
        frame,
        POWERBALL,
        n_perm=19,
        n_sim=19,
        seq_len=4,
        seed=0,
        training={
            "epochs": 1,
            "patience": 0,
            "batch_size": 8,
            "d_model": 8,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "max_seq_len": 16,
            "val_split": 0.25,
        },
    )
    models = report["models"]
    assert models["holdout_draws"] > 0
    assert models["chance"]["mean_white_matches"] == pytest.approx(
        POWERBALL.expected_white_matches()
    )
    assert 0.0 <= models["lunar_vs_chance"]["bonus_hit_p"] <= 1.0
    assert 0.0 <= models["lunar_minus_plain"]["p_permutation"] <= 1.0
    assert report["correction"]["n_tests"] == 5
    assert report["correction"]["permutation_floor"] == pytest.approx(1 / 20)


def test_published_report_finds_no_corrected_lunar_edge():
    report = json.loads(
        (REPO_ROOT / "results" / "lunar_lottery_analysis.json").read_text(encoding="utf-8")
    )
    assert report["correction"]["n_tests"] == 10
    assert report["correction"]["n_raw_below_alpha"] == 0
    assert report["correction"]["n_bh_rejects"] == 0
    assert report["correction"]["n_bonferroni_rejects"] == 0
    assert report["model_vs_chance"]["n_bh_rejects"] == 0
    names = {game["game"] for game in report["games"]}
    assert names == {"Powerball", "Mega Millions"}
    assert "not a reason to expect a number" in report["summary"]


def test_high_ball_split_includes_the_powerball_midpoint():
    whites = np.array([[34, 35, 69, 1, 2]])
    assert high_ball_count(whites, POWERBALL).tolist() == [2.0]
