"""Parsers, the 730-day window, and the near-Earth forecast file."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest
import yaml

import src.space.approaches as approaches
from src.space.approaches import (
    AU_PER_LD,
    CLOSE_AU,
    LOOKAHEAD_DAYS,
    OUTLOOK_MONTHS,
    cad_rows_frame,
    classify_kind,
    fireball_events,
    monthly_approaches,
    monthly_sum_count,
    near_earth_report,
    satcat_decays,
    sentry_summary,
    window_bounds,
)
from src.space.baselines import discovery_log_regression, mean_forecast
from src.space.evaluate import extra_baseline_scores, validation_frame
from src.space.near_earth import NEAR_EARTH_SOURCES, cad_query_url
from src.space.outlook import build_outlook
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config


def test_classify_kind_keeps_comets_asteroids_and_interstellar():
    keys = {"1I", "2017 U1"}
    assert classify_kind("289P", "289P", keys) == "comet"
    assert classify_kind("C/2019 Q4", "C/2019 Q4", set()) == "comet"
    assert classify_kind("2026 TU1", "2026 TU1", keys) == "asteroid"
    assert classify_kind("A/2017 U1", "A/2017 U1", set()) == "asteroid"
    assert classify_kind("1I", "1I/'Oumuamua", set()) == "interstellar"
    assert classify_kind("2017 U1", "A/2017 U1", keys) == "interstellar"


def test_cad_rows_frame_parses_jpl_fields():
    payload = {
        "fields": ["des", "cd", "dist", "h", "fullname"],
        "data": [["2026 TU1", "2026-Oct-22 00:10", "0.01", "22.5", "(2026 TU1)"]],
    }
    frame = cad_rows_frame(payload, body="Earth")
    assert frame.loc[0, "body"] == "Earth"
    assert frame.loc[0, "designation"] == "2026 TU1"
    assert frame.loc[0, "cd"] == pd.Timestamp("2026-10-22 00:10")
    assert frame.loc[0, "dist_au"] == pytest.approx(0.01)
    assert frame.loc[0, "h"] == pytest.approx(22.5)


def test_monthly_approaches_controls_censoring_and_future_rows():
    rows = [
        ("Earth", "2020 AA", "2020-01-15 00:10", 0.001, 21.0, True, "asteroid"),
        ("Earth", "2020 BB", "2020-01-20 00:10", 0.04, 26.0, False, "asteroid"),
        ("Earth", "2020 CC", "2020-02-10 00:10", 0.01, 21.0, True, "asteroid"),
        ("Moon", "2020 AA", "2020-01-05 00:10", 0.002, 21.0, True, "asteroid"),
        ("Moon", "2020 DD", "2020-02-01 00:10", 0.03, 26.0, True, "asteroid"),
        ("Moon", "289P", "2020-03-02 00:10", 0.02, np.nan, True, "comet"),
        ("Earth", "2020 EE", "2020-04-02 00:10", 0.001, 18.0, True, "asteroid"),
    ]
    events = pd.DataFrame(
        rows,
        columns=["body", "designation", "cd", "dist_au", "h", "known", "kind"],
    )
    events["cd"] = pd.to_datetime(events["cd"])
    discoveries = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-11-01", "2019-12-01", "2020-01-01"]),
            "discoveries": [5, 5, 2],
        }
    )
    frame = monthly_approaches(events, discoveries, through=pd.Timestamp("2020-04-01"))
    assert list(frame["date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-03-01"),
    ]
    january = frame.iloc[0]
    assert january["earth_within_0_05au"] == 2
    assert january["earth_within_1ld"] == 1
    assert january["earth_large_h22"] == 1
    assert january["earth_known_0_05au"] == 1
    assert january["earth_closest_au"] == pytest.approx(0.001)
    assert january["earth_closest_censored"] == 0
    assert january["known_neos_prev"] == 10
    assert january["earth_0_05au_per_1000"] == pytest.approx(200.0)
    assert january["moon_within_1ld"] == 1
    march = frame.iloc[2]
    assert march["earth_within_0_05au"] == 0
    assert march["earth_closest_au"] == pytest.approx(CLOSE_AU)
    assert march["earth_closest_censored"] == 1
    assert march["moon_comets_0_05au"] == 1
    assert march["moon_large_h22"] == 0
    assert 0.001 < AU_PER_LD < 0.04


def test_discovery_regression_fits_only_before_the_holdout():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 100.0])
    covariate = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    rows = np.array([4, 5])
    original = discovery_log_regression(values, covariate, rows)
    changed_labels = values.copy()
    changed_labels[4:] = -50.0
    refit = discovery_log_regression(changed_labels, covariate, rows)
    assert np.allclose(original, refit)
    changed_catalog = covariate.copy()
    changed_catalog[:4] = [1.0, 1.0, 1.0, 8.0]
    shifted = discovery_log_regression(values, changed_catalog, rows)
    assert not np.allclose(original, shifted)


def test_mean_forecast_uses_the_prefix_mean():
    values = np.array([1.0, 3.0, 5.0, 100.0])
    forecast = mean_forecast(values, np.array([3]))
    assert forecast[0] == pytest.approx(3.0)


def test_fireball_null_energy_still_counts():
    payload = {
        "fields": ["date", "energy", "impact-e"],
        "data": [
            ["2020-01-02 01:00:00", "5.4", "0.18"],
            ["2020-01-03 01:00:00", None, None],
            ["2020-02-01 00:00:00", "1", "2"],
            ["2020-03-01 00:00:00", "9", "9"],
        ],
    }
    events = fireball_events(payload)
    monthly = monthly_sum_count(
        events["date"],
        events["energy_kt"],
        "fireball_energy_kt",
        through=pd.Timestamp("2020-03-01"),
        count_column="fireball_count",
    )
    assert list(monthly["fireball_count"]) == [2, 1]
    assert monthly["fireball_energy_kt"].iloc[0] == pytest.approx(0.18)
    assert monthly["fireball_energy_kt"].iloc[1] == pytest.approx(2.0)


def test_satcat_keeps_dated_decays():
    table = pd.DataFrame(
        {
            "OBJECT_NAME": ["A", "B"],
            "NORAD_CAT_ID": [1, 2],
            "DECAY_DATE": ["2020-01-02", ""],
        }
    )
    decays = satcat_decays(table)
    assert list(decays["name"]) == ["A"]
    assert decays.loc[0, "date"] == pd.Timestamp("2020-01-02")


def test_sentry_sorts_palermo_toward_zero():
    payload = {
        "data": [
            {"des": "a", "fullname": "A", "ps_cum": "-3.0"},
            {"des": "b", "fullname": "B", "ps_cum": "-1.0"},
            {"des": "c", "fullname": "C", "ps_cum": None},
        ]
    }
    summary = sentry_summary(payload)
    assert [row["designation"] for row in summary["objects"]] == ["b", "a", "c"]
    assert summary["count"] == 3


def test_window_bounds_defaults_to_730_inclusive_days():
    today = pd.Timestamp("2026-10-22")
    start, end = window_bounds(today, LOOKAHEAD_DAYS)
    assert LOOKAHEAD_DAYS == 730
    assert OUTLOOK_MONTHS == 24
    assert start == today
    assert (end - start).days == 730
    with pytest.raises(ValueError):
        window_bounds(today, 731)
    with pytest.raises(ValueError):
        window_bounds(today, 0)


def test_cad_query_url_sets_body_and_distance():
    earth = parse_qs(
        urlparse(
            cad_query_url(
                start="1900-01-01",
                end="2028-10-21",
                body="Earth",
                dist_max="0.05",
            )
        ).query
    )
    moon = parse_qs(
        urlparse(
            cad_query_url(
                start="1900-01-01",
                end="2028-10-21",
                body="Moon",
                dist_max="0.05",
            )
        ).query
    )
    assert earth["body"] == ["Earth"]
    assert moon["body"] == ["Moon"]
    assert earth["dist-max"] == ["0.05"]
    assert earth["date-max"] == ["2028-10-21"]
    assert {row["id"] for row in NEAR_EARTH_SOURCES} == {
        "cad",
        "fireballs",
        "sentry",
        "sbdb",
        "satcat",
    }


def _point_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    mapping = {
        "EVENTS_PATH": root / "events.csv",
        "FORECAST_PATH": root / "forecast.json",
        "REENTRY_EVENTS_PATH": root / "reentries.csv",
        "FIREBALL_EVENTS_PATH": root / "fireballs.csv",
        "SENTRY_PATH": root / "sentry.json",
        "INTERSTELLAR_PATH": root / "interstellar.csv",
        "MANIFEST_PATH": root / "manifest.json",
    }
    for name, path in mapping.items():
        monkeypatch.setattr(approaches, name, str(path))
    monkeypatch.setattr(approaches, "utc_today", lambda: pd.Timestamp("2026-10-22"))


def test_report_filters_days_and_keeps_a_wide_interstellar_pass(tmp_path, monkeypatch):
    _point_paths(monkeypatch, tmp_path)
    events = pd.DataFrame(
        [
            {
                "body": "Earth",
                "designation": "2026 TU1",
                "name": "2026 TU1",
                "date": "2026-11-01T00:10",
                "dist_au": 0.01,
                "dist_ld": 0.01 / AU_PER_LD,
                "h": 20.0,
                "kind": "asteroid",
            },
            {
                "body": "Moon",
                "designation": "2026 UA1",
                "name": "2026 UA1",
                "date": "2026-12-01T01:00",
                "dist_au": 0.002,
                "dist_ld": 0.002 / AU_PER_LD,
                "h": 21.0,
                "kind": "asteroid",
            },
            {
                "body": "Earth",
                "designation": "2028 XY",
                "name": "2028 XY",
                "date": "2028-06-01T00:00",
                "dist_au": 0.02,
                "dist_ld": 0.02 / AU_PER_LD,
                "h": 19.0,
                "kind": "asteroid",
            },
            {
                "body": "Earth",
                "designation": "3I",
                "name": "3I",
                "date": "2026-11-15T00:00",
                "dist_au": 0.2,
                "dist_ld": 0.2 / AU_PER_LD,
                "h": np.nan,
                "kind": "interstellar",
            },
            {
                "body": "Earth",
                "designation": "past",
                "name": "past",
                "date": "2026-09-01T00:00",
                "dist_au": 0.001,
                "dist_ld": 0.4,
                "h": 18.0,
                "kind": "asteroid",
            },
        ]
    )
    events.to_csv(tmp_path / "events.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"date_max": "2028-10-21"}), encoding="utf-8"
    )
    (tmp_path / "interstellar.csv").write_text(
        "interstellar_id,designation,name,orbit_class,first_obs\n"
        "3I,2025 N1,C/2025 N1,HYP,2025-07-01\n",
        encoding="utf-8",
    )
    short = near_earth_report(days=90, today=pd.Timestamp("2026-10-22"))
    names = {row["name"] for row in short["approaches"]}
    assert names == {"2026 TU1", "2026 UA1", "3I"}
    assert short["days"] == 90
    assert (
        pd.Timestamp(short["window_end"]) - pd.Timestamp(short["window_start"])
    ).days == 90
    assert any(row["body"] == "Moon" for row in short["approaches"])
    assert short["closest"][0]["name"] == "2026 TU1"
    assert short["interstellar"]["upcoming"][0]["designation"] == "3I"
    wide = near_earth_report(days=730, today=pd.Timestamp("2026-10-22"))
    assert {row["name"] for row in wide["approaches"]} == {
        "2026 TU1",
        "2026 UA1",
        "2028 XY",
        "3I",
    }
    assert wide["outlook_months"] == 24
    assert wide["available"] is True


def test_report_without_files_is_available_false(tmp_path, monkeypatch):
    _point_paths(monkeypatch, tmp_path)
    report = near_earth_report(today=pd.Timestamp("2026-10-22"))
    assert report["available"] is False
    assert report["days"] == 730
    assert report["approaches"] == []
    assert report["forecasts"] == []
    assert report["outlook_months"] == 24


def test_outlook_bands_stay_ordered_on_a_short_run(tmp_path: Path):
    rows = 80
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=rows, freq="MS"),
            "value": 2.0 + 0.05 * np.arange(rows),
            "known_neos_prev": np.linspace(10.0, 80.0, rows),
        }
    )
    csv_path = tmp_path / "series.csv"
    frame.to_csv(csv_path, index=False)
    cfg = {
        "model": {
            "name": "time_series_transformer",
            "input_dim": 1,
            "output_dim": 1,
            "d_model": 16,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "max_seq_len": 32,
        },
        "data": {
            "source": "csv",
            "path": str(csv_path),
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 8,
            "val_split": 0.2,
            "split_mode": "temporal",
            "normalize": True,
            "report_baselines": True,
            "seed": 0,
            "season_length": 12,
        },
        "training": {
            "experiment_name": "near_earth_tiny",
            "batch_size": 16,
            "learning_rate": 0.01,
            "epochs": 2,
            "patience": 0,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_interval": 1,
            "seed": 0,
        },
        "inference": {
            "device": "cpu",
            "checkpoint_path": str(tmp_path / "checkpoints" / "best_model.pt"),
        },
        "storage": {"db_path": str(tmp_path / "experiments.db")},
        "space": {
            "id": "tiny_outlook",
            "lags": [12],
            "drift": True,
            "mean": True,
            "chart": False,
            "discovery_column": "known_neos_prev",
            "outlook_months": 3,
            "outlook_min": 0,
            "outlook_max": 20,
        },
    }
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    TimeSeriesTrainer(str(config_path)).train()
    trace = validation_frame(cfg)
    extra = extra_baseline_scores(cfg, trace)
    assert extra["mean"]["mae"] >= 0
    assert extra["discovery"]["mae"] >= 0
    outlook = build_outlook(cfg)
    assert len(outlook["months"]) == 3
    for month in outlook["months"]:
        assert month["p10"] <= month["p50"] <= month["p90"]
        assert 0 <= month["p50"] <= 20


def test_recorded_near_earth_forecast_covers_every_config():
    recorded = json.loads(
        Path("results/near_earth_forecast.json").read_text(encoding="utf-8")
    )
    configs = []
    for path in sorted(Path("config/space").glob("*.yaml")):
        space = load_config(path)["space"]
        if space.get("group") == "near_earth":
            configs.append(space["id"])
    series = recorded["series"]
    assert {row["id"] for row in series} == set(configs)
    assert recorded["lookahead_days"] == 730
    assert recorded["outlook_months"] == 24
    assert recorded["seed"] == 0
    for row in series:
        assert row["model"]["mae"] >= 0
        assert row["trainer_baselines"]["persistence"]["mae"] >= 0
        assert row["trainer_baselines"]["seasonal_naive"]["mae"] >= 0
        assert row["mean"]["mae"] >= 0
        months = row["outlook"]["months"]
        assert len(months) == 24
        for month in months:
            assert month["p10"] <= month["p50"] <= month["p90"]
