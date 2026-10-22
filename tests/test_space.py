"""Parsers, baselines, configs, and a tiny train for the space series."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.space.baselines import (
    SCHWABE_CYCLE_DAYS,
    SCHWABE_CYCLE_MONTHS,
    SOLAR_ROTATION_DAYS,
    drift_forecast,
    pearson,
    seasonal_naive_lag,
    storm_day_scores,
)
from src.space.charts import write_forecast_svg
from src.space.evaluate import extra_baseline_scores, validation_frame
from src.space.fetch import (
    CAD_API_URL,
    GCAT_COLUMNS_URL,
    SOURCES,
    cad_url,
    exoplanet_url,
    sbdb_url,
    xrs_report_names,
)
from src.space.parse import (
    parse_close_approaches,
    parse_exoplanet_years,
    parse_gcat_launches,
    parse_goes_xrs_reports,
    parse_neo_discoveries,
    parse_omni_daily,
    parse_space_weather_csv,
    regularize_daily,
    trim_open_period,
)
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config

_SW_CSV = """\
DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81
1957-10-01,1700,19,43,40,30,20,37,23,43,37,273,32,27,15,7,22,9,32,22,21,1.1,5,334,269.3,269.8,OBS,266.6,230.9,266.8,235.5
1957-10-02,1700,20,50,40,30,20,37,23,43,37,280,32,27,15,7,22,9,32,22,22,1.1,5,300,250.0,250.0,INT,0,0,0,0
2041-10-01,2837,1,,,,,,,,,,,,,,,,,,,,,10,69.8,70.0,PRM,68.8,69.0,69.2,70.5
"""

_XRS = """\
31777750901  1512E1519 1512 N03E54SN                       C 0     GOES           633 7509 5.7
31777890101  0612E0657D0620 S21W53SF                       M 14    GOES          5292 881228.3
31777170104  0234 0250 0239                                B 18    G15  1.2E-04
31777170131  0234 0250 0239                                X 21    G15  2.2E-04
"""

_LAUNCHES = (
    "#Launch_Tag\tLaunch_Date\tLaunchCode\tCategory\n"
    "1957-A01\t1957 Oct 4 1928\tOS\tSat LEO O\n"
    "1957-A02\t1957 Oct 4 2000\tOF\tSat LEO O\n"
    "1942-A01\t1942 Jun 13 1052\tMF\tTest\n"
    "1960-S01\t1960 Jan 1 0000\tSS\tAeron\n"
    "1962-D01\t1962 Aug 31 0640\tDS\tProbe\n"
    "1964-U01\t1964 Jun  6 0600?\tOS\tSat LEO O\n"
    "2012-U01\t2012 May 31?\tDS\tProbe\n"
    "2024-N01\t2024 Mar 14 0000\tNS\tTest\n"
)


def _omni_line() -> str:
    words = ["999.9"] * 55
    words[0] = "2000"
    words[1] = "10"
    words[2] = "0"
    words[7] = "12"
    words[23] = "5.5"
    words[24] = "420"
    words[40] = "-25"
    return " ".join(words)


def test_sources_are_keyless_and_skip_silso():
    assert SOURCES
    for source in SOURCES:
        assert source["url"].startswith("https://")
        assert "api_key" not in source["url"]
        assert "sidc.be" not in source["url"].lower()
        assert "silso" not in source["url"].lower()
        assert source["licence"]
    joined = " ".join(source["licence"] for source in SOURCES).lower()
    assert "cc by 4.0" in joined
    assert "public domain" in joined
    assert "cc-by" in joined
    assert GCAT_COLUMNS_URL.startswith("https://")


def test_request_urls_match_the_documented_services():
    assert cad_url("2000-01-01", "2000-01-31").startswith(CAD_API_URL)
    assert "dist-max=0.05" in cad_url("2000-01-01", "2000-01-31")
    assert "sb-class=APO" in sbdb_url("APO")
    assert "first_obs" in sbdb_url("AMO")
    tap = exoplanet_url()
    assert "pscomppars" in tap
    assert "disc_year" in tap
    assert "api_key" not in tap
    names = xrs_report_names()
    assert names[0] == "goes-xrs-report_1975.txt"
    assert names[-1] == "goes-xrs-report_2017-ytd.txt"
    assert "goes-xrs-report_2017.txt" not in names


def test_space_weather_parser_scales_kp_and_drops_predictions_and_sunspots():
    kp, f107 = parse_space_weather_csv(_SW_CSV)
    assert list(kp.columns) == ["date", "kp_max", "ap"]
    assert "ISN" not in kp.columns
    assert "isn" not in f107.columns
    assert kp["kp_max"].tolist() == pytest.approx([4.3, 5.0])
    assert kp["ap"].tolist() == pytest.approx([21.0, 22.0])
    assert f107["f107"].tolist() == pytest.approx([269.3, 250.0])
    assert f107["date"].dt.year.max() == 1957


def test_omni_daily_parser_reads_speed_density_and_dst():
    wind, dst = parse_omni_daily(_omni_line() + "\n", min_plasma_hours=4)
    assert wind["date"].dt.strftime("%Y-%m-%d").tolist() == ["2000-01-10"]
    assert wind["speed"].tolist() == pytest.approx([420.0])
    assert wind["density"].tolist() == pytest.approx([5.5])
    assert dst["dst"].tolist() == pytest.approx([-25.0])


def test_omni_parser_rejects_a_low_plasma_hour_count():
    words = _omni_line().split()
    words[7] = "1"
    # Density and speed are valid, but one hour is below the cutoff, and the
    # only Dst value is still kept. A file with no wind days is an error.
    text = " ".join(words) + "\n"
    with pytest.raises(ValueError, match="solar-wind"):
        parse_omni_daily(text, min_plasma_hours=4)


def test_flare_reports_count_classes_and_fill_empty_months():
    monthly = parse_goes_xrs_reports([_XRS])
    by_month = {
        stamp.strftime("%Y-%m"): row
        for stamp, row in zip(monthly["date"], monthly.to_dict("records"), strict=True)
    }
    assert by_month["1975-09"]["c_class"] == 1
    assert by_month["1989-01"]["m_class"] == 1
    assert by_month["2017-01"]["b_class"] == 1
    assert by_month["2017-01"]["x_class"] == 1
    assert by_month["1980-06"]["c_class"] == 0
    assert monthly["date"].max() == pd.Timestamp("2017-01-01")


def test_gcat_keeps_orbital_and_deep_space_attempts():
    monthly = parse_gcat_launches(_LAUNCHES)
    counts = {
        stamp.strftime("%Y-%m"): int(value)
        for stamp, value in zip(monthly["date"], monthly["attempts"], strict=True)
    }
    assert counts["1957-10"] == 2
    assert counts["1962-08"] == 1
    # A ``?`` on the clock or the day still counts in that month.
    assert counts["1964-06"] == 1
    assert counts["2012-05"] == 1
    # Months between the first and last orbital attempt are explicit zeros.
    # The 1960 sounding rocket and the 2024 near-orbital row are not attempts.
    assert counts["1960-01"] == 0
    assert "1942-06" not in counts
    assert "2024-03" not in counts


def test_cad_and_sbdb_and_exoplanet_counts():
    approaches = parse_close_approaches(
        {
            "fields": ["cd"],
            "data": [
                ["2020-Jan-01 00:54"],
                ["2020-Jan-15 02:06"],
                ["2020-Feb-29 12:00"],
            ],
        }
    )
    assert approaches["approaches"].tolist() == [2, 1]

    discoveries = parse_neo_discoveries(
        [
            pd.DataFrame(
                {"pdes": ["433", "1566"], "first_obs": ["1893-10-29", "1949-06-30"]}
            ),
            pd.DataFrame({"pdes": ["433"], "first_obs": ["1893-10-29"]}),
        ]
    )
    labelled = {
        stamp.strftime("%Y-%m"): int(value)
        for stamp, value in zip(
            discoveries["date"], discoveries["discoveries"], strict=True
        )
    }
    assert labelled["1893-10"] == 1
    assert labelled["1949-06"] == 1
    assert int(discoveries["discoveries"].sum()) == 2

    yearly = parse_exoplanet_years("disc_year,n\n1992,2\n1994,1\n")
    assert yearly["discoveries"].tolist() == [2, 0, 1]
    assert yearly["date"].dt.year.tolist() == [1992, 1993, 1994]


def test_regularize_daily_fills_short_gaps_and_cuts_long_ones():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-06", "2020-01-07", "2020-01-08"]
            ),
            "value": [1.0, 2.0, 6.0, 7.0, 8.0],
        }
    )
    # Jan 3-5 is a 3-day hole (gap of 4 days between the 2nd and the 6th),
    # which is the largest hole max_gap_days=3 still interpolates. A bigger
    # hole is a break, so the earlier January stretch is what we want here.
    regular, filled = regularize_daily(frame, ["value"], max_gap_days=3)
    assert filled == 3
    assert regular["date"].iloc[0] == pd.Timestamp("2020-01-01")
    assert len(regular) == 8
    assert regular["value"].iloc[2] == pytest.approx(3.0)

    broken = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-01-11"]),
            "value": [1.0, 10.0, 11.0],
        }
    )
    suffix, filled_suffix = regularize_daily(broken, ["value"], max_gap_days=3)
    assert filled_suffix == 0
    assert suffix["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-10",
        "2020-01-11",
    ]


def test_trim_open_period_drops_the_current_bucket():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-01", "2026-10-01", "2026-10-21"]),
            "value": [1, 2, 3],
        }
    )
    today = pd.Timestamp("2026-10-21")
    daily = trim_open_period(frame, "D", today=today)
    assert daily["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-09-01",
        "2026-10-01",
    ]
    monthly = trim_open_period(frame.iloc[:2], "MS", today=today)
    assert monthly["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-09-01"]


def test_lag_drift_and_storm_scores():
    values = np.arange(10, dtype=np.float64)
    rows = np.array([6, 7, 8, 9])
    assert seasonal_naive_lag(values, rows, 3).tolist() == [3.0, 4.0, 5.0, 6.0]
    with pytest.raises(ValueError, match="history"):
        seasonal_naive_lag(values, rows, SOLAR_ROTATION_DAYS)
    assert drift_forecast(values, rows).tolist() == pytest.approx([6.0, 7.0, 8.0, 9.0])
    with pytest.raises(ValueError, match="history"):
        seasonal_naive_lag(values, rows, SCHWABE_CYCLE_DAYS)
    assert SCHWABE_CYCLE_MONTHS == 132

    scores = storm_day_scores(
        np.array([4.0, 6.0, 5.0, 1.0]),
        np.array([6.0, 6.0, 2.0, 1.0]),
        5.0,
    )
    assert scores["tp"] == 1
    assert scores["fp"] == 1
    assert scores["fn"] == 1
    assert scores["tn"] == 1
    assert scores["precision"] == pytest.approx(0.5)
    assert scores["recall"] == pytest.approx(0.5)
    quiet = storm_day_scores(np.array([1.0, 1.0]), np.array([1.0, 2.0]), 5.0)
    assert quiet["alarms"] == 0
    assert np.isnan(quiet["precision"])
    assert pearson(
        np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
    ) == pytest.approx(1.0)


def test_chart_svg_contains_the_title_and_series(tmp_path: Path):
    destination = write_forecast_svg(
        tmp_path / "f107.svg",
        title="F10.7 validation",
        series=[
            ("actual", "#111111", [1.0, 2.0, 3.0, 2.5]),
            ("model", "#c2410c", [1.2, 1.8, 2.4, 2.2]),
        ],
        hline=5.0,
        hline_label="Kp 5",
    )
    text = destination.read_text(encoding="utf-8")
    assert text.startswith("<?xml")
    assert "<svg" in text
    assert "F10.7 validation" in text
    assert "Kp 5" in text


def test_space_configs_use_the_trainer_contract():
    paths = sorted(Path("config/space").glob("*.yaml"))
    assert len(paths) >= 10
    seen: set[str] = set()
    for path in paths:
        cfg = load_config(path)
        data = cfg["data"]
        training = cfg["training"]
        space = cfg["space"]
        assert data["source"] == "csv"
        assert data["split_mode"] == "temporal"
        assert data["val_split"] == 0.2
        assert data["normalize"] is True
        assert data["report_baselines"] is True
        assert data["seed"] == 0
        assert training["seed"] == 0
        assert training["patience"] > 0
        assert training["max_grad_norm"] > 0
        assert training["device"] == "cpu"
        assert str(data["path"]).startswith("data/space/")
        assert str(training["checkpoint_dir"]).startswith("checkpoints/space/")
        assert cfg["model"]["input_dim"] == len(data["feature_columns"])
        assert "isn" not in data["feature_columns"]
        assert "sunspot" not in " ".join(data["feature_columns"])
        assert space["id"] not in seen
        seen.add(space["id"])
        if space["id"] == "exoplanets":
            assert "date_column" not in data
            assert space["drift"] is True
            assert cfg["model"]["num_encoder_layers"] == 1
        if space["id"] in {"kp", "kp_solar_wind", "kp_omni_window"}:
            assert space["storm_threshold"] == 5.0
        if space["id"] == "kp_omni_window":
            assert data["feature_columns"] == ["kp_max"]
            assert data["path"] == "data/space/kp_solar_wind.csv"
        if space["id"] == "kp_solar_wind":
            assert data["feature_columns"] == ["kp_max", "speed", "density"]
            assert data["path"] == "data/space/kp_solar_wind.csv"
        if space["id"] == "f107":
            assert space["lags"] == [SOLAR_ROTATION_DAYS, SCHWABE_CYCLE_DAYS]
        if space["id"] == "flares_m":
            assert space["lags"] == [SCHWABE_CYCLE_MONTHS]
        if space.get("group") == "near_earth":
            assert cfg["retrain"]["enabled"] is True
            assert cfg["retrain"]["baselines"] == ["persistence", "mean", "seasonal"]
            assert data["season_length"] == 12
            assert space["outlook_months"] == 24
            assert "fetch_near_earth" in str(data["fetch"])
        elif space.get("group") == "solar_system":
            assert cfg["retrain"]["enabled"] is True
            assert cfg["retrain"]["baselines"] == ["persistence", "mean", "seasonal"]
            assert data["season_length"] == 12
            assert space["outlook_months"] == 24
            assert "fetch_solar_system" in str(data["fetch"])
        else:
            assert cfg["retrain"]["enabled"] is False
    assert "f107" in seen
    assert "kp" in seen
    assert "launches" in seen
    assert "exoplanets" in seen


def test_recorded_forecast_covers_every_space_config():
    recorded = json.loads(
        Path("results/space_forecast.json").read_text(encoding="utf-8")
    )
    configs = {
        load_config(path)["space"]["id"]
        for path in Path("config/space").glob("*.yaml")
        if load_config(path)["space"].get("group") not in {"near_earth", "solar_system"}
    }
    series = recorded["series"]
    assert {row["id"] for row in series} == configs
    assert recorded["seed"] == 0
    assert recorded["total_train_seconds"] > 0
    for row in series:
        assert row["model"]["mae"] >= 0
        assert row["trainer_baselines"]["persistence"]["mae"] >= 0


def test_checkpoint_scores_match_a_short_training_run(tmp_path: Path):
    rows = 80
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "value": np.linspace(0.0, 8.0, rows),
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
            "seq_len": 4,
            "val_split": 0.2,
            "split_mode": "temporal",
            "normalize": True,
            "report_baselines": True,
            "seed": 0,
        },
        "training": {
            "experiment_name": "space_test",
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
            "id": "tiny",
            "lags": [3, 100],
            "storm_threshold": 5.0,
            "drift": True,
            "chart": False,
        },
    }
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    trained = TimeSeriesTrainer(str(config_path)).train()
    trace = validation_frame(cfg)
    extra = extra_baseline_scores(cfg, trace)
    assert abs(float(trained["metrics"]["mae"]) - float(extra["model"]["mae"])) < 1e-3
    assert "persistence" in trained["baselines"]
    assert extra["lags"]["lag_3"] is not None
    assert extra["lags"]["lag_100"] is None
    assert extra["drift"]["mae"] >= 0.0
    assert extra["storms"]["model"]["n"] == len(trace["rows"])
    assert json.dumps({"ok": True})
