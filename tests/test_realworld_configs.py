"""The shipped real-world configs all use the same honest training setup."""

from pathlib import Path

import yaml

from src.realworld.website import read_visit_csv, synthetic_daily_visits
from src.utils.paths import REPO_ROOT

EXPECTED = {
    "air_ozone_berlin",
    "air_pm25_berlin",
    "air_pollen_berlin",
    "earthquakes_m45",
    "electricity_us48",
    "fred_cpi_inflation",
    "fred_gas",
    "fred_unemployment",
    "sports_arsenal",
    "sunspots_silso",
    "ucdp_monthly_events",
    "ucdp_monthly_fatalities",
    "ucdp_onset_gaps",
    "website_visits",
}

# Holdout MAE beat persistence, seasonal naive, and the mean, and the fetch
# fits the weekly job. Conflict series, the synthetic site, the CC BY-NC
# sunspot file, and the EIA bulk extract stay out.
WEEKLY_RETRAIN = {
    "air_ozone_berlin",
    "air_pm25_berlin",
    "earthquakes_m45",
}


def test_every_realworld_config_is_temporal_normalized_and_seeded():
    paths = sorted((REPO_ROOT / "config" / "realworld").glob("*.yaml"))
    ids = set()
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        data = cfg["data"]
        training = cfg["training"]
        model = cfg["model"]
        meta = cfg["realworld"]
        assert data["source"] == "csv"
        assert data["split_mode"] == "temporal"
        assert data["val_split"] == 0.2
        assert data["normalize"] is True
        assert int(data["seasonal_period"]) >= 1
        assert int(data["seq_len"]) >= 1
        assert int(data["seq_len"]) <= int(model["max_seq_len"])
        assert int(training["seed"]) == 0
        assert int(training["patience"]) > 0
        assert float(training["max_grad_norm"]) > 0
        assert training["device"] == "cpu"
        weekly = meta["id"] in WEEKLY_RETRAIN
        assert cfg["retrain"]["enabled"] is weekly
        if weekly:
            assert cfg["retrain"]["baselines"] == ["persistence", "mean", "seasonal"]
            assert int(data["season_length"]) == int(data["seasonal_period"])
            assert str(data["fetch"]).startswith("python -m scripts.realworld.fetch_")
        assert model["name"] == "time_series_transformer"
        assert meta["id"]
        assert meta["title"]
        assert meta["units"]
        assert meta["licence"]
        ids.add(meta["id"])
        assert Path(str(data["path"]))
    assert ids == EXPECTED


def test_website_example_csv_is_the_synthetic_series():
    path = REPO_ROOT / "examples" / "realworld" / "website_visits.csv"
    loaded = read_visit_csv(path)
    generated = synthetic_daily_visits()
    assert loaded == [(day, float(value)) for day, value in generated]
