"""Conflict intervals: gaps, peace spells, durations, and the distribution checks."""

from datetime import date

import numpy as np

from src.realworld.spells import (
    Episode,
    annual_onset_counts,
    conflict_onsets,
    durations,
    episodes_from_acd,
    exponential_dispersion,
    exponential_trend_mean,
    fit_exponential_trend,
    fit_poisson_trend,
    fit_weibull,
    inter_onset_gaps,
    kaplan_meier,
    linear_trend,
    mann_kendall,
    peace_spells_by_country,
    peace_spells_by_group,
    poisson_mean_gap,
)


def _episode(**kwargs: object) -> Episode:
    base = {
        "group_id": "1",
        "region": "Africa",
        "countries": ("10",),
        "conflict_start": date(2000, 1, 1),
        "start": date(2000, 1, 1),
        "end": date(2000, 6, 1),
        "outcome": "",
    }
    base.update(kwargs)
    return Episode(**base)  # type: ignore[arg-type]


def test_acd_rows_collapse_to_one_episode_with_an_end_date():
    rows = [
        {
            "conflict_id": "7",
            "year": "1990",
            "start_date": "1990-01-01",
            "start_date2": "1990-01-15",
            "ep_end": "0",
            "ep_end_date": "",
            "gwno_loc": "10, 20",
            "region": "4",
        },
        {
            "conflict_id": "7",
            "year": "1991",
            "start_date": "1990-01-01",
            "start_date2": "1990-01-15",
            "ep_end": "1",
            "ep_end_date": "1991-03-01",
            "gwno_loc": "10, 20",
            "region": "4",
        },
    ]
    episodes = episodes_from_acd(rows)
    assert len(episodes) == 1
    assert episodes[0].region == "Africa"
    assert episodes[0].countries == ("10", "20")
    assert episodes[0].end == date(1991, 3, 1)
    assert episodes[0].conflict_start == date(1990, 1, 1)


def test_onset_gaps_drop_pre_window_starts_and_keep_same_day_zeros():
    episodes = [
        _episode(group_id="old", conflict_start=date(1939, 3, 31)),
        _episode(group_id="a", conflict_start=date(1946, 1, 1), region="Europe"),
        _episode(group_id="b", conflict_start=date(1946, 1, 1), region="Europe"),
        _episode(group_id="c", conflict_start=date(1946, 1, 31), region="Asia"),
    ]
    onsets = conflict_onsets(episodes)
    gaps = inter_onset_gaps(onsets, not_before=date(1946, 1, 1))
    assert [gap["value"] for gap in gaps] == [0, 30]
    assert [gap["conflict_id"] for gap in gaps] == ["b", "c"]


def test_peace_spell_and_duration_censoring():
    episodes = [
        _episode(
            group_id="1",
            start=date(2000, 1, 1),
            end=date(2000, 1, 11),
            conflict_start=date(2000, 1, 1),
        ),
        _episode(
            group_id="1",
            start=date(2000, 2, 1),
            end=date(2000, 2, 10),
            conflict_start=date(2000, 1, 1),
        ),
    ]
    censor = date(2000, 3, 1)
    peace = peace_spells_by_group(episodes, censor)
    assert [(spell.length_days, spell.observed) for spell in peace] == [
        ((date(2000, 2, 1) - date(2000, 1, 11)).days, True),
        ((censor - date(2000, 2, 10)).days, False),
    ]
    lengths = durations(episodes, censor)
    assert all(spell.observed for spell in lengths)
    assert lengths[0].length_days == (date(2000, 1, 11) - date(2000, 1, 1)).days


def test_country_peace_merges_overlapping_wars():
    episodes = [
        _episode(
            group_id="1",
            countries=("10",),
            start=date(2000, 1, 1),
            end=date(2000, 6, 1),
        ),
        _episode(
            group_id="2",
            countries=("10",),
            start=date(2000, 5, 1),
            end=date(2001, 1, 1),
        ),
        _episode(
            group_id="3",
            countries=("10",),
            start=date(2002, 1, 1),
            end=date(2002, 6, 1),
        ),
    ]
    censor = date(2003, 1, 1)
    spells = peace_spells_by_country(episodes, censor)
    assert [(spell.length_days, spell.observed) for spell in spells] == [
        ((date(2002, 1, 1) - date(2001, 1, 1)).days, True),
        ((censor - date(2002, 6, 1)).days, False),
    ]


def test_kaplan_meier_median_with_a_censored_point():
    curve = kaplan_meier(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([True, False, True, True]),
    )
    assert curve["survival"][0] == 0.75
    assert curve["median"] == 3.0
    assert curve["survival"][-1] == 0.0


def test_weibull_shape_is_near_one_for_an_exponential_sample():
    sample = np.random.default_rng(0).exponential(10.0, size=600)
    fitted = fit_weibull(sample, np.ones(sample.shape, dtype=bool))
    assert fitted["ok"] is True
    assert 0.75 < float(fitted["shape"]) < 1.35


def test_clustered_gaps_have_cv_above_one():
    values = np.array([1.0] * 40 + [100.0] * 10)
    report = exponential_dispersion(values, n_boot=30, seed=0)
    assert float(report["cv_squared"]) > 2


def test_trend_detects_a_steady_increase():
    years = np.arange(1980, 2010, dtype=float)
    values = years - 1980
    trend = linear_trend(years, values, n_perm=200)
    assert trend["slope_per_year"] == 1
    assert trend["permutation_p"] < 0.05
    kendall = mann_kendall(values)
    assert kendall["s"] > 0
    assert kendall["p_value"] < 0.01


def test_exponential_and_poisson_means():
    years = np.array([2000, 2001, 2002, 2003], dtype=float)
    gaps = np.array([10, 10, 10, 10], dtype=float)
    model = fit_exponential_trend(years, gaps)
    assert abs(exponential_trend_mean(model, 2001.5) - 10) < 0.2
    counts = annual_onset_counts(
        [
            ("a", date(2000, 1, 1), ""),
            ("b", date(2000, 6, 1), ""),
            ("c", date(2001, 1, 1), ""),
        ],
        2000,
        2002,
    )
    assert counts == [(2000, 2), (2001, 1), (2002, 0)]
    poisson = fit_poisson_trend([(2000, 2), (2001, 2), (2002, 2), (2003, 2)])
    assert (
        abs(poisson_mean_gap(poisson, float(poisson["year_mean"])) - 365.25 / 2) < 1e-6
    )
