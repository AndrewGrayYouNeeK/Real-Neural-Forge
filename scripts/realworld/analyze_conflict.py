"""Aggregate history of time between wars. Not a forecast of a specific conflict.

Reads the cached UCDP/PRIO Armed Conflict Dataset and the UCDP Conflict
Termination Dataset written by ``fetch_ucdp``. Writes onset gaps (the series
the transformer trains on), a JSON summary, and SVG charts.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from scripts.realworld.fetch_ucdp import (
    ACD_VERSION,
    GED_VERSION,
    TERMINATION_VERSION,
    cache_dir,
)
from scripts.realworld.plots import write_png_copies
from src.realworld.baselines import (
    mean_forecast,
    persistence,
    regression_scores,
    seasonal_naive,
    target_positions,
)
from src.realworld.charts import svg_bars, svg_histogram, svg_km, svg_scatter
from src.realworld.spells import (
    Spell,
    annual_onset_counts,
    conflict_onsets,
    decade_summary,
    durations,
    episodes_from_acd,
    episodes_from_termination,
    exponential_dispersion,
    fit_poisson_trend,
    fit_weibull,
    inter_onset_gaps,
    kaplan_meier,
    lag1_correlation,
    linear_trend,
    mann_kendall,
    outcome_counts,
    peace_spells_by_country,
    peace_spells_by_group,
    poisson_mean_gap,
    spell_arrays,
    summarize,
)
from src.realworld.tabular import write_csv
from src.utils.config import load_config
from src.utils.paths import REPO_ROOT

logger = logging.getLogger(__name__)

ONSET_NOT_BEFORE = date(1946, 1, 1)
SUMMARY_PATH = REPO_ROOT / "reports" / "realworld" / "conflict_summary.json"
ARTIFACT_DIR = Path("/opt/cursor/artifacts/conflict-spells")
GAP_CONFIG = "config/realworld/ucdp_onset_gaps.yaml"

OBSERVATION_NOTE = (
    "Observed-only means drop right-censored spells and are biased short. "
    "The Kaplan-Meier median and the Weibull mean keep those spells in the "
    "risk set. This is an aggregate historical description, not a prediction "
    "of when or where a specific war will start."
)


def read_zip_csv(path: Path) -> list[dict[str, str]]:
    """Read the single CSV member of a UCDP zip."""
    with zipfile.ZipFile(path) as archive:
        name = next(item for item in archive.namelist() if item.endswith(".csv"))
        with archive.open(name) as handle:
            reader = csv.DictReader(
                io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
            )
            return list(reader)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def observation_end(rows: list[dict[str, str]]) -> date:
    """31 December of the latest 4-digit ``year`` in a conflict-year file."""
    years = [
        int(row["year"])
        for row in rows
        if (row.get("year") or "").strip().isdigit() and len(row["year"].strip()) == 4
    ]
    if not years:
        raise ValueError("conflict file has no observation year")
    return date(max(years), 12, 31)


def spell_report(spells: list[Spell]) -> dict[str, Any]:
    """Summary, Kaplan-Meier median, and Weibull fit for one set of spells."""
    lengths, observed = spell_arrays(spells)
    curve = kaplan_meier(lengths, observed)
    fitted = fit_weibull(lengths, observed)
    observed_lengths = lengths[observed] if observed.size else lengths
    return {
        "n": int(len(spells)),
        "n_observed": int(observed.sum()) if observed.size else 0,
        "n_censored": int((~observed).sum()) if observed.size else 0,
        "observed_only": summarize(observed_lengths),
        "kaplan_meier_median_days": curve["median"],
        "kaplan_meier_survival_at_end": curve["survival_at_end"],
        "weibull": fitted,
        "note": OBSERVATION_NOTE,
    }


def gap_forecasts(
    gaps: list[dict[str, object]],
    onsets: list[tuple[str, date, str]],
) -> dict[str, Any]:
    """Score mean, persistence, seasonal-naive, and Poisson on the config holdout."""
    cfg = load_config(GAP_CONFIG)
    seq_len = int(cfg["data"]["seq_len"])
    val_split = float(cfg["data"]["val_split"])
    period = int(cfg["data"]["seasonal_period"])
    values = np.array([float(row["value"]) for row in gaps])
    years = np.array([int(row["year"]) for row in gaps])
    _n_train, targets = target_positions(len(values), seq_len, val_split)
    actual = values[targets]
    first = int(targets[0])
    holdout_year = int(years[first])
    scores = {
        "mean": regression_scores(actual, mean_forecast(values, targets)),
        "persistence": regression_scores(actual, persistence(values, targets)),
        "seasonal_naive": regression_scores(
            actual, seasonal_naive(values, targets, period)
        ),
    }
    counted = annual_onset_counts(
        onsets, min(item[1].year for item in onsets), holdout_year - 1
    )
    poisson = fit_poisson_trend(counted)
    poisson_pred = np.array(
        [poisson_mean_gap(poisson, float(years[index])) for index in targets]
    )
    scores["poisson_trend"] = regression_scores(actual, poisson_pred)
    return {
        "n_val": int(len(targets)),
        "seq_len": seq_len,
        "seasonal_period": period,
        "holdout_starts": gaps[first]["date"],
        "baselines": scores,
        "poisson_model": poisson,
        "constant_rate_note": (
            "Under squared error a constant-rate exponential forecast is the "
            "training mean, so it is the mean baseline rather than a fourth number."
        ),
    }


def write_charts(
    gaps: list[dict[str, object]],
    peace_conflict: list[Spell],
    peace_country: list[Spell],
    episode_lengths: list[Spell],
    decades: list[dict[str, float]],
) -> None:
    """Write SVG charts. PNG copies are added when matplotlib is installed."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    gap_values = [float(row["value"]) for row in gaps]
    gap_years = [float(row["year"]) for row in gaps]
    positive = [value for value in gap_values if value > 0]
    mean_positive = sum(positive) / len(positive) if positive else None
    charts = {
        "onset_gap_histogram.svg": svg_histogram(
            gap_values,
            "Days between new conflict onsets (global, from 1946)",
            "gap (days)",
            exponential_mean=mean_positive,
        ),
        "onset_gaps_over_time.svg": svg_scatter(
            gap_years,
            gap_values,
            "Inter-onset gap by year",
            "year of the new onset",
            "days since the previous onset",
        ),
        "peace_kaplan_meier.svg": svg_km(
            [
                ("conflict / incompatibility", *_km_xy(peace_conflict)),
                ("country", *_km_xy(peace_country)),
            ],
            "Peace spells after fighting stops (Kaplan-Meier)",
        ),
        "duration_histogram.svg": svg_histogram(
            [float(spell.length_days) for spell in episode_lengths if spell.observed],
            "Observed conflict-episode durations",
            "duration (days)",
        ),
        "gap_by_decade.svg": svg_bars(
            [str(int(row["decade"])) for row in decades],
            [row["mean"] for row in decades],
            "Mean days between new onsets, by decade",
            "mean gap (days)",
        ),
    }
    for name, svg in charts.items():
        (ARTIFACT_DIR / name).write_text(svg)
    write_png_copies(
        ARTIFACT_DIR,
        gap_values,
        gap_years,
        peace_conflict,
        peace_country,
        episode_lengths,
        decades,
        mean_positive,
    )


def _km_xy(spells: list[Spell]) -> tuple[list[float], list[float]]:
    lengths, observed = spell_arrays(spells)
    curve = kaplan_meier(lengths, observed)
    times = curve["times"]
    survival = curve["survival"]
    assert isinstance(times, list)
    assert isinstance(survival, list)
    return times, survival


def main() -> dict[str, Any]:
    """Build the interval report from the cached UCDP files."""
    cache = cache_dir()
    acd_rows = read_zip_csv(cache / "ucdp-prio-acd-261-csv.zip")
    termination_rows = read_csv(cache / "ucdp-termination-v4-2024-conflict.csv")
    dyad_rows = read_csv(cache / "ucdp-termination-v4-2024-dyad.csv")
    acd_end = observation_end(acd_rows)
    termination_end = observation_end(termination_rows)
    dyad_end = observation_end(dyad_rows)

    episodes = episodes_from_acd(acd_rows)
    onsets = conflict_onsets(episodes)
    in_window = [item for item in onsets if item[1] >= ONSET_NOT_BEFORE]
    gaps = inter_onset_gaps(onsets, not_before=ONSET_NOT_BEFORE)
    peace_conflict = peace_spells_by_group(episodes, acd_end)
    peace_country = peace_spells_by_country(episodes, acd_end)
    episode_lengths = durations(episodes, acd_end)

    termination_episodes = episodes_from_termination(
        termination_rows,
        id_field="conflict_id",
        episode_field="c_epid",
        end_field="c_ependdate",
        outcome_field="c_outcome",
    )
    dyad_episodes = episodes_from_termination(
        dyad_rows,
        id_field="dyad_id",
        episode_field="d_epid",
        end_field="d_ependdate",
        outcome_field="d_outcome",
    )

    gap_values = np.array([float(row["value"]) for row in gaps])
    gap_years = np.array([float(row["year"]) for row in gaps])
    observed_durations = np.array(
        [spell.length_days for spell in episode_lengths if spell.observed], dtype=float
    )
    duration_years = np.array(
        [spell.start.year for spell in episode_lengths if spell.observed], dtype=float
    )
    regions = sorted({item[2] for item in in_window if item[2]})
    regional = {}
    for region in regions:
        regional_gaps = inter_onset_gaps(
            onsets, region=region, not_before=ONSET_NOT_BEFORE
        )
        regional_values = np.array([float(row["value"]) for row in regional_gaps])
        regional[region] = {
            "n_gaps": int(regional_values.size),
            "summary": summarize(regional_values),
            "lag1": lag1_correlation(regional_values),
        }

    summary: dict[str, Any] = {
        "disclaimer": OBSERVATION_NOTE,
        "sources": {
            "ged_version": GED_VERSION,
            "acd_version": ACD_VERSION,
            "acd_observation_end": acd_end.isoformat(),
            "acd_rows": len(acd_rows),
            "conflicts": len(onsets),
            "episodes": len(episodes),
            "onsets_before_1946_excluded": len(onsets) - len(in_window),
            "termination_version": TERMINATION_VERSION,
            "termination_observation_end": termination_end.isoformat(),
            "termination_corresponds_to_acd": "25.1",
            "dyad_observation_end": dyad_end.isoformat(),
            "licence": "CC BY 4.0",
        },
        "onset_gaps": {
            "summary": summarize(gap_values),
            "n_zero_day": int(np.sum(gap_values == 0)),
            "lag1_correlation": lag1_correlation(gap_values),
            "exponential": exponential_dispersion(gap_values),
            "weibull": fit_weibull(gap_values, np.ones(gap_values.shape, dtype=bool)),
            "ols_trend": linear_trend(gap_years, gap_values),
            "mann_kendall": mann_kendall(gap_values),
            "decades": decade_summary(gap_years, gap_values),
            "by_region": regional,
            "forecast_baselines": gap_forecasts(gaps, in_window),
        },
        "peace_spells_conflict": spell_report(peace_conflict),
        "peace_spells_country": spell_report(peace_country),
        "peace_spells_termination_conflict": spell_report(
            peace_spells_by_group(termination_episodes, termination_end)
        ),
        "peace_spells_termination_dyad": spell_report(
            peace_spells_by_group(dyad_episodes, dyad_end)
        ),
        "termination_outcomes_conflict": outcome_counts(termination_episodes),
        "termination_outcomes_dyad": outcome_counts(dyad_episodes),
        "durations": spell_report(episode_lengths),
        "duration_trend_observed": linear_trend(duration_years, observed_durations),
    }

    gap_path = REPO_ROOT / "data" / "realworld" / "ucdp_onset_gaps.csv"
    write_csv(
        gap_path,
        ["date", "value", "year", "region", "conflict_id"],
        gaps,
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    write_charts(
        gaps,
        peace_conflict,
        peace_country,
        episode_lengths,
        summary["onset_gaps"]["decades"],
    )
    logger.info(
        "onset gaps %d, conflict peace spells %d, durations %d, summary %s",
        len(gaps),
        len(peace_conflict),
        len(episode_lengths),
        SUMMARY_PATH,
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
