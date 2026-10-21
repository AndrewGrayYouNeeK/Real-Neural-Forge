"""Build a monthly frontier-compute series from Epoch AI's notable-models CSV.

Epoch AI releases the notable-models table under Creative Commons Attribution.
The fetch stores the upstream file under ``data/raw`` (gitignored) and writes
the small derived series that training actually reads.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request

import pandas as pd

from src.domain_forecasts.baselines import doubling_time_years, linear_fit, r_squared
from src.utils.paths import resolve_repo_path

SOURCE_URL = "https://epoch.ai/data/notable_ai_models.csv"
LICENCE = "CC BY 4.0"
ATTRIBUTION = (
    "Epoch AI, 'Data on AI Models' (notable AI models), "
    "https://epoch.ai/data/ai-models, CC BY."
)


def _frontier_events(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["Publication date"], errors="coerce")
    work["compute"] = pd.to_numeric(work["Training compute (FLOP)"], errors="coerce")
    work["parameters"] = pd.to_numeric(work["Parameters"], errors="coerce")
    frontier = work["Frontier model"].fillna(False).astype(bool)
    events = work.loc[
        frontier & work["date"].notna() & (work["compute"] > 0),
        [
            "Model",
            "date",
            "compute",
            "parameters",
            "Organization",
        ],
    ]
    events = events.sort_values("date")
    events["record_compute"] = events["compute"].cummax()
    return events.reset_index(drop=True)


def _monthly_record(events: pd.DataFrame) -> pd.DataFrame:
    monthly_rows: list[dict[str, object]] = []
    cursor = events["date"].min().to_period("M")
    last = events["date"].max().to_period("M")
    best = 0.0
    event_index = 0
    dates = events["date"].tolist()
    computes = events["compute"].tolist()
    while cursor <= last:
        month_end = cursor.to_timestamp(how="end")
        while event_index < len(dates) and dates[event_index] <= month_end:
            best = max(best, float(computes[event_index]))
            event_index += 1
        monthly_rows.append(
            {
                "date": cursor.to_timestamp().strftime("%Y-%m-%d"),
                "compute_flop": best,
                "log10_compute": math.log10(best),
            }
        )
        cursor += 1
    return pd.DataFrame(monthly_rows)


def descriptive_doubling(events: pd.DataFrame) -> dict[str, float | None]:
    """Log-linear doubling time on frontier publication dates, full sample.

    This describes the historical cloud of models. The training script refits
    the same regression on the training window only and that is the number
    compared with the transformer.
    """
    years = (events["date"] - events["date"].iloc[0]).dt.total_seconds() / (
        365.25 * 24 * 3600
    )
    log_compute = events["compute"].map(math.log10)
    intercept, slope = linear_fit(years.to_numpy(), log_compute.to_numpy())
    fitted = intercept + slope * years.to_numpy()
    return {
        "slope_log10_per_year": slope,
        "doubling_time_years": doubling_time_years(slope, "log10"),
        "r2": r_squared(log_compute.to_numpy(), fitted),
        "n_models": int(len(events)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--raw", default="data/raw/notable_ai_models.csv")
    parser.add_argument(
        "--monthly-out",
        default="datasets/ai_progress/frontier_compute_monthly.csv",
    )
    parser.add_argument(
        "--events-out",
        default="datasets/ai_progress/frontier_models.csv",
    )
    args = parser.parse_args()
    raw_path = resolve_repo_path(args.raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        urllib.request.urlretrieve(args.url, raw_path)
    frame = pd.read_csv(raw_path)
    events = _frontier_events(frame)
    # Epoch's frontier-compute trend is the post-2010 record. Earlier systems
    # (Theseus, the perceptron, TD-Gammon) stay in the events file.
    modern = events.loc[events["date"] >= "2010-01-01"].reset_index(drop=True)
    monthly = _monthly_record(modern)
    monthly_path = resolve_repo_path(args.monthly_out)
    events_path = resolve_repo_path(args.events_out)
    monthly_path.parent.mkdir(parents=True, exist_ok=True)
    monthly["log10_compute"] = monthly["compute_flop"].map(math.log10)
    monthly.to_csv(monthly_path, index=False)
    exported = events.copy()
    exported["date"] = exported["date"].dt.strftime("%Y-%m-%d")
    exported.to_csv(events_path, index=False)
    growth = descriptive_doubling(modern)
    growth_full = descriptive_doubling(events)
    provenance = {
        "source": args.url,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "n_notable_rows": int(len(frame)),
        "n_frontier_with_compute": int(len(events)),
        "n_frontier_post_2010": int(len(modern)),
        "monthly_rows": int(len(monthly)),
        "date_start": monthly["date"].iloc[0],
        "date_end": monthly["date"].iloc[-1],
        "series_cutoff": "2010-01-01",
        "descriptive_doubling_post_2010": growth,
        "descriptive_doubling_all_frontier_models": growth_full,
        "note": (
            "The monthly series is the cumulative maximum of frontier-model "
            "training compute from 2010 onward. descriptive_doubling figures "
            "use every model in that scope, including months later held out "
            "for testing. The training script refits the slope on the "
            "training window only."
        ),
    }
    (monthly_path.parent / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
