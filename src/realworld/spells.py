"""Intervals between wars in UCDP conflict data.

This is aggregate history: how long conflicts last, how long peace lasts after
one ends, and how the gaps between new onsets are distributed. It is not a
prediction of a specific war.

Dates are UCDP's published estimates. Episode ends use ``ep_end_date`` when
``ep_end`` is 1. An episode with no end date is treated as ongoing and is
censored at ``censor_date`` for durations. Peace spells after the last episode
are right-censored the same way.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np

REGION_NAMES = {
    "1": "Europe",
    "2": "Middle East",
    "3": "Asia",
    "4": "Africa",
    "5": "Americas",
}

OUTCOME_LABELS = {
    "1": "peace agreement",
    "2": "ceasefire agreement",
    "3": "victory for side A (government)",
    "4": "victory for side B (non-state)",
    "5": "low activity",
    "6": "actor ceases to exist",
}

# Continuous fits cannot take a log of a same-day (zero-length) spell.
_FIT_FLOOR_DAYS = 0.5


@dataclass(frozen=True)
class Episode:
    """One continuous spell of fighting for a conflict or a dyad."""

    group_id: str
    region: str
    countries: tuple[str, ...]
    conflict_start: date
    start: date
    end: date | None
    outcome: str = ""


@dataclass(frozen=True)
class Spell:
    """A duration in days. ``observed`` is false when the spell is censored."""

    length_days: int
    observed: bool
    start: date
    group: str


def parse_ucdp_date(raw: object) -> date | None:
    """Parse ``YYYY-MM-DD`` or a datetime string. Blank values are ``None``."""
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def region_name(raw: object) -> str:
    """Map an ACD region code to a name. Names already in the file pass through."""
    text = "" if raw is None else str(raw).strip()
    return REGION_NAMES.get(text, text)


def episodes_from_acd(rows: Iterable[Mapping[str, str]]) -> list[Episode]:
    """Collapse conflict-year rows into episodes keyed by start_date2."""
    grouped: dict[tuple[str, str], Episode] = {}
    for row in rows:
        conflict_id = (row.get("conflict_id") or "").strip()
        started = parse_ucdp_date(row.get("start_date"))
        episode_start = parse_ucdp_date(row.get("start_date2"))
        if not conflict_id or started is None or episode_start is None:
            continue
        ended = (
            parse_ucdp_date(row.get("ep_end_date"))
            if (row.get("ep_end") or "").strip() == "1"
            else None
        )
        countries = _split_codes(row.get("gwno_loc"))
        key = (conflict_id, episode_start.isoformat())
        current = grouped.get(key)
        if current is None:
            grouped[key] = Episode(
                group_id=conflict_id,
                region=region_name(row.get("region")),
                countries=countries,
                conflict_start=started,
                start=episode_start,
                end=ended,
            )
            continue
        grouped[key] = Episode(
            group_id=current.group_id,
            region=current.region or region_name(row.get("region")),
            countries=current.countries or countries,
            conflict_start=min(current.conflict_start, started),
            start=current.start,
            end=ended if ended is not None else current.end,
        )
    episodes = list(grouped.values())
    episodes.sort(
        key=lambda episode: (episode.conflict_start, episode.start, episode.group_id)
    )
    return episodes


def episodes_from_termination(
    rows: Iterable[Mapping[str, str]],
    *,
    id_field: str,
    episode_field: str,
    end_field: str,
    outcome_field: str,
) -> list[Episode]:
    """Collapse termination-dataset rows into episodes.

    Conflict files use ``c_epid`` / ``c_ependdate`` / ``c_outcome``. Dyad files
    use the ``d_`` equivalents. Pass those names in. An episode with no end
    date is ongoing.
    """
    grouped: dict[tuple[str, str], Episode] = {}
    for row in rows:
        group_id = (row.get(id_field) or "").strip()
        started = parse_ucdp_date(row.get("start_date2")) or parse_ucdp_date(
            row.get("start_date")
        )
        if not group_id or started is None:
            continue
        episode_id = (row.get(episode_field) or "").strip() or started.isoformat()
        conflict_start = parse_ucdp_date(row.get("start_date")) or started
        ended = parse_ucdp_date(row.get(end_field))
        outcome = (row.get(outcome_field) or "").strip()
        countries = _split_codes(row.get("gwno_loc"))
        key = (group_id, episode_id)
        current = grouped.get(key)
        if current is None:
            grouped[key] = Episode(
                group_id=group_id,
                region=region_name(row.get("region")),
                countries=countries,
                conflict_start=conflict_start,
                start=started,
                end=ended,
                outcome=outcome,
            )
            continue
        grouped[key] = Episode(
            group_id=current.group_id,
            region=current.region or region_name(row.get("region")),
            countries=current.countries or countries,
            conflict_start=min(current.conflict_start, conflict_start),
            start=min(current.start, started),
            end=ended if ended is not None else current.end,
            outcome=outcome or current.outcome,
        )
    episodes = list(grouped.values())
    episodes.sort(key=lambda episode: (episode.start, episode.group_id))
    return episodes


def conflict_onsets(episodes: Iterable[Episode]) -> list[tuple[str, date, str]]:
    """First start date of each group, sorted by date then id."""
    first: dict[str, Episode] = {}
    for episode in episodes:
        current = first.get(episode.group_id)
        if current is None or episode.conflict_start < current.conflict_start:
            first[episode.group_id] = episode
    rows = [
        (episode.group_id, episode.conflict_start, episode.region)
        for episode in first.values()
    ]
    rows.sort(key=lambda item: (item[1], item[0]))
    return rows


def inter_onset_gaps(
    onsets: list[tuple[str, date, str]],
    *,
    region: str | None = None,
    not_before: date | None = None,
) -> list[dict[str, object]]:
    """Days between successive onset dates.

    Same-day onsets produce a gap of 0. ``not_before`` drops onsets earlier
    than the dataset's observation window (UCDP's window opens in 1946).
    """
    chosen = [
        item
        for item in onsets
        if (region is None or item[2] == region)
        and (not_before is None or item[1] >= not_before)
    ]
    chosen.sort(key=lambda item: (item[1], item[0]))
    gaps: list[dict[str, object]] = []
    for previous, current in zip(chosen, chosen[1:], strict=False):
        gaps.append(
            {
                "date": current[1].isoformat(),
                "year": current[1].year,
                "value": (current[1] - previous[1]).days,
                "conflict_id": current[0],
                "region": current[2],
                "previous_conflict_id": previous[0],
            }
        )
    return gaps


def peace_spells_by_group(episodes: Iterable[Episode], censor: date) -> list[Spell]:
    """Days from an episode end until the next episode of the same group.

    No following episode produces a right-censored spell running to ``censor``.
    Overlapping episodes are not a peace spell.
    """
    grouped: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.group_id].append(episode)
    spells: list[Spell] = []
    for group_id, group in grouped.items():
        ordered = sorted(
            group, key=lambda episode: (episode.start, episode.end or censor)
        )
        for index, episode in enumerate(ordered):
            if episode.end is None or episode.end > censor:
                continue
            follower = ordered[index + 1] if index + 1 < len(ordered) else None
            if follower is not None and follower.start > episode.end:
                length = (follower.start - episode.end).days
                if length > 0:
                    spells.append(Spell(length, True, episode.end, group_id))
            elif follower is None:
                length = (censor - episode.end).days
                if length > 0:
                    spells.append(Spell(length, False, episode.end, group_id))
    return spells


def peace_spells_by_country(episodes: Iterable[Episode], censor: date) -> list[Spell]:
    """Gaps between merged active intervals in each country code.

    A conflict listed in several ``gwno_loc`` codes contributes its interval
    to each of those countries. Ongoing episodes run to ``censor``. Peace
    still underway at ``censor`` is right-censored.
    """
    intervals: dict[str, list[tuple[date, date]]] = defaultdict(list)
    for episode in episodes:
        active_end = censor if episode.end is None else min(episode.end, censor)
        if active_end < episode.start:
            continue
        for country in episode.countries:
            intervals[country].append((episode.start, active_end))
    spells: list[Spell] = []
    for country, spans in intervals.items():
        merged = _merge_intervals(spans)
        for (_left_start, left_end), (right_start, _right_end) in zip(
            merged, merged[1:], strict=False
        ):
            if right_start > left_end:
                length = (right_start - left_end).days
                if length > 0:
                    spells.append(Spell(length, True, left_end, country))
        if merged and merged[-1][1] < censor:
            length = (censor - merged[-1][1]).days
            if length > 0:
                spells.append(Spell(length, False, merged[-1][1], country))
    return spells


def durations(episodes: Iterable[Episode], censor: date) -> list[Spell]:
    """Episode length in days. Ongoing episodes are censored at ``censor``."""
    spells: list[Spell] = []
    for episode in episodes:
        if episode.end is None or episode.end > censor:
            length = (censor - episode.start).days
            observed = False
        else:
            length = (episode.end - episode.start).days
            observed = True
        if length < 0:
            continue
        spells.append(Spell(length, observed, episode.start, episode.group_id))
    return spells


def outcome_counts(episodes: Iterable[Episode]) -> dict[str, int]:
    """Count termination outcomes, labeled when the code is one of 1–6."""
    counts: dict[str, int] = defaultdict(int)
    for episode in episodes:
        code = episode.outcome.strip()
        if not code:
            continue
        counts[OUTCOME_LABELS.get(code, f"code {code}")] += 1
    return dict(sorted(counts.items()))


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    """n, mean, median, std, min, max, and the coefficient of variation."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {"n": 0}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    cv = std / mean if mean else None
    return {
        "n": int(array.size),
        "mean": mean,
        "median": float(np.median(array)),
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
        "cv": cv,
        "cv_squared": None if cv is None else cv**2,
    }


def spell_arrays(spells: Iterable[Spell]) -> tuple[np.ndarray, np.ndarray]:
    """Lengths and a boolean observed mask. Empty input yields empty arrays."""
    items = list(spells)
    if not items:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=bool)
    lengths = np.array([item.length_days for item in items], dtype=float)
    observed = np.array([item.observed for item in items], dtype=bool)
    return lengths, observed


def kaplan_meier(lengths: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    """Product-limit survival curve. Deaths at time t are applied before censoring."""
    times = np.asarray(lengths, dtype=float)
    events = np.asarray(observed, dtype=bool)
    if times.size == 0:
        return {"times": [], "survival": [], "median": None, "survival_at_end": None}
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]
    at_risk = int(times.size)
    survival = 1.0
    curve_t: list[float] = []
    curve_s: list[float] = []
    index = 0
    while index < times.size:
        stamp = float(times[index])
        deaths = 0
        censored = 0
        while index < times.size and float(times[index]) == stamp:
            if events[index]:
                deaths += 1
            else:
                censored += 1
            index += 1
        if deaths and at_risk:
            survival *= 1.0 - deaths / at_risk
        curve_t.append(stamp)
        curve_s.append(survival)
        at_risk -= deaths + censored
    median = next(
        (stamp for stamp, level in zip(curve_t, curve_s, strict=True) if level <= 0.5),
        None,
    )
    return {
        "times": curve_t,
        "survival": curve_s,
        "median": median,
        "survival_at_end": curve_s[-1],
    }


def fit_weibull(
    lengths: np.ndarray, observed: np.ndarray
) -> dict[str, float | bool | None]:
    """Right-censored Weibull MLE profiled over the shape.

    Lengths are floored at half a day so same-day spells do not make the log
    likelihood infinite. Shape 1 is the exponential (memoryless) model. The
    likelihood-ratio p-value compares the fitted shape with that exponential.
    """
    raw = np.asarray(lengths, dtype=float)
    events = np.asarray(observed, dtype=bool)
    if raw.size < 2 or int(events.sum()) == 0:
        return {"ok": False}
    times = np.maximum(raw, _FIT_FLOOR_DAYS)
    n_events = float(events.sum())

    def neg_loglik(shape: float) -> float:
        powered = times**shape
        scale_power = float(powered.sum()) / n_events
        log_scale = math.log(scale_power) / shape
        log_time = np.log(times)
        ratio = powered / scale_power
        event_ll = (
            np.log(shape) - log_scale + (shape - 1.0) * (log_time - log_scale) - ratio
        )
        censored_ll = -ratio
        return -float(np.where(events, event_ll, censored_ll).sum())

    shape, nll = _golden(neg_loglik, 0.15, 8.0)
    exponential_nll = neg_loglik(1.0)
    # Profile the scale at the chosen shape for the reported mean.
    scale_power = float((times**shape).sum()) / n_events
    scale = scale_power ** (1.0 / shape)
    statistic = max(0.0, 2.0 * (exponential_nll - nll))
    return {
        "ok": True,
        "shape": shape,
        "scale_days": scale,
        "mean_days": scale * math.gamma(1.0 + 1.0 / shape),
        "median_days": scale * (math.log(2.0) ** (1.0 / shape)),
        "nll": nll,
        "exponential_nll": exponential_nll,
        "lr_stat": statistic,
        "lr_p_value": math.erfc(math.sqrt(statistic / 2.0)),
        "floor_days": _FIT_FLOOR_DAYS,
    }


def exponential_dispersion(
    values: np.ndarray, n_boot: int = 400, seed: int = 0
) -> dict[str, float]:
    """Compare positive durations with an exponential fitted on the same sample.

    The p-value is a parametric bootstrap that re-estimates the rate, because
    a textbook Kolmogorov-Smirnov p-value assumes a pre-specified rate.
    ``cv_squared`` near 1 is what a memoryless exponential looks like. Values
    well above 1 are more clustered (some very short gaps and some very long).
    """
    array = np.asarray(values, dtype=float)
    array = array[array > 0]
    if array.size < 5:
        return {"n": float(array.size)}
    rate = 1.0 / float(array.mean())
    observed = _ks_exponential(array, rate)
    rng = np.random.default_rng(seed)
    larger = 0
    for _ in range(n_boot):
        sample = rng.exponential(1.0 / rate, size=array.size)
        sample_rate = 1.0 / float(sample.mean())
        if _ks_exponential(sample, sample_rate) >= observed - 1e-15:
            larger += 1
    std = float(array.std(ddof=1))
    mean = float(array.mean())
    return {
        "n": float(array.size),
        "rate_per_day": rate,
        "mean": mean,
        "cv_squared": (std / mean) ** 2,
        "ks": observed,
        "p_value": (larger + 1) / (n_boot + 1),
        "n_boot": float(n_boot),
    }


def linear_trend(
    years: np.ndarray, values: np.ndarray, n_perm: int = 2000
) -> dict[str, float]:
    """OLS slope of ``values`` on calendar year, with a permutation p-value."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size != y.size or x.size < 3:
        raise ValueError("trend needs at least three paired observations")
    centered = x - x.mean()
    variance = float(np.sum(centered**2))
    slope = float(np.sum(centered * (y - y.mean())) / variance)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    degrees = x.size - 2
    sigma2 = float(np.sum(residual**2) / degrees)
    standard_error = math.sqrt(sigma2 / variance) if variance else math.nan
    t_stat = slope / standard_error if standard_error else math.nan
    rng = np.random.default_rng(0)
    larger = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        permuted = float(np.sum(centered * (shuffled - shuffled.mean())) / variance)
        if abs(permuted) >= abs(slope):
            larger += 1
    return {
        "n": float(x.size),
        "slope_per_year": slope,
        "slope_per_decade": slope * 10.0,
        "intercept": intercept,
        "t": t_stat,
        "permutation_p": (larger + 1) / (n_perm + 1),
    }


def mann_kendall(values: np.ndarray) -> dict[str, float]:
    """Mann-Kendall trend test with the tie-corrected normal approximation."""
    y = np.asarray(values, dtype=float).reshape(-1)
    n = int(y.size)
    if n < 3:
        raise ValueError("Mann-Kendall needs at least three observations")
    s_stat = 0.0
    for index in range(n - 1):
        s_stat += float(np.sign(y[index + 1 :] - y[index]).sum())
    _, tie_counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if variance <= 0 or s_stat == 0:
        z_value = 0.0
    elif s_stat > 0:
        z_value = (s_stat - 1.0) / math.sqrt(variance)
    else:
        z_value = (s_stat + 1.0) / math.sqrt(variance)
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return {"n": float(n), "s": s_stat, "z": z_value, "p_value": p_value}


def lag1_correlation(values: np.ndarray) -> float | None:
    """Pearson correlation of a series with itself shifted by one step."""
    y = np.asarray(values, dtype=float).reshape(-1)
    if y.size < 3 or float(y.std()) == 0:
        return None
    left, right = y[:-1], y[1:]
    if float(left.std()) == 0 or float(right.std()) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def decade_summary(years: np.ndarray, values: np.ndarray) -> list[dict[str, float]]:
    """Mean and median by calendar decade (1940 means 1940–1949)."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for year, value in zip(
        np.asarray(years), np.asarray(values, dtype=float), strict=True
    ):
        buckets[(int(year) // 10) * 10].append(float(value))
    rows: list[dict[str, float]] = []
    for decade in sorted(buckets):
        sample = np.array(buckets[decade], dtype=float)
        rows.append(
            {
                "decade": float(decade),
                "n": float(sample.size),
                "mean": float(sample.mean()),
                "median": float(np.median(sample)),
            }
        )
    return rows


def fit_exponential_trend(years: np.ndarray, gaps: np.ndarray) -> dict[str, float]:
    """Exponential gaps with log-rate linear in year: ``λ = exp(a + b (year-mean))``.

    The squared-error point forecast is the mean ``1/λ``. A constant rate
    (``b = 0``) reproduces the sample mean.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(gaps, dtype=float)
    if x.size != y.size or x.size < 3:
        raise ValueError("exponential trend needs at least three gaps")
    if np.any(y < 0):
        raise ValueError("gaps must be non-negative")
    center = float(x.mean())
    centered = x - center
    log_rate = math.log(1.0 / max(float(y.mean()), _FIT_FLOOR_DAYS))
    slope = 0.0
    for _ in range(30):
        lam = np.exp(log_rate + slope * centered)
        residual = 1.0 - lam * y
        grad_a = -float(np.sum(residual))
        grad_b = -float(np.sum(centered * residual))
        weight = lam * np.maximum(y, _FIT_FLOOR_DAYS)
        haa = float(np.sum(weight))
        hab = float(np.sum(centered * weight))
        hbb = float(np.sum(centered**2 * weight))
        det = haa * hbb - hab**2
        if det <= 1e-12:
            break
        step_a = (hbb * grad_a - hab * grad_b) / det
        step_b = (haa * grad_b - hab * grad_a) / det
        log_rate -= step_a
        slope -= step_b
        if abs(step_a) + abs(step_b) < 1e-12:
            break
    return {
        "log_rate_at_mean_year": log_rate,
        "slope_log_rate_per_year": slope,
        "year_mean": center,
        "n": float(x.size),
    }


def exponential_trend_mean(model: Mapping[str, float], year: float) -> float:
    """Expected gap in days at ``year`` from :func:`fit_exponential_trend`."""
    centered = year - float(model["year_mean"])
    lam = math.exp(
        float(model["log_rate_at_mean_year"])
        + float(model["slope_log_rate_per_year"]) * centered
    )
    return 1.0 / lam


def fit_poisson_trend(year_counts: list[tuple[int, int]]) -> dict[str, float]:
    """Annual onset counts ~ Poisson with log-mean linear in year.

    The implied mean gap is ``365.25 / λ(year)`` onsets per year.
    """
    if len(year_counts) < 3:
        raise ValueError("Poisson trend needs at least three years")
    years = np.array([year for year, _count in year_counts], dtype=float)
    counts = np.array([count for _year, count in year_counts], dtype=float)
    center = float(years.mean())
    centered = years - center
    log_mean = math.log(max(float(counts.mean()), 1e-3))
    slope = 0.0
    for _ in range(30):
        lam = np.exp(log_mean + slope * centered)
        residual = lam - counts
        grad_a = float(np.sum(residual))
        grad_b = float(np.sum(centered * residual))
        haa = float(np.sum(lam))
        hab = float(np.sum(centered * lam))
        hbb = float(np.sum(centered**2 * lam))
        det = haa * hbb - hab**2
        if det <= 1e-12:
            break
        step_a = (hbb * grad_a - hab * grad_b) / det
        step_b = (haa * grad_b - hab * grad_a) / det
        log_mean -= step_a
        slope -= step_b
        if abs(step_a) + abs(step_b) < 1e-12:
            break
    return {
        "log_mean_at_mean_year": log_mean,
        "slope_log_mean_per_year": slope,
        "year_mean": center,
        "n_years": float(years.size),
    }


def poisson_mean_gap(model: Mapping[str, float], year: float) -> float:
    """Days per onset implied by the annual Poisson rate."""
    centered = year - float(model["year_mean"])
    lam = math.exp(
        float(model["log_mean_at_mean_year"])
        + float(model["slope_log_mean_per_year"]) * centered
    )
    return 365.25 / lam


def annual_onset_counts(
    onsets: list[tuple[str, date, str]], first_year: int, last_year: int
) -> list[tuple[int, int]]:
    """Onset counts for every year in ``first_year`` … ``last_year``, zeros included."""
    counts: dict[int, int] = defaultdict(int)
    for _group, started, _region in onsets:
        if first_year <= started.year <= last_year:
            counts[started.year] += 1
    return [(year, counts.get(year, 0)) for year in range(first_year, last_year + 1)]


def _split_codes(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _merge_intervals(spans: list[tuple[date, date]]) -> list[tuple[date, date]]:
    ordered = sorted(spans)
    merged: list[tuple[date, date]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _ks_exponential(sample: np.ndarray, rate: float) -> float:
    ordered = np.sort(np.asarray(sample, dtype=float))
    cdf = 1.0 - np.exp(-rate * ordered)
    n = ordered.size
    high = np.arange(1, n + 1) / n
    low = np.arange(0, n) / n
    return float(max(np.max(np.abs(high - cdf)), np.max(np.abs(cdf - low))))


def _golden(
    fn: Callable[[float], float], lo: float, hi: float, steps: int = 50
) -> tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = lo, hi
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = fn(c), fn(d)
    for _ in range(steps):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = fn(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = fn(d)
    if fc < fd:
        return c, fc
    return d, fd
