"""Run the Powerball comparison and write the report files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.lottery import WHITE_BALL_COUNT, fetch_and_write, read_draw_table
from src.evaluation.lottery import temporal_window_split
from src.lottery_study.align import DRAW_TIMEZONE, align_observations
from src.lottery_study.closecall import (
    SCALAR_TARGETS,
    ball_indicators,
    binomial_two_sided_p,
    chance_reference,
    close_call_scorecard,
    decode_ticket,
    decode_tickets,
    frequency_ticket,
    lottery_target_frame,
    random_tickets,
    raw_sum_abs_error,
    score_tickets,
    shift_ticket,
)
from src.lottery_study.closecall import (
    attach_close_call_scoring as attach_scorecard,
)
from src.lottery_study.modeling import build_windows, train_holdout
from src.lottery_study.registry import DatasetLoad, resolve_datasets
from src.lottery_study.report import (
    conclusion_text,
    headline_table,
    render_chart,
    render_markdown,
    to_jsonable,
)
from src.lottery_study.stats import (
    benjamini_hochberg,
    bonferroni_adjust,
    mutual_info_permutation,
    paired_difference_p,
    pearson_p_value,
    pearson_permutation_p,
    pearson_r,
    summarize_corrections,
)
from src.utils.paths import resolve_repo_path

_WHITE_COLUMNS = ("white_1", "white_2", "white_3", "white_4", "white_5")


def run_study(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compare every available dataset with Powerball and write the report."""
    alpha = float(cfg.get("alpha", 0.05))
    min_aligned = max(3, int(cfg.get("min_aligned", 80)))
    n_perm = int(cfg.get("n_permutations", 499))
    n_bins = int(cfg.get("mi_bins", 8))
    rng = np.random.default_rng(int(cfg.get("permutation_seed", 0)))
    training = dict(cfg.get("training") or {})
    seq_len = int(training.get("seq_len", 32))
    val_split = float(training.get("val_split", 0.2))

    lottery_path = _ensure_lottery(cfg)
    frame = read_draw_table(lottery_path)
    draw_days = [pd.Timestamp(value).date() for value in frame["draw_date"]]
    whites = frame.loc[:, list(_WHITE_COLUMNS)].to_numpy(dtype=np.int64)
    powerball = frame["powerball"].to_numpy(dtype=np.int64)
    balls = np.column_stack([whites, powerball]).astype(np.float64)
    n_draws = int(balls.shape[0])
    split = temporal_window_split(n_draws, seq_len, val_split)
    scorecard_module = attach_scorecard()

    loads = resolve_datasets(cfg)
    aligned = _align_loads(loads, draw_days, cfg)
    _log(
        cfg,
        f"Lottery draws: {n_draws} ({draw_days[0].isoformat()} to {draw_days[-1].isoformat()})",
    )
    _log(
        cfg,
        "Datasets available: "
        + (", ".join(str(item["dataset_id"]) for item in aligned) or "none"),
    )

    targets = lottery_target_frame(whites, powerball)
    indicator_names, indicators = ball_indicators(whites, powerball)
    association = _association_tests(
        aligned,
        targets,
        indicator_names,
        indicators,
        min_aligned=min_aligned,
        alpha=alpha,
        n_perm=n_perm,
        n_bins=n_bins,
        rng=rng,
        log=lambda message: _log(cfg, message),
    )

    scored, per_draw = _score_predictors(
        balls,
        aligned,
        split,
        training,
        cfg,
    )
    chance = chance_reference(
        per_draw["actual"],
        n_sim=int(cfg.get("chance_simulations", 2000)),
        seed=int(cfg.get("chance_seed", 0)),
    )
    predictive = _predictive_tests(scored, per_draw, chance, alpha)
    _apply_corrections(association, predictive, alpha, 1.0 / (n_perm + 1))
    _mark_dataset_signals(scored, predictive, alpha)

    report: dict[str, Any] = {
        "schema_version": 1,
        "draw_cutoff": f"22:59 {DRAW_TIMEZONE}",
        "lottery": {
            "path": str(lottery_path),
            "n_draws": n_draws,
            "start": draw_days[0].isoformat(),
            "end": draw_days[-1].isoformat(),
            "holdout_start_index": int(split["first_val_target"]),
            "n_holdout_scored": int(per_draw["actual"].shape[0]),
            "seq_len": seq_len,
            "val_split": val_split,
        },
        "scorecard_module": scorecard_module,
        "datasets": [_dataset_row(load, aligned) for load in loads],
        "chance": _chance_dict(chance),
        "association_tests": association,
        "predictive_tests": predictive,
        "predictors": scored,
        "corrections": {},
    }
    report["corrections"] = {
        "association": _summary_dict(
            association, "association", alpha, 1.0 / (n_perm + 1)
        ),
        "predictive": _summary_dict(predictive, "predictive", alpha, None),
        "combined": _summary_dict(
            association + predictive, "combined", alpha, 1.0 / (n_perm + 1)
        ),
    }
    report["conclusion"] = conclusion_text(report)
    report["headline_table"] = headline_table(report)
    chart_path = _output_path(cfg, "chart")
    report["chart_path"] = str(chart_path) if chart_path else None
    markdown = render_markdown(report)
    report["markdown"] = markdown
    _write_outputs(cfg, report, markdown, render_chart(report))
    return report


def _ensure_lottery(cfg: dict[str, Any]) -> Path:
    path = resolve_repo_path(str(cfg.get("lottery_csv", "data/powerball.csv")))
    if path.is_file():
        return path
    if not cfg.get("fetch_lottery_if_missing", False):
        raise FileNotFoundError(
            f"Powerball CSV is missing at {path}. Run scripts/fetch_powerball.py "
            "or set fetch_lottery_if_missing: true."
        )
    _log(cfg, f"Fetching Powerball draws into {path}")
    fetch_and_write(path)
    return path


def _align_loads(
    loads: list[DatasetLoad],
    draw_days: list[Any],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    timezone = str(cfg.get("timezone", DRAW_TIMEZONE))
    aligned: list[dict[str, Any]] = []
    for load in loads:
        if not load.available or load.values is None or load.observed_at is None:
            continue
        matrix, rule = align_observations(
            draw_days,
            load.observed_at,
            load.values,
            availability=load.availability,
            timezone=timezone,
        )
        keep = [
            index
            for index, column in enumerate(load.columns)
            if np.isfinite(matrix[:, index]).any()
        ]
        if not keep:
            load.available = False
            load.detail = "no finite values aligned to a draw"
            continue
        aligned.append(
            {
                "dataset_id": load.dataset_id,
                "group": load.group,
                "description": load.description,
                "columns": tuple(load.columns[index] for index in keep),
                "values": matrix[:, keep],
                "rule": rule,
                "source": load.source,
            }
        )
    return aligned


def _association_tests(
    aligned: list[dict[str, Any]],
    targets: dict[str, np.ndarray],
    indicator_names: tuple[str, ...],
    indicators: np.ndarray,
    *,
    min_aligned: int,
    alpha: float,
    n_perm: int,
    n_bins: int,
    rng: np.random.Generator,
    log: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in aligned:
        values = dataset["values"]
        for column_index, column in enumerate(dataset["columns"]):
            series_id = f"{dataset['dataset_id']}.{column}"
            log(f"Association tests for {series_id}")
            series = values[:, column_index]
            for target_name in SCALAR_TARGETS:
                records.extend(
                    _scalar_tests(
                        series_id,
                        dataset["dataset_id"],
                        series,
                        targets[target_name],
                        target_name,
                        min_aligned=min_aligned,
                        alpha=alpha,
                        n_perm=n_perm,
                        n_bins=n_bins,
                        rng=rng,
                    )
                )
            for indicator_index, indicator_name in enumerate(indicator_names):
                record = _pearson_test(
                    series_id,
                    dataset["dataset_id"],
                    series,
                    indicators[:, indicator_index],
                    indicator_name,
                    min_aligned=min_aligned,
                    alpha=alpha,
                    n_perm=n_perm,
                    rng=rng,
                )
                if record is not None:
                    records.append(record)
    return records


def _scalar_tests(
    series_id: str,
    dataset_id: str,
    series: np.ndarray,
    target: np.ndarray,
    target_name: str,
    *,
    min_aligned: int,
    alpha: float,
    n_perm: int,
    n_bins: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    pearson = _pearson_test(
        series_id,
        dataset_id,
        series,
        target,
        target_name,
        min_aligned=min_aligned,
        alpha=alpha,
        n_perm=n_perm,
        rng=rng,
    )
    if pearson is None:
        return []
    mask = np.isfinite(series) & np.isfinite(target)
    mi_bits, mi_p, exceedances = mutual_info_permutation(
        series[mask],
        target[mask],
        n_bins=n_bins,
        n_perm=n_perm,
        rng=rng,
    )
    mutual = {
        "family": "association",
        "kind": "mutual_info",
        "series_id": series_id,
        "dataset_id": dataset_id,
        "target": target_name,
        "n": int(mask.sum()),
        "pearson_r": pearson["pearson_r"],
        "mutual_info_bits": mi_bits,
        "p": mi_p,
        "permutation_p": mi_p,
        "permutation_exceedances": exceedances,
    }
    return [pearson, mutual]


def _pearson_test(
    series_id: str,
    dataset_id: str,
    series: np.ndarray,
    target: np.ndarray,
    target_name: str,
    *,
    min_aligned: int,
    alpha: float,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    mask = np.isfinite(series) & np.isfinite(target)
    n = int(mask.sum())
    if n < min_aligned:
        return None
    x = series[mask]
    y = target[mask]
    correlation = pearson_r(x, y)
    if correlation is None:
        return None
    p_value = pearson_p_value(correlation, n)
    perm_p = None
    exceedances = None
    if p_value <= alpha:
        perm_p, exceedances = pearson_permutation_p(x, y, n_perm=n_perm, rng=rng)
    return {
        "family": "association",
        "kind": "pearson",
        "series_id": series_id,
        "dataset_id": dataset_id,
        "target": target_name,
        "n": n,
        "pearson_r": correlation,
        "mutual_info_bits": None,
        "p": float(min(1.0, max(0.0, p_value))),
        "permutation_p": perm_p,
        "permutation_exceedances": exceedances,
    }


def _score_predictors(
    balls: np.ndarray,
    aligned: list[dict[str, Any]],
    split: dict[str, int],
    training: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_draws = int(balls.shape[0])
    holdout = np.arange(int(split["first_val_target"]), n_draws)
    history = balls[: int(split["first_val_target"])]
    fits: dict[str, dict[str, Any]] = {}
    if training.get("enabled", True):
        fits["plain"] = _fit_variant(
            "Plain transformer", "plain", None, balls, training, cfg
        )
        matrices = [dataset["values"] for dataset in aligned]
        for dataset in aligned:
            fits[dataset["dataset_id"]] = _fit_variant(
                dataset["dataset_id"].replace("_", " ").title(),
                "dataset",
                dataset["values"],
                balls,
                training,
                cfg,
            )
        if len(matrices) == 1:
            only = aligned[0]["dataset_id"]
            fits["combined"] = {
                **fits[only],
                "name": "All datasets",
                "kind": "combined",
                "alias_of": only,
                "note": f"Same fit as {only}; it is the only external dataset on main.",
            }
        elif len(matrices) > 1:
            fits["combined"] = _fit_variant(
                "All datasets",
                "combined",
                np.column_stack(matrices),
                balls,
                training,
                cfg,
            )
    common = holdout
    for fit in fits.values():
        if fit.get("alias_of"):
            continue
        if "target_index" not in fit:
            continue
        common = np.intersect1d(common, fit["target_index"])
    if common.size == 0:
        raise ValueError("no holdout draws left after alignment")
    actual = balls[common].astype(np.int64)
    per_draw: dict[str, Any] = {"index": common, "actual": actual, "rows": {}}
    scored: list[dict[str, Any]] = []

    def _add(
        name: str,
        kind: str,
        tickets: np.ndarray,
        *,
        raw: np.ndarray | None,
        note: str,
        dataset_id: str | None,
    ) -> None:
        row = _predictor_row(
            name, kind, tickets, actual, raw=raw, note=note, dataset_id=dataset_id
        )
        scored.append(row)
        parts = _per_draw(tickets, actual, raw)
        per_draw["rows"][name] = parts

    if "plain" in fits and "error" not in fits["plain"]:
        raw = _raw_on(fits["plain"], common)
        _add(
            "Plain transformer",
            "plain",
            decode_tickets(raw),
            raw=raw,
            note="",
            dataset_id=None,
        )
    elif training.get("enabled", True):
        raise ValueError(
            fits.get("plain", {}).get("error", "plain model was not trained")
        )

    for dataset in aligned:
        variant = fits.get(str(dataset["dataset_id"]))
        if variant is None or "error" in variant:
            detail = "" if variant is None else str(variant.get("error", ""))
            _log(cfg, f"Skipping model {dataset['dataset_id']}: {detail}")
            continue
        raw = _raw_on(variant, common)
        _add(
            str(variant["name"]),
            "dataset",
            decode_tickets(raw),
            raw=raw,
            note=str(variant.get("note", "")),
            dataset_id=str(dataset["dataset_id"]),
        )
    if "combined" in fits and "error" not in fits["combined"] and "plain" in fits:
        fit = fits["combined"]
        if fit.get("alias_of"):
            source = next(
                row for row in scored if row.get("dataset_id") == fit["alias_of"]
            )
            alias = dict(source)
            alias["name"] = "All datasets"
            alias["kind"] = "combined"
            alias["dataset_id"] = "combined"
            alias["note"] = str(fit["note"])
            alias["alias_of"] = fit["alias_of"]
            scored.append(alias)
            per_draw["rows"]["All datasets"] = per_draw["rows"][source["name"]]
        else:
            raw = _raw_on(fit, common)
            _add(
                "All datasets",
                "combined",
                decode_tickets(raw),
                raw=raw,
                note="",
                dataset_id="combined",
            )

    mean_ticket = decode_ticket(history.mean(axis=0))
    mean_tickets = np.repeat(mean_ticket.reshape(1, -1), common.size, axis=0)
    mean_raw = np.repeat(history.mean(axis=0).reshape(1, -1), common.size, axis=0)
    _add(
        "Historical mean",
        "historical_mean",
        mean_tickets,
        raw=mean_raw,
        note="Positional mean of draws before the holdout, then repaired into a legal ticket.",
        dataset_id=None,
    )
    previous = balls[common - 1].astype(np.int64)
    _add(
        "Persistence",
        "pattern",
        previous,
        raw=None,
        note="Replay the previous draw.",
        dataset_id=None,
    )
    hot = frequency_ticket(
        history[:, :WHITE_BALL_COUNT],
        history[:, WHITE_BALL_COUNT].astype(np.int64),
        cold=False,
    )
    cold = frequency_ticket(
        history[:, :WHITE_BALL_COUNT],
        history[:, WHITE_BALL_COUNT].astype(np.int64),
        cold=True,
    )
    _add(
        "Hot numbers",
        "pattern",
        np.repeat(hot.reshape(1, -1), common.size, axis=0),
        raw=None,
        note="Most common whites and Powerball before the holdout.",
        dataset_id=None,
    )
    _add(
        "Cold numbers",
        "pattern",
        np.repeat(cold.reshape(1, -1), common.size, axis=0),
        raw=None,
        note="Least common whites and Powerball before the holdout.",
        dataset_id=None,
    )
    shifted = np.vstack([shift_ticket(row) for row in previous])
    _add(
        "Neighbor shift",
        "pattern",
        shifted,
        raw=None,
        note="Previous draw with every number moved up by one.",
        dataset_id=None,
    )
    random_seed = int(cfg.get("chance_seed", 0)) + 1
    random = random_tickets(int(common.size), np.random.default_rng(random_seed))
    _add(
        "Random tickets",
        "random",
        random,
        raw=None,
        note="One seeded uniform ticket per holdout draw.",
        dataset_id=None,
    )
    if not any(row["kind"] == "plain" for row in scored):
        # Training was turned off. The conclusion still compares tickets with
        # the historical mean; duplicate that row under the plain kind only
        # when the transformer was not requested.
        if training.get("enabled", True):
            raise ValueError("plain model row is missing")
    return scored, per_draw


def _fit_variant(
    name: str,
    kind: str,
    covariates: np.ndarray | None,
    balls: np.ndarray,
    training: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    seq_len = int(training.get("seq_len", 32))
    _log(cfg, f"Training {name}")
    try:
        inputs, targets, _index = build_windows(balls, covariates, seq_len)
        target_index, raw = train_holdout(
            inputs, targets, n_draws=int(balls.shape[0]), training=training
        )
    except (RuntimeError, ValueError) as exc:
        return {"name": name, "kind": kind, "error": str(exc)}
    return {
        "name": name,
        "kind": kind,
        "target_index": target_index,
        "raw": raw,
        "note": "",
    }


def _raw_on(fit: dict[str, Any], common: np.ndarray) -> np.ndarray:
    lookup = {
        int(index): row
        for index, row in zip(fit["target_index"], fit["raw"], strict=True)
    }
    return np.vstack([lookup[int(index)] for index in common])


def _predictor_row(
    name: str,
    kind: str,
    tickets: np.ndarray,
    actual: np.ndarray,
    *,
    raw: np.ndarray | None,
    note: str,
    dataset_id: str | None,
) -> dict[str, Any]:
    card = close_call_scorecard(tickets, actual)
    actual_powerball = actual[:, WHITE_BALL_COUNT].astype(np.float64)
    if raw is not None:
        powerball_mae = float(
            np.mean(np.abs(raw[:, WHITE_BALL_COUNT] - actual_powerball))
        )
        powerball_rmse = float(
            np.sqrt(np.mean((raw[:, WHITE_BALL_COUNT] - actual_powerball) ** 2))
        )
        raw_sum = float(np.mean(raw_sum_abs_error(raw, actual[:, :WHITE_BALL_COUNT])))
    else:
        powerball_mae = float(card["mean_powerball_abs_error"])
        powerball_rmse = powerball_mae
        raw_sum = float(card["white_sum_mae"])
    return {
        **card,
        "name": name,
        "kind": kind,
        "dataset_id": dataset_id,
        "powerball_mae": powerball_mae,
        "powerball_rmse": powerball_rmse,
        "raw_white_sum_mae": raw_sum,
        "note": note,
        "dataset_signal": False,
        "alias_of": None,
    }


def _per_draw(
    tickets: np.ndarray, actual: np.ndarray, raw: np.ndarray | None
) -> dict[str, np.ndarray]:
    parts = score_tickets(tickets, actual)
    matched = parts["whites_matched"]
    return {
        "white_sum_abs_error": parts["white_sum_abs_error"],
        "at_least_3": (matched >= 3).astype(np.float64),
        "at_least_4": (matched >= 4).astype(np.float64),
        "within_1": parts["within_1"].astype(np.float64),
        "powerball_matched": parts["powerball_matched"].astype(np.float64),
        "raw_white_sum_abs_error": (
            raw_sum_abs_error(raw, actual[:, :WHITE_BALL_COUNT])
            if raw is not None
            else parts["white_sum_abs_error"]
        ),
    }


def _predictive_tests(
    scored: list[dict[str, Any]],
    per_draw: dict[str, Any],
    chance: Any,
    alpha: float,
) -> list[dict[str, Any]]:
    del alpha
    tests: list[dict[str, Any]] = []
    actual_n = int(per_draw["actual"].shape[0])
    for row in scored:
        if row.get("alias_of"):
            continue
        n = int(row["n"])
        if n != actual_n:
            raise ValueError("predictor length does not match the scored holdout")
        specs = (
            ("rate_at_least_2", int(row["n_at_least_2"]), chance.rate_at_least_2),
            ("rate_at_least_3", int(row["n_at_least_3"]), chance.rate_at_least_3),
            ("rate_at_least_4", int(row["n_at_least_4"]), chance.rate_at_least_4),
            (
                "powerball_match_rate",
                int(row["powerball_exact_matches"]),
                chance.powerball_exact_rate,
            ),
        )
        for metric, successes, p0 in specs:
            tests.append(
                _predictive_record(
                    str(row["name"]),
                    metric + "_vs_chance",
                    binomial_two_sided_p(successes, n, p0),
                    float(row[metric]) - p0,
                    higher_is_better=True,
                )
            )
        for metric, expected, se in (
            ("mean_within_1", chance.mean_within_1, chance.within_1_se),
            ("mean_within_2", chance.mean_within_2, chance.within_2_se),
            ("white_sum_mae", chance.white_sum_mae, chance.white_sum_mae_se),
        ):
            observed = float(row[metric])
            if se <= 0.0:
                p_value = 1.0
            else:
                z_value = (observed - expected) / se
                p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
            tests.append(
                _predictive_record(
                    str(row["name"]),
                    metric + "_vs_chance",
                    p_value,
                    observed - expected,
                    higher_is_better=metric != "white_sum_mae",
                )
            )
    if "Historical mean" not in per_draw["rows"]:
        return tests
    mean_rows = per_draw["rows"]["Historical mean"]
    plain_rows = per_draw["rows"].get("Plain transformer")
    comparable = (
        "white_sum_abs_error",
        "at_least_3",
        "at_least_4",
        "within_1",
        "powerball_matched",
    )
    higher_is_better = {
        "white_sum_abs_error": False,
        "at_least_3": True,
        "at_least_4": True,
        "within_1": True,
        "powerball_matched": True,
    }
    for row in scored:
        if row.get("alias_of") or row["kind"] not in {"dataset", "combined", "plain"}:
            continue
        left = per_draw["rows"][row["name"]]
        references = [("historical_mean", mean_rows)]
        if row["kind"] != "plain" and plain_rows is not None:
            references.append(("plain", plain_rows))
        for metric in comparable:
            for reference_name, reference in references:
                diff = left[metric] - reference[metric]
                tests.append(
                    _predictive_record(
                        str(row["name"]),
                        f"{metric}_vs_{reference_name}",
                        paired_difference_p(diff),
                        float(np.mean(diff)),
                        higher_is_better=higher_is_better[metric],
                    )
                )
    return tests


def _predictive_record(
    name: str,
    target: str,
    p_value: float,
    effect: float,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    return {
        "family": "predictive",
        "kind": "paired"
        if "_vs_historical_mean" in target or "_vs_plain" in target
        else "chance",
        "series_id": name,
        "dataset_id": None,
        "target": target,
        "n": None,
        "pearson_r": None,
        "mutual_info_bits": None,
        "p": float(min(1.0, max(0.0, p_value))),
        "effect": float(effect),
        "higher_is_better": higher_is_better,
        "permutation_p": None,
        "permutation_exceedances": None,
    }


def _apply_corrections(
    association: list[dict[str, Any]],
    predictive: list[dict[str, Any]],
    alpha: float,
    permutation_floor: float,
) -> None:
    del permutation_floor
    pooled = association + predictive
    if not pooled:
        return
    p_values = np.array([float(row["p"]) for row in pooled], dtype=np.float64)
    bh = benjamini_hochberg(p_values)
    bonferroni = bonferroni_adjust(p_values)
    family_adjusted = _family_adjustments(pooled)
    for index, row in enumerate(pooled):
        row["p_bh"] = float(bh[index])
        row["p_bonferroni"] = float(bonferroni[index])
        row["p_bh_family"] = family_adjusted[index][0]
        row["p_bonferroni_family"] = family_adjusted[index][1]
        perm = row.get("permutation_p")
        confirmed = perm is None or float(perm) <= alpha
        row["survives_bh"] = bool(row["p_bh"] <= alpha and confirmed)
        row["survives_bonferroni"] = bool(row["p_bonferroni"] <= alpha and confirmed)


def _family_adjustments(
    pooled: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    adjusted = [(1.0, 1.0)] * len(pooled)
    for family in ("association", "predictive"):
        indexes = [index for index, row in enumerate(pooled) if row["family"] == family]
        if not indexes:
            continue
        p_values = np.array(
            [float(pooled[index]["p"]) for index in indexes], dtype=np.float64
        )
        bh = benjamini_hochberg(p_values)
        bonferroni = bonferroni_adjust(p_values)
        for position, index in enumerate(indexes):
            adjusted[index] = (float(bh[position]), float(bonferroni[position]))
    return adjusted


def _mark_dataset_signals(
    scored: list[dict[str, Any]], predictive: list[dict[str, Any]], alpha: float
) -> None:
    metrics = (
        "white_sum_abs_error",
        "at_least_3",
        "at_least_4",
        "within_1",
        "powerball_matched",
    )
    by_key = {
        (str(row["series_id"]), str(row["target"])): row
        for row in predictive
        if row["kind"] == "paired"
    }
    for predictor in scored:
        if predictor["kind"] not in {"dataset", "combined"} or predictor.get(
            "alias_of"
        ):
            continue
        name = str(predictor["name"])
        for metric in metrics:
            against_mean = by_key.get((name, f"{metric}_vs_historical_mean"))
            against_plain = by_key.get((name, f"{metric}_vs_plain"))
            if against_mean is None or against_plain is None:
                continue
            if not against_mean.get("survives_bh") or not against_plain.get(
                "survives_bh"
            ):
                continue
            if (
                float(against_mean["p_bh"]) > alpha
                or float(against_plain["p_bh"]) > alpha
            ):
                continue
            if not _beneficial(against_mean) or not _beneficial(against_plain):
                continue
            predictor["dataset_signal"] = True
            predictor["signal_metric"] = metric
            break
    for predictor in scored:
        alias = predictor.get("alias_of")
        if not alias:
            continue
        source = next(row for row in scored if row.get("dataset_id") == alias)
        predictor["dataset_signal"] = bool(source.get("dataset_signal"))
        if source.get("signal_metric"):
            predictor["signal_metric"] = source["signal_metric"]


def _beneficial(test: dict[str, Any]) -> bool:
    effect = float(test["effect"])
    if test.get("higher_is_better"):
        return effect > 0.0
    return effect < 0.0


def _dataset_row(load: DatasetLoad, aligned: list[dict[str, Any]]) -> dict[str, Any]:
    match = next(
        (item for item in aligned if item["dataset_id"] == load.dataset_id), None
    )
    return {
        "dataset_id": load.dataset_id,
        "group": load.group,
        "description": load.description,
        "available": bool(match is not None and load.available),
        "detail": load.detail
        if match is None
        else f"{load.detail}; alignment {match['rule']}",
        "availability": load.availability,
        "columns": list(match["columns"]) if match else [],
        "source": load.source,
    }


def _chance_dict(chance: Any) -> dict[str, Any]:
    payload = {
        "rate_at_least_2": chance.rate_at_least_2,
        "rate_at_least_3": chance.rate_at_least_3,
        "rate_at_least_4": chance.rate_at_least_4,
        "rate_exact_whites": chance.rate_exact_whites,
        "powerball_exact_rate": chance.powerball_exact_rate,
        "full_ticket_exact_rate": chance.full_ticket_exact_rate,
        "mean_within_1": chance.mean_within_1,
        "mean_within_2": chance.mean_within_2,
        "mean_near_miss_1": chance.mean_near_miss_1,
        "mean_near_miss_2": chance.mean_near_miss_2,
        "within_1_se": chance.within_1_se,
        "within_2_se": chance.within_2_se,
        "powerball_within_1_rate": chance.powerball_within_1_rate,
        "powerball_within_2_rate": chance.powerball_within_2_rate,
        "white_sum_mae": chance.white_sum_mae,
        "white_sum_mae_se": chance.white_sum_mae_se,
        "white_pmf": {str(key): value for key, value in chance.white_pmf.items()},
    }
    return payload


def _summary_dict(
    tests: list[dict[str, Any]],
    family: str,
    alpha: float,
    permutation_floor: float | None,
) -> dict[str, Any]:
    if family == "combined":
        rows = tests
    else:
        rows = [row for row in tests if row["family"] == family]
    if not rows:
        summary = summarize_corrections(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            family=family,
            alpha=alpha,
            permutation_floor=permutation_floor,
        )
        return summary.__dict__
    p_values = np.array([float(row["p"]) for row in rows], dtype=np.float64)
    if "p_bh_family" in rows[0] and family != "combined":
        bh = np.array([float(row["p_bh_family"]) for row in rows], dtype=np.float64)
        bonferroni = np.array(
            [float(row["p_bonferroni_family"]) for row in rows], dtype=np.float64
        )
    else:
        bh = np.array([float(row["p_bh"]) for row in rows], dtype=np.float64)
        bonferroni = np.array(
            [float(row["p_bonferroni"]) for row in rows], dtype=np.float64
        )
    summary = summarize_corrections(
        p_values,
        bh,
        bonferroni,
        family=family,
        alpha=alpha,
        permutation_floor=permutation_floor,
    )
    return summary.__dict__


def _output_path(cfg: dict[str, Any], key: str) -> Path | None:
    output = cfg.get("output") or {}
    value = output.get(key)
    if not value:
        return None
    return resolve_repo_path(str(value))


def _write_outputs(
    cfg: dict[str, Any], report: dict[str, Any], markdown: str, chart: str
) -> None:
    payload = to_jsonable(
        {key: value for key, value in report.items() if key != "markdown"}
    )
    json_path = _output_path(cfg, "json")
    markdown_path = _output_path(cfg, "markdown")
    chart_path = _output_path(cfg, "chart")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown)
    if chart_path is not None:
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path.write_text(chart)
    _log(cfg, "Wrote the lottery dataset study report")


def _log(cfg: dict[str, Any], message: str) -> None:
    if cfg.get("verbose", True):
        print(message, flush=True)
