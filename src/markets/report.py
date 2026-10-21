"""Tables and plain-language summaries for the market evaluation.

The wording is driven by the measured ratios. A median RMSE at or above the
no-change forecast is described as not beating a random walk, which is the
usual result for these series.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DISCLAIMER = (
    "Not financial advice. This is an educational evaluation of a small "
    "transformer against naive baselines on past prices. It is not a forecast "
    "you should trade on, and it is not a claim about future returns."
)

GROUP_ORDER = ("stocks", "fx", "crypto", "commodities")

GROUP_TITLES = {
    "stocks": "Stocks and the S&P 500",
    "fx": "Currencies (per 1 USD)",
    "crypto": "Cryptocurrencies",
    "commodities": "Commodities",
}


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty group")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median skill of one asset class. Only successful training rows count."""
    if not rows:
        return {
            "n": 0,
            "n_beat_persistence": 0,
            "n_beat_mean": 0,
            "median_rmse_ratio_vs_persistence": None,
            "median_rmse_ratio_vs_mean": None,
            "median_directional_accuracy": None,
            "median_mean_directional_accuracy": None,
            "median_positive_fraction": None,
            "train_seconds": 0.0,
        }
    ratios = [float(row["rmse_ratio_vs_persistence"]) for row in rows]
    mean_ratios = [float(row["rmse_ratio_vs_mean"]) for row in rows]
    directions = [float(row["directional_accuracy"]) for row in rows]
    mean_directions = [float(row["mean_directional_accuracy"]) for row in rows]
    positives = [float(row["positive_fraction"]) for row in rows]
    return {
        "n": len(rows),
        "n_beat_persistence": sum(bool(row["beats_persistence_rmse"]) for row in rows),
        "n_beat_mean": sum(bool(row["beats_mean_rmse"]) for row in rows),
        "median_rmse_ratio_vs_persistence": _median(ratios),
        "median_rmse_ratio_vs_mean": _median(mean_ratios),
        "median_directional_accuracy": _median(directions),
        "median_mean_directional_accuracy": _median(mean_directions),
        "median_positive_fraction": _median(positives),
        "train_seconds": float(
            sum(float(row.get("train_seconds") or 0.0) for row in rows)
        ),
    }


def skill_sentence(summary: dict[str, Any]) -> str:
    """One sentence on whether the typical series beat a no-change forecast."""
    n = int(summary["n"])
    if n == 0:
        return "No series in this group were trained."
    ratio = float(summary["median_rmse_ratio_vs_persistence"])
    direction = float(summary["median_directional_accuracy"])
    beat = int(summary["n_beat_persistence"])
    direction_pct = 100.0 * direction
    if ratio < 0.98 and direction > 0.52:
        base = (
            f"The median RMSE was {ratio:.3f} times persistence and the median "
            f"directional accuracy was {direction_pct:.1f}%."
        )
    elif ratio >= 1.0:
        base = (
            f"The typical series did not beat a no-change forecast "
            f"(median RMSE {ratio:.3f} times persistence, {beat} of {n} series lower). "
            f"Median directional accuracy was {direction_pct:.1f}%."
        )
    else:
        base = (
            f"{beat} of {n} series had a lower RMSE than persistence, but the median "
            f"ratio was {ratio:.3f} and median directional accuracy was {direction_pct:.1f}%."
        )
    mean_direction = summary.get("median_mean_directional_accuracy")
    if mean_direction is not None and float(mean_direction) > direction:
        base += (
            " The sign of the training-set mean return matched the holdout more "
            f"often (median {100.0 * float(mean_direction):.1f}%) than the model did."
        )
    return base


def group_lead(group: str, summary: dict[str, Any]) -> str:
    """Fixed claim plus the measured sentence. FX states the random-walk result."""
    ratio = summary.get("median_rmse_ratio_vs_persistence")
    direction = summary.get("median_directional_accuracy")
    strong = (
        ratio is not None
        and direction is not None
        and float(ratio) < 0.95
        and float(direction) > 0.55
    )
    if group == "fx":
        if strong:
            lead = (
                "On this split the FX panel beat a no-change forecast by more than "
                "a rounding error. Treat that as a property of the holdout, not as "
                "a reason to trade."
            )
        else:
            lead = "FX reference rates are close to a random walk."
    elif group == "crypto":
        lead = (
            "Daily crypto closes are volatile, and a useful forecast still has to "
            "beat predicting that tomorrow's return is zero."
        )
    elif group == "commodities":
        lead = (
            "Commodity results use the same return baseline. Front-month futures "
            "include contract rolls, so they are not a pure spot series."
        )
    else:
        lead = (
            "Equity and index results use the same return baseline. A single "
            "recent holdout is not evidence of a tradable edge."
        )
    if int(summary.get("n") or 0) == 0:
        return lead
    return f"{lead} {skill_sentence(summary)}"


def _fmt(value: float | None, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _cell(value: object) -> str:
    return str(value).replace("|", "/")


def summary_table(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Group | Series | Beat persistence | Beat mean return | Median RMSE / persistence | Median RMSE / mean | Median directional accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUP_ORDER:
        summary = summaries.get(group)
        if not summary:
            continue
        lines.append(
            "| {title} | {n} | {beat_p}/{n} | {beat_m}/{n} | {ratio} | {mean_ratio} | {direction} |".format(
                title=GROUP_TITLES[group],
                n=summary["n"],
                beat_p=summary["n_beat_persistence"],
                beat_m=summary["n_beat_mean"],
                ratio=_fmt(summary["median_rmse_ratio_vs_persistence"], 3),
                mean_ratio=_fmt(summary["median_rmse_ratio_vs_mean"], 3),
                direction=_pct(summary["median_directional_accuracy"]),
            )
        )
    return "\n".join(lines)


def metrics_table(rows: list[dict[str, Any]]) -> str:
    """Per-series holdout scores. RMSE is in daily (or monthly) log-return units."""
    lines = [
        "| Series | Source | Coverage | Holdout | N | Model RMSE | Persistence RMSE | Mean RMSE | Dir. acc. |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("group")),
            int(row.get("rank") or 10**9),
            str(row["id"]),
        ),
    )
    for row in ordered:
        coverage = f"{row.get('start')} → {row.get('end')}"
        holdout = f"{row.get('val_start')} → {row.get('val_end')}"
        lines.append(
            "| {name} | {source} | {coverage} | {holdout} | {n} | {model} | {persist} | {mean} | {direction} |".format(
                name=_cell(row.get("name")),
                source=_cell(row.get("source")),
                coverage=coverage,
                holdout=holdout,
                n=row.get("n_val"),
                model=_fmt(row.get("model_rmse")),
                persist=_fmt(row.get("persistence_rmse")),
                mean=_fmt(row.get("mean_rmse")),
                direction=_pct(row.get("directional_accuracy")),
            )
        )
    return "\n".join(lines)


def coverage_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Series | Source | Start | End | Observations | Trained | Note |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("group")),
            int(row.get("rank") or 10**9),
            str(row["id"]),
        ),
    )
    for row in ordered:
        lines.append(
            "| {name} | {source} | {start} | {end} | {n} | {trained} | {note} |".format(
                name=_cell(row.get("name")),
                source=_cell(row.get("source")),
                start=row.get("start") or "",
                end=row.get("end") or "",
                n=row.get("observations") or 0,
                trained="yes" if row.get("train") else "no",
                note=_cell(row.get("skip_reason") or row.get("quote") or ""),
            )
        )
    return "\n".join(lines)


def _largest_gap_note(rows: list[dict[str, Any]]) -> str:
    """Explain the biggest persistence gap so a single drift is not read as skill."""
    if not rows:
        return "No series were scored."
    best = min(rows, key=lambda row: float(row["rmse_ratio_vs_persistence"]))
    ratio = float(best["rmse_ratio_vs_persistence"])
    if ratio >= 0.99:
        return (
            "No series beat persistence by as much as 1% of RMSE. A count such as "
            "19 of 41 is a split around a tie."
        )
    return (
        f"The largest improvement on persistence is {best.get('name')} "
        f"(RMSE {ratio:.3f} times the zero forecast, directional accuracy "
        f"{_pct(best.get('directional_accuracy'))}). The training-set mean sign hits "
        f"{_pct(best.get('mean_directional_accuracy'))} of that same holdout, and the "
        f"model RMSE is {float(best['rmse_ratio_vs_mean']):.3f} times the historical-mean "
        "forecast. A gap that shrinks once the average drift is the baseline is the "
        "drift itself. The same reading applies when directional accuracy sits next "
        "to the fraction of up days."
    )


def readme_section(results: dict[str, Any], coverage: dict[str, Any]) -> str:
    """Full README section. The numbers come from the training JSON."""
    protocol = results.get("protocol") or {}
    summaries = results.get("summaries") or {}
    trained = results.get("series") or []
    covered = coverage.get("series") or []
    notes = coverage.get("notes") or []
    lines = [
        "## Market forecasts (stocks, currencies, crypto, commodities)",
        "",
        DISCLAIMER,
        "",
        "The model predicts the next **log return**, not the price level. Features are that return and a 10-observation rolling standard deviation. Returns use consecutive published observations only (business days for ECB rates and equity sessions, exchange sessions for futures, calendar days for crypto) and only when both prices are positive. Inputs and the target are z-scored with statistics fit on the training windows. The holdout is the most recent "
        f"{protocol.get('val_split', 0.2):.0%} of windows (`split_mode: temporal`). Early stopping keeps the best validation checkpoint (`patience: {protocol.get('patience')}`, at most `{protocol.get('epochs')}` epochs, seed `{protocol.get('seed')}`). The network is a small transformer (`d_model={protocol.get('d_model')}`, `{protocol.get('layers')}` encoder layer, sequence length `{protocol.get('seq_len')}`).",
        "",
        "Two baselines are scored in those same return units:",
        "",
        "- **Persistence (random walk).** Predict a return of zero. On the price, that is \"tomorrow's price equals today's price.\"",
        "- **Historical mean.** Predict the training-window average return, held constant.",
        "",
        "Directional accuracy is the fraction of holdout observations with a nonzero return whose sign the model matches. A coin flip is 50%. The fraction of up moves in the holdout is the hit rate of always predicting an increase, so a print near that rate is not skill. RMSE below is the root mean squared **log return** on the holdout. A ratio above 1 means the model was worse than that baseline.",
        "",
        "### Sources and terms",
        "",
        "Prices are not committed. `scripts/fetch_markets.py` downloads them into `data/markets/` (gitignored by `/data/`). Checkpoints written by training are gitignored as well.",
        "",
        "- **ECB euro foreign-exchange reference rates**, file `eurofxref-hist.zip`. The file quotes foreign currency units per 1 euro. This project converts each rate to units of that currency per 1 USD using the ECB dollar rate, and stores the euro as EUR per 1 USD. Reuse is allowed with attribution to the European Central Bank. The file is newest-first; the loader sorts it into time order before the split.",
        "- **Frankfurter** (`api.frankfurter.app`) mirrors those ECB rates. The fetch records a same-day cross-check of the USD crosses. It is not a second training source.",
        "- **FRED** `fredgraph.csv` (no API key) is the preferred source for the S&P 500 (`SP500`), WTI (`DCOILWTICO`), Brent (`DCOILBRENTEU`), Henry Hub natural gas (`DHHNGSP`), and the H.10 New Taiwan dollar rate (`DEXTWUS`), which the ECB file does not carry. S&P 500 values are S&P Dow Jones Indices data via FRED. H.10 rates are Board of Governors data. Oil and gas are EIA data via FRED. If those downloads time out, the script says so and uses the Yahoo fallback below where one exists. IMF monthly aluminum (`PALUMUSDM`) is not used: a daily aluminum futures history is available.",
        "- **Stooq** daily CSVs were requested first for equities. The host returned an HTML browser check rather than a file, so equities were not taken from Stooq.",
        "- **Yahoo Finance chart** (no API key) supplies AAPL, MSFT, and NVDA, plus commodity futures (gold `GC=F`, silver `SI=F`, copper `HG=F`, aluminum `ALI=F`, platinum `PL=F`, palladium `PA=F`, and the oil and gas contracts when FRED is unavailable) and the S&P 500 (`^GSPC`) when FRED is unavailable. Adjusted close is used when the payload includes it. Front-month futures include roll gaps. Yahoo's terms apply. The series are not redistributed in this repository.",
        "- **CoinGecko** `coins/markets` (no API key) supplies the market-cap order. If it is unavailable, CoinPaprika tickers are the fallback. Prices are not taken from either ranking source.",
        "- **Binance** public spot klines from `data-api.binance.vision` (no API key), daily, quote asset USDT. `api.binance.com` returned HTTP 451 from this environment, so it is not used. A `1000…USDT` contract is accepted when that is the listed market; its log return matches the token while the scale factor is constant. **Coinbase** exchange candles are the fallback for a ranked coin with no Binance USDT market.",
        "- Stablecoins are skipped. Wrapped and staked duplicates (WBTC, stETH, and the same pattern) are skipped when that still leaves 50 coins with at least two years of daily candles.",
        "",
        "Run notes from the fetch that produced these tables:",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No fetch notes were recorded.")
    lines.extend(
        [
            "",
            "### Fetch and train",
            "",
            "From the repository root, with the project virtualenv:",
            "",
            "```bash",
            "python scripts/fetch_markets.py",
            "python scripts/train_markets.py",
            "```",
            "",
            "Training reads `config/stocks.yaml`, `config/fx.yaml`, `config/crypto.yaml`, and `config/commodities.yaml`. Those files hold the shared model, the temporal split, the seed, and early stopping. The trainer writes one ordinary CSV config per series under `data/markets/runs/` and evaluates the best checkpoint, not the weights from the final epoch. One currency or one coin is one model. Total CPU training time for this run was "
            f"{float(results.get('train_seconds') or 0.0):.0f} seconds.",
            "",
            "### Results",
            "",
            "Holdout metrics are log returns. Persistence RMSE is the RMSE of predicting zero. Mean RMSE predicts the training-set mean return. Directional accuracy is compared with 50% in the text under each table.",
            "",
            summary_table(summaries),
            "",
            _largest_gap_note(trained),
            "",
        ]
    )
    by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUP_ORDER}
    for row in trained:
        by_group.setdefault(str(row.get("group")), []).append(row)
    covered_by_group: dict[str, list[dict[str, Any]]] = {
        group: [] for group in GROUP_ORDER
    }
    for row in covered:
        covered_by_group.setdefault(str(row.get("group")), []).append(row)

    for group in GROUP_ORDER:
        summary = summaries.get(group) or summarize_group([])
        lines.extend(
            [f"#### {GROUP_TITLES[group]}", "", group_lead(group, summary), ""]
        )
        metrics = by_group.get(group) or []
        if metrics:
            lines.extend([metrics_table(metrics), ""])
        else:
            lines.extend(["No series in this group finished training.", ""])
        if group in {"fx", "crypto"}:
            lines.extend(
                [
                    f"Coverage for every {GROUP_TITLES[group].lower()} series the fetch considered:",
                    "",
                    coverage_table(covered_by_group.get(group) or []),
                    "",
                ]
            )

    lines.extend(
        [
            "### Limits",
            "",
            "A lower RMSE than persistence on one holdout, especially by a fraction of a percent, is not a trading edge. The model does not see order flow, news, or funding costs, and the commodity futures are front-month contracts. Discontinued ECB currencies (for example the old Romanian leu, or the ruble after the ECB stopped publishing it) are scored on the last 20% of their own sample, which is not 2026. Crypto coins with less than two years of daily candles are listed and not trained. Pegged currencies (the Danish krone, the Hong Kong dollar, and others) mostly import the dollar's move against the peg partner, so their errors are not independent of the euro or the dollar.",
            "",
            "Retrain with the commands above. Live rankings and the latest ECB file move, so a later run will not reproduce these digits exactly; the seed, the split, and the checkpoint selection are fixed.",
            "",
        ]
    )
    return "\n".join(lines)
