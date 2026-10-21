"""Markdown, SVG, and the plain-English conclusion for the lottery study."""

from __future__ import annotations

import html
import math
from typing import Any

import numpy as np


def conclusion_text(report: dict[str, Any]) -> str:
    """Say what the comparison showed, in ordinary language."""
    association = report["corrections"]["association"]
    predictive = report["corrections"]["predictive"]
    combined = report["corrections"]["combined"]
    available = [row["dataset_id"] for row in report["datasets"] if row["available"]]
    missing = [row["dataset_id"] for row in report["datasets"] if not row["available"]]
    signals = [row for row in report["predictors"] if row.get("dataset_signal")]
    association_hits = [
        row for row in report["association_tests"] if row.get("survives_bh")
    ]
    best = _strongest_association(report["association_tests"])
    lines = [
        (
            "This study asks whether any series Neural Forge can train on "
            "moves with Powerball draws. The draw is at 10:59 pm Eastern. "
            "Each series contributes only the last value that would already "
            "have been known at that moment, and the models are scored on a "
            "later block of draws they were not fit on."
        ),
        _data_sentence(available, missing),
        _luck_sentence(association, predictive, combined),
    ]
    if best is None:
        lines.append(
            "There was no usable association test. With no external series "
            "aligned to the draws, there is nothing here that could be a signal."
        )
    else:
        lines.append(
            f"The strongest Pearson correlation was {best['pearson_r']:+.3f} "
            f"between {best['series_id']} and {best['target']} "
            f"(raw p {best['p']:.3g}, Benjamini–Hochberg p {best['p_bh']:.3g})."
        )
    if association_hits:
        names = ", ".join(
            f"{row['series_id']} vs {row['target']}" for row in association_hits
        )
        lines.append(
            "These association tests still sat under the Benjamini–Hochberg "
            f"line after the full set of comparisons: {names}. "
            "That is a statistical threshold on this one sample, not a reason "
            "to buy a ticket."
        )
    else:
        lines.append(
            "No association survived Benjamini–Hochberg or Bonferroni. "
            "That includes the white-ball sum, the Powerball, the odd count, "
            "each individual ball, near-repeats (a draw sharing 3 or 4 white "
            "balls with the draw before it), and neighbor numbers (a ball one "
            "away from the previous draw, or a consecutive pair inside the draw)."
        )
    lines.append(_close_call_sentence(report, signals))
    predictive_hits = [
        row for row in report.get("predictive_tests", []) if row.get("survives_bh")
    ]
    sum_hits = [
        str(row["series_id"])
        for row in predictive_hits
        if row.get("target") == "white_sum_mae_vs_chance"
    ]
    other_predictive_hits = [
        row for row in predictive_hits if row.get("target") != "white_sum_mae_vs_chance"
    ]
    if not signals and not association_hits and not other_predictive_hits:
        if sum_hits:
            names = ", ".join(sum_hits)
            lines.append(
                "The only rows that stay under the Benjamini–Hochberg line are "
                f"white-ball sum errors beating a random ticket: {names}. "
                "A forecast that sits near the average sum will do that, and the "
                "historical mean does it with no outside data at all. It is not "
                "evidence that a dataset predicts the draw. No association survived, "
                "and no dataset model beat both the plain transformer and the "
                "historical-mean ticket after the correction. Close calls — "
                "matching 2, 3, or 4 whites, or landing within 1 or 2 of a drawn "
                "ball — sit on the chance rates. The draws still look independent. "
                "Re-run this study after the other fetchers merge; the registry "
                "will pick them up without a new design."
            )
        else:
            lines.append(
                "Nothing here beats luck once the number of comparisons is taken "
                "into account. The draws still look like independent lottery draws. "
                "Close calls are common for a random ticket, and landing nearer "
                "the average white-ball sum is what any forecast that hugs the "
                "historical average will do. Re-run this study after the other "
                "fetchers merge; the registry will pick them up without a new design."
            )
    else:
        lines.append(
            "Treat any surviving row as a lead for a pre-registered repeat, "
            "not as evidence that an outside series predicts Powerball. "
            "The rest of the table is what chance and the historical average produce."
        )
    return "\n\n".join(lines)


def headline_table(report: dict[str, Any]) -> str:
    """Markdown table of played tickets, ranked by white-ball sum error."""
    chance = report["chance"]
    lines = [
        (
            "| Rank | Predictor | PB exact | PB MAE | Sum MAE | "
            "Mean whites | 2+ | 3+ | 4+ | Within ±1 | Within ±2 | Signal |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ranked = sorted(
        report["predictors"],
        key=lambda row: (float(row["white_sum_mae"]), -float(row["rate_at_least_3"])),
    )
    for rank, row in enumerate(ranked, start=1):
        signal = "yes" if row.get("dataset_signal") else "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    _md(str(row["name"])),
                    _rate(
                        row["powerball_match_rate"],
                        row["powerball_exact_matches"],
                        row["n"],
                    ),
                    f"{float(row['powerball_mae']):.3f}",
                    f"{float(row['white_sum_mae']):.2f}",
                    f"{float(row['mean_whites_matched']):.3f}",
                    _pct(row["rate_at_least_2"]),
                    _pct(row["rate_at_least_3"]),
                    _pct(row["rate_at_least_4"]),
                    f"{float(row['mean_within_1']):.2f}",
                    f"{float(row['mean_within_2']):.2f}",
                    signal,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Chance for a uniform ticket on this holdout: "
        f"Powerball exact {_pct(chance['powerball_exact_rate'])}, "
        f"2+ {_pct(chance['rate_at_least_2'])}, "
        f"3+ {_pct(chance['rate_at_least_3'])}, "
        f"4+ {_pct(chance['rate_at_least_4'])}, "
        f"mean whites within ±1 {float(chance['mean_within_1']):.2f}, "
        f"within ±2 {float(chance['mean_within_2']):.2f}, "
        f"white-sum MAE {float(chance['white_sum_mae']):.2f}."
    )
    return "\n".join(lines)


def association_table(report: dict[str, Any], limit: int = 15) -> str:
    """The association rows with the smallest Benjamini–Hochberg p-values."""
    rows = [
        row
        for row in report["association_tests"]
        if row.get("p") is not None and row.get("kind") != "skipped"
    ]
    rows.sort(key=lambda row: (float(row["p_bh"]), float(row["p"]), -abs(_r(row))))
    lines = [
        "| Rank | Series | Target | Test | r | MI bits | p | BH | Bonferroni | Perm p | Survives |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    shown = rows[:limit]
    survivors = [row for row in rows if row.get("survives_bh")]
    for extra in survivors:
        if extra not in shown:
            shown.append(extra)
    if not shown:
        return "No association tests were run."
    for rank, row in enumerate(shown, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    _md(str(row["series_id"])),
                    _md(str(row["target"])),
                    _md(str(row["kind"])),
                    _fmt_r(row),
                    _fmt_optional(row.get("mutual_info_bits")),
                    f"{float(row['p']):.3g}",
                    f"{float(row['p_bh']):.3g}",
                    f"{float(row['p_bonferroni']):.3g}",
                    _fmt_optional(row.get("permutation_p")),
                    "yes" if row.get("survives_bh") else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    """Full written report."""
    dataset_lines = []
    for row in report["datasets"]:
        state = "available" if row["available"] else "TODO"
        columns = ", ".join(row["columns"]) if row["columns"] else "—"
        dataset_lines.append(
            f"- **{row['dataset_id']}** ({state}, {row['group']}): {row['detail']} "
            f"Columns: {columns}. {row['description']}"
        )
    corrections = report["corrections"]
    return "\n".join(
        [
            "# Powerball dataset comparison",
            "",
            report["conclusion"],
            "",
            "## Ranked tickets",
            "",
            report["headline_table"],
            "",
            "## Strongest associations",
            "",
            association_table(report),
            "",
            "## Multiple comparisons",
            "",
            _correction_line(corrections["association"]),
            _correction_line(corrections["predictive"]),
            _correction_line(corrections["combined"]),
            "",
            (
                "A dataset signal in the ticket table requires a paired improvement "
                "on the same metric against both the plain transformer and the "
                "historical-mean ticket, with the Benjamini–Hochberg p-value from "
                "the combined family at or below alpha. Beating a random ticket "
                "is not enough: the historical mean already gets a closer white-ball "
                "sum, and more Powerball exact hits, by sitting near the middle."
            ),
            "",
            "## Datasets",
            "",
            "\n".join(dataset_lines),
            "",
            "## Definitions",
            "",
            (
                "White-ball sum, Powerball, and odd count are properties of the "
                "draw being predicted. Odd count is the number of odd white balls. "
                "Individual ball rows are indicators that the number was drawn. "
                "A near-repeat shares 3 or 4 white balls with the previous draw. "
                "A neighbor event is a white ball whose nearest ball in the previous "
                "draw is exactly one away (an exact repeat is not a neighbor), "
                "or two consecutive whites in the same draw. "
                "Within ±1 and ±2 count predicted whites whose nearest drawn white "
                "is at most that far away, so exact matches are included. "
                "Near-miss counts in the JSON are the same comparison at distance "
                "exactly 1 or exactly 2. Partial-match chance rates are hypergeometric. "
                "The chart shows white-sum error against the random-ticket baseline "
                "and the smallest association p-values."
            ),
            "",
            f"Chart: `{report['chart_path']}`" if report.get("chart_path") else "",
            "",
        ]
    )


def render_chart(report: dict[str, Any]) -> str:
    """Two-panel SVG: ticket sum error, and the strongest association p-values."""
    predictors = sorted(
        report["predictors"], key=lambda row: float(row["white_sum_mae"])
    )
    associations = [
        row
        for row in report["association_tests"]
        if row.get("kind") == "pearson" and row.get("pearson_r") is not None
    ]
    associations.sort(key=lambda row: float(row["p"]))
    associations = associations[:12]
    width = 960
    height = 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 960 720">',
        '<rect width="960" height="720" fill="#fbfaf6"/>',
        '<text x="32" y="36" font-family="Georgia, serif" font-size="22" fill="#1c1915">'
        "Powerball dataset study</text>",
        '<text x="32" y="58" font-family="Georgia, serif" font-size="13" fill="#5c564c">'
        "White-ball sum error (lower is closer) and the strongest correlations</text>",
    ]
    parts.append(
        _bar_panel(
            predictors,
            x=32,
            y=80,
            width=900,
            height=300,
            value_of=lambda row: float(row["white_sum_mae"]),
            label_of=lambda row: str(row["name"]),
            title="Played-ticket white-sum MAE",
            reference=float(report["chance"]["white_sum_mae"]),
            reference_label="random ticket",
        )
    )
    if associations:
        floor = 1e-6
        parts.append(
            _bar_panel(
                associations,
                x=32,
                y=400,
                width=900,
                height=280,
                value_of=lambda row: -math.log10(max(float(row["p"]), floor)),
                label_of=lambda row: f"{row['series_id']} · {row['target']}",
                title="-log10 Pearson p (taller is less likely under no relationship)",
                reference=-math.log10(0.05),
                reference_label="raw p = 0.05",
            )
        )
    else:
        parts.append(
            '<text x="32" y="460" font-family="Georgia, serif" font-size="14" fill="#5c564c">'
            "No association tests."
            "</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def to_jsonable(value: Any) -> Any:
    """Convert numpy containers into plain JSON types."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer | np.bool_):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _bar_panel(
    rows: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    value_of: Any,
    label_of: Any,
    title: str,
    reference: float,
    reference_label: str,
) -> str:
    values = [float(value_of(row)) for row in rows] or [0.0]
    peak = max(max(values), reference, 1e-9)
    label_width = 260
    plot_x = x + label_width
    plot_width = width - label_width - 16
    row_height = min(28, (height - 36) / max(len(rows), 1))
    parts = [
        f'<text x="{x}" y="{y + 16}" font-family="Georgia, serif" font-size="15" '
        f'fill="#1c1915">{html.escape(title)}</text>'
    ]
    baseline = plot_x + plot_width * (reference / peak)
    parts.append(
        f'<line x1="{baseline:.1f}" y1="{y + 28}" x2="{baseline:.1f}" '
        f'y2="{y + 28 + row_height * max(len(rows), 1):.1f}" stroke="#b4532a" '
        'stroke-dasharray="4 3"/>'
    )
    parts.append(
        f'<text x="{min(baseline + 4, x + width - 120):.1f}" y="{y + 26}" '
        'font-family="Georgia, serif" font-size="11" fill="#b4532a">'
        f"{html.escape(reference_label)}</text>"
    )
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        top = y + 32 + index * row_height
        bar = plot_width * (value / peak)
        parts.append(
            f'<text x="{x}" y="{top + 12:.1f}" font-family="Georgia, serif" '
            f'font-size="12" fill="#1c1915">{html.escape(str(label_of(row))[:42])}</text>'
        )
        parts.append(
            f'<rect x="{plot_x}" y="{top:.1f}" width="{bar:.1f}" height="{row_height - 8:.1f}" '
            'fill="#2f5d50"/>'
        )
    return "\n".join(parts)


def _data_sentence(available: list[str], missing: list[str]) -> str:
    have = ", ".join(available) if available else "no external dataset"
    waiting = ", ".join(missing) if missing else "nothing"
    return (
        f"Series actually compared: {have}. "
        f"Still waiting on a fetcher, so not in the numbers: {waiting}."
    )


def _luck_sentence(
    association: dict[str, Any], predictive: dict[str, Any], combined: dict[str, Any]
) -> str:
    floor = association.get("permutation_floor")
    floor_text = (
        f" Permutation p-values cannot go below {floor:.4f} with this shuffle count."
        if floor
        else ""
    )
    return (
        f"The association family had {association['n_tests']} tests. "
        f"If none of the series had anything to do with the lottery, and the "
        f"tests were independent, about {association['expected_luck_hits']:.1f} "
        f"of them would fall under p = {association['alpha']:.2f} by luck "
        f"(dependence makes the usual count smaller). "
        f"{association['n_raw_below_alpha']} did. "
        f"Benjamini–Hochberg kept {association['n_bh_rejects']}; "
        f"Bonferroni kept {association['n_bonferroni_rejects']}. "
        f"The ticket-scoring family had {predictive['n_tests']} tests and "
        f"would expect about {predictive['expected_luck_hits']:.1f} raw hits; "
        f"it produced {predictive['n_raw_below_alpha']} raw, "
        f"{predictive['n_bh_rejects']} after Benjamini–Hochberg, and "
        f"{predictive['n_bonferroni_rejects']} after Bonferroni. "
        f"Pooling every test ({combined['n_tests']}) raises the luck expectation "
        f"to about {combined['expected_luck_hits']:.1f} raw hits." + floor_text
    )


def _close_call_sentence(report: dict[str, Any], signals: list[dict[str, Any]]) -> str:
    chance = report["chance"]
    mean = next(row for row in report["predictors"] if row["kind"] == "historical_mean")
    plain = next((row for row in report["predictors"] if row["kind"] == "plain"), mean)
    text = (
        f"On the holdout, a random ticket is expected to match 2 or more whites "
        f"about {_pct(chance['rate_at_least_2'])} of the time, 3 or more "
        f"{_pct(chance['rate_at_least_3'])}, and 4 or more "
        f"{_pct(chance['rate_at_least_4'])}. "
        f"The plain transformer matched 3 or more on {_pct(plain['rate_at_least_3'])} "
        f"of draws and its played white-sum MAE was {plain['white_sum_mae']:.2f}. "
        f"The historical-mean ticket was at {mean['white_sum_mae']:.2f}. "
        f"Chance white-sum MAE on these draws is {chance['white_sum_mae']:.2f}. "
        f"Powerball exact-match chance is {_pct(chance['powerball_exact_rate'])}; "
        f"the plain model hit {_pct(plain['powerball_match_rate'])} and the "
        f"historical mean hit {_pct(mean['powerball_match_rate'])}."
    )
    if signals:
        names = ", ".join(str(row["name"]) for row in signals)
        text += f" Dataset rows that cleared the paired-test bar: {names}."
    else:
        text += (
            " No dataset-trained model beat both the plain model and the "
            "historical-mean ticket on a pre-specified close-call metric after "
            "the multiple-comparison correction."
        )
    return text


def _correction_line(summary: dict[str, Any]) -> str:
    return (
        f"- **{summary['family']}**: {summary['n_tests']} tests, "
        f"expected luck hits {summary['expected_luck_hits']:.2f}, "
        f"raw p ≤ {summary['alpha']:.2f}: {summary['n_raw_below_alpha']}, "
        f"Benjamini–Hochberg: {summary['n_bh_rejects']}, "
        f"Bonferroni: {summary['n_bonferroni_rejects']} "
        f"(cutoff {summary['bonferroni_cutoff']:.3g})."
    )


def _strongest_association(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    pearson = [
        row
        for row in rows
        if row.get("kind") == "pearson"
        and isinstance(row.get("pearson_r"), int | float)
    ]
    if not pearson:
        return None
    return max(pearson, key=lambda row: abs(float(row["pearson_r"])))


def _r(row: dict[str, Any]) -> float:
    value = row.get("pearson_r")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _fmt_r(row: dict[str, Any]) -> str:
    value = row.get("pearson_r")
    if not isinstance(value, int | float):
        return "—"
    return f"{float(value):+.3f}"


def _fmt_optional(value: Any) -> str:
    if not isinstance(value, int | float):
        return "—"
    return f"{float(value):.3g}"


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _rate(rate: Any, matches: Any, n: Any) -> str:
    return f"{float(matches):.0f}/{float(n):.0f} ({_pct(rate)})"


def _md(text: str) -> str:
    return text.replace("|", "\\|")
