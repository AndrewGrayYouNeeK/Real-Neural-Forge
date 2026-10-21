"""Fetch monthly arXiv submission counts, one request every few seconds.

arXiv asks callers to wait at least three seconds between API requests
(https://info.arxiv.org/help/api/user-manual.html). This script does that,
retries transient failures, and resumes from the long CSV it already wrote.

Counts come from ``opensearch:totalResults`` with ``max_results=1``, so each
month is one small response rather than a page through every paper.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.domain_forecasts.baselines import iter_months
from src.utils.paths import resolve_repo_path

USER_AGENT = (
    "RealNeuralForge/1.0 "
    "(monthly category counts; https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge)"
)
_TOTAL = re.compile(r"<opensearch:totalResults>(\d+)</opensearch:totalResults>")
DEFAULT_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL", "quant-ph")


def month_bounds(month: str) -> tuple[str, str]:
    """Return inclusive arXiv ``submittedDate`` endpoints for one month.

    The end stamp is 23:59 on the last calendar day, so adjacent months do
    not share an instant.
    """
    year_s, month_s = month.split("-")
    year, month_i = int(year_s), int(month_s)
    last_day = calendar.monthrange(year, month_i)[1]
    start = f"{year:04d}{month_i:02d}010000"
    end = f"{year:04d}{month_i:02d}{last_day:02d}2359"
    return start, end


def count_url(category: str, month: str) -> str:
    start, end = month_bounds(month)
    query = f"cat:{category} AND submittedDate:[{start} TO {end}]"
    return "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": 1}
    )


def fetch_count(category: str, month: str, timeout: float) -> int:
    request = urllib.request.Request(
        count_url(category, month), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    match = _TOTAL.search(body)
    if match is None:
        raise RuntimeError(f"no totalResults for {category} {month}")
    return int(match.group(1))


def load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            done.add((row["month"], row["category"]))
    return done


def pivot_wide(long_path: Path, wide_path: Path, categories: list[str]) -> None:
    table: dict[str, dict[str, int]] = {}
    with long_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            table.setdefault(row["month"], {})[row["category"]] = int(row["count"])
    wide_path.parent.mkdir(parents=True, exist_ok=True)
    with wide_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["month", *categories])
        writer.writeheader()
        for month in sorted(table):
            writer.writerow(
                {
                    "month": month,
                    **{cat: table[month].get(cat, "") for cat in categories},
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2012-01")
    parser.add_argument("--end", default="2026-09")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--long-out",
        default="datasets/ai_progress/arxiv_counts_long.csv",
    )
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if args.delay < 3.0:
        raise SystemExit("arXiv asks for at least 3 seconds between requests")

    long_path = resolve_repo_path(args.long_out)
    long_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(long_path)
    write_header = not long_path.exists()
    months = iter_months(args.start, args.end)
    pending = [
        (month, category)
        for month in months
        for category in args.categories
        if (month, category) not in done
    ]
    print(f"{len(pending)} requests remaining ({len(done)} already cached)")

    with long_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["month", "category", "count"])
        if write_header:
            writer.writeheader()
        for index, (month, category) in enumerate(pending, start=1):
            count = _fetch_with_retries(category, month, args.timeout, args.retries)
            writer.writerow({"month": month, "category": category, "count": count})
            handle.flush()
            print(f"[{index}/{len(pending)}] {month} {category} {count}", flush=True)
            if index != len(pending):
                time.sleep(args.delay)

    ai_categories = [c for c in args.categories if c.startswith("cs.")]
    if ai_categories:
        pivot_wide(
            long_path,
            long_path.with_name("arxiv_counts_monthly.csv"),
            ai_categories,
        )
    if "quant-ph" in args.categories:
        quantum_path = resolve_repo_path("datasets/quantum/arxiv_quant_ph_monthly.csv")
        pivot_wide(long_path, quantum_path, ["quant-ph"])
        _rename_quant_ph_column(quantum_path)
    provenance = {
        "source": "https://export.arxiv.org/api/query",
        "query": "cat:{category} AND submittedDate:[YYYYMM010000 TO last-day 2359]",
        "categories": list(args.categories),
        "start": args.start,
        "end": args.end,
        "delay_seconds": args.delay,
        "licence": (
            "Monthly totals derived from the public arXiv API. Paper texts are "
            "not stored. arXiv asks for at least three seconds between requests."
        ),
    }
    (long_path.parent / "arxiv_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def _rename_quant_ph_column(path: Path) -> None:
    """Store the quant-ph series under a plain ``count`` column."""
    rows = list(csv.DictReader(path.open(newline="")))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["month", "count"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"month": row["month"], "count": row["quant-ph"]})


def _fetch_with_retries(category: str, month: str, timeout: float, retries: int) -> int:
    delay = 10.0
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            return fetch_count(category, month, timeout)
        except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            last_error = exc
            print(f"retry {category} {month}: {exc}", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed {category} {month}") from last_error


if __name__ == "__main__":
    main()
