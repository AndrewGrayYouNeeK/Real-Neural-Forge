"""Monthly counts of new public GitHub repos tagged topic:machine-learning.

The search API allows 10 unauthenticated requests per minute. This script
waits when the remaining budget hits zero. Repos that never set the topic
are missing, so the level is an undercount; the month-to-month shape is
what the forecast uses. No token is read or required.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from src.domain_forecasts.baselines import iter_months
from src.utils.paths import resolve_repo_path

USER_AGENT = "RealNeuralForge/1.0 (topic counts; no token)"


def search_url(month: str) -> str:
    year_s, month_s = month.split("-")
    year, month_i = int(year_s), int(month_s)
    last = calendar.monthrange(year, month_i)[1]
    query = (
        f"topic:machine-learning created:{year:04d}-{month_i:02d}-01"
        f"..{year:04d}-{month_i:02d}-{last:02d}"
    )
    return "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": query, "per_page": 1}
    )


def fetch_month(month: str, timeout: float) -> tuple[int, bool, int, int]:
    request = urllib.request.Request(
        search_url(month),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        remaining = int(response.headers.get("X-RateLimit-Remaining", "0"))
        reset = int(response.headers.get("X-RateLimit-Reset", "0"))
        payload = json.load(response)
    return (
        int(payload["total_count"]),
        bool(payload.get("incomplete_results")),
        remaining,
        reset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01")
    parser.add_argument("--end", default="2026-09")
    parser.add_argument(
        "--out", default="datasets/ai_progress/github_ml_repos_monthly.csv"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    path = resolve_repo_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if path.exists():
        with path.open(newline="") as handle:
            done = {row["month"] for row in csv.DictReader(handle)}
    pending = [
        month for month in iter_months(args.start, args.end) if month not in done
    ]
    print(f"{len(pending)} GitHub searches remaining")
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["month", "count", "incomplete"])
        if write_header:
            writer.writeheader()
        for index, month in enumerate(pending, start=1):
            count, incomplete, remaining, reset = _with_retries(month, args.timeout)
            writer.writerow(
                {"month": month, "count": count, "incomplete": str(incomplete).lower()}
            )
            handle.flush()
            print(
                f"[{index}/{len(pending)}] {month} {count} incomplete={incomplete}",
                flush=True,
            )
            if index != len(pending) and remaining <= 1:
                wait = max(0.0, reset - time.time()) + 1.0
                print(f"rate limit, sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
            elif index != len(pending):
                time.sleep(0.5)
    provenance = {
        "source": "https://api.github.com/search/repositories",
        "query": "topic:machine-learning created:YYYY-MM-01..YYYY-MM-last",
        "licence": (
            "GitHub repository metadata returned by the public search API. "
            "Counts are facts derived from that API; the underlying repository "
            "contents are not redistributed."
        ),
        "caveat": (
            "Unauthenticated search is limited to 10 requests per minute. "
            "total_count includes only public repositories that set the "
            "machine-learning topic, so the level is an undercount."
        ),
    }
    (path.parent / "github_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def _with_retries(month: str, timeout: float) -> tuple[int, bool, int, int]:
    delay = 5.0
    last: Exception | None = None
    for _ in range(5):
        try:
            return fetch_month(month, timeout)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 403:
                time.sleep(max(delay, 20.0))
            else:
                time.sleep(delay)
            delay *= 2
        except urllib.error.URLError as exc:
            last = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"GitHub search failed for {month}") from last


if __name__ == "__main__":
    main()
