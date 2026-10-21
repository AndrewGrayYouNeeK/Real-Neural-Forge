"""Monthly means of the Oulu neutron monitor, via NMDB's keyless ASCII endpoint.

NMDB asks users to acknowledge the database and the station PIs. Oulu's own
readme limits the data to research, education, and public non-commercial
services. This fetch requests daily averages (``tresolution=1440``) in
five-year chunks and reduces them to a month.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import cast

from src.utils.paths import resolve_repo_path

USER_AGENT = "RealNeuralForge/1.0 (Oulu NM monthly means; research)"
ACKNOWLEDGEMENT = (
    "We acknowledge the NMDB database (www.nmdb.eu), founded under the "
    "European Union's FP7 programme (contract no. 213007), and the PIs of "
    "the neutron monitor at Oulu (Sodankyla Geophysical Observatory of the "
    "University of Oulu)."
)


def chunk_url(start_year: int, end_year: int) -> str:
    params = {
        "wget": "1",
        "stations[]": "OULU",
        "output": "ascii",
        "tabchoice": "ori",
        "dtype": "corr_for_efficiency",
        "tresolution": "1440",
        "date_choice": "bydate",
        "start_year": str(start_year),
        "start_month": "1",
        "start_day": "1",
        "start_hour": "0",
        "start_min": "0",
        "end_year": str(end_year),
        "end_month": "12",
        "end_day": "31",
        "end_hour": "23",
        "end_min": "59",
        "yunits": "0",
    }
    return "https://www.nmdb.eu/nest/draw_graph.php?" + urllib.parse.urlencode(params)


def parse_daily(body: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in body.splitlines():
        if ";" not in line or not line[0].isdigit():
            continue
        stamp, value = line.split(";", 1)
        day = stamp.strip().split(" ", 1)[0]
        try:
            rows.append((day, float(value.strip())))
        except ValueError:
            continue
    return rows


def fetch_chunk(start_year: int, end_year: int, timeout: float) -> str:
    request = urllib.request.Request(
        chunk_url(start_year, end_year), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return cast(str, response.read().decode("utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1964)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out", default="datasets/quantum/oulu_neutrons_monthly.csv")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    totals: dict[str, list[float]] = defaultdict(list)
    year = args.start_year
    while year <= args.end_year:
        end = min(year + 4, args.end_year)
        print(f"Oulu {year}-{end}", flush=True)
        body = fetch_chunk(year, end, args.timeout)
        daily = parse_daily(body)
        if len(daily) < 10:
            raise RuntimeError(
                f"NMDB returned {len(daily)} daily rows for {year}-{end}"
            )
        for day, value in daily:
            totals[day[:7]].append(value)
        year = end + 1
        if year <= args.end_year:
            time.sleep(2.0)
    months = sorted(totals)
    path = resolve_repo_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("date,count_rate\n")
        for month in months:
            values = totals[month]
            # A trailing month with only a few days is still in progress.
            if len(values) < 20:
                continue
            mean = sum(values) / len(values)
            handle.write(f"{month}-01,{mean:.6f}\n")
        months = [month for month in months if len(totals[month]) >= 20]
    provenance = {
        "source": "https://www.nmdb.eu/nest/draw_graph.php",
        "station": "OULU",
        "data_type": "corr_for_efficiency",
        "resolution": "daily averages reduced to monthly means",
        "licence": (
            "Free for non-commercial research, education, and public services. "
            "Acknowledge NMDB and the Oulu PIs. Oulu readme: "
            "https://cosmicrays.oulu.fi/readme.html"
        ),
        "acknowledgement": ACKNOWLEDGEMENT,
        "n_months": len(months),
        "date_start": f"{months[0]}-01",
        "date_end": f"{months[-1]}-01",
    }
    (path.parent / "oulu_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
