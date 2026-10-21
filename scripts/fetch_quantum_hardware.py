"""Annual record series from the MIT-licensed quantum-experiment tables.

Source: https://github.com/francois-marie/awesome-quantum-computing-experiments
(SPDX MIT). ``qubit_count.csv`` is the best reported physical qubit count in
each paper. ``entangled_state_error_exp.csv`` mixes two-qubit gate errors and
entangled-state errors; the annual series keeps the lowest reported error.
Quantum volume is not in this dataset.
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from src.utils.paths import resolve_repo_path

REPO = "https://raw.githubusercontent.com/francois-marie/awesome-quantum-computing-experiments/main/data"
QUBIT_URL = f"{REPO}/qubit_count.csv"
ERROR_URL = f"{REPO}/entangled_state_error_exp.csv"
LICENCE = "MIT"
ATTRIBUTION = (
    "François-Marie Le Régent and contributors, "
    "'Awesome Quantum Computing Experiments', MIT licence, "
    "https://github.com/francois-marie/awesome-quantum-computing-experiments"
)


def _annual(frame: pd.DataFrame, value: str, how: str) -> pd.DataFrame:
    grouped = frame.groupby("Year", as_index=False)[value].agg(how)
    start = int(grouped["Year"].min())
    end = int(grouped["Year"].max())
    years = pd.DataFrame({"year": list(range(start, end + 1))})
    merged = years.merge(
        grouped.rename(columns={"Year": "year"}), on="year", how="left"
    )
    best: float | None = None
    record: list[float | None] = []
    for observed in merged[value].tolist():
        if pd.notna(observed):
            number = float(observed)
            if best is None:
                best = number
            elif how == "max":
                best = max(best, number)
            else:
                best = min(best, number)
        record.append(best)
    merged["record"] = record
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="datasets/quantum")
    args = parser.parse_args()
    out = resolve_repo_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qubits = pd.read_csv(QUBIT_URL)
    errors = pd.read_csv(ERROR_URL)
    qubits["Year"] = pd.to_numeric(qubits["Year"], errors="coerce")
    qubits["Number of qubits"] = pd.to_numeric(
        qubits["Number of qubits"], errors="coerce"
    )
    qubits = qubits.dropna(subset=["Year", "Number of qubits"])
    qubits["Year"] = qubits["Year"].astype(int)
    notes = qubits.get("Notes", pd.Series([""] * len(qubits))).fillna("").str.lower()
    rearranged = qubits.loc[~notes.str.contains("not rearranged")].copy()

    qubit_record = _annual(qubits, "Number of qubits", "max")
    rearranged_record = _annual(rearranged, "Number of qubits", "max")
    qubit_out = pd.DataFrame(
        {
            "date": qubit_record["year"].map(lambda year: f"{int(year):04d}-01-01"),
            "record_qubits": qubit_record["record"],
            "log10_qubits": qubit_record["record"].map(math.log10),
        }
    )
    rearranged_map = dict(
        zip(rearranged_record["year"], rearranged_record["record"], strict=True)
    )
    qubit_out["record_qubits_excluding_unrearranged"] = [
        rearranged_map.get(int(year), pd.NA) for year in qubit_record["year"]
    ]
    # Forward-fill the filtered record onto the same annual grid, then log it.
    filled = pd.to_numeric(
        qubit_out["record_qubits_excluding_unrearranged"], errors="coerce"
    ).ffill()
    qubit_out["record_qubits_excluding_unrearranged"] = filled
    qubit_out["log10_qubits_excluding_unrearranged"] = filled.map(
        lambda value: math.log10(value) if pd.notna(value) and value > 0 else pd.NA
    )
    qubit_out.to_csv(out / "qubit_record_annual.csv", index=False)

    errors["Year"] = pd.to_numeric(errors["Year"], errors="coerce")
    errors["Entangled State Error"] = pd.to_numeric(
        errors["Entangled State Error"], errors="coerce"
    )
    errors = errors.dropna(subset=["Year", "Entangled State Error"])
    errors = errors[errors["Entangled State Error"] > 0]
    errors["Year"] = errors["Year"].astype(int)
    error_record = _annual(errors, "Entangled State Error", "min")
    error_out = pd.DataFrame(
        {
            "date": error_record["year"].map(lambda year: f"{int(year):04d}-01-01"),
            "best_error": error_record["record"],
            "log10_error": error_record["record"].map(math.log10),
            "best_fidelity": 1.0 - error_record["record"],
        }
    )
    error_out.to_csv(out / "two_qubit_error_annual.csv", index=False)
    provenance = {
        "qubit_source": QUBIT_URL,
        "error_source": ERROR_URL,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "quantum_volume": (
            "Not present in this dataset. The fidelity series is the running "
            "minimum of the entangled-state / two-qubit error column."
        ),
        "unrearranged_note": (
            "Some 2024-2026 neutral-atom rows are notes as 'Not rearranged'. "
            "record_qubits includes them. The excluding column drops those rows "
            "before the cumulative maximum."
        ),
    }
    (out / "hardware_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"qubits {len(qubit_out)} years, errors {len(error_out)} years")


if __name__ == "__main__":
    main()
