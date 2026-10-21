"""Download a published quantum-randomness file and turn it into a series.

ANU's live QRNG API was not used: connections to qrng.anu.edu.au timed out
from this environment, and the current API requires a key. The file below is
a keyless MIT-licensed dump whose README attributes the bytes to the ANU
generator. Each row is a short slice of one entropy dump; concatenating the
slices in timestamp order is the series the sanity-check model tries (and
should fail) to predict.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

from src.utils.paths import resolve_repo_path

SOURCE_URL = (
    "https://huggingface.co/datasets/ProCreations/quantum-randomness"
    "/resolve/main/quantum_random_1000_final.csv"
)
LICENCE = "MIT"
ATTRIBUTION = (
    "ProCreations, 'quantum-randomness' (1,000 slices attributed to the ANU "
    "QRNG), MIT licence, "
    "https://huggingface.co/datasets/ProCreations/quantum-randomness"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--raw-out", default="datasets/quantum/qrng_slices.csv")
    parser.add_argument("--series-out", default="datasets/quantum/qrng_symbols.csv")
    args = parser.parse_args()
    raw_path = resolve_repo_path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        urllib.request.urlretrieve(args.url, raw_path)
    frame = pd.read_csv(raw_path)
    symbols = "".join(str(value) for value in frame["value"].tolist())
    series = pd.DataFrame(
        {
            "step": list(range(len(symbols))),
            "symbol": list(symbols),
            "code": [ord(character) for character in symbols],
        }
    )
    series_path = resolve_repo_path(args.series_out)
    series.to_csv(series_path, index=False)
    alphabet = sorted(set(symbols))
    provenance = {
        "source": args.url,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "n_slices": int(len(frame)),
        "n_symbols": int(len(symbols)),
        "alphabet_size": len(alphabet),
        "anu_api": "not used; the keyless endpoint did not respond and now requires a key",
        "framing": (
            "Slices are concatenated in file order. Timestamps in the source "
            "are microseconds apart inside a single dump, not a calendar series."
        ),
    }
    (raw_path.parent / "qrng_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
