"""Fail unless a running API serves baked space events and the other panels.

The image test starts the server with ``NEURAL_FORGE_SPACE_REFRESH=0`` and
runs this module inside the container, so a green result means the snapshot
and the committed forecast and scorecard files are in the image.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def require_served_events(
    near: dict[str, Any],
    solar: dict[str, Any],
    scorecard: dict[str, Any],
    series: dict[str, Any],
) -> None:
    """Raise ``SystemExit`` when a panel that should be baked is empty."""
    _require_rows(
        near.get("approaches"), "GET /space/near-earth returned no approaches."
    )
    _require_rows(
        solar.get("approaches"),
        "GET /space/solar-system returned no approaches.",
    )
    _require_rows(
        solar.get("perihelia"),
        "GET /space/solar-system returned no perihelia.",
    )
    _require_rows(
        solar.get("alignments"),
        "GET /space/solar-system returned no alignments.",
    )
    _require_rows(
        near.get("forecasts"), "Near-Earth forecasts are missing from the image."
    )
    _require_rows(
        solar.get("forecasts"),
        "Solar-system forecasts are missing from the image.",
    )
    moons = solar.get("moons")
    by_planet = moons.get("by_planet") if isinstance(moons, dict) else None
    _require_rows(by_planet, "Solar-system moon counts are missing from the image.")
    if scorecard.get("available") is not True:
        raise SystemExit("Powerball scorecard is missing from the image.")
    _require_rows(series.get("history"), "Dashboard series history is empty.")
    _require_rows(series.get("forecast"), "Dashboard series forecast is empty.")


def _require_rows(value: object, message: str) -> None:
    if not isinstance(value, list) or not value:
        raise SystemExit(message)


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{url} did not return a JSON object.")
    return payload


def main() -> None:
    base = os.environ.get("NEURAL_FORGE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    near = _get_json(f"{base}/space/near-earth?days=730")
    solar = _get_json(f"{base}/space/solar-system?days=730")
    scorecard = _get_json(f"{base}/powerball/scorecard")
    series = _get_json(f"{base}/series/recent?points=32")
    require_served_events(near, solar, scorecard, series)


if __name__ == "__main__":
    main()
