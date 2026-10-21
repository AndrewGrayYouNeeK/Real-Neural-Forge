"""Visit CSV import and the optional Cloudflare client."""

import io
import json
from datetime import date
from urllib.request import Request

from src.realworld.website import (
    fetch_zone_analytics,
    read_visit_csv,
    synthetic_daily_visits,
    visits_from_env,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode()
        self.request: Request | None = None

    def __call__(self, request: Request, timeout: int) -> "_Response":
        self.request = request
        self.timeout = timeout
        return self

    def __enter__(self) -> io.BytesIO:
        return io.BytesIO(self._payload)

    def __exit__(self, *exc: object) -> bool:
        return False


def test_synthetic_visits_are_reproducible_and_weekly(tmp_path):
    first = synthetic_daily_visits()
    second = synthetic_daily_visits()
    assert first == second
    assert len(first) == 1096
    weekend = [value for day, value in first[:14] if day.weekday() >= 5]
    weekday = [value for day, value in first[:14] if day.weekday() < 5]
    assert sum(weekend) / len(weekend) < sum(weekday) / len(weekday)
    path = tmp_path / "visits.csv"
    path.write_text(
        "date,visits\n"
        + "\n".join(f"{day.isoformat()},{value}" for day, value in first[:3])
    )
    loaded = read_visit_csv(path)
    assert loaded[0] == (first[0][0], float(first[0][1]))


def test_visits_from_env_without_credentials(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ZONE_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    assert visits_from_env() is None


def test_zone_analytics_prefers_uniques():
    opener = _Response(
        {
            "data": {
                "viewer": {
                    "zones": [
                        {
                            "httpRequests1dGroups": [
                                {
                                    "dimensions": {"date": "2024-01-02"},
                                    "sum": {"requests": 50},
                                    "uniq": {"uniques": 7},
                                },
                                {
                                    "dimensions": {"date": "2024-01-01"},
                                    "sum": {"requests": 40},
                                    "uniq": {"uniques": 5},
                                },
                            ]
                        }
                    ]
                }
            }
        }
    )
    rows = fetch_zone_analytics(
        "token",
        "zone",
        date(2024, 1, 1),
        date(2024, 1, 3),
        opener=opener,
    )
    assert rows == [(date(2024, 1, 1), 5.0), (date(2024, 1, 2), 7.0)]
    assert opener.request is not None
    assert b"zone" in opener.request.data
