"""Daily visit-count CSV import and an optional Cloudflare analytics fetch.

Credentials are never required. :func:`visits_from_env` returns ``None`` when
``CLOUDFLARE_API_TOKEN`` and a zone or account id are not both set.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np

GRAPHQL_URL = "https://api.cloudflare.com/client/v4/graphql"

ZONE_QUERY = """
query($zone: String!, $start: Date!, $end: Date!) {
  viewer {
    zones(filter: { zoneTag: $zone }) {
      httpRequests1dGroups(
        limit: 10000
        filter: { date_geq: $start, date_lt: $end }
        orderBy: [date_ASC]
      ) {
        dimensions { date }
        sum { requests }
        uniq { uniques }
      }
    }
  }
}
"""

ACCOUNT_QUERY = """
query($account: String!, $start: Date!, $end: Date!) {
  viewer {
    accounts(filter: { accountTag: $account }) {
      rumPageloadEventsAdaptiveGroups(
        limit: 10000
        filter: { date_geq: $start, date_lt: $end }
        orderBy: [date_ASC]
      ) {
        count
        dimensions { date }
      }
    }
  }
}
"""

ACCOUNT_SITE_QUERY = """
query($account: String!, $start: Date!, $end: Date!, $site: String!) {
  viewer {
    accounts(filter: { accountTag: $account }) {
      rumPageloadEventsAdaptiveGroups(
        limit: 10000
        filter: { date_geq: $start, date_lt: $end, siteTag: $site }
        orderBy: [date_ASC]
      ) {
        count
        dimensions { date }
      }
    }
  }
}
"""


class _Body(Protocol):
    def __enter__(self) -> Any: ...

    def __exit__(self, *exc: object) -> bool: ...


class Opener(Protocol):
    def __call__(self, request: Request, *, timeout: int) -> _Body: ...


def read_visit_csv(
    path: str | Path,
    date_column: str = "date",
    value_column: str = "value",
) -> list[tuple[date, float]]:
    """Read a daily visit CSV. ``value`` may be named ``visits`` when omitted."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        fields = {name.strip(): name for name in reader.fieldnames if name}
        if date_column not in fields:
            raise ValueError(f"{path} is missing a {date_column!r} column")
        value_name = value_column if value_column in fields else None
        if value_name is None and "visits" in fields and value_column == "value":
            value_name = "visits"
        if value_name is None:
            raise ValueError(f"{path} is missing a {value_column!r} column")
        rows: list[tuple[date, float]] = []
        for raw in reader:
            stamp = (raw.get(fields[date_column]) or "").strip()
            value = (raw.get(fields[value_name]) or "").strip()
            if not stamp or not value:
                continue
            rows.append((date.fromisoformat(stamp[:10]), float(value)))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError(f"{path} has no visit rows")
    return rows


def synthetic_daily_visits(
    n_days: int = 1096,
    start: date | None = None,
    seed: int = 0,
) -> list[tuple[date, int]]:
    """A reproducible example series with a weekly pattern. Not a real site."""
    if n_days < 1:
        raise ValueError("n_days must be positive")
    origin = start or date(2023, 1, 1)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 35.0, size=n_days)
    rows: list[tuple[date, int]] = []
    for index in range(n_days):
        day = origin + timedelta(days=index)
        weekend = day.weekday() >= 5
        level = 620.0 if weekend else 1040.0
        trend = index * 0.15
        annual = 80.0 * np.sin(2.0 * np.pi * index / 365.25)
        value = max(1, int(round(level + trend + annual + float(noise[index]))))
        rows.append((day, value))
    return rows


def visits_from_env(
    *,
    start: date | None = None,
    end: date | None = None,
    opener: Opener | None = None,
) -> list[tuple[date, float]] | None:
    """Fetch daily visits when Cloudflare credentials are set, else ``None``.

    ``CLOUDFLARE_ACCOUNT_ID`` uses Web Analytics page loads (optional
    ``CLOUDFLARE_SITE_TAG``). Otherwise ``CLOUDFLARE_ZONE_ID`` uses GraphQL
    zone analytics and prefers unique visitors, falling back to requests.
    """
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    zone = os.environ.get("CLOUDFLARE_ZONE_ID", "").strip()
    site = os.environ.get("CLOUDFLARE_SITE_TAG", "").strip()
    if not token or not (account or zone):
        return None
    end_day = end or date.today()
    start_day = start or (end_day - timedelta(days=365))
    if account:
        return fetch_web_analytics(
            token,
            account,
            start_day,
            end_day,
            site_tag=site or None,
            opener=opener,
        )
    return fetch_zone_analytics(token, zone, start_day, end_day, opener=opener)


def fetch_zone_analytics(
    token: str,
    zone_id: str,
    start: date,
    end: date,
    opener: Opener | None = None,
) -> list[tuple[date, float]]:
    """Daily unique visitors for a zone, or requests when uniques are absent."""
    payload = _graphql(
        token,
        ZONE_QUERY,
        {"zone": zone_id, "start": start.isoformat(), "end": end.isoformat()},
        opener,
    )
    zones = payload["data"]["viewer"]["zones"]
    if not zones:
        raise ValueError(f"Cloudflare returned no zone for {zone_id}")
    groups = zones[0]["httpRequests1dGroups"]
    rows: list[tuple[date, float]] = []
    for group in groups:
        stamp = date.fromisoformat(group["dimensions"]["date"])
        uniques = (group.get("uniq") or {}).get("uniques")
        requests = (group.get("sum") or {}).get("requests")
        value = uniques if uniques is not None else requests
        if value is None:
            continue
        rows.append((stamp, float(value)))
    rows.sort(key=lambda item: item[0])
    return rows


def fetch_web_analytics(
    token: str,
    account_id: str,
    start: date,
    end: date,
    site_tag: str | None = None,
    opener: Opener | None = None,
) -> list[tuple[date, float]]:
    """Daily Web Analytics page loads for an account, optionally one site tag."""
    variables: dict[str, object] = {
        "account": account_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    query = ACCOUNT_QUERY
    if site_tag:
        query = ACCOUNT_SITE_QUERY
        variables["site"] = site_tag
    payload = _graphql(token, query, variables, opener)
    accounts = payload["data"]["viewer"]["accounts"]
    if not accounts:
        raise ValueError(f"Cloudflare returned no account for {account_id}")
    groups = accounts[0]["rumPageloadEventsAdaptiveGroups"]
    rows = [
        (date.fromisoformat(group["dimensions"]["date"]), float(group["count"]))
        for group in groups
        if group.get("count") is not None
    ]
    rows.sort(key=lambda item: item[0])
    return rows


def _graphql(
    token: str,
    query: str,
    variables: dict[str, object],
    opener: Opener | None,
) -> dict[str, Any]:
    body = json.dumps({"query": query, "variables": variables}).encode()
    request = Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "real-neural-forge-dataset-fetch/1.0",
        },
        method="POST",
    )
    if opener is None:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    else:
        with opener(request, timeout=60) as response:
            payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Cloudflare response was not a JSON object")
    errors = payload.get("errors")
    if errors:
        message = "Cloudflare GraphQL error"
        if isinstance(errors, list) and errors and isinstance(errors[0], dict):
            message = str(errors[0].get("message", message))
        raise RuntimeError(message)
    return payload
