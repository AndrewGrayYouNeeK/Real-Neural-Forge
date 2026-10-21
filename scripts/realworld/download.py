"""HTTP download helper for the public dataset files."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

USER_AGENT = "real-neural-forge-dataset-fetch/1.0"


def fetch_bytes(url: str, timeout: int = 180, retries: int = 3) -> bytes:
    """GET ``url`` and return the body. Retries transient network errors."""
    last_error: Exception | None = None
    for attempt in range(retries):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                return bytes(response.read())
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}") from last_error


def download_file(url: str, dest: Path, timeout: int = 600) -> Path:
    """Download ``url`` to ``dest`` unless ``dest`` already exists and is non-empty."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_bytes(url, timeout=timeout)
    temporary = dest.with_suffix(dest.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(dest)
    return dest
