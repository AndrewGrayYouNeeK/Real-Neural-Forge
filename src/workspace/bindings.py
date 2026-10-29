"""HTTP client for the Worker D1 and R2 binding proxy.

The container cannot hold D1 or R2 bindings itself. It calls back to the
Worker at ``NEURAL_FORGE_BINDINGS_URL``. Those routes never forward into the
container, so a storage call cannot deadlock the request that triggered it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from src.workspace.keys import valid_object_key


class StorageError(Exception):
    """The binding proxy rejected a storage call."""


class BindingTransport:
    """Speak the Worker ``/_bindings`` protocol."""

    def __init__(self, base_url: str, secret: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.secret = secret
        self.timeout = timeout

    def _call(
        self,
        method: str,
        path: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        request_headers = {
            "Authorization": f"Bearer {self.secret}",
            "Accept": "application/json",
        }
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            return int(exc.code), body

    def query(
        self, statements: Sequence[tuple[str, Sequence[Any]]]
    ) -> list[dict[str, Any]]:
        payload = {
            "statements": [
                {"sql": sql, "params": list(params)} for sql, params in statements
            ]
        }
        status, body = self._call(
            "POST",
            "/_bindings/sql",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if status != 200:
            detail = body.decode("utf-8", errors="replace")[:500]
            raise StorageError(f"D1 query failed ({status}): {detail}")
        parsed = json.loads(body.decode("utf-8"))
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(results, list):
            raise StorageError("D1 query returned an unexpected payload.")
        return [item if isinstance(item, dict) else {} for item in results]

    def put_object(self, key: str, data: bytes) -> None:
        if not valid_object_key(key):
            raise StorageError("Refusing to write an unsafe object key.")
        status, body = self._call(
            "PUT",
            "/_bindings/object",
            data=data,
            headers={"Content-Type": "application/octet-stream", "X-Object-Key": key},
        )
        if status not in (200, 201, 204):
            detail = body.decode("utf-8", errors="replace")[:300]
            raise StorageError(f"R2 put failed ({status}): {detail}")

    def get_object(self, key: str) -> bytes | None:
        if not valid_object_key(key):
            raise StorageError("Refusing to read an unsafe object key.")
        status, body = self._call(
            "GET",
            "/_bindings/object",
            headers={"X-Object-Key": key},
        )
        if status == 404:
            return None
        if status != 200:
            detail = body.decode("utf-8", errors="replace")[:300]
            raise StorageError(f"R2 get failed ({status}): {detail}")
        return body

    def delete_object(self, key: str) -> None:
        if not valid_object_key(key):
            raise StorageError("Refusing to delete an unsafe object key.")
        status, body = self._call(
            "DELETE",
            "/_bindings/object",
            headers={"X-Object-Key": key},
        )
        if status not in (200, 204, 404):
            detail = body.decode("utf-8", errors="replace")[:300]
            raise StorageError(f"R2 delete failed ({status}): {detail}")
