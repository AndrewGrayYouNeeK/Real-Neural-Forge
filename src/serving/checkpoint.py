"""Download the champion checkpoint before the API loads a model.

``NEURAL_FORGE_CHECKPOINT_URL`` is an https URL (a GitHub Release asset, or any
other https file). ``NEURAL_FORGE_CHECKPOINT_TOKEN`` is sent only to GitHub
hosts, and it is dropped when a redirect leaves those hosts. ``NEURAL_FORGE_CHECKPOINT_PATH``
overrides the destination. Unset variables leave the configured local file alone.
"""

from __future__ import annotations

import copy
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)

URL_ENV = "NEURAL_FORGE_CHECKPOINT_URL"
TOKEN_ENV = "NEURAL_FORGE_CHECKPOINT_TOKEN"
PATH_ENV = "NEURAL_FORGE_CHECKPOINT_PATH"
MAX_BYTES_ENV = "NEURAL_FORGE_CHECKPOINT_MAX_BYTES"
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_GITHUB_HOSTS = frozenset({"github.com", "api.github.com", "uploads.github.com"})
_USER_AGENT = "real-neural-forge"


class CheckpointDownloadError(RuntimeError):
    """The champion URL could not be stored as a checkpoint file."""


def checkpoint_destination(cfg: dict[str, Any]) -> Path:
    """Return the checkpoint path the server should load."""
    inference = cfg.get("inference")
    raw = inference.get("checkpoint_path") if isinstance(inference, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        raw = "checkpoints/best_model.pt"
    return resolve_repo_path(raw)


def build_request(url: str, token: str | None) -> urllib.request.Request:
    """Build a GET request. The token is attached only for GitHub hosts."""
    headers = {"User-Agent": _USER_AGENT}
    host = (urlparse(url).hostname or "").lower()
    if token and host in _GITHUB_HOSTS:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/octet-stream"
    return urllib.request.Request(url, headers=headers, method="GET")


class _GitHubTokenRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow https redirects and rebuild headers for the new host."""

    def __init__(self, token: str | None) -> None:
        super().__init__()
        self._token = token

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        del req, fp, msg, headers
        if code not in {301, 302, 303, 307, 308}:
            return None
        if urlparse(newurl).scheme != "https":
            raise CheckpointDownloadError("Refusing to follow a non-https redirect.")
        return build_request(newurl, self._token)


def _max_bytes() -> int:
    raw = os.environ.get(MAX_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise CheckpointDownloadError(
            f"{MAX_BYTES_ENV} must be an integer number of bytes."
        ) from exc
    if value < 1:
        raise CheckpointDownloadError(f"{MAX_BYTES_ENV} must be positive.")
    return value


def download_checkpoint(
    url: str,
    dest: Path,
    token: str | None = None,
    *,
    max_bytes: int | None = None,
) -> None:
    """Download ``url`` to ``dest``. A failed download leaves ``dest`` unchanged."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise CheckpointDownloadError("Checkpoint URL must be an https URL.")
    limit = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes
    if limit < 1:
        raise CheckpointDownloadError("max_bytes must be positive.")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    opener = urllib.request.build_opener(_GitHubTokenRedirectHandler(token))
    try:
        with opener.open(build_request(url, token), timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > limit:
                raise CheckpointDownloadError(
                    f"Checkpoint is larger than {limit} bytes."
                )
            total = 0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise CheckpointDownloadError(
                            f"Checkpoint is larger than {limit} bytes."
                        )
                    handle.write(chunk)
        if total == 0:
            raise CheckpointDownloadError("Checkpoint download was empty.")
        os.replace(partial, dest)
    except CheckpointDownloadError:
        partial.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise CheckpointDownloadError(f"Could not download checkpoint: {exc}") from exc
    logger.info("Downloaded champion checkpoint to %s", dest)


def prepare_runtime_checkpoint(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply checkpoint env vars and download the champion when a URL is set."""
    path_override = os.environ.get(PATH_ENV, "").strip()
    url = os.environ.get(URL_ENV, "").strip()
    token = os.environ.get(TOKEN_ENV, "").strip() or None
    if path_override:
        cfg = merge_inference_path(cfg, path_override)
    if url:
        download_checkpoint(
            url, checkpoint_destination(cfg), token, max_bytes=_max_bytes()
        )
    return cfg


def merge_inference_path(cfg: dict[str, Any], path: str) -> dict[str, Any]:
    """Return a copy of ``cfg`` whose inference checkpoint path is ``path``."""
    updated = copy.deepcopy(cfg)
    inference = updated.get("inference")
    if inference is None:
        inference = {}
        updated["inference"] = inference
    if not isinstance(inference, dict):
        raise CheckpointDownloadError("inference config must be a mapping.")
    inference["checkpoint_path"] = path
    return updated
