"""Champion download and startup config selection."""

from pathlib import Path

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from src.api import app, load_model
from src.models.registry import build_model
from src.serving.checkpoint import (
    CheckpointDownloadError,
    _GitHubTokenRedirectHandler,
    build_request,
    download_checkpoint,
    prepare_runtime_checkpoint,
)
from src.utils.config import load_config


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._sent = False
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, _size: int) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        return self._payload

    def __enter__(self) -> "_Body":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def test_download_rejects_non_https(tmp_path: Path):
    with pytest.raises(CheckpointDownloadError, match="https"):
        download_checkpoint("http://example.com/model.pt", tmp_path / "model.pt")


def test_download_writes_the_body_and_rejects_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Opener:
        def open(self, request: object, timeout: int = 60) -> _Body:
            del timeout
            assert getattr(request, "full_url", "") == "https://example.com/model.pt"
            return _Body(b"weights")

    monkeypatch.setattr(
        "src.serving.checkpoint.urllib.request.build_opener", lambda *_a, **_k: Opener()
    )
    dest = tmp_path / "nested" / "model.pt"
    download_checkpoint("https://example.com/model.pt", dest)
    assert dest.read_bytes() == b"weights"

    class Empty:
        def open(self, _request: object, timeout: int = 60) -> _Body:
            del timeout
            return _Body(b"")

    monkeypatch.setattr(
        "src.serving.checkpoint.urllib.request.build_opener", lambda *_a, **_k: Empty()
    )
    with pytest.raises(CheckpointDownloadError, match="empty"):
        download_checkpoint("https://example.com/model.pt", tmp_path / "other.pt")
    assert not (tmp_path / "other.pt").exists()


def test_token_stays_on_github_hosts():
    authed = build_request(
        "https://github.com/a/b/releases/download/champions/m.pt", "secret"
    )
    assert authed.get_header("Authorization") == "Bearer secret"
    public = build_request("https://example.com/m.pt", "secret")
    assert public.has_header("Authorization") is False

    handler = _GitHubTokenRedirectHandler("secret")
    original = build_request("https://github.com/a/b", "secret")
    stayed = handler.redirect_request(
        original, None, 302, "Found", {}, "https://api.github.com/repos/a/b"
    )
    assert stayed is not None
    assert stayed.get_header("Authorization") == "Bearer secret"
    left = handler.redirect_request(
        original,
        None,
        302,
        "Found",
        {},
        "https://release-assets.githubusercontent.com/file",
    )
    assert left is not None
    assert left.has_header("Authorization") is False
    with pytest.raises(CheckpointDownloadError, match="non-https"):
        handler.redirect_request(
            original, None, 302, "Found", {}, "http://example.com/x"
        )


def test_prepare_runtime_checkpoint_uses_the_path_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, object] = {}

    def fake(
        url: str, dest: Path, token: str | None = None, *, max_bytes: int = 0
    ) -> None:
        seen["url"] = url
        seen["dest"] = dest
        seen["token"] = token
        seen["max_bytes"] = max_bytes

    monkeypatch.setattr("src.serving.checkpoint.download_checkpoint", fake)
    monkeypatch.setenv("NEURAL_FORGE_CHECKPOINT_URL", "https://example.com/c.pt")
    monkeypatch.setenv("NEURAL_FORGE_CHECKPOINT_TOKEN", "secret")
    monkeypatch.setenv("NEURAL_FORGE_CHECKPOINT_PATH", str(tmp_path / "c.pt"))
    monkeypatch.setenv("NEURAL_FORGE_CHECKPOINT_MAX_BYTES", "1024")
    cfg = prepare_runtime_checkpoint(load_config("config/config.yaml"))
    assert cfg["inference"]["checkpoint_path"] == str(tmp_path / "c.pt")
    assert seen["url"] == "https://example.com/c.pt"
    assert seen["dest"] == tmp_path / "c.pt"
    assert seen["token"] == "secret"
    assert seen["max_bytes"] == 1024


def test_load_model_downloads_before_reading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("config/config.yaml")
    destination = tmp_path / "champion.pt"
    state = build_model(cfg).state_dict()

    def fake(
        url: str, dest: Path, token: str | None = None, *, max_bytes: int = 0
    ) -> None:
        del url, token, max_bytes
        torch.save({"model_state_dict": state, "config": cfg, "scaler": None}, dest)

    monkeypatch.setenv("NEURAL_FORGE_CHECKPOINT_URL", "https://example.com/champion.pt")
    monkeypatch.setenv("NEURAL_FORGE_CHECKPOINT_PATH", str(destination))
    monkeypatch.setattr("src.serving.checkpoint.download_checkpoint", fake)
    load_model("config/config.yaml")
    from src.api import _state

    assert _state["checkpoint_loaded"] is True
    assert _state["checkpoint_path"] == destination


def test_startup_config_env_selects_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = load_config("config/config.yaml")
    cfg["model"]["d_model"] = 32
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("NEURAL_FORGE_CONFIG", str(path))
    monkeypatch.delenv("NEURAL_FORGE_CHECKPOINT_URL", raising=False)
    with TestClient(app) as client:
        info = client.get("/model/info").json()
    expected = sum(param.numel() for param in build_model(cfg).parameters())
    assert info["parameters"] == expected
    assert info["checkpoint_loaded"] is False


def test_oversized_content_length_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Huge:
        def open(self, _request: object, timeout: int = 60) -> _Body:
            del timeout
            body = _Body(b"x")
            body.headers = {"Content-Length": "999999"}
            return body

    monkeypatch.setattr(
        "src.serving.checkpoint.urllib.request.build_opener", lambda *_a, **_k: Huge()
    )
    with pytest.raises(CheckpointDownloadError, match="larger"):
        download_checkpoint(
            "https://example.com/model.pt", tmp_path / "model.pt", max_bytes=10
        )


def test_redirect_handler_ignores_unknown_status():
    handler = _GitHubTokenRedirectHandler(None)
    request = build_request("https://example.com/a", None)
    assert (
        handler.redirect_request(request, None, 304, "Not Modified", {}, "https://x")
        is None
    )
