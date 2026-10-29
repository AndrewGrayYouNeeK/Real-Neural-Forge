"""The container's binding client must not look like Python-urllib."""

from __future__ import annotations

import json
import urllib.request

from src.workspace.bindings import USER_AGENT, BindingTransport


def test_binding_calls_send_a_custom_user_agent(monkeypatch):
    captured: dict[str, str] = {}

    class _Response:
        status = 200

        def read(self) -> bytes:
            return json.dumps({"results": [{"rows": []}]}).encode()

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def _urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        del timeout
        header = request.get_header("User-agent")
        assert header is not None
        captured["user_agent"] = header
        captured["url"] = request.full_url
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    BindingTransport("http://bindings.neuralforge", "secret").query([("SELECT 1", ())])
    assert captured["user_agent"] == USER_AGENT
    assert captured["user_agent"] == "NeuralForge-Container/1.0"
    assert "Python-urllib" not in captured["user_agent"]
    assert captured["url"] == "http://bindings.neuralforge/_bindings/sql"
