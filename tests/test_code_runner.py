"""The in-browser Python runner's worker script ships with a locked-down CSP."""

from fastapi.testclient import TestClient

from src.api import app


def test_pyworker_is_served_with_a_strict_csp() -> None:
    client = TestClient(app)
    response = client.get("/assets/pyworker.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "connect-src https://cdn.jsdelivr.net" in csp
    assert "'self'" not in csp
    assert "lockDown" in response.text


def test_chat_page_loads_the_code_runner() -> None:
    client = TestClient(app)
    html = client.get("/assets/index.html").text
    assert "/assets/coderunner.js" in html
    assert "/assets/abilities.js" in html
