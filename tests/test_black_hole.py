"""The black hole is part of the page and keeps moving when Nous is idle."""

from fastapi.testclient import TestClient

from src.api import app


def test_black_hole_ships_in_the_landing_and_the_app() -> None:
    client = TestClient(app)
    html = client.get("/assets/index.html").text
    assert "/assets/black-hole.css" in html
    # One on the landing page, one above the app greeting.
    assert html.count('class="bh"') == 2
    assert 'class="bh-disk bh-disk-2"' in html


def test_black_hole_css_animates_when_idle_and_calms_for_reduce_motion() -> None:
    client = TestClient(app)
    response = client.get("/assets/black-hole.css")
    assert response.status_code == 200
    css = response.text
    assert ".bh .bh-disk {" in css and "animation: bh-spin" in css
    assert ".nous-wordmark .nous-o {" in css
    assert "prefers-reduced-motion: reduce" in css
    # Reduce Motion keeps a calm glow instead of going dead.
    assert "bh-glow" in css
