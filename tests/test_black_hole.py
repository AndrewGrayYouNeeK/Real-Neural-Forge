"""The only black hole is the O in the wordmark; only its center moves."""

from fastapi.testclient import TestClient

from src.api import app


def test_one_wordmark_black_hole_and_no_second_hole() -> None:
    client = TestClient(app)
    html = client.get("/assets/index.html").text
    assert "/assets/black-hole.css" in html
    # The extra standalone hole above the wordmark is gone.
    assert 'class="bh"' not in html
    assert "bh-disk" not in html
    # Every wordmark, including the app greeting, uses the layered O.
    assert html.count("nous-o-core") >= 6
    assert 'class="nous-wordmark greeting-wordmark"' in html


def test_only_the_o_center_spins_and_reduce_motion_is_static() -> None:
    client = TestClient(app)
    response = client.get("/assets/black-hole.css")
    assert response.status_code == 200
    css = response.text
    assert "bh-core-spin" in css
    assert ".nous-wordmark .nous-o-core" in css
    # The S must not shimmer, warp, or change shape.
    assert "nous-s" not in css
    assert "bh-shimmer" not in css
    assert "prefers-reduced-motion: reduce" in css
    # Reduce Motion is fully still (no leftover glow or spin).
    reduce_block = css.split("prefers-reduced-motion: reduce", 1)[1]
    assert "animation: none" in reduce_block
