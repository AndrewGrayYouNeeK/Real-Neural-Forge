"""Static checks on the Docker build files (Docker itself is not needed)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerignore_excludes_local_artifacts():
    entries = (ROOT / ".dockerignore").read_text().split()
    for item in (".git", ".venv", "checkpoints", "*.pt", "__pycache__"):
        assert item in entries
    # Things the image needs at runtime must not be excluded.
    for needed in ("src", "config", "frontend", "requirements.txt"):
        assert needed not in entries


def test_dockerfile_healthcheck_hits_health_endpoint():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "HEALTHCHECK" in dockerfile
    assert "http://127.0.0.1:8000/health" in dockerfile
