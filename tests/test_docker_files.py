"""Static checks on the Docker build files (Docker itself is not needed)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerignore_excludes_local_artifacts():
    entries = (ROOT / ".dockerignore").read_text().split()
    for item in (
        ".git",
        ".venv",
        "checkpoints",
        "*.pt",
        "__pycache__",
        "/data",
        "*.db",
        ".env",
    ):
        assert item in entries
    # A bare "data" pattern would also drop the src/data package.
    assert "data" not in entries
    # Things the image needs at runtime must not be excluded.
    for needed in (
        "src",
        "config",
        "frontend",
        "requirements.txt",
        "workers_ci",
        "space_snapshot",
        "predictions",
        "results",
    ):
        assert needed not in entries


def test_dockerfile_healthcheck_hits_health_endpoint():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "HEALTHCHECK" in dockerfile
    assert "http://127.0.0.1:8000/health" in dockerfile


def test_dockerfile_installs_workers_hook_without_writing_config():
    dockerfile = (ROOT / "Dockerfile").read_text()
    copy_at = dockerfile.index("COPY workers_ci ./workers_ci")
    install_at = dockerfile.index(
        "pip install --no-cache-dir -r /tmp/requirements-notorch.txt"
    )
    assert copy_at < install_at
    assert "test ! -e /app/wrangler.jsonc" in dockerfile
    assert "-e ./workers_ci" in (ROOT / "requirements.in").read_text()
    assert "-e ./workers_ci" in (ROOT / "requirements.txt").read_text()


def test_dockerfile_uses_cpu_torch_without_build_tools():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu" in dockerfile
    assert "build-essential" not in dockerfile
    assert "USER forge" in dockerfile
    assert "/app/checkpoints" in dockerfile
    assert "/app/data" in dockerfile
    assert "COPY --chown=forge:forge space_snapshot/ /app/data/space/" in dockerfile


def test_publish_workflow_waits_for_ci_and_keeps_tags():
    text = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "workflow_run:" in text
    assert 'workflows: ["CI"]' in text
    assert 'tags: ["v*.*.*"]' in text
    assert "needs: verify" in text
    assert "main-${SHA::7}" in text
    assert "steps.rev.outputs.main_tag" in text
    for line in text.splitlines():
        if "outputs.main_tag" in line:
            assert "enable=" not in line
    assert "uses: ./.github/workflows/ci.yml" in text
    assert (
        "branches: [main]"
        not in text.split("push:", 1)[-1].split("workflow_dispatch:", 1)[0]
    )
