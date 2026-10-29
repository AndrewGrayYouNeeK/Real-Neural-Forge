"""Deploy files stay beside the existing Workers builds and name the right app."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_WORKERS = {"real-neural-forge", "real-neural-forge-frontend"}


def _load_jsonc(path: Path) -> dict:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("//")
    ]
    return json.loads("\n".join(lines))


def test_repository_root_has_no_wrangler_config():
    for name in ("wrangler.toml", "wrangler.json", "wrangler.jsonc"):
        assert not (ROOT / name).exists()


def test_container_worker_is_not_a_dashboard_worker():
    config = _load_jsonc(ROOT / "deploy" / "cloudflare" / "wrangler.jsonc")
    assert config["name"] == "real-neural-forge-api"
    assert config["name"] not in DASHBOARD_WORKERS
    assert "assets" not in config
    container = config["containers"][0]
    assert container["class_name"] == "NeuralForgeContainer"
    assert container["instance_type"] == "standard-1"
    assert container["max_instances"] == 1
    assert container["image"] == (
        "registry.cloudflare.com/d74b0de637f4d9d9eb3efb9e13142784/"
        "real-neural-forge:main-6ce3b6f"
    )
    assert ":latest" not in container["image"]
    assert config["migrations"] == [
        {"tag": "v1", "new_sqlite_classes": ["NeuralForgeContainer"]}
    ]
    assert "exports" not in config
    assert config["vars"]["NEURAL_FORGE_CHECKPOINT_URL"].endswith(
        "/champion-default.pt"
    )
    assert config["workers_dev"] is True
    assert config["vars"]["NEURAL_FORGE_STORAGE"] == "cloudflare"
    assert config["d1_databases"][0]["binding"] == "WORKSPACES_DB"
    assert config["d1_databases"][0]["database_name"] == "neural-forge-workspaces"
    assert config["d1_databases"][0]["database_id"] == (
        "13f8ec26-e2a0-41a6-b49d-a4ba4623082a"
    )
    assert config["vars"]["NEURAL_FORGE_BINDINGS_URL"] == "http://bindings.neuralforge"
    assert config["d1_databases"][0]["migrations_dir"] == "migrations"
    assert config["r2_buckets"][0]["binding"] == "WORKSPACES"
    assert config["r2_buckets"][0]["bucket_name"] == "neural-forge-workspaces"
    hostnames = {route["pattern"]: route for route in config["routes"]}
    assert hostnames["forge.youneek.xyz"]["custom_domain"] is True
    assert hostnames["forge.youneek.xyz"]["zone_name"] == "youneek.xyz"
    assert hostnames["forge.youneek.dev"]["custom_domain"] is True
    assert hostnames["forge.youneek.dev"]["zone_name"] == "youneek.dev"
    worker = (ROOT / "deploy" / "cloudflare" / "src" / "index.ts").read_text()
    assert "NEURAL_FORGE_API_TOKEN" in worker
    assert "X-Forwarded-Host" in worker
    assert "X-Forwarded-Proto" in worker
    assert 'headers.delete("X-Neural-Forge-Watch")' in worker
    assert "/_bindings/sql" in worker
    assert "/_bindings/object" in worker
    assert "bindings.neuralforge" in worker
    assert "outboundByHost" in worker
    assert "ContainerProxy" in worker
    assert "youneek-xyz" not in worker
    assert "WORKSPACES_DB" in worker
    assert "Set-Cookie" not in worker
    schema = (ROOT / "src" / "workspace" / "schema.sql").read_text(encoding="utf-8")
    migration = (
        ROOT / "deploy" / "cloudflare" / "migrations" / "0001_workspaces.sql"
    ).read_text(encoding="utf-8")
    assert schema == migration
    assert "localhost/health" in worker
    assert "defaultPort = 8000" in worker
    assert 'sleepAfter = "10m"' in worker
    assert "renewActivityTimeout" in worker
    assert 'TRAINING_WATCH_CALLBACK = "watchTraining"' in worker
    assert "TRAINING_POLL_SECONDS = 3 * 60" in worker
    assert "http://localhost/training/status" in worker
    assert "override async alarm" not in worker
    assert "async alarm(" not in worker


def test_fly_and_render_use_the_published_image_and_no_baked_secrets():
    image = "ghcr.io/andrewgrayyouneek/real-neural-forge:latest"
    fly = (ROOT / "fly.toml").read_text(encoding="utf-8")
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert image in fly
    assert image in render
    assert "internal_port = 8000" in fly
    assert 'path = "/health"' in fly
    assert "healthCheckPath: /health" in render
    assert "NEURAL_FORGE_API_TOKEN" in render
    assert "sync: false" in render
    assert not re.search(r"NEURAL_FORGE_API_TOKEN\s*[:=]\s*\S+", fly)
    for secret in ("ghp_", "github_pat_", "sk-"):
        assert secret not in fly
        assert secret not in render


def test_retrain_workflow_is_scheduled_and_manual_only():
    text = (ROOT / ".github" / "workflows" / "retrain.yml").read_text(encoding="utf-8")
    assert 'cron: "0 6 * * 1"' in text
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "contents: write" in text
    assert "timeout-minutes: 60" in text
    assert "--publish" in text
    assert "--pull-release" in text
    assert "--max-epochs" in text
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "actionlint" in ci


def test_compose_can_forward_champion_env():
    for name in ("docker-compose.yml", "docker-compose.ghcr.yml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "NEURAL_FORGE_CHECKPOINT_URL" in text
        assert "NEURAL_FORGE_API_TOKEN" in text
