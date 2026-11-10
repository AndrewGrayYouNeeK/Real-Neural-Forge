"""Visitor workspaces stay off the owner's files and off each other."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app, load_model
from src.retrain.discover import discover
from src.utils.paths import REPO_ROOT
from src.workspace.runner import execute_member_job
from src.workspace.runtime import runtime
from src.workspace.service import watch_authorized
from src.workspace.store import open_workspace

STARTER_IDS = ("weather", "stocks", "crypto", "lottery", "near-earth", "lunar")


@pytest.fixture(scope="module")
def workspace_client(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("workspaces")
    updates = {
        "NEURAL_FORGE_API_TOKEN": "owner-secret",
        "NEURAL_FORGE_WORKSPACE_DIR": str(root),
        "NEURAL_FORGE_QUEUE_RUNNER": "0",
        "NEURAL_FORGE_STORAGE": "local",
        "NEURAL_FORGE_GLOBAL_QUEUE_LIMIT": "8",
        "NEURAL_FORGE_MAX_DATASET_BYTES": "2000000",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    load_model("config/config.yaml")
    try:
        with TestClient(app) as client:
            yield client, root
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _signup(client: TestClient, email: str) -> None:
    client.cookies.clear()
    response = client.post(
        "/auth/signup",
        json={"email": email, "password": "password-a-ok"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["role"] == "member"
    assert "password" not in response.text


def _login(client: TestClient, email: str) -> None:
    client.cookies.clear()
    response = client.post(
        "/auth/login",
        json={"email": email, "password": "password-a-ok"},
    )
    assert response.status_code == 200, response.text


def _db(root: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(root / "workspaces.db")
    connection.row_factory = sqlite3.Row
    return connection


def test_cloudflare_mode_does_not_create_local_files(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEURAL_FORGE_STORAGE", "cloudflare")
    monkeypatch.delenv("NEURAL_FORGE_BINDINGS_URL", raising=False)
    monkeypatch.delenv("NEURAL_FORGE_STORAGE_SECRET", raising=False)
    store, error = open_workspace()
    assert store is None
    assert error
    assert list(tmp_path.iterdir()) == []


def test_starter_templates_are_not_champion_datasets():
    specs, _errors = discover(REPO_ROOT / "config")
    discovered = {spec.path.resolve() for spec in specs}
    for starter_id in STARTER_IDS:
        template = (
            REPO_ROOT / "starters" / "templates" / f"{starter_id}.yaml"
        ).resolve()
        assert template not in discovered


def test_public_routes_stay_open_and_private_routes_do_not(workspace_client):
    client, _root = workspace_client
    client.cookies.clear()
    assert client.get("/health").status_code == 200
    limits = client.get("/workspace/limits")
    assert limits.status_code == 200, limits.text
    assert limits.json()["limits"]["max_epochs"] >= 1
    assert limits.json()["limits"]["max_dataset_bytes"] >= 1
    assert client.get("/lunar").status_code == 200
    assert client.get("/space/near-earth").status_code == 200
    assert client.get("/space/solar-system").status_code == 200
    assert client.get("/series/recent", params={"points": 16}).status_code == 200
    private = client.get("/experiments")
    assert private.status_code == 401
    assert "token" in private.json()["detail"].lower()
    watched = client.get(
        "/training/status",
        headers={"X-Neural-Forge-Watch": "owner-secret"},
    )
    assert watched.status_code == 401
    assert watch_authorized("localhost", "owner-secret")
    assert not watch_authorized("forge.youneek.xyz", "owner-secret")
    assert not watch_authorized("forge.youneek.dev", "owner-secret")


def test_accounts_are_isolated_and_starters_are_copy_on_write(workspace_client):
    client, root = workspace_client
    _signup(client, "a@example.com")
    secure = client.post(
        "/auth/login",
        json={"email": "a@example.com", "password": "password-a-ok"},
        headers={"X-Forwarded-Proto": "https"},
    )
    cookie = secure.headers["set-cookie"]
    assert "nf_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" in cookie
    assert "Domain=" not in cookie
    # A Secure cookie is not stored for this HTTP test client. Sign in again.
    _login(client, "a@example.com")

    starters = client.get("/workspace/starters")
    assert starters.status_code == 200
    cards = {card["id"]: card for card in starters.json()["starters"]}
    assert set(cards) == set(STARTER_IDS)
    weather_dataset = cards["weather"]["dataset_id"]
    weather_config = cards["weather"]["config_id"]

    with _db(root) as connection:
        stored = connection.execute(
            "SELECT password_hash FROM users WHERE email = ?",
            ("a@example.com",),
        ).fetchone()
        keys = connection.execute(
            """
            SELECT object_key, shared FROM datasets
            WHERE user_id = (SELECT id FROM users WHERE email = ?)
            AND starter_id = 'weather'
            """,
            ("a@example.com",),
        ).fetchone()
    assert stored["password_hash"].startswith("scrypt$")
    assert "password-a-ok" not in stored["password_hash"]
    assert keys["shared"] == 1
    assert keys["object_key"] == "bases/starters/weather/dataset.csv"

    base_before = (REPO_ROOT / "starters" / "data" / "weather.csv").read_bytes()
    template_before = (REPO_ROOT / "starters" / "templates" / "weather.yaml").read_text(
        encoding="utf-8"
    )
    model_before = (REPO_ROOT / "starters" / "models" / "weather.pt").read_bytes()
    base_hash = hashlib.sha256(base_before).hexdigest()

    replacement = "date,value\n" + "".join(
        f"2024-01-{day:02d},{float(day)}\n" for day in range(1, 33)
    )
    replaced = client.put(
        f"/workspace/datasets/{weather_dataset}",
        content=replacement.encode(),
        headers={"Content-Type": "text/csv"},
    )
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["shared"] is False

    edited = client.put(
        f"/workspace/configs/{weather_config}",
        json={"body": template_before.replace("epochs: 4", "epochs: 3", 1)},
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["shared"] is False
    assert (REPO_ROOT / "starters" / "templates" / "weather.yaml").read_text(
        encoding="utf-8"
    ) == template_before
    assert (REPO_ROOT / "starters" / "data" / "weather.csv").read_bytes() == base_before
    assert (
        REPO_ROOT / "starters" / "models" / "weather.pt"
    ).read_bytes() == model_before

    _signup(client, "b@example.com")
    other = client.get("/workspace/starters").json()["starters"]
    other_weather = next(card for card in other if card["id"] == "weather")
    missing = client.get(f"/workspace/datasets/{weather_dataset}")
    assert missing.status_code == 404
    assert client.delete(f"/workspace/datasets/{weather_dataset}").status_code == 404
    assert (
        client.get(f"/workspace/predictions/{other_weather['model_id']}").status_code
        == 404
    )

    with _db(root) as connection:
        shared_keys = connection.execute(
            """
            SELECT DISTINCT object_key FROM models
            WHERE starter_id = 'weather' AND shared = 1
            """
        ).fetchall()
        base_blob = (
            root / "objects" / "bases" / "starters" / "weather" / "dataset.csv"
        ).read_bytes()
    assert len(shared_keys) == 1
    assert shared_keys[0]["object_key"] == "bases/starters/weather/model.pt"
    assert hashlib.sha256(base_blob).hexdigest() == base_hash

    b_config = client.get("/workspace/configs").json()["configs"]
    b_weather = next(row for row in b_config if row["starter_id"] == "weather")
    assert "epochs: 4" in b_weather["body"]
    assert b_weather["shared"] is True

    _login(client, "a@example.com")
    owner_store = api_module._store
    assert owner_store is not None
    before = len(owner_store.list_experiments(limit=500))
    owner_id = owner_store.create_experiment(
        "owner-only",
        {"model": {"name": "time_series_transformer"}, "training": {}},
        "time_series_transformer",
    )
    listed = client.get("/experiments")
    assert listed.status_code == 200
    assert all(row.get("name") != "owner-only" for row in listed.json()["experiments"])
    assert client.get(f"/experiments/{owner_id}").status_code == 404
    assert client.get(f"/experiments/{owner_id}/predictions").status_code == 404
    assert len(owner_store.list_experiments(limit=500)) == before + 1

    client.cookies.clear()
    owner_list = client.get(
        "/experiments",
        headers={"Authorization": "Bearer owner-secret"},
    )
    assert owner_list.status_code == 200
    assert any(row["name"] == "owner-only" for row in owner_list.json()["experiments"])
    assert (
        client.get(
            "/workspace/datasets",
            headers={"Authorization": "Bearer owner-secret"},
        ).json()["datasets"]
        == []
    )
    assert (
        client.get(
            "/workspace/starters",
            headers={"Authorization": "Bearer owner-secret"},
        ).json()["starters"]
        == []
    )


def test_training_queue_cannot_touch_another_user_or_the_owner(workspace_client):
    client, root = workspace_client
    _login(client, "a@example.com")
    model_before = (REPO_ROOT / "starters" / "models" / "weather.pt").read_bytes()
    owner_checkpoint = REPO_ROOT / "checkpoints" / "best_model.pt"
    owner_mtime = (
        owner_checkpoint.stat().st_mtime_ns if owner_checkpoint.is_file() else None
    )

    class FakeTrainer:
        def __init__(self, path: str) -> None:
            self.path = Path(path)

        def train(self, epoch_callback=None):
            del epoch_callback
            cfg = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            directory = Path(cfg["training"]["checkpoint_dir"])
            assert directory.is_absolute()
            assert not directory.is_relative_to(REPO_ROOT)
            assert Path(cfg["storage"]["db_path"]).is_absolute()
            assert not Path(cfg["storage"]["db_path"]).is_relative_to(REPO_ROOT)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "best_model.pt").write_bytes(b"private-weights")
            return {
                "best_loss": 0.25,
                "metrics": {"mae": 0.5},
                "history": [{"epoch": 1, "train_loss": 0.25}],
                "elapsed_seconds": 0.01,
                "best_epoch": 1,
            }

    with patch("src.api._run_training") as owner_runner:
        posted = client.post("/train", json={"config_path": "config/config.yaml"})
    assert posted.status_code == 200, posted.text
    assert posted.json()["status"] == "queued"
    owner_runner.assert_not_called()
    posted_job = client.get("/workspace/jobs").json()["jobs"][0]["id"]
    assert client.delete(f"/workspace/jobs/{posted_job}").status_code == 200

    queued = client.post("/workspace/starters/weather/train")
    assert queued.status_code == 200, queued.text
    assert queued.json()["status"] == "queued"
    again = client.post("/workspace/starters/weather/train")
    assert again.status_code == 409

    store = runtime.store
    assert store is not None
    job = store.get_job(
        store.user_by_email("a@example.com")["id"],
        queued.json()["id"],
    )
    assert job is not None
    execute_member_job(store, job, trainer_factory=FakeTrainer)
    finished = client.get(f"/workspace/jobs/{queued.json()['id']}")
    assert finished.json()["status"] == "completed"
    private_models = [
        model
        for model in client.get("/workspace/models").json()["models"]
        if model["starter_id"] == "weather" and model["shared"] is False
    ]
    assert len(private_models) == 1
    assert (
        REPO_ROOT / "starters" / "models" / "weather.pt"
    ).read_bytes() == model_before
    if owner_mtime is None:
        assert not owner_checkpoint.is_file()
    else:
        assert owner_checkpoint.stat().st_mtime_ns == owner_mtime

    _login(client, "b@example.com")
    assert client.get(f"/workspace/jobs/{queued.json()['id']}").status_code == 404
    assert client.delete(f"/workspace/jobs/{queued.json()['id']}").status_code == 404
    b_models = client.get("/workspace/models").json()["models"]
    assert all(
        model["shared"] for model in b_models if model["starter_id"] == "weather"
    )
    predicted = client.post("/workspace/starters/lunar/predict")
    assert predicted.status_code == 200, predicted.text
    assert isinstance(predicted.json()["prediction"], list)
    _login(client, "a@example.com")
    assert (
        client.get(f"/workspace/predictions/{predicted.json()['id']}").status_code
        == 404
    )

    os.environ["NEURAL_FORGE_GLOBAL_QUEUE_LIMIT"] = "1"
    try:
        _login(client, "a@example.com")
        # The completed job no longer occupies the active slot. Queue one,
        # then the other account is told the global queue is full.
        first = client.post("/workspace/starters/lunar/train")
        assert first.status_code == 200, first.text
        _login(client, "b@example.com")
        blocked = client.post("/workspace/starters/lunar/train")
        assert blocked.status_code == 429
        _login(client, "a@example.com")
        assert client.delete(f"/workspace/jobs/{first.json()['id']}").status_code == 200
    finally:
        os.environ["NEURAL_FORGE_GLOBAL_QUEUE_LIMIT"] = "8"

    os.environ["NEURAL_FORGE_MAX_DATASET_BYTES"] = "40"
    try:
        _login(client, "b@example.com")
        too_big = client.post(
            "/workspace/datasets?name=x.csv",
            content=b"date,value\n" + b"2024-01-01,1\n" * 40,
            headers={"Content-Type": "text/csv"},
        )
        assert too_big.status_code == 413
    finally:
        os.environ["NEURAL_FORGE_MAX_DATASET_BYTES"] = "2000000"

    with _db(root) as connection:
        owners = connection.execute(
            "SELECT id, role FROM users WHERE role = 'owner'"
        ).fetchall()
    assert [row["id"] for row in owners] == ["owner"]


def _numeric_csv(
    column: str = "temp", rows: int = 40, extra: list[str] | None = None
) -> bytes:
    headers = ["date", column, *(extra or [])]
    lines = [",".join(headers)]
    for index in range(rows):
        values = [f"2024-01-{(index % 28) + 1:02d}", f"{10 + index * 0.25}"]
        for offset, _name in enumerate(extra or []):
            values.append(str(100 + index + offset))
        lines.append(",".join(values))
    return ("\n".join(lines) + "\n").encode()


def test_one_click_csv_train_flow_is_capped_and_private(workspace_client):
    client, _root = workspace_client
    client.cookies.clear()
    guest = client.post(
        "/workspace/datasets?name=temps.csv",
        content=_numeric_csv(),
        headers={"Content-Type": "text/csv"},
    )
    assert guest.status_code == 401
    guest_train = client.post(
        "/workspace/datasets/dst_missing/train", json={"target_column": "temp"}
    )
    assert guest_train.status_code == 401

    _signup(client, "csv.user@example.com")
    _login(client, "csv.user@example.com")

    too_short = client.post(
        "/workspace/datasets?name=short.csv",
        content=_numeric_csv(rows=10),
        headers={"Content-Type": "text/csv"},
    )
    assert too_short.status_code == 422

    no_numbers = client.post(
        "/workspace/datasets?name=labels.csv",
        content=b"date,label\n" + b"2024-01-01,hot\n" * 40,
        headers={"Content-Type": "text/csv"},
    )
    assert no_numbers.status_code == 422

    uploaded = client.post(
        "/workspace/datasets?name=temps.csv",
        content=_numeric_csv(extra=["humidity"]),
        headers={"Content-Type": "text/csv"},
    )
    assert uploaded.status_code == 201, uploaded.text
    body = uploaded.json()
    names = {column["name"]: column for column in body["columns"]}
    assert "temp" in names and names["temp"]["usable"] is True
    assert "humidity" in names and names["humidity"]["usable"] is True
    assert names["date"]["usable"] is False
    assert body["suggested_target"] == "temp"
    assert body["limits"]["max_epochs"] >= 1
    dataset_id = body["id"]

    missing = client.post(
        f"/workspace/datasets/{dataset_id}/train",
        json={"target_column": "not-a-column"},
    )
    assert missing.status_code == 422
    assert "temp" in missing.json()["detail"]

    queued = client.post(
        f"/workspace/datasets/{dataset_id}/train",
        json={"target_column": "humidity"},
    )
    assert queued.status_code == 200, queued.text
    payload = queued.json()
    assert payload["status"] == "queued"
    assert payload["dataset_id"] == dataset_id
    assert payload["target_column"] == "humidity"
    assert payload["limits"]["max_run_seconds"] >= 1
    job_id = payload["id"]
    config_id = payload["config_id"]

    again = client.post(
        f"/workspace/datasets/{dataset_id}/train",
        json={"target_column": "temp"},
    )
    assert again.status_code == 409

    stored = client.get("/workspace/configs").json()["configs"]
    custom = next(row for row in stored if row["id"] == config_id)
    parsed = yaml.safe_load(custom["body"])
    assert parsed["data"]["target_column"] == "humidity"
    assert parsed["data"]["feature_columns"] == ["humidity"]
    assert parsed["data"]["source"] == "csv"
    assert parsed["training"]["epochs"] <= payload["limits"]["max_epochs"]
    assert parsed["training"]["device"] == "cpu"

    class FakeTrainer:
        def __init__(self, path: str) -> None:
            self.path = Path(path)

        def train(self, epoch_callback=None):
            del epoch_callback
            cfg = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            assert cfg["data"]["source"] == "csv"
            assert Path(cfg["data"]["path"]).name == "series.csv"
            assert cfg["data"]["target_column"] == "humidity"
            assert cfg["training"]["epochs"] <= 5
            directory = Path(cfg["training"]["checkpoint_dir"])
            assert not directory.is_relative_to(REPO_ROOT)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "best_model.pt").write_bytes(b"private-weights")
            return {
                "best_loss": 0.1,
                "metrics": {"mae": 0.2},
                "history": [{"epoch": 1, "train_loss": 0.1}],
                "elapsed_seconds": 0.01,
                "best_epoch": 1,
            }

    store = runtime.store
    assert store is not None
    job = store.get_job(store.user_by_email("csv.user@example.com")["id"], job_id)
    assert job is not None
    execute_member_job(store, job, trainer_factory=FakeTrainer)
    finished = client.get(f"/workspace/jobs/{job_id}")
    assert finished.json()["status"] == "completed"
    model_id = finished.json()["result"]["model_id"]
    assert model_id

    # The fake checkpoint is not a real model, but starter predict-latest is.
    starters = client.get("/workspace/starters").json()["starters"]
    weather = next(card for card in starters if card["id"] == "weather")
    latest = client.post(f"/workspace/models/{weather['model_id']}/predict-latest")
    assert latest.status_code == 200, latest.text
    assert isinstance(latest.json()["prediction"], list)
    assert latest.json()["target_column"]
    assert latest.json()["dataset_id"] == weather["dataset_id"]

    _signup(client, "csv.other@example.com")
    _login(client, "csv.other@example.com")
    assert (
        client.post(
            f"/workspace/datasets/{dataset_id}/train",
            json={"target_column": "humidity"},
        ).status_code
        == 404
    )
    assert client.get(f"/workspace/jobs/{job_id}").status_code == 404
