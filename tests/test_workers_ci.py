"""Tests for the Workers Builds wrangler.jsonc writer."""

import importlib.util
import json
from pathlib import Path

import pytest


def _writer():
    path = Path(__file__).resolve().parents[1] / "workers_ci" / "write_wrangler.py"
    spec = importlib.util.spec_from_file_location("write_wrangler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skips_when_worker_name_missing(tmp_path):
    written = _writer().write_wrangler_config(tmp_path, None)
    assert written is None
    assert not (tmp_path / "wrangler.jsonc").exists()
    written = _writer().write_wrangler_config(tmp_path, "   ")
    assert written is None


def test_writes_config_for_each_connected_worker(tmp_path):
    writer = _writer()
    path = writer.write_wrangler_config(tmp_path, "real-neural-forge")
    assert path == tmp_path / "wrangler.jsonc"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["name"] == "real-neural-forge"
    assert cfg["assets"]["directory"] == "frontend"
    assert cfg["compatibility_date"] == "2026-10-12"
    assert "routes" not in cfg


def test_youneek_xyz_build_publishes_a_different_worker(tmp_path):
    """A push must not deploy the portfolio Worker or claim its hostnames."""
    writer = _writer()
    path = writer.write_wrangler_config(tmp_path, "youneek-xyz")
    assert path is not None
    text = path.read_text(encoding="utf-8")
    cfg = json.loads(text)
    assert cfg["name"] == "real-neural-forge-frontend"
    assert cfg["name"] != "youneek-xyz"
    assert "youneek-xyz" not in text
    assert "routes" not in cfg
    for host in (
        "youneek.xyz",
        "www.youneek.xyz",
        "forge.youneek.xyz",
        "forge.youneek.dev",
    ):
        assert host not in text


def test_rejects_invalid_worker_name(tmp_path):
    writer = _writer()
    for name in ("../etc", "Real-Neural-Forge", "-bad", "bad-", "a" * 64):
        with pytest.raises(ValueError):
            writer.write_wrangler_config(tmp_path, name)
    assert not (tmp_path / "wrangler.jsonc").exists()
