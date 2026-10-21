"""Scheduled retraining: discovery, gates, artifacts, and a tiny end-to-end run."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src.retrain.__main__ import main
from src.retrain.artifacts import (
    _trim_records,
    append_history,
    export_release,
    import_release,
    promote_checkpoint,
    publish_release,
    pull_release,
    release_files,
    write_index,
)
from src.retrain.baselines import (
    beats,
    decide_promotion,
    regression_metrics,
    score_baselines,
)
from src.retrain.discover import discover, load_dataset_file
from src.retrain.run import (
    DatasetSpec,
    FetchError,
    FetchPlan,
    RunOptions,
    child_argv,
    run_dataset,
    run_fetch,
    supervise,
)
from src.utils.paths import REPO_ROOT


def _write_config(directory: Path, name: str, body: dict) -> Path:
    config_dir = directory / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / name
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _series_config(tmp_path: Path, *, fetch: object | None = None) -> dict:
    csv_path = tmp_path / "series.csv"
    rows = ["value", *[str(float(1000 + index)) for index in range(40)]]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    body: dict = {
        "dataset": {"id": "series"},
        "model": {
            "name": "time_series_transformer",
            "input_dim": 1,
            "output_dim": 1,
            "d_model": 16,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "max_seq_len": 32,
        },
        "data": {
            "source": "csv",
            "path": str(csv_path),
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 8,
            "val_split": 0.25,
        },
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.01,
            "device": "cpu",
            "seed": 0,
        },
        "storage": {"db_path": str(tmp_path / "experiments.db")},
        "retrain": {"metric": "mae"},
    }
    if fetch is not None:
        body["data"]["fetch"] = fetch
    return body


def _options(tmp_path: Path, **overrides: object) -> RunOptions:
    values: dict = {
        "champions_dir": tmp_path / "champions",
        "forecasts_dir": tmp_path / "forecasts",
        "in_process": True,
        "max_epochs": 1,
    }
    values.update(overrides)
    return RunOptions(**values)


class TestDiscovery:
    def test_shipped_configs_include_fetch_scripts(self):
        specs, errors = discover(REPO_ROOT / "config")
        assert errors == []
        by_id = {spec.id: spec for spec in specs}
        assert {"default", "lottery", "weather"} <= set(by_id)
        assert by_id["default"].fetch is None
        weather = by_id["weather"].fetch
        lottery = by_id["lottery"].fetch
        assert weather is not None and weather.argv[-1].endswith(
            "scripts/fetch_weather.py"
        )
        assert lottery is not None and lottery.argv[-1].endswith(
            "scripts/fetch_powerball.py"
        )

    def test_picks_up_a_new_file_and_skips_notes(self, tmp_path: Path):
        _write_config(
            tmp_path,
            "weather.yaml",
            {"model": {"name": "time_series_transformer"}, "training": {"epochs": 1}},
        )
        _write_config(tmp_path, "notes.yaml", {"title": "not a dataset"})
        _write_config(
            tmp_path,
            "lottery.yaml",
            {
                "dataset": {"id": "lottery", "enabled": False},
                "model": {"name": "time_series_transformer"},
                "training": {"epochs": 1},
            },
        )
        specs, errors = discover(tmp_path / "config")
        assert errors == []
        assert [spec.id for spec in specs] == ["weather"]

    def test_duplicate_ids_and_bad_flags_are_errors(self, tmp_path: Path):
        body = {
            "dataset": {"id": "shared"},
            "model": {"name": "time_series_transformer"},
            "training": {"epochs": 1},
        }
        _write_config(tmp_path, "a.yaml", body)
        _write_config(tmp_path, "b.yaml", body)
        _write_config(
            tmp_path,
            "c.yaml",
            {
                "dataset": {"enabled": "yes"},
                "model": {"name": "time_series_transformer"},
                "training": {"epochs": 1},
            },
        )
        _specs, errors = discover(tmp_path / "config")
        assert any("shared" in error for error in errors)
        assert any("enabled" in error for error in errors)

    def test_conventional_fetch_script(self, tmp_path: Path):
        script = tmp_path / "scripts" / "fetch_weather.py"
        script.parent.mkdir()
        script.write_text("print('ok')\n", encoding="utf-8")
        _write_config(
            tmp_path,
            "weather.yaml",
            {"model": {"name": "m"}, "training": {"epochs": 1}},
        )
        specs, errors = discover(tmp_path / "config")
        assert errors == []
        assert specs[0].fetch is not None
        assert specs[0].fetch.argv == [sys.executable, str(script.resolve())]

    def test_csv_path_stem_finds_a_fetch_script(self, tmp_path: Path):
        script = tmp_path / "scripts" / "fetch_powerball.py"
        script.parent.mkdir()
        script.write_text("print('ok')\n", encoding="utf-8")
        _write_config(
            tmp_path,
            "lottery.yaml",
            {
                "model": {"name": "m"},
                "training": {"epochs": 1},
                "data": {"source": "csv", "path": "data/powerball.csv"},
            },
        )
        specs, errors = discover(tmp_path / "config")
        assert errors == []
        assert specs[0].id == "lottery"
        assert specs[0].fetch is not None
        assert specs[0].fetch.argv[-1] == str(script.resolve())

    def test_explicit_fetch_string_is_split_without_a_shell(self, tmp_path: Path):
        _write_config(
            tmp_path,
            "fx.yaml",
            {
                "model": {"name": "m"},
                "training": {"epochs": 1},
                "data": {"fetch": "python scripts/fetch_fx.py --out data/fx.csv"},
            },
        )
        specs, _errors = discover(tmp_path / "config")
        assert specs[0].fetch is not None
        assert specs[0].fetch.argv == [
            "python",
            "scripts/fetch_fx.py",
            "--out",
            "data/fx.csv",
        ]


class TestGates:
    def test_persistence_and_mean_on_a_known_window(self):
        inputs = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
        targets = torch.tensor([[10.0], [20.0], [30.0]])
        scores = score_baselines(
            ["persistence", "mean", "seasonal"],
            inputs,
            targets,
            [0, 1],
            [2],
            {
                "data": {
                    "season_length": 1,
                    "feature_columns": ["value"],
                    "target_column": "value",
                }
            },
        )
        assert scores["persistence"]["mae"] == pytest.approx(24.0)
        assert scores["mean"]["mae"] == pytest.approx(15.0)
        assert scores["seasonal"]["mae"] == pytest.approx(10.0)

    def test_promotion_requires_beating_champion_and_baselines(self):
        baselines = {
            "persistence": {"mae": 0.4, "rmse": 0.4, "mse": 0.16},
            "mean": {"mae": 0.8, "rmse": 0.8, "mse": 0.64},
        }
        decision, lost = decide_promotion(
            candidate={"mae": 0.1, "rmse": 0.2, "mse": 0.04},
            champion={"mae": 0.3, "rmse": 0.3, "mse": 0.09},
            baselines=baselines,
            metric="mae",
            champion_error=None,
        )
        assert decision == "promoted"
        assert lost == []

        decision, lost = decide_promotion(
            candidate={"mae": 0.2, "rmse": 0.2, "mse": 0.04},
            champion={"mae": 0.1, "rmse": 0.1, "mse": 0.01},
            baselines=baselines,
            metric="mae",
            champion_error=None,
        )
        assert decision == "kept_champion"

        decision, lost = decide_promotion(
            candidate={"mae": 0.5, "rmse": 0.5, "mse": 0.25},
            champion=None,
            baselines=baselines,
            metric="mae",
            champion_error=None,
        )
        assert decision == "rejected_baseline"
        assert "persistence" in lost

        decision, _lost = decide_promotion(
            candidate={"mae": 0.01, "rmse": 0.01, "mse": 0.0},
            champion=None,
            baselines=baselines,
            metric="mae",
            champion_error="architecture changed",
        )
        assert decision == "champion_unevaluable"
        assert not beats(0.2, 0.2)

    def test_metrics_reject_a_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            regression_metrics(torch.zeros(2, 1), torch.zeros(3, 1))


class TestRun:
    def test_fetch_failure_is_recorded_and_does_not_promote(self, tmp_path: Path):
        script = tmp_path / "fail.py"
        script.write_text("raise SystemExit(2)\n", encoding="utf-8")
        path = _write_config(tmp_path, "series.yaml", _series_config(tmp_path))
        spec = DatasetSpec(
            "series",
            path,
            tmp_path,
            FetchPlan([sys.executable, str(script)], timeout=30),
        )
        result = run_dataset(spec, _options(tmp_path))
        assert result.decision == "error"
        assert result.promoted is False
        assert not (tmp_path / "champions" / "series.pt").exists()
        document = json.loads((tmp_path / "forecasts" / "series.json").read_text())
        assert document["decision"] == "error"
        assert "2" in document["error"]

    def test_fetch_then_train_writes_a_forecast_summary(self, tmp_path: Path):
        csv_path = tmp_path / "fresh.csv"
        script = tmp_path / "fetch.py"
        rows = ["value", *[str(float(index)) for index in range(40)]]
        script.write_text(
            "from pathlib import Path\n"
            f"Path({str(csv_path)!r}).write_text({chr(10)!r}.join({rows!r}) + '\\n')\n",
            encoding="utf-8",
        )
        body = _series_config(tmp_path, fetch=[sys.executable, str(script)])
        body["data"]["path"] = str(csv_path)
        path = _write_config(tmp_path, "series.yaml", body)
        spec, error = load_dataset_file(path)
        assert error is None and spec is not None
        result = run_dataset(spec, _options(tmp_path))
        assert result.decision != "error"
        assert csv_path.is_file()
        document = json.loads(result.forecast_path.read_text(encoding="utf-8"))
        assert document["schema_version"] == 1
        assert document["dataset_id"] == "series"
        assert document["held_out"]["split"] == "temporal"
        assert document["metric"] == "mae"
        assert set(document["baselines"]) == {"persistence", "mean"}
        assert "mae" in document["candidate"]
        forecast = document["forecast"]
        assert forecast["as_of"] == "beyond_last_observation"
        assert forecast["horizon"] == 1
        assert len(forecast["values"]) == 1
        champion = tmp_path / "champions" / "series.pt"
        assert champion.is_file() is document["promoted"]

    def test_promotion_keeps_the_champion_file_until_the_challenger_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        path = _write_config(tmp_path, "series.yaml", _series_config(tmp_path))
        spec = DatasetSpec("series", path, tmp_path, None)
        scores = iter([0.1, 0.2])

        def fake_train(_spec, _cfg, _options, _fraction, directory: Path) -> None:
            (directory / "candidate.pt").write_bytes(b"new-weights")

        def fake_score(checkpoint: Path, *_args: object, **_kwargs: object) -> dict:
            mae = next(scores) if checkpoint.name == "candidate.pt" else 0.05
            return {"mae": mae, "rmse": mae, "mse": mae}

        monkeypatch.setattr("src.retrain.run._train_candidate", fake_train)
        monkeypatch.setattr("src.retrain.run.score_checkpoint", fake_score)
        monkeypatch.setattr(
            "src.retrain.run.score_baselines",
            lambda *_args, **_kwargs: {
                "persistence": {"mae": 0.4, "rmse": 0.4, "mse": 0.16},
                "mean": {"mae": 0.8, "rmse": 0.8, "mse": 0.64},
            },
        )
        monkeypatch.setattr(
            "src.retrain.run.forward_window",
            lambda _cfg: (torch.zeros(8, 1), "last_window"),
        )
        monkeypatch.setattr(
            "src.retrain.run.forecast_values", lambda *_args, **_kwargs: [1.25]
        )

        first = run_dataset(spec, _options(tmp_path))
        champion = tmp_path / "champions" / "series.pt"
        assert first.decision == "promoted"
        assert champion.read_bytes() == b"new-weights"

        def fake_train_again(_spec, _cfg, _options, _fraction, directory: Path) -> None:
            (directory / "candidate.pt").write_bytes(b"worse-weights")

        monkeypatch.setattr("src.retrain.run._train_candidate", fake_train_again)
        second = run_dataset(spec, _options(tmp_path))
        assert second.decision == "kept_champion"
        assert second.promoted is False
        assert champion.read_bytes() == b"new-weights"

    def test_supervise_timeout_writes_an_error_forecast(self, tmp_path: Path):
        path = _write_config(tmp_path, "series.yaml", _series_config(tmp_path))
        spec = DatasetSpec("series", path, tmp_path, None)
        result = supervise(
            spec, _options(tmp_path, dataset_timeout=1, in_process=False)
        )
        assert result.decision == "error"
        assert result.error is not None
        assert "Timed out" in result.error

    def test_dry_run_lists_the_shipped_dataset(
        self, capsys: pytest.CaptureFixture[str]
    ):
        code = main(["--all", "--dry-run", "--config-dir", "config"])
        document = json.loads(capsys.readouterr().out)
        assert code == 0
        assert document["datasets"][0]["id"] == "default"
        assert document["datasets"][0]["fetch"] is None

    def test_child_argv_stays_in_process(self, tmp_path: Path):
        spec = DatasetSpec("series", tmp_path / "config.yaml", tmp_path, None)
        argv = child_argv(spec, _options(tmp_path, max_epochs=4))
        assert "--in-process" in argv
        assert "--release-dir" not in argv
        assert argv[argv.index("--max-epochs") + 1] == "4"


class TestArtifacts:
    def test_release_round_trip_and_index_merge(self, tmp_path: Path):
        champions = tmp_path / "champions"
        forecasts = tmp_path / "forecasts"
        release = tmp_path / "release"
        champions.mkdir()
        forecasts.mkdir()
        (champions / "default.pt").write_bytes(b"weights")
        (forecasts / "default.json").write_text("{}\n", encoding="utf-8")
        write_index(
            forecasts,
            [
                {
                    "id": "default",
                    "promoted": True,
                    "decision": "promoted",
                    "error": None,
                }
            ],
        )
        export_release(champions, forecasts, release)
        assert (release / "champion-default.pt").read_bytes() == b"weights"
        assert (release / "forecast-default.json").is_file()
        assert (release / "forecasts-index.json").is_file()

        other = tmp_path / "imported"
        import_release(release, other)
        assert (other / "default.pt").read_bytes() == b"weights"

        merged = write_index(
            forecasts,
            [{"id": "weather", "promoted": False, "decision": "error", "error": "x"}],
            release_dir=release,
        )
        assert [item["id"] for item in merged["datasets"]] == ["default", "weather"]
        names = {path.name for path in release_files(release)}
        assert "champion-default.pt" in names
        (release / "notes.txt").write_text("skip", encoding="utf-8")
        assert "notes.txt" not in {path.name for path in release_files(release)}

    def test_history_keeps_the_newest_records(self, tmp_path: Path):
        path = tmp_path / "metrics-history.json"
        append_history(path, {"dataset_id": "default", "decision": "promoted"})
        records = [{"dataset_id": "default", "n": index} for index in range(405)]
        trimmed = _trim_records(records)
        assert len(trimmed) == 400
        assert trimmed[0]["n"] == 5
        assert trimmed[-1]["n"] == 404

    def test_promote_replaces_atomically(self, tmp_path: Path):
        source = tmp_path / "candidate.pt"
        dest = tmp_path / "champions" / "default.pt"
        source.write_bytes(b"new")
        promote_checkpoint(source, dest)
        assert dest.read_bytes() == b"new"
        assert not dest.with_name(dest.name + ".promoting").exists()

    def test_publish_creates_a_prerelease_and_uploads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        asset = tmp_path / "forecast-default.json"
        asset.write_text("{}\n", encoding="utf-8")
        calls: list[list[str]] = []

        def fake_run(
            args: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            returncode = 1 if args[1:3] == ["release", "view"] else 0
            return subprocess.CompletedProcess(args, returncode, stdout="", stderr="")

        monkeypatch.setattr("src.retrain.artifacts.subprocess.run", fake_run)
        monkeypatch.setenv("GITHUB_REPOSITORY", "AndrewGrayYouNeeK/Real-Neural-Forge")
        publish_release(tmp_path)
        create = next(args for args in calls if "create" in args)
        upload = next(args for args in calls if "upload" in args)
        assert "--prerelease" in create
        assert "--latest=false" in create
        assert "--clobber" in upload
        assert str(asset) in upload
        assert "--repo" in upload

    def test_pull_release_is_quiet_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def fake_run(
            args: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="not found")

        monkeypatch.setattr("src.retrain.artifacts.subprocess.run", fake_run)
        assert pull_release(tmp_path) is False

    def test_fetch_timeout(self, tmp_path: Path):
        with pytest.raises(FetchError, match="timed out"):
            run_fetch(
                FetchPlan(
                    [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1
                ),
                cwd=tmp_path,
            )

    def test_fetch_script_can_import_the_package(self, tmp_path: Path):
        script = tmp_path / "fetch_check.py"
        script.write_text(
            "import src.data.prepare\nprint('imported')\n",
            encoding="utf-8",
        )
        run_fetch(
            FetchPlan([sys.executable, str(script)], timeout=30),
            cwd=REPO_ROOT,
        )
