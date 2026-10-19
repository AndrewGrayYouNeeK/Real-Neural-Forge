"""Tests for experiment storage."""

import gc
import sqlite3
import warnings

from src.storage.experiments import ExperimentStore, db_path_from_config


class TestExperimentStore:
    def test_create_and_list_experiments(self, tmp_path):
        store = ExperimentStore(tmp_path / "experiments.db")
        exp_id = store.create_experiment(
            name="test-run",
            config={"epochs": 1},
            model_name="time_series_transformer",
        )
        experiments = store.list_experiments()
        assert len(experiments) == 1
        assert experiments[0]["id"] == exp_id
        assert experiments[0]["name"] == "test-run"

    def test_log_metrics_and_complete(self, tmp_path):
        store = ExperimentStore(tmp_path / "experiments.db")
        exp_id = store.create_experiment(
            name="metrics-run",
            config={"epochs": 2},
            model_name="time_series_transformer",
        )
        store.log_metric(exp_id, "train_loss", 0.5, epoch=1)
        store.complete_experiment(
            exp_id,
            status="completed",
            best_loss=0.25,
            checkpoint_path="checkpoints/best_model.pt",
            metrics={"mse": 0.25},
        )

        experiment = store.get_experiment(exp_id)
        assert experiment is not None
        assert experiment["status"] == "completed"
        assert experiment["metrics"]["mse"] == 0.25
        assert len(store.get_metrics(exp_id)) == 1

    def test_connections_are_closed(self, tmp_path):
        store = ExperimentStore(tmp_path / "experiments.db")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            exp_id = store.create_experiment(
                name="closed",
                config={"epochs": 1},
                model_name="time_series_transformer",
            )
            store.log_metric(exp_id, "train_loss", 0.5, epoch=1)
            store.complete_experiment(
                exp_id,
                status="completed",
                best_loss=0.5,
                metrics={"mse": 0.5},
            )
            assert store.get_experiment(exp_id) is not None
            assert store.list_experiments()
            assert store.get_metrics(exp_id)
            del store
            gc.collect()
        leaked = [w for w in caught if "unclosed database" in str(w.message).lower()]
        assert leaked == []

    def test_failed_query_closes_connection(self, tmp_path):
        store = ExperimentStore(tmp_path / "experiments.db")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            try:
                with store._connect() as conn:
                    conn.execute("SELECT * FROM missing_table")
            except sqlite3.OperationalError:
                pass
            else:
                raise AssertionError("expected the query to fail")
            del store
            gc.collect()
        leaked = [w for w in caught if "unclosed database" in str(w.message).lower()]
        assert leaked == []

    def test_relative_db_path_resolves_from_repo_root(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("src.utils.paths.REPO_ROOT", repo)
        store = ExperimentStore("custom/experiments.db")
        assert store.db_path == (repo / "custom" / "experiments.db").resolve()
        assert store.db_path.is_file()
        assert not (tmp_path / "custom").exists()


def test_db_path_from_config():
    assert db_path_from_config({"storage": {"db_path": "a/b.db"}}) == "a/b.db"
    assert db_path_from_config({}) == "data/experiments.db"
    assert db_path_from_config(None) == "data/experiments.db"
    assert db_path_from_config({"storage": {"db_path": "  "}}) == "data/experiments.db"
