"""Tests for experiment storage."""


from src.storage.experiments import ExperimentStore


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
