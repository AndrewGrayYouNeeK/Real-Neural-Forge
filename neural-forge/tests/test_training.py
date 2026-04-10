"""Unit tests for training utilities (optimizer, scheduler, trainer)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_forge.models.mlp import MLP
from neural_forge.training.optimizer import build_optimizer
from neural_forge.training.scheduler import build_scheduler
from neural_forge.training.trainer import Trainer


class TestBuildOptimizer:
    def _model(self):
        return MLP(4, [8], 2)

    def test_adamw(self):
        opt = build_optimizer(self._model(), name="adamw", lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_adam(self):
        opt = build_optimizer(self._model(), name="adam", lr=1e-3)
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd(self):
        opt = build_optimizer(self._model(), name="sgd", lr=1e-2, momentum=0.9)
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_optimizer(self._model(), name="rmsprop")


class TestBuildScheduler:
    def _opt(self):
        model = MLP(4, [8], 2)
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def test_cosine(self):
        sched = build_scheduler(self._opt(), name="cosine", T_max=10)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step(self):
        sched = build_scheduler(self._opt(), name="step", step_size=5)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_scheduler(self._opt(), name="cyclic")


class TestTrainer:
    def _setup(self, n=80):
        model = MLP(4, [16], 3)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, lr=1e-2)

        X = torch.randn(n, 4)
        y = torch.randint(0, 3, (n,))
        train_ds = TensorDataset(X[:60], y[:60])
        val_ds = TensorDataset(X[60:], y[60:])
        train_loader = DataLoader(train_ds, batch_size=16)
        val_loader = DataLoader(val_ds, batch_size=16)
        return model, criterion, optimizer, train_loader, val_loader

    def test_fit_returns_history(self):
        model, crit, opt, train_dl, val_dl = self._setup()
        trainer = Trainer(model, crit, opt, max_epochs=2)
        history = trainer.fit(train_dl, val_dl)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2

    def test_fit_without_val(self):
        model, crit, opt, train_dl, _ = self._setup()
        trainer = Trainer(model, crit, opt, max_epochs=1)
        history = trainer.fit(train_dl)
        assert history["val_loss"] == []
