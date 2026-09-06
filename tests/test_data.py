"""Tests for dataset loading and windowing."""

from pathlib import Path

import pytest
import torch

from src.data import (
    load_csv_series,
    load_train_val,
    make_sine_dataset,
    temporal_split,
    window_series,
)
from src.train import evaluate, load_config


def _write_csv(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


class TestLoadCsvSeries:
    def test_header_value_column(self, tmp_path):
        csv_path = _write_csv(
            tmp_path / "series.csv",
            ["t,value", "0,1.0", "1,2.0", "2,3.5"],
        )
        series = load_csv_series(csv_path)
        assert torch.allclose(series, torch.tensor([1.0, 2.0, 3.5]))

    def test_explicit_column(self, tmp_path):
        csv_path = _write_csv(
            tmp_path / "ohlc.csv",
            ["open,close,volume", "10,11,100", "12,13,110"],
        )
        series = load_csv_series(csv_path, value_column="close")
        assert torch.allclose(series, torch.tensor([11.0, 13.0]))

    def test_single_column_no_header(self, tmp_path):
        csv_path = _write_csv(tmp_path / "plain.csv", ["0.1", "0.2", "0.3"])
        series = load_csv_series(csv_path)
        assert torch.allclose(series, torch.tensor([0.1, 0.2, 0.3]))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv_series(tmp_path / "missing.csv")

    def test_unknown_column(self, tmp_path):
        csv_path = _write_csv(tmp_path / "series.csv", ["t,value", "0,1"])
        with pytest.raises(ValueError, match="not found"):
            load_csv_series(csv_path, value_column="price")

    def test_empty_file(self, tmp_path):
        csv_path = _write_csv(tmp_path / "empty.csv", [])
        with pytest.raises(ValueError, match="empty"):
            load_csv_series(csv_path)


class TestWindowAndSplit:
    def test_window_shapes(self):
        series = torch.arange(10, dtype=torch.float32)
        inputs, targets = window_series(series, seq_len=4)
        assert inputs.shape == (6, 4, 1)
        assert targets.shape == (6, 1)
        assert torch.equal(inputs[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
        assert targets[0].item() == 4.0
        assert targets[-1].item() == 9.0

    def test_window_too_short(self):
        with pytest.raises(ValueError, match="Need more than"):
            window_series(torch.arange(4, dtype=torch.float32), seq_len=4)

    def test_temporal_split_is_contiguous(self):
        series = torch.arange(20, dtype=torch.float32)
        inputs, targets = window_series(series, seq_len=5)
        train_x, train_y, val_x, val_y = temporal_split(inputs, targets, val_split=0.25)
        assert train_x.size(0) + val_x.size(0) == inputs.size(0)
        # Last train target is immediately before first val target
        assert train_y[-1].item() < val_y[0].item()

    def test_temporal_split_rejects_bad_ratio(self):
        x = torch.zeros(4, 2, 1)
        y = torch.zeros(4, 1)
        with pytest.raises(ValueError, match="val_split"):
            temporal_split(x, y, val_split=1.0)


class TestLoadTrainVal:
    def test_sine_fallback_default_config(self):
        cfg = load_config("config/config.yaml")
        train_x, train_y, val_x, val_y, source = load_train_val(cfg)
        assert source == "sine"
        assert train_x.size(0) > val_x.size(0) > 0
        assert train_x.shape[1:] == (64, 1)
        assert train_y.shape[1:] == (1,)
        assert val_x.shape[1:] == (64, 1)

    def test_csv_source(self, tmp_path):
        values = [f"{i},{i * 0.1:.4f}" for i in range(80)]
        csv_path = _write_csv(tmp_path / "series.csv", ["t,value", *values])
        cfg = {
            "data": {
                "csv_path": str(csv_path),
                "seq_len": 8,
                "val_split": 0.25,
            }
        }
        train_x, train_y, val_x, val_y, source = load_train_val(cfg)
        assert source.startswith("csv:")
        assert train_x.size(0) + val_x.size(0) == 80 - 8
        assert val_x.size(0) == pytest.approx((80 - 8) * 0.25, abs=1)


class TestEvaluate:
    def test_perfect_model_is_zero(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x[:, -1, :]

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.ones(6, 4, 1),
                torch.ones(6, 1),
            ),
            batch_size=3,
        )
        mse, mae = evaluate(Identity(), loader, torch.device("cpu"))
        assert mse == pytest.approx(0.0)
        assert mae == pytest.approx(0.0)
