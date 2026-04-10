"""Tests for the TimeSeriesTransformer model."""

import pytest
import torch

from src.model import PositionalEncoding, TimeSeriesTransformer


class TestPositionalEncoding:
    def test_output_shape(self):
        batch, seq_len, d_model = 4, 20, 64
        pe = PositionalEncoding(d_model=d_model)
        x = torch.zeros(batch, seq_len, d_model)
        out = pe(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_adds_information(self):
        """Positional encoding should change the values of the input tensor."""
        batch, seq_len, d_model = 2, 10, 32
        pe = PositionalEncoding(d_model=d_model, dropout=0.0)
        x = torch.zeros(batch, seq_len, d_model)
        out = pe(x)
        assert not torch.allclose(out, x)

    def test_different_positions_differ(self):
        """Different positions must receive different encoding vectors."""
        d_model = 16
        pe = PositionalEncoding(d_model=d_model, dropout=0.0)
        x = torch.zeros(1, 5, d_model)
        out = pe(x)
        # Positions 0 and 1 should differ
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestTimeSeriesTransformer:
    @pytest.fixture
    def default_model(self):
        return TimeSeriesTransformer(
            input_dim=1,
            output_dim=1,
            d_model=32,
            nhead=2,
            num_encoder_layers=2,
            dim_feedforward=64,
            dropout=0.0,
        )

    def test_forward_output_shape(self, default_model):
        batch, seq_len = 8, 30
        x = torch.randn(batch, seq_len, 1)
        out = default_model(x)
        assert out.shape == (batch, 1)

    def test_multi_feature_output_shape(self):
        model = TimeSeriesTransformer(
            input_dim=3,
            output_dim=2,
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=64,
            dropout=0.0,
        )
        x = torch.randn(4, 20, 3)
        out = model(x)
        assert out.shape == (4, 2)

    def test_forward_with_padding_mask(self, default_model):
        batch, seq_len = 4, 15
        x = torch.randn(batch, seq_len, 1)
        mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        mask[:, -3:] = True  # mark last 3 steps as padding
        out = default_model(x, src_key_padding_mask=mask)
        assert out.shape == (batch, 1)

    def test_eval_mode_deterministic(self, default_model):
        default_model.eval()
        x = torch.randn(2, 10, 1)
        with torch.no_grad():
            out1 = default_model(x)
            out2 = default_model(x)
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self, default_model):
        default_model.train()
        x = torch.randn(2, 10, 1)
        out = default_model(x)
        loss = out.sum()
        loss.backward()
        # Check at least one gradient is non-zero
        grads = [
            p.grad
            for p in default_model.parameters()
            if p.grad is not None
        ]
        assert len(grads) > 0
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_parameter_count(self, default_model):
        n_params = sum(p.numel() for p in default_model.parameters())
        assert n_params > 0
