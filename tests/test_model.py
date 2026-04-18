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


class TestPositionalEncodingEdgeCases:
    def test_max_sequence_length(self):
        """Test positional encoding with maximum sequence length."""
        d_model, max_len = 32, 512
        pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
        x = torch.zeros(1, max_len, d_model)
        out = pe(x)
        assert out.shape == (1, max_len, d_model)

    def test_dropout_in_training_mode(self):
        """Test that dropout is applied in training mode."""
        d_model, dropout = 32, 0.5
        pe = PositionalEncoding(d_model=d_model, dropout=dropout)
        pe.train()
        x = torch.zeros(2, 10, d_model)

        # Run multiple times - dropout should produce different results
        outputs = [pe(x.clone()) for _ in range(3)]
        # At least some outputs should differ due to dropout
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Dropout should produce varied outputs in training mode"

    def test_batch_independence(self):
        """Test that positional encoding is batch-independent."""
        d_model = 32
        pe = PositionalEncoding(d_model=d_model, dropout=0.0)

        # Single batch item
        x1 = torch.zeros(1, 10, d_model)
        out1 = pe(x1)

        # Multiple batch items
        x2 = torch.zeros(3, 10, d_model)
        out2 = pe(x2)

        # First batch item should match single item
        assert torch.allclose(out1[0], out2[0])


class TestTimeSeriesTransformerEdgeCases:
    def test_very_long_sequence(self):
        """Test transformer with a very long sequence."""
        model = TimeSeriesTransformer(
            input_dim=1,
            output_dim=1,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_seq_len=1024,
        )
        batch, seq_len = 2, 500
        x = torch.randn(batch, seq_len, 1)
        out = model(x)
        assert out.shape == (batch, 1)

    def test_single_timestep(self):
        """Test transformer with single timestep input."""
        model = TimeSeriesTransformer(
            input_dim=1,
            output_dim=1,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        x = torch.randn(4, 1, 1)
        out = model(x)
        assert out.shape == (4, 1)

    def test_large_dimensions(self):
        """Test transformer with large embedding dimensions."""
        model = TimeSeriesTransformer(
            input_dim=10,
            output_dim=5,
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
        )
        x = torch.randn(2, 20, 10)
        out = model(x)
        assert out.shape == (2, 5)

    def test_model_state_dict_saveable(self):
        """Test that model state dict can be saved and loaded."""
        model = TimeSeriesTransformer(
            input_dim=1,
            output_dim=1,
            d_model=32,
            nhead=2,
            num_encoder_layers=2,
            dim_feedforward=64,
            dropout=0.0,
        )

        # Save state
        state_dict = model.state_dict()

        # Create new model and load state
        new_model = TimeSeriesTransformer(
            input_dim=1,
            output_dim=1,
            d_model=32,
            nhead=2,
            num_encoder_layers=2,
            dim_feedforward=64,
            dropout=0.0,
        )
        new_model.load_state_dict(state_dict)

        # Test that outputs match
        model.eval()
        new_model.eval()
        x = torch.randn(2, 10, 1)
        with torch.no_grad():
            out1 = model(x)
            out2 = new_model(x)
        assert torch.allclose(out1, out2)

    def test_backward_pass_with_mask(self):
        """Test that gradients flow correctly with padding mask."""
        model = TimeSeriesTransformer(
            input_dim=1,
            output_dim=1,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model.train()

        x = torch.randn(2, 10, 1, requires_grad=True)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, -3:] = True  # Mask last 3 positions

        out = model(x, src_key_padding_mask=mask)
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0
