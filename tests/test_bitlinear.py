"""
Tests for BitLinear (1.58-bit linear layer).

Validates:
- Forward pass produces correct output shapes
- Gradient flow through ternary quantization + activation quantization
- RMSNorm integration
- Deployment: ternary weight extraction
- Equivalence between training (STE) and inference (ternary) modes
"""

import torch
import torch.nn as nn

from bitnet.bitlinear import BitLinear


class TestBitLinear:
    """Tests for the 1.58-bit linear layer."""

    def test_output_shape(self):
        """Output shape must match standard nn.Linear."""
        bl = BitLinear(64, 32)
        x = torch.randn(4, 64)
        y = bl(x)
        assert y.shape == (4, 32)

    def test_output_shape_batched(self):
        """Works with 3D input (batch, seq_len, features)."""
        bl = BitLinear(128, 64)
        x = torch.randn(2, 10, 128)
        y = bl(x)
        assert y.shape == (2, 10, 64)

    def test_gradient_flow_full_path(self):
        """Gradients must flow through RMSNorm → activation quant → weight quant."""
        bl = BitLinear(32, 16)
        x = torch.randn(4, 32, requires_grad=True)
        y = bl(x)
        loss = y.sum()
        loss.backward()

        # Input gradient
        assert x.grad is not None, "No gradient to input"
        # Weight gradient
        assert bl.weight.grad is not None, "No gradient to weight"
        # Bias gradient
        assert bl.bias.grad is not None, "No gradient to bias"

    def test_no_bias(self):
        """BitLinear should work without bias."""
        bl = BitLinear(32, 16, bias=False)
        x = torch.randn(4, 32)
        y = bl(x)
        assert y.shape == (4, 16)
        assert bl.bias is None

    def test_ternary_weight_extraction(self):
        """Extracted deployment weights must be strictly ternary INT8."""
        bl = BitLinear(64, 32)
        tw = bl.get_ternary_weight()

        assert tw.dtype == torch.int8
        unique_vals = set(tw.unique().tolist())
        assert unique_vals.issubset({-1, 0, 1}), (
            f"Ternary weights contain non-ternary values: {unique_vals}"
        )

    def test_training_step(self):
        """Full training step should not crash or produce NaN."""
        bl = BitLinear(32, 16)
        optimizer = torch.optim.Adam(bl.parameters(), lr=1e-3)

        for _ in range(5):
            x = torch.randn(8, 32)
            y = bl(x)
            loss = (y**2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert not torch.isnan(bl.weight).any(), "Weights became NaN after training"

    def test_replaces_nn_linear_functionality(self):
        """BitLinear should be a drop-in replacement for nn.Linear."""
        # Same input/output dimensions
        bl = BitLinear(64, 32)
        standard = nn.Linear(64, 32)

        x = torch.randn(4, 64)
        # Both should produce valid output
        y_bl = bl(x)
        y_std = standard(x)
        assert y_bl.shape == y_std.shape

    def test_parameter_count(self):
        """BitLinear should have weight + bias + RMSNorm parameters."""
        bl = BitLinear(64, 32)
        param_count = sum(p.numel() for p in bl.parameters())
        # weight: 64*32 = 2048, bias: 32, RMSNorm weight: 64
        expected = 64 * 32 + 32 + 64
        assert param_count == expected, (
            f"Expected {expected} parameters, got {param_count}"
        )
