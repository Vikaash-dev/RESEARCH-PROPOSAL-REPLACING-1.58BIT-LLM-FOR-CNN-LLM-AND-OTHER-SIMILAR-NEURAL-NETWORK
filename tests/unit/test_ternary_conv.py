"""Unit tests for TernaryConv2d and TernaryQuantize (STE).

Validates:
- Output shape matches standard Conv2d
- Weights are constrained to {-1, 0, +1} after quantization
- Gradients flow correctly through Straight-Through Estimator
- Threshold behaviour and gradient clipping
"""

import torch
import pytest

from src.layers.ternary_conv import TernaryConv2d, TernaryQuantize


class TestTernaryQuantize:
    """Tests for the ternary quantization autograd function."""

    def test_output_values_in_ternary_set(self):
        """Quantized weights must be in {-1, 0, +1}."""
        w = torch.randn(64, 32, 3, 3)
        q = TernaryQuantize.apply(w)
        unique = q.unique().tolist()
        assert all(v in [-1.0, 0.0, 1.0] for v in unique)

    def test_output_shape_preserved(self):
        """Quantization preserves tensor shape."""
        w = torch.randn(16, 8, 3, 3)
        q = TernaryQuantize.apply(w)
        assert q.shape == w.shape

    def test_gradient_flow_ste(self):
        """Gradients pass through STE (straight-through estimator)."""
        w = torch.randn(8, 4, 3, 3, requires_grad=True)
        q = TernaryQuantize.apply(w)
        loss = q.sum()
        loss.backward()
        # Gradient should exist and be non-zero for weights in [-1, 1]
        assert w.grad is not None
        assert w.grad.abs().sum() > 0

    def test_gradient_clipped_outside_range(self):
        """Gradients are zeroed for weights with |w| > 1.0."""
        w = torch.tensor([0.5, 2.0, -0.3, -1.5], requires_grad=True)
        q = TernaryQuantize.apply(w)
        loss = q.sum()
        loss.backward()
        # Weights at index 1 and 3 have |w| > 1, gradient should be 0
        assert w.grad[1].item() == 0.0
        assert w.grad[3].item() == 0.0
        # Weights at index 0 and 2 should have gradient 1.0
        assert w.grad[0].item() == 1.0
        assert w.grad[2].item() == 1.0

    def test_threshold_adaptive(self):
        """Threshold is 5% of mean absolute value."""
        # Large weights: threshold is higher, more zeros possible
        w_large = torch.ones(10) * 10.0
        w_large[0] = 0.01  # This is below 5% of mean(10) = 0.5
        q = TernaryQuantize.apply(w_large)
        assert q[0].item() == 0.0  # Below threshold → zero

    def test_all_zero_input(self):
        """All-zero input produces all-zero output."""
        w = torch.zeros(4, 4, 3, 3)
        q = TernaryQuantize.apply(w)
        assert (q == 0).all()


class TestTernaryConv2d:
    """Tests for the TernaryConv2d layer."""

    def test_output_shape_matches_conv2d(self):
        """Output shape must match standard Conv2d."""
        layer = TernaryConv2d(16, 32, kernel_size=3, padding=1)
        x = torch.randn(2, 16, 8, 8)
        y = layer(x)
        assert y.shape == (2, 32, 8, 8)

    def test_stride_changes_spatial(self):
        """Stride reduces spatial dimensions correctly."""
        layer = TernaryConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 16, 8, 8)
        y = layer(x)
        assert y.shape == (2, 32, 4, 4)

    def test_gradients_propagate(self):
        """Backpropagation works through TernaryConv2d."""
        layer = TernaryConv2d(8, 16, kernel_size=3, padding=1)
        x = torch.randn(1, 8, 4, 4, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.conv.weight.grad is not None

    def test_weight_statistics(self):
        """Weight statistics report valid fractions summing to ~1.0."""
        layer = TernaryConv2d(16, 32, kernel_size=3, padding=1)
        stats = layer.weight_statistics()
        total = stats["positive"] + stats["zero"] + stats["negative"]
        assert abs(total - 1.0) < 1e-6

    def test_get_ternary_weights(self):
        """get_ternary_weights returns tensor in {-1, 0, +1}."""
        layer = TernaryConv2d(8, 16, kernel_size=3, padding=1)
        w = layer.get_ternary_weights()
        unique = w.unique().tolist()
        assert all(v in [-1.0, 0.0, 1.0] for v in unique)

    def test_groups_support(self):
        """TernaryConv2d supports grouped convolutions."""
        layer = TernaryConv2d(16, 16, kernel_size=3, padding=1, groups=4)
        x = torch.randn(1, 16, 8, 8)
        y = layer(x)
        assert y.shape == (1, 16, 8, 8)

    def test_properties(self):
        """Properties expose correct channel counts."""
        layer = TernaryConv2d(16, 32, kernel_size=3)
        assert layer.in_channels == 16
        assert layer.out_channels == 32
