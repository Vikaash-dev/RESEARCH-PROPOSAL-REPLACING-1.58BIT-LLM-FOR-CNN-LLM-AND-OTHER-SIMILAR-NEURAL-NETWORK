"""Unit tests for ResiBitBlock, GroupMix, and integration of components."""

import torch
import pytest

from src.layers.resibit_block import ResiBitBlock
from src.layers.group_mix import GroupMix


class TestGroupMix:
    """Tests for the GroupMix aggregation module."""

    def test_output_shape(self):
        """Fused output has correct channel count."""
        mix = GroupMix(channels=32, num_streams=3)
        s1 = torch.randn(2, 32, 8, 8)
        s2 = torch.randn(2, 32, 8, 8)
        s3 = torch.randn(2, 32, 8, 8)
        out = mix(s1, s2, s3)
        assert out.shape == (2, 32, 8, 8)

    def test_two_streams(self):
        """GroupMix works with 2 streams."""
        mix = GroupMix(channels=16, num_streams=2)
        s1 = torch.randn(1, 16, 4, 4)
        s2 = torch.randn(1, 16, 4, 4)
        out = mix(s1, s2)
        assert out.shape == (1, 16, 4, 4)

    def test_wrong_stream_count_raises(self):
        """Passing wrong number of streams raises ValueError."""
        mix = GroupMix(channels=16, num_streams=3)
        s1 = torch.randn(1, 16, 4, 4)
        s2 = torch.randn(1, 16, 4, 4)
        with pytest.raises(ValueError, match="Expected 3 streams"):
            mix(s1, s2)

    def test_gradient_flow(self):
        """Gradients flow through GroupMix."""
        mix = GroupMix(channels=8, num_streams=2)
        s1 = torch.randn(1, 8, 4, 4, requires_grad=True)
        s2 = torch.randn(1, 8, 4, 4, requires_grad=True)
        out = mix(s1, s2)
        out.sum().backward()
        assert s1.grad is not None
        assert s2.grad is not None


class TestResiBitBlock:
    """Tests for the ResiBit block (dual expert + INT8 highway)."""

    def test_output_shape(self):
        """Block output has correct shape."""
        block = ResiBitBlock(32, 32, kernel_size=3, padding=1)
        x = torch.randn(2, 32, 8, 8)
        output, texture, shape = block(x)
        assert output.shape == (2, 32, 8, 8)

    def test_stride_reduces_spatial(self):
        """Stride=2 halves spatial dims."""
        block = ResiBitBlock(16, 32, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 16, 8, 8)
        output, _, _ = block(x)
        assert output.shape == (2, 32, 4, 4)

    def test_returns_expert_features(self):
        """Block returns texture and shape features for spectral loss."""
        block = ResiBitBlock(32, 32, kernel_size=3, padding=1)
        x = torch.randn(2, 32, 8, 8)
        _, texture, shape = block(x)
        assert texture.ndim == 4
        assert shape.ndim == 4

    def test_int8_scale_default(self):
        """Default INT8 scale is 1.0."""
        block = ResiBitBlock(16, 16, kernel_size=3, padding=1)
        assert block.int8_scale == 1.0

    def test_set_int8_scale(self):
        """set_int8_scale clamps to [0, 1]."""
        block = ResiBitBlock(16, 16, kernel_size=3, padding=1)
        block.set_int8_scale(0.5)
        assert block.int8_scale == 0.5
        block.set_int8_scale(-0.1)
        assert block.int8_scale == 0.0
        block.set_int8_scale(1.5)
        assert block.int8_scale == 1.0

    def test_int8_scale_zero_still_works(self):
        """Block produces valid output with INT8 highway disabled."""
        block = ResiBitBlock(16, 16, kernel_size=3, padding=1)
        block.set_int8_scale(0.0)
        x = torch.randn(1, 16, 4, 4)
        output, _, _ = block(x)
        assert output.shape == (1, 16, 4, 4)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Gradients propagate through all paths."""
        block = ResiBitBlock(16, 16, kernel_size=3, padding=1)
        x = torch.randn(1, 16, 4, 4, requires_grad=True)
        output, _, _ = block(x)
        output.sum().backward()
        assert x.grad is not None
        # Check that both ternary and INT8 paths received gradients
        assert block.dual_expert.texture_expert.conv.weight.grad is not None
        assert block.int8_highway[0].weight.grad is not None

    def test_weight_statistics(self):
        """Weight statistics include dual_expert and int8_scale."""
        block = ResiBitBlock(16, 16, kernel_size=3, padding=1)
        stats = block.weight_statistics()
        assert "dual_expert" in stats
        assert "int8_scale" in stats
