"""Unit tests for DualExpertBlock with Static Channel Splitting."""

import torch
import pytest

from src.layers.dual_expert import DualExpertBlock


class TestDualExpertBlock:
    """Tests for the DualExpertBlock layer."""

    def test_output_shape(self):
        """Block output has correct shape (B, C_out, H', W')."""
        block = DualExpertBlock(32, 32, kernel_size=3, padding=1)
        x = torch.randn(2, 32, 8, 8)
        merged, texture, shape = block(x)
        assert merged.shape == (2, 32, 8, 8)

    def test_channel_split_ratio(self):
        """SCS splits channels according to ratio."""
        block = DualExpertBlock(32, 32, scs_ratio=0.75)
        assert block.texture_in == 24  # 32 * 0.75
        assert block.shape_in == 8     # 32 - 24

    def test_expert_independence(self):
        """Texture and shape experts have separate weight parameters."""
        block = DualExpertBlock(16, 16, kernel_size=3, padding=1)
        # They should be different nn.Module instances
        assert block.texture_expert is not block.shape_expert
        # Their weights should not be the same object
        assert block.texture_expert.weight is not block.shape_expert.weight

    def test_texture_shape_features_returned(self):
        """Forward returns texture and shape features for spectral loss."""
        block = DualExpertBlock(32, 32, kernel_size=3, padding=1)
        x = torch.randn(2, 32, 8, 8)
        merged, texture, shape = block(x)
        assert texture.shape[1] == block.texture_out
        assert shape.shape[1] == block.shape_out
        assert texture.shape[1] + shape.shape[1] == 32

    def test_stride_reduces_spatial(self):
        """Stride=2 halves spatial dimensions."""
        block = DualExpertBlock(16, 32, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 16, 8, 8)
        merged, _, _ = block(x)
        assert merged.shape == (2, 32, 4, 4)

    def test_gradients_flow(self):
        """Gradients propagate through both experts."""
        block = DualExpertBlock(16, 16, kernel_size=3, padding=1)
        x = torch.randn(1, 16, 4, 4, requires_grad=True)
        merged, _, _ = block(x)
        loss = merged.sum()
        loss.backward()
        assert x.grad is not None
        assert block.texture_expert.conv.weight.grad is not None
        assert block.shape_expert.conv.weight.grad is not None

    def test_weight_statistics(self):
        """Weight statistics report valid fractions for both experts."""
        block = DualExpertBlock(16, 16, kernel_size=3, padding=1)
        stats = block.weight_statistics()
        assert "texture_expert" in stats
        assert "shape_expert" in stats
        for expert_stats in stats.values():
            total = expert_stats["positive"] + expert_stats["zero"] + expert_stats["negative"]
            assert abs(total - 1.0) < 1e-6

    def test_invalid_scs_ratio(self):
        """Invalid SCS ratios raise ValueError."""
        with pytest.raises(ValueError, match="scs_ratio"):
            DualExpertBlock(16, 16, scs_ratio=0.0)
        with pytest.raises(ValueError, match="scs_ratio"):
            DualExpertBlock(16, 16, scs_ratio=1.0)

    def test_too_few_channels_raises(self):
        """Very small channel count with extreme ratio raises ValueError."""
        with pytest.raises(ValueError):
            DualExpertBlock(2, 2, scs_ratio=0.01)
