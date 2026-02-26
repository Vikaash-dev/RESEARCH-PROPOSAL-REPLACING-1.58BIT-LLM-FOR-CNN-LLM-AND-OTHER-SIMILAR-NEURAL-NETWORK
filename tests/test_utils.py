"""
Tests for model surgery utilities and compression analysis.

Validates:
- Layer replacement (nn.Linear → BitLinear, nn.Conv2d → MoQE/ResiBit)
- Sensitivity-aware exclusion (skip narrow-channel layers)
- Weight initialization transfer from original model
- Symmetry-breaking perturbation for dual experts
- Compression ratio calculations matching the paper's Table 1
- Weight distribution analysis for collapse detection
"""

import math

import torch
import torch.nn as nn

from bitnet.bitlinear import BitLinear
from bitnet.moqe_conv import MoQEConv
from bitnet.resbit_conv import ResiBitConv
from bitnet.utils import (
    replace_linear_with_bitlinear,
    replace_conv2d_with_moqe,
    replace_conv2d_with_resbit,
    compute_compression_ratio,
    compute_weight_distribution,
)


class TestReplaceLinear:
    """Tests for nn.Linear → BitLinear replacement."""

    def test_replaces_linear_layers(self):
        """All nn.Linear layers should be replaced with BitLinear."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        model = replace_linear_with_bitlinear(model)

        assert isinstance(model[0], BitLinear)
        assert isinstance(model[2], BitLinear)

    def test_preserves_non_linear_layers(self):
        """Non-Linear layers should be untouched."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        model = replace_linear_with_bitlinear(model)

        assert isinstance(model[1], nn.ReLU)
        assert isinstance(model[2], nn.BatchNorm1d)

    def test_excludes_by_name(self):
        """Layers matching exclude pattern should not be replaced."""
        model = nn.ModuleDict({
            "encoder": nn.Linear(64, 32),
            "head": nn.Linear(32, 10),
        })
        model = replace_linear_with_bitlinear(model, exclude_names=["head"])

        assert isinstance(model["encoder"], BitLinear)
        assert isinstance(model["head"], nn.Linear)

    def test_weights_transferred(self):
        """Original weights should be transferred to BitLinear."""
        model = nn.Sequential(nn.Linear(32, 16))
        original_weight = model[0].weight.data.clone()

        model = replace_linear_with_bitlinear(model)
        assert torch.allclose(model[0].weight.data, original_weight)


class TestReplaceConv2dMoQE:
    """Tests for nn.Conv2d → MoQEConv replacement."""

    def test_replaces_conv_layers(self):
        """Conv2d layers should be replaced with MoQEConv."""
        model = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
        )
        model = replace_conv2d_with_moqe(model)

        assert isinstance(model[0], MoQEConv)
        assert isinstance(model[2], MoQEConv)

    def test_skips_narrow_channels(self):
        """Layers with few output channels should be preserved."""
        model = nn.Sequential(
            nn.Conv2d(16, 4, 3, padding=1),   # Too narrow (4 < 8)
            nn.Conv2d(16, 32, 3, padding=1),  # Should be replaced
        )
        model = replace_conv2d_with_moqe(model, min_channels=8)

        assert isinstance(model[0], nn.Conv2d)  # Preserved
        assert isinstance(model[1], MoQEConv)   # Replaced

    def test_expert_symmetry_breaking(self):
        """Expert B should use 0.1× scaling relative to Expert A (per paper).

        Paper Listing 2, line 56: resibit.w_b.copy_(module.weight * 0.1)
        This breaks symmetry by giving Expert B smaller initial magnitude,
        encouraging it to specialize in low-frequency features.
        """
        model = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1))
        original_weight = model[0].weight.data.clone()
        model = replace_conv2d_with_moqe(model)

        moqe = model[0]
        # Expert A should match original
        assert torch.allclose(moqe.w_expert_A.data, original_weight)
        # Expert B should be 0.1× original
        assert torch.allclose(moqe.w_expert_B.data, original_weight * 0.1)


class TestReplaceConv2dResiBit:
    """Tests for nn.Conv2d → ResiBitConv replacement."""

    def test_replaces_conv_layers(self):
        """Conv2d layers should be replaced with ResiBitConv."""
        model = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
        )
        model = replace_conv2d_with_resbit(model)

        assert isinstance(model[0], ResiBitConv)
        assert isinstance(model[1], ResiBitConv)

    def test_highway_initialized_from_original(self):
        """INT8 highway should be initialized from the original weights."""
        model = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1))
        original_weight = model[0].weight.data.clone()
        model = replace_conv2d_with_resbit(model)

        resbit = model[0]
        # w_a and w_r should match original
        assert torch.allclose(resbit.w_a.data, original_weight)
        assert torch.allclose(resbit.w_r.data, original_weight)

    def test_model_still_runs(self):
        """Replaced model should produce valid output."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Conv2d(16, 32, 3, padding=1),
        )
        model = replace_conv2d_with_resbit(model)

        x = torch.randn(1, 3, 8, 8)
        y = model(x)
        assert y.shape[0] == 1
        assert y.shape[1] == 32
        assert not torch.isnan(y).any()


class TestCompressionRatio:
    """Tests for compression ratio calculations (Paper Table 1)."""

    def test_base5_compression(self):
        """Base-5: 32/log2(5) ≈ 13.78×"""
        result = compute_compression_ratio(1_000_000, method="base5")
        assert abs(result["compression_ratio"] - 13.78) < 0.01

    def test_ternary_compression(self):
        """Ternary: 32/log2(3) ≈ 20.19×"""
        result = compute_compression_ratio(1_000_000, method="ternary")
        assert abs(result["compression_ratio"] - 20.19) < 0.01

    def test_int8_compression(self):
        """INT8: 32/8 = 4.0×"""
        result = compute_compression_ratio(1_000_000, method="int8")
        assert result["compression_ratio"] == 4.0

    def test_int4_compression(self):
        """INT4: 32/4 = 8.0×"""
        result = compute_compression_ratio(1_000_000, method="int4")
        assert result["compression_ratio"] == 8.0

    def test_storage_calculation(self):
        """FP32 storage should be num_weights * 4 bytes."""
        result = compute_compression_ratio(1_000_000, method="base5")
        expected_fp32_mb = 1_000_000 * 32 / 8 / (1024**2)
        assert abs(result["fp32_storage_mb"] - expected_fp32_mb) < 0.01

    def test_yolo26n_compression(self):
        """Validate compression for YOLO26n-like model (2.4M weights).

        Paper Table 2: 2,546,480 quantised weights → 0.705 MB in Base-5
        FP32 baseline: 9.714 MB
        """
        result = compute_compression_ratio(2_546_480, method="base5")
        assert abs(result["fp32_storage_mb"] - 9.714) < 0.01
        assert abs(result["compressed_storage_mb"] - 0.705) < 0.01

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        try:
            compute_compression_ratio(1000, method="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestWeightDistribution:
    """Tests for weight distribution analysis."""

    def test_healthy_distribution(self):
        """Uniform ternary distribution should be flagged as healthy."""
        # Create a balanced ternary tensor
        W = torch.zeros(1000)
        W[:333] = -1
        W[333:666] = 0
        W[666:] = 1

        dist = compute_weight_distribution(W)
        assert dist["is_healthy"]
        assert abs(dist["fraction_negative"] - 0.333) < 0.01
        assert abs(dist["fraction_zero"] - 0.333) < 0.01
        assert abs(dist["fraction_positive"] - 0.334) < 0.01

    def test_collapsed_distribution(self):
        """>50% zeros should be flagged as unhealthy (collapse)."""
        W = torch.zeros(1000)
        W[:100] = -1
        W[900:] = 1
        # 80% zeros

        dist = compute_weight_distribution(W)
        assert not dist["is_healthy"]
        assert dist["fraction_zero"] == 0.8

    def test_paper_collapse_signature(self):
        """Reproduce the paper's collapse signature: ~59% zeros.

        Paper Table 4: Post-QAT Expert A had 59.28% zeros,
        Expert B had 60.52% zeros. Both flagged as collapse.
        """
        # Simulate Expert A distribution from the paper
        W = torch.zeros(10000)
        n_neg = int(0.2106 * 10000)  # 21.06% negative
        n_pos = int(0.1966 * 10000)  # 19.66% positive
        W[:n_neg] = -1
        W[n_neg:n_neg + n_pos] = 1
        # Rest (~59%) stays zero

        dist = compute_weight_distribution(W)
        assert not dist["is_healthy"], (
            f"59% zeros should be detected as collapse, "
            f"got is_healthy={dist['is_healthy']}"
        )
        assert abs(dist["fraction_zero"] - 0.5928) < 0.01
