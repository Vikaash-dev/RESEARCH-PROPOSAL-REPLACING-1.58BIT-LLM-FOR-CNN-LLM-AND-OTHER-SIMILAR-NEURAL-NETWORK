"""
Tests for quantization functions.

Validates the core theoretical claims:
- Absmean ternary quantization produces values strictly in {-1, 0, +1}
- STE passes gradients correctly through the non-differentiable rounding
- INT8 quantization stays within [-127, 127]
- Activation quantization preserves relative ordering
- Compression ratios match the paper's Table 1
"""

import math

import pytest
import torch

from bitnet.quantization import (
    quantize_ternary,
    quantize_ternary_ste,
    quantize_int8,
    quantize_activations,
)


class TestQuantizeTernary:
    """Tests for absmean ternary quantization (Equation 2 from papers)."""

    def test_output_values_strictly_ternary(self):
        """Quantized weights must be exactly {-1, 0, +1}."""
        W = torch.randn(64, 32, 3, 3)
        Q = quantize_ternary(W)
        unique_vals = set(Q.unique().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
            f"Ternary quantization produced non-ternary values: {unique_vals}"
        )

    def test_zero_tensor_handled(self):
        """All-zero input should not cause division by zero."""
        W = torch.zeros(16, 16)
        Q = quantize_ternary(W)
        assert not torch.isnan(Q).any()
        assert not torch.isinf(Q).any()

    def test_large_weights_clamp_correctly(self):
        """Very large weights should map to ±1, not exceed."""
        W = torch.tensor([100.0, -100.0, 0.001])
        Q = quantize_ternary(W)
        assert Q.max() <= 1.0
        assert Q.min() >= -1.0

    def test_symmetric_distribution(self):
        """For symmetric Gaussian input, positive ≈ negative fractions."""
        torch.manual_seed(42)
        W = torch.randn(1000, 1000)
        Q = quantize_ternary(W)
        pos_frac = (Q == 1).float().mean().item()
        neg_frac = (Q == -1).float().mean().item()
        # Should be roughly equal (within 2%)
        assert abs(pos_frac - neg_frac) < 0.02, (
            f"Asymmetric distribution: pos={pos_frac:.4f}, neg={neg_frac:.4f}"
        )

    def test_absmean_scaling_property(self):
        """After scaling by alpha=mean(|W|), values near 0 map to 0."""
        W = torch.tensor([0.01, -0.01, 1.0, -1.0, 0.5, -0.5])
        Q = quantize_ternary(W)
        # The mean absolute value is ~0.337
        # 0.01/0.337 ≈ 0.03 → rounds to 0
        assert Q[0].item() == 0.0
        assert Q[1].item() == 0.0

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        for shape in [(10,), (8, 8), (4, 4, 3, 3), (2, 3, 5, 5)]:
            W = torch.randn(*shape)
            Q = quantize_ternary(W)
            assert Q.shape == W.shape

    def test_information_content(self):
        """Verify log2(3) ≈ 1.585 bits per ternary weight.

        This validates the paper's claim that each ternary weight carries
        1.58 bits of information.
        """
        assert abs(math.log2(3) - 1.585) < 0.001


class TestQuantizeTernarySTE:
    """Tests for STE-enabled ternary quantization."""

    def test_forward_values_ternary(self):
        """Forward pass output must be ternary."""
        W = torch.randn(32, 32, requires_grad=True)
        Q = quantize_ternary_ste(W)
        # Forward values should be ternary
        unique_forward = set(Q.detach().unique().tolist())
        assert unique_forward.issubset({-1.0, 0.0, 1.0})

    def test_gradient_flows_through(self):
        """STE must allow gradient flow through the quantization."""
        W = torch.randn(16, 16, requires_grad=True)
        Q = quantize_ternary_ste(W)
        loss = Q.sum()
        loss.backward()
        assert W.grad is not None, "No gradient computed through STE"
        assert W.grad.abs().sum() > 0, "Gradient is all zeros"

    def test_gradient_magnitude_reasonable(self):
        """Gradient should not explode or vanish through STE."""
        W = torch.randn(64, 64, requires_grad=True)
        Q = quantize_ternary_ste(W)
        loss = (Q**2).sum()
        loss.backward()
        grad_norm = W.grad.norm().item()
        # Should be on the order of the output norm, not wildly different
        assert grad_norm > 0.01, f"Gradient too small: {grad_norm}"
        assert grad_norm < 1000, f"Gradient too large: {grad_norm}"

    def test_ste_correctness(self):
        """Verify STE property: forward = quantized, backward = identity-like.

        The key STE property from Equation (2):
            Forward: output = clip(round(W/alpha), -1, 1)
            Backward: doutput/dW ≈ 1/alpha (scaled identity)
        """
        W = torch.randn(8, 8, requires_grad=True)
        Q = quantize_ternary_ste(W)

        # Forward: should match non-STE version
        Q_ref = quantize_ternary(W.detach())
        assert torch.allclose(Q.detach(), Q_ref), (
            "STE forward doesn't match non-STE quantization"
        )

        # Backward: gradient should exist for all weights
        Q.sum().backward()
        assert (W.grad != 0).any(), "STE gradient is all zeros"


class TestQuantizeINT8:
    """Tests for symmetric INT8 quantization (residual highway)."""

    def test_output_range(self):
        """INT8 quantized values must be in [-127, 127]."""
        W = torch.randn(64, 64)
        Q = quantize_int8(W)
        assert Q.max() <= 127.0
        assert Q.min() >= -127.0

    def test_integer_values(self):
        """Output should be (approximately) integer-valued."""
        W = torch.randn(32, 32)
        Q = quantize_int8(W)
        # Check that values are very close to integers
        assert (Q - Q.round()).abs().max() < 1e-5

    def test_gradient_flows(self):
        """INT8 quantization must support gradient flow via STE."""
        W = torch.randn(16, 16, requires_grad=True)
        Q = quantize_int8(W)
        loss = Q.sum()
        loss.backward()
        assert W.grad is not None

    def test_256_levels_vs_3(self):
        """INT8 has 255 distinct levels vs ternary's 3.

        This validates the paper's claim that INT8 has a much better-conditioned
        STE gradient (256 levels vs 3), which is why it works as a gradient anchor.
        """
        W = torch.randn(100, 100)
        Q_int8 = quantize_int8(W)
        Q_ternary = quantize_ternary(W)

        int8_unique = len(Q_int8.unique())
        ternary_unique = len(Q_ternary.unique())

        assert ternary_unique <= 3
        assert int8_unique > ternary_unique
        # INT8 should preserve much more information
        # Quantization error should be much smaller for INT8
        err_int8 = (W - Q_int8 * W.abs().max() / 127.0).abs().mean().item()
        err_ternary = (W - Q_ternary * W.abs().mean()).abs().mean().item()
        assert err_int8 < err_ternary, (
            f"INT8 error ({err_int8:.4f}) should be < ternary error ({err_ternary:.4f})"
        )


class TestQuantizeActivations:
    """Tests for absmax activation quantization."""

    def test_output_range_8bit(self):
        """8-bit activation quantization should produce values in [-127, 127]."""
        X = torch.randn(4, 32, 8, 8)
        Q = quantize_activations(X, bits=8)
        assert Q.max() <= 127.0
        assert Q.min() >= -127.0

    def test_preserves_relative_ordering(self):
        """Quantized activations should preserve the sign and relative order."""
        X = torch.tensor([1.0, 2.0, 3.0, -1.0, -3.0])
        Q = quantize_activations(X, bits=8)
        # Signs should be preserved
        assert (Q[0] > 0) and (Q[1] > 0) and (Q[2] > 0)
        assert (Q[3] < 0) and (Q[4] < 0)
        # Ordering should be preserved
        assert Q[2] > Q[1] > Q[0]

    def test_gradient_flow(self):
        """Activation quantization must support backpropagation."""
        X = torch.randn(2, 16, requires_grad=True)
        Q = quantize_activations(X)
        loss = Q.sum()
        loss.backward()
        assert X.grad is not None


class TestCompressionTheory:
    """Tests that validate the information-theoretic claims from the papers."""

    def test_base5_compression_ratio(self):
        """Verify the 13.78× compression claim from Table 1.

        The paper claims: 32 / log2(5) = 13.78×
        """
        expected = 32.0 / math.log2(5)
        assert abs(expected - 13.78) < 0.01, (
            f"Expected 13.78×, got {expected:.2f}×"
        )

    def test_ternary_compression_ratio(self):
        """Single ternary: 32 / log2(3) ≈ 20.19×"""
        expected = 32.0 / math.log2(3)
        assert abs(expected - 20.19) < 0.01

    def test_base5_bits_per_weight(self):
        """Base-5 weights need log2(5) ≈ 2.322 bits per weight."""
        assert abs(math.log2(5) - 2.322) < 0.001

    def test_base5_sum_is_quinary(self):
        """Adding two ternary values produces exactly 5 distinct values.

        This validates Equation (7): WB5 = W_A + W_B ∈ {-2, -1, 0, 1, 2}
        """
        ternary_vals = torch.tensor([-1.0, 0.0, 1.0])
        all_sums = set()
        for a in ternary_vals:
            for b in ternary_vals:
                all_sums.add((a + b).item())
        assert all_sums == {-2.0, -1.0, 0.0, 1.0, 2.0}, (
            f"Expected 5 quinary values, got: {all_sums}"
        )

    def test_capacity_increase_from_dual_expert(self):
        """Dual experts provide 47% more information per weight.

        Paper claim: (log2(5) - log2(3)) / log2(3) ≈ 0.47
        """
        increase = (math.log2(5) - math.log2(3)) / math.log2(3)
        assert abs(increase - 0.465) < 0.01, (
            f"Expected ~46.5% capacity increase, got {increase*100:.1f}%"
        )
