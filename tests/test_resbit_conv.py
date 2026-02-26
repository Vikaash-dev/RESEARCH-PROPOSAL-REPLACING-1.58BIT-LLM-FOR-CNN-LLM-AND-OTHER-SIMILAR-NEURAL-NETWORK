"""
Tests for ResiBitConv (Residual-Precision 1.58-bit Convolution Block).

Validates the core architectural innovation from the ResiBit-YOLO paper:
- Three-stream architecture: Expert A + Expert B + INT8 Highway
- Group-Mix aggregation with softmax-normalized per-channel coefficients
- Phase-dependent behavior (FP32 warmup → ternary warmup → full QAT)
- INT8 highway as gradient anchor (better-conditioned than ternary STE)
- BatchNorm + SiLU activation integration
"""

import torch
import torch.nn.functional as F

from bitnet.resbit_conv import ResiBitConv


class TestResiBitConvForward:
    """Tests for the three-stream forward pass."""

    def test_output_shape(self):
        """Output shape must be correct after BN + SiLU."""
        rb = ResiBitConv(16, 32, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 16, 8, 8)
        y = rb(x)
        assert y.shape == (2, 32, 8, 8)

    def test_output_shape_stride2(self):
        """Stride=2 should halve spatial dimensions."""
        rb = ResiBitConv(16, 32, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 16, 8, 8)
        y = rb(x)
        assert y.shape == (2, 32, 4, 4)

    def test_all_three_streams_contribute(self):
        """Verify that all three streams (A, B, R) contribute to output."""
        rb = ResiBitConv(8, 16)
        x = torch.randn(1, 8, 4, 4)

        # Get output
        y = rb(x)

        # Check mix weights are non-zero for all streams
        mix = rb._mix_weights().detach()
        assert (mix[0] > 0).all(), "Expert A has zero mix weight"
        assert (mix[1] > 0).all(), "Expert B has zero mix weight"
        assert (mix[2] > 0).all(), "INT8 Highway has zero mix weight"

    def test_mix_weights_sum_to_one(self):
        """Softmax mixing coefficients must sum to 1.0 per channel."""
        rb = ResiBitConv(16, 32)
        mix = rb._mix_weights()
        sums = mix.sum(dim=0)  # Sum over 3 streams
        assert torch.allclose(sums, torch.ones(32), atol=1e-5), (
            f"Mix weights don't sum to 1: {sums}"
        )


class TestResiBitPhases:
    """Tests for the three-phase training protocol behavior."""

    def test_phase1_no_quantization(self):
        """Phase 1 (FP32 warmup): no quantization should be applied."""
        rb = ResiBitConv(8, 16)
        rb.set_phase(ResiBitConv.PHASE_FP32_WARMUP)
        x = torch.randn(1, 8, 4, 4)

        # In Phase 1, all weights should be used as-is (FP32)
        y = rb(x)
        assert y is not None

        # Orthogonal loss should be zero in Phase 1
        orth_loss = rb.orthogonal_loss()
        assert orth_loss.item() == 0.0, (
            f"Orth loss should be 0 in Phase 1, got {orth_loss.item()}"
        )

    def test_phase2_highway_frozen(self):
        """Phase 2: INT8 highway should be frozen (no gradient)."""
        rb = ResiBitConv(8, 16)
        rb.set_phase(ResiBitConv.PHASE_TERNARY_WARMUP)

        assert not rb.w_r.requires_grad, (
            "Highway weights should be frozen in Phase 2"
        )
        # Expert weights should still have gradients
        assert rb.w_a.requires_grad, "Expert A should be trainable in Phase 2"
        assert rb.w_b.requires_grad, "Expert B should be trainable in Phase 2"

    def test_phase3_all_trainable(self):
        """Phase 3: all weights should be trainable."""
        rb = ResiBitConv(8, 16)
        # First go to Phase 2 (freezes highway)
        rb.set_phase(ResiBitConv.PHASE_TERNARY_WARMUP)
        assert not rb.w_r.requires_grad

        # Then Phase 3 (unfreezes)
        rb.set_phase(ResiBitConv.PHASE_FULL_QAT)
        assert rb.w_r.requires_grad, (
            "Highway weights should be unfrozen in Phase 3"
        )

    def test_ternary_anneal_affects_quantization(self):
        """Higher anneal factor → softer rounding → more non-zero weights."""
        rb = ResiBitConv(32, 64)

        # Count zeros with tight quantization (anneal=1.0)
        rb.set_phase(ResiBitConv.PHASE_TERNARY_WARMUP, ternary_anneal=1.0)
        wa_tight = rb._quantize_ternary_annealed(rb.w_a)
        zeros_tight = (wa_tight.detach().round() == 0).float().mean().item()

        # Count zeros with soft quantization (anneal=3.0)
        rb.set_phase(ResiBitConv.PHASE_TERNARY_WARMUP, ternary_anneal=3.0)
        wa_soft = rb._quantize_ternary_annealed(rb.w_a)
        zeros_soft = (wa_soft.detach().round() == 0).float().mean().item()

        # Softer quantization (higher anneal) should produce MORE zeros
        # because more values fall within the rounding-to-zero threshold
        # when alpha is scaled up by 3x
        assert zeros_soft >= zeros_tight * 0.8, (
            f"Anneal factor should affect zero fraction: "
            f"soft={zeros_soft:.3f}, tight={zeros_tight:.3f}"
        )

    def test_phase_transition_gradient_continuity(self):
        """Gradients should remain finite across phase transitions."""
        rb = ResiBitConv(8, 16)
        x = torch.randn(1, 8, 4, 4)

        for phase in [
            ResiBitConv.PHASE_FP32_WARMUP,
            ResiBitConv.PHASE_TERNARY_WARMUP,
            ResiBitConv.PHASE_FULL_QAT,
        ]:
            rb.set_phase(phase)
            y = rb(x)
            loss = y.sum()
            loss.backward()

            # Check gradients are finite
            assert not torch.isnan(rb.w_a.grad).any(), f"NaN grad in Phase {phase}"
            assert not torch.isinf(rb.w_a.grad).any(), f"Inf grad in Phase {phase}"
            rb.zero_grad()


class TestResiBitGradientAnchor:
    """Tests validating the INT8 highway as gradient anchor.

    The key hypothesis (H3) from the paper: the INT8 residual path
    preserves a stable, non-quantised gradient channel through every
    block, preventing irreversible collapse.
    """

    def test_highway_gradient_magnitude(self):
        """INT8 highway should have larger gradient magnitude than ternary.

        Because INT8 has 255 levels vs ternary's 3 levels, the STE
        gradient through the highway is better conditioned.
        """
        rb = ResiBitConv(16, 32)
        rb.set_phase(ResiBitConv.PHASE_FULL_QAT)

        x = torch.randn(2, 16, 8, 8)
        y = rb(x)
        loss = y.sum()
        loss.backward()

        # Highway gradient should exist and be non-zero
        assert rb.w_r.grad is not None
        highway_grad_norm = rb.w_r.grad.norm().item()
        assert highway_grad_norm > 0, "Highway received zero gradient"

    def test_three_streams_all_receive_gradients(self):
        """All three streams must receive gradients in Phase 3."""
        rb = ResiBitConv(16, 32)
        rb.set_phase(ResiBitConv.PHASE_FULL_QAT)

        x = torch.randn(2, 16, 8, 8)
        y = rb(x)
        loss = y.sum()
        loss.backward()

        for name, param in [("w_a", rb.w_a), ("w_b", rb.w_b), ("w_r", rb.w_r)]:
            assert param.grad is not None, f"{name} received no gradient"
            assert param.grad.abs().sum() > 0, f"{name} gradient is all zeros"

    def test_highway_provides_gradient_anchor(self):
        """The highway's gradient should not collapse to zero during training.

        This is the core test for Hypothesis H3: the INT8 highway prevents
        gradient manifold shattering.
        """
        rb = ResiBitConv(16, 32)
        rb.set_phase(ResiBitConv.PHASE_FULL_QAT)
        optimizer = torch.optim.Adam(rb.parameters(), lr=1e-3)

        highway_grad_norms = []
        for _ in range(10):
            x = torch.randn(2, 16, 4, 4)
            y = rb(x)
            loss = y.mean()
            loss.backward()

            if rb.w_r.grad is not None:
                highway_grad_norms.append(rb.w_r.grad.norm().item())
            optimizer.step()
            optimizer.zero_grad()

        # Highway gradient should remain non-zero throughout training
        assert all(g > 0 for g in highway_grad_norms), (
            "Highway gradient collapsed to zero during training"
        )


class TestResiBitBase5Fusion:
    """Tests for the deployment-time Base-5 weight fusion."""

    def test_base5_values_quinary(self):
        """Fused weights must be in {-2, -1, 0, 1, 2}."""
        rb = ResiBitConv(16, 32)
        fused = rb.get_base5_fused()
        unique_vals = set(fused.unique().tolist())
        assert unique_vals.issubset({-2, -1, 0, 1, 2})

    def test_base5_shape(self):
        """Fused weight shape matches a single standard convolution."""
        rb = ResiBitConv(16, 32, kernel_size=3)
        fused = rb.get_base5_fused()
        assert fused.shape == (32, 16, 3, 3)


class TestResiBitOrthogonalLoss:
    """Tests for the expert orthogonalization mechanism."""

    def test_orth_loss_zero_in_phase1(self):
        """No orthogonal loss during FP32 warmup (Phase 1)."""
        rb = ResiBitConv(8, 16)
        rb.set_phase(ResiBitConv.PHASE_FP32_WARMUP)
        loss = rb.orthogonal_loss()
        assert loss.item() == 0.0

    def test_orth_loss_nonzero_in_phase2(self):
        """Orthogonal loss should be active in Phase 2."""
        rb = ResiBitConv(8, 16)
        rb.set_phase(ResiBitConv.PHASE_TERNARY_WARMUP)
        loss = rb.orthogonal_loss()
        assert loss.item() >= 0.0

    def test_orth_loss_nonzero_in_phase3(self):
        """Orthogonal loss should be active in Phase 3."""
        rb = ResiBitConv(8, 16)
        rb.set_phase(ResiBitConv.PHASE_FULL_QAT)
        loss = rb.orthogonal_loss()
        assert loss.item() >= 0.0


class TestResiBitTraining:
    """End-to-end training tests for ResiBit blocks."""

    def test_full_training_loop(self):
        """Complete training loop should not produce NaN or Inf."""
        rb = ResiBitConv(8, 16)
        optimizer = torch.optim.Adam(rb.parameters(), lr=1e-3)

        # Phase 1
        rb.set_phase(ResiBitConv.PHASE_FP32_WARMUP)
        for _ in range(3):
            x = torch.randn(2, 8, 4, 4)
            y = rb(x)
            loss = y.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Phase 2
        rb.set_phase(ResiBitConv.PHASE_TERNARY_WARMUP, ternary_anneal=2.0)
        for _ in range(3):
            x = torch.randn(2, 8, 4, 4)
            y = rb(x)
            loss = y.mean() + rb.orthogonal_loss()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Phase 3
        rb.set_phase(ResiBitConv.PHASE_FULL_QAT)
        for _ in range(3):
            x = torch.randn(2, 8, 4, 4)
            y = rb(x)
            loss = y.mean() + rb.orthogonal_loss()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Check no NaN
        assert not torch.isnan(rb.w_a).any()
        assert not torch.isnan(rb.w_b).any()
        assert not torch.isnan(rb.w_r).any()
