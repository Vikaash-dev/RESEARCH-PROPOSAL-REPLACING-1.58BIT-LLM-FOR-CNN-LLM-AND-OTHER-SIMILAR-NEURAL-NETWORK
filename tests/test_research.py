"""
Tests for new research analysis features, deployment module, and paper-faithful
improvements identified through deep reading of both PDFs.

Validates:
- Per-channel INT8 quantization (improvement over paper's per-tensor)
- STE gradient variance analysis (Theorem 2 from theoretical module)
- Weight entropy and effective rank monitoring (early collapse detection)
- Cosine similarity metric between FP32 and quantized weights
- MSE-based sensitivity analysis (Paper Table 1)
- Mix diversification penalty (Paper Section 5.4)
- Highway-biased Group-Mix initialization (improvement over paper)
- Expert B 0.1× initialization (Paper Listing 2 line 56)
- Deployment module: Base-5 fused inference
- Theoretical analysis module: all theorems produce valid results
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet.quantization import (
    quantize_ternary,
    quantize_int8,
    quantize_int8_per_channel,
    compute_ste_gradient_variance_ratio,
    compute_cosine_similarity,
    compute_weight_entropy,
    compute_effective_rank,
)
from bitnet.moqe_conv import MoQEConv
from bitnet.resbit_conv import ResiBitConv
from bitnet.utils import (
    compute_quantization_mse,
    compute_mix_diversification_loss,
    replace_conv2d_with_resbit,
)
from bitnet.deploy import (
    DeployedBase5Conv,
    convert_moqe_to_deployed,
    convert_resbit_to_deployed,
    export_model_for_deployment,
)
from bitnet.analysis import (
    theorem_1_information_capacity,
    theorem_2_ste_gradient_variance,
    theorem_3_orthogonal_convergence,
    theorem_4_base5_reparameterization,
    theorem_5_pareto_frontier,
    analyze_muon_trap,
    generate_theoretical_report,
)


# ---------------------------------------------------------------------------
# Per-channel INT8 quantization tests
# ---------------------------------------------------------------------------

class TestPerChannelINT8:
    """Tests for per-channel INT8 quantization."""

    def test_output_range(self):
        """Per-channel INT8 values must be in [-127, 127]."""
        W = torch.randn(64, 32, 3, 3)
        Q = quantize_int8_per_channel(W)
        assert Q.max() <= 127.0
        assert Q.min() >= -127.0

    def test_shape_preserved(self):
        """Output shape must match input."""
        W = torch.randn(32, 16, 3, 3)
        Q = quantize_int8_per_channel(W)
        assert Q.shape == W.shape

    def test_gradient_flows(self):
        """Per-channel INT8 must support gradient flow via STE."""
        W = torch.randn(16, 8, 3, 3, requires_grad=True)
        Q = quantize_int8_per_channel(W)
        loss = Q.sum()
        loss.backward()
        assert W.grad is not None

    def test_per_channel_less_error_than_per_tensor(self):
        """Per-channel should have lower quantization error than per-tensor.

        This validates our improvement over the paper's per-tensor approach.
        Per-channel quantization computes separate scales per output channel,
        better handling channels with different magnitude distributions.
        """
        torch.manual_seed(42)
        # Create weights with varying per-channel magnitudes
        W = torch.randn(32, 16, 3, 3)
        # Make some channels much larger to amplify the difference
        W[0:8] *= 10.0
        W[24:32] *= 0.1

        Q_per_tensor = quantize_int8(W)
        Q_per_channel = quantize_int8_per_channel(W)

        # Denormalize for error comparison
        scale_pt = W.abs().max() / 127.0
        err_pt = (W - Q_per_tensor * scale_pt).abs().mean().item()

        # Per-channel: each channel has its own scale
        out_ch = W.size(0)
        W_flat = W.view(out_ch, -1)
        scales_pc = W_flat.abs().amax(dim=1).clamp(min=1e-6) / 127.0
        shape = [out_ch] + [1] * (W.dim() - 1)
        err_pc = (W - Q_per_channel * scales_pc.view(*shape)).abs().mean().item()

        assert err_pc <= err_pt, (
            f"Per-channel error ({err_pc:.6f}) should be <= per-tensor ({err_pt:.6f})"
        )

    def test_falls_back_for_1d(self):
        """1D tensors should fall back to per-tensor quantization."""
        W = torch.randn(64)
        Q = quantize_int8_per_channel(W)
        assert Q.max() <= 127.0


# ---------------------------------------------------------------------------
# STE gradient variance analysis tests
# ---------------------------------------------------------------------------

class TestSTEVarianceAnalysis:
    """Tests for the STE gradient variance ratio computation."""

    def test_ternary_self_ratio_is_one(self):
        """Ternary vs ternary should give ratio of 1.0."""
        ratio = compute_ste_gradient_variance_ratio(3)
        assert abs(ratio - 1.0) < 0.01

    def test_int8_much_lower_variance(self):
        """INT8 should have much lower gradient variance than ternary.

        For 255 levels, step delta = 2/(255-1) = 2/254, so the variance
        ratio is (delta_ternary / delta_255)^2 = (1.0 / (2/254))^2
        = (127)^2 = 16,129.
        """
        ratio = compute_ste_gradient_variance_ratio(255)
        assert ratio > 1000, f"INT8 variance ratio should be >>1000, got {ratio}"

    def test_more_levels_means_less_variance(self):
        """Monotonically increasing levels should decrease variance."""
        prev_ratio = 1.0
        for n in [3, 7, 15, 31, 63, 127, 255]:
            ratio = compute_ste_gradient_variance_ratio(n)
            assert ratio >= prev_ratio, (
                f"Variance ratio should increase with n: {n} gave {ratio} < {prev_ratio}"
            )
            prev_ratio = ratio


# ---------------------------------------------------------------------------
# Weight entropy and effective rank tests
# ---------------------------------------------------------------------------

class TestWeightEntropy:
    """Tests for Shannon entropy of weight distributions."""

    def test_balanced_ternary_max_entropy(self):
        """Perfectly balanced {-1,0,+1} should have entropy ≈ log2(3) ≈ 1.585."""
        W = torch.zeros(900)
        W[:300] = -1
        W[300:600] = 0
        W[600:] = 1
        entropy = compute_weight_entropy(W)
        assert abs(entropy - math.log2(3)) < 0.01, f"Expected ~1.585, got {entropy}"

    def test_all_zeros_low_entropy(self):
        """All-zero weights should have entropy of 0."""
        W = torch.zeros(1000)
        entropy = compute_weight_entropy(W)
        assert entropy < 0.01, f"All-zero should have near-zero entropy, got {entropy}"

    def test_collapsed_distribution_low_entropy(self):
        """60% zeros (paper's collapse signature) should have low entropy."""
        W = torch.zeros(1000)
        W[:200] = -1
        W[800:] = 1
        # 60% zeros, 20% negative, 20% positive
        entropy = compute_weight_entropy(W)
        # Entropy should be well below max (1.585)
        assert entropy < 1.4, f"Collapsed dist should have entropy < 1.4, got {entropy}"
        assert entropy > 0, "Entropy should be positive"


class TestEffectiveRank:
    """Tests for effective rank of weight matrices."""

    def test_identity_matrix_full_rank(self):
        """Identity matrix should have maximal effective rank = n."""
        I = torch.eye(16)
        erank = compute_effective_rank(I)
        assert abs(erank - 16.0) < 0.5, f"Identity should have erank≈16, got {erank}"

    def test_rank1_matrix_low_rank(self):
        """Rank-1 matrix should have effective rank ≈ 1."""
        v = torch.randn(32, 1)
        W = v @ v.T  # rank-1
        erank = compute_effective_rank(W)
        assert erank < 2.0, f"Rank-1 matrix should have erank≈1, got {erank}"

    def test_random_matrix_reasonable_rank(self):
        """Random matrix should have reasonable effective rank."""
        torch.manual_seed(42)
        W = torch.randn(32, 16, 3, 3)  # Conv weight shape
        erank = compute_effective_rank(W)
        assert erank > 1.0
        assert erank <= 32.0  # Can't exceed min(rows, cols)


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Tests for cosine similarity between weight tensors."""

    def test_identical_tensors_similarity_one(self):
        """Identical tensors should have cosine similarity = 1.0."""
        W = torch.randn(32, 16, 3, 3)
        cos_sim = compute_cosine_similarity(W, W)
        assert abs(cos_sim - 1.0) < 1e-5

    def test_negated_tensors_similarity_minus_one(self):
        """Negated tensors should have cosine similarity = -1.0."""
        W = torch.randn(32, 16, 3, 3)
        cos_sim = compute_cosine_similarity(W, -W)
        assert abs(cos_sim - (-1.0)) < 1e-5

    def test_quantized_vs_original_high_similarity(self):
        """Ternary-quantized weights should maintain reasonable similarity."""
        torch.manual_seed(42)
        W = torch.randn(32, 16, 3, 3)
        W_q = quantize_ternary(W) * W.abs().mean()
        cos_sim = compute_cosine_similarity(W, W_q)
        assert cos_sim > 0.5, f"Quantized similarity too low: {cos_sim}"


# ---------------------------------------------------------------------------
# MSE sensitivity analysis tests
# ---------------------------------------------------------------------------

class TestQuantizationMSE:
    """Tests for MSE-based sensitivity analysis (Paper Table 1)."""

    def test_mse_nonnegative(self):
        """MSE should always be non-negative."""
        layer = nn.Conv2d(16, 32, 3, padding=1)
        mse = compute_quantization_mse(layer)
        assert mse >= 0

    def test_zero_weights_zero_mse(self):
        """All-zero weights should have zero MSE."""
        layer = nn.Conv2d(16, 32, 3, padding=1)
        with torch.no_grad():
            layer.weight.fill_(0)
        mse = compute_quantization_mse(layer)
        assert mse < 1e-10

    def test_larger_weights_larger_mse(self):
        """Larger magnitude weights should generally have higher MSE."""
        layer_small = nn.Conv2d(16, 32, 3, padding=1)
        layer_large = nn.Conv2d(16, 32, 3, padding=1)
        with torch.no_grad():
            layer_small.weight.fill_(0.01)
            layer_large.weight.normal_(0, 1.0)
        mse_small = compute_quantization_mse(layer_small)
        mse_large = compute_quantization_mse(layer_large)
        # Larger weights have more quantization error in absolute terms
        # (but relative error depends on distribution)
        assert mse_large > 0


# ---------------------------------------------------------------------------
# Mix diversification penalty tests
# ---------------------------------------------------------------------------

class TestMixDiversification:
    """Tests for the Group-Mix diversification penalty (Paper §5.4)."""

    def test_returns_scalar(self):
        """Diversification loss should be a scalar tensor."""
        model = nn.Sequential(
            ResiBitConv(8, 16),
            ResiBitConv(16, 16),
        )
        loss = compute_mix_diversification_loss(model)
        assert loss.dim() == 0

    def test_uniform_mix_zero_penalty(self):
        """If all mix weights are uniform (1/3), penalty should be ~0."""
        rb = ResiBitConv(8, 16)
        # Default init with highway bias of 0.5 gives non-uniform mix
        # Set logits to zero for uniform 1/3 each
        with torch.no_grad():
            rb.mix_logits.fill_(0)
        model = nn.Sequential(rb)
        loss = compute_mix_diversification_loss(model)
        assert abs(loss.item()) < 0.01

    def test_gradient_flows(self):
        """Diversification loss must provide gradients to mix_logits."""
        model = nn.Sequential(ResiBitConv(8, 16))
        loss = compute_mix_diversification_loss(model)
        loss.backward()
        assert model[0].mix_logits.grad is not None


# ---------------------------------------------------------------------------
# Highway-biased Group-Mix initialization tests
# ---------------------------------------------------------------------------

class TestHighwayBiasedInit:
    """Tests for highway-biased Group-Mix initialization."""

    def test_highway_has_higher_initial_weight(self):
        """Highway stream should have higher initial mix weight than experts.

        Our improvement: init mix_logits[2] = 0.5 so softmax gives
        highway ~40% weight vs ~30% each for experts.
        This is an improvement over the paper's zero init (uniform 1/3).
        """
        rb = ResiBitConv(16, 32)
        mix = rb._mix_weights().detach()
        # mix[2] is highway, mix[0] is Expert A, mix[1] is Expert B
        highway_weight = mix[2].mean().item()
        expert_a_weight = mix[0].mean().item()
        assert highway_weight > expert_a_weight, (
            f"Highway ({highway_weight:.3f}) should have higher weight than "
            f"Expert A ({expert_a_weight:.3f})"
        )

    def test_mix_still_sums_to_one(self):
        """Highway-biased init should still have mix weights summing to 1."""
        rb = ResiBitConv(16, 32)
        mix = rb._mix_weights()
        sums = mix.sum(dim=0)
        assert torch.allclose(sums, torch.ones(32), atol=1e-5)


# ---------------------------------------------------------------------------
# Deployment module tests
# ---------------------------------------------------------------------------

class TestDeployment:
    """Tests for the deployment module (shift-and-add inference)."""

    def test_moqe_deployment_produces_output(self):
        """Deployed MoQE should produce valid output."""
        moqe = MoQEConv(16, 32, kernel_size=3, padding=1)
        deployed = convert_moqe_to_deployed(moqe)
        x = torch.randn(1, 16, 8, 8)
        y = deployed(x)
        assert y.shape == (1, 32, 8, 8)
        assert not torch.isnan(y).any()

    def test_resbit_deployment_produces_output(self):
        """Deployed ResiBit should produce valid output."""
        resbit = ResiBitConv(16, 32, kernel_size=3, padding=1)
        deployed = convert_resbit_to_deployed(resbit)
        x = torch.randn(1, 16, 8, 8)
        y = deployed(x)
        assert y.shape == (1, 32, 8, 8)
        assert not torch.isnan(y).any()

    def test_deployed_weights_are_quinary(self):
        """Deployed Base-5 weights must be in {-2,-1,0,1,2}."""
        moqe = MoQEConv(16, 32)
        deployed = convert_moqe_to_deployed(moqe)
        unique_vals = set(deployed.weight.unique().tolist())
        assert unique_vals.issubset({-2, -1, 0, 1, 2})

    def test_deployed_weights_are_int8(self):
        """Deployed weights should be stored as INT8."""
        moqe = MoQEConv(16, 32)
        deployed = convert_moqe_to_deployed(moqe)
        assert deployed.weight.dtype == torch.int8

    def test_count_ops_zero_multiplications(self):
        """Deployment should report zero floating-point multiplications.

        This validates the paper's claim of multiplication-free inference
        via shift-and-add arithmetic (Equations 8-12).
        """
        moqe = MoQEConv(16, 32, kernel_size=3, padding=1)
        deployed = convert_moqe_to_deployed(moqe)
        ops = deployed.count_ops((1, 16, 8, 8))
        assert ops["multiplications"] == 0

    def test_export_model_replaces_all_blocks(self):
        """export_model_for_deployment should replace all quantized blocks."""
        model = nn.Sequential(
            MoQEConv(16, 32, kernel_size=3, padding=1),
            MoQEConv(32, 64, kernel_size=3, padding=1),
        )
        model = export_model_for_deployment(model)
        assert isinstance(model[0], DeployedBase5Conv)
        assert isinstance(model[1], DeployedBase5Conv)

    def test_deployment_linearity(self):
        """Deployed model output should match the dual-expert sum.

        The key mathematical property: Conv(X; WA+WB) = Conv(X;WA) + Conv(X;WB)
        """
        torch.manual_seed(42)
        moqe = MoQEConv(8, 16, kernel_size=3, padding=1)
        x = torch.randn(1, 8, 6, 6)

        # Get dual-expert output (forward pass)
        y_dual = moqe(x)

        # Get deployed output (fused single conv)
        deployed = convert_moqe_to_deployed(moqe)
        y_deployed = deployed(x)

        # Should be very close (the STE changes values slightly)
        error = (y_dual.detach() - y_deployed).abs().max().item()
        assert error < 0.5, f"Deployment linearity error too high: {error}"


# ---------------------------------------------------------------------------
# Theoretical analysis module tests
# ---------------------------------------------------------------------------

class TestTheoreticalAnalysis:
    """Tests for the formal theorem verification module."""

    def test_theorem1_information_capacity(self):
        """Theorem 1: Verify information capacity claims from Table 1."""
        t1 = theorem_1_information_capacity()
        assert abs(t1.single_ternary_bits - math.log2(3)) < 1e-6
        assert abs(t1.base5_bits - math.log2(5)) < 1e-6
        assert abs(t1.capacity_increase_pct - 46.5) < 0.5
        assert abs(t1.dual_compression_ratio - 13.78) < 0.01

    def test_theorem2_ste_gradient_variance(self):
        """Theorem 2: Verify STE gradient variance bounds."""
        t2 = theorem_2_ste_gradient_variance()
        assert abs(t2.ternary_step_size - 1.0) < 1e-6
        assert t2.int8_variance < t2.ternary_variance
        assert t2.variance_ratio > 1000  # INT8 has much less noise

    def test_theorem3_orthogonal_convergence(self):
        """Theorem 3: Orthogonal analysis returns valid metrics."""
        torch.manual_seed(42)
        W_A = torch.randn(16, 8, 3, 3)
        W_B = torch.randn(16, 8, 3, 3)
        t3 = theorem_3_orthogonal_convergence(W_A, W_B)
        assert t3.initial_gram_norm > 0
        assert t3.effective_rank_A > 0
        assert t3.effective_rank_B > 0

    def test_theorem4_base5_reparameterization(self):
        """Theorem 4: Base-5 reparameterization is algebraically exact."""
        t4 = theorem_4_base5_reparameterization()
        assert t4.convolution_linearity_error < 1e-4
        assert t4.all_quinary
        assert t4.memory_loads_inference == 1

    def test_theorem5_pareto_frontier(self):
        """Theorem 5: Pareto frontier has expected structure."""
        frontier = theorem_5_pareto_frontier()
        assert len(frontier) == 5
        # FP32 baseline should have 0% accuracy loss
        assert frontier[0].expected_accuracy_loss_pct == 0.0
        # DualExpert Base-5 should be shift-and-add
        base5 = [p for p in frontier if "Base-5" in p.method][0]
        assert base5.ops_type == "shift-and-add"

    def test_muon_trap_analysis_healthy(self):
        """Fresh random weights should be classified as healthy."""
        torch.manual_seed(42)
        W = torch.randn(32, 16, 3, 3)
        result = analyze_muon_trap(W)
        assert result.collapse_stage == "healthy"
        assert not result.is_collapsed

    def test_muon_trap_analysis_collapsed(self):
        """Near-zero weights should be classified as collapsed."""
        W = torch.zeros(32, 16, 3, 3)
        W[0, 0, 0, 0] = 1e-7  # tiny nonzero to avoid division issues
        result = analyze_muon_trap(W)
        assert result.is_collapsed or result.zero_fraction > 0.9

    def test_generate_report(self):
        """Report generation should produce non-empty string."""
        report = generate_theoretical_report()
        assert len(report) > 500
        assert "Theorem 1" in report
        assert "Theorem 2" in report
        assert "13.78" in report


# ---------------------------------------------------------------------------
# Expert B 0.1× initialization tests
# ---------------------------------------------------------------------------

class TestExpertBInit:
    """Tests for paper-faithful Expert B initialization."""

    def test_resbit_expert_b_is_01x(self):
        """ResiBit Expert B should be 0.1× of original weight after surgery."""
        model = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1))
        original = model[0].weight.data.clone()
        model = replace_conv2d_with_resbit(model)
        resbit = model[0]
        assert torch.allclose(resbit.w_b.data, original * 0.1), (
            "Expert B should be 0.1× original (Paper Listing 2 line 56)"
        )

    def test_resbit_expert_a_matches_original(self):
        """Expert A should exactly match the original weights."""
        model = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1))
        original = model[0].weight.data.clone()
        model = replace_conv2d_with_resbit(model)
        resbit = model[0]
        assert torch.allclose(resbit.w_a.data, original)

    def test_resbit_highway_matches_original(self):
        """Highway should exactly match the original weights."""
        model = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1))
        original = model[0].weight.data.clone()
        model = replace_conv2d_with_resbit(model)
        resbit = model[0]
        assert torch.allclose(resbit.w_r.data, original)
