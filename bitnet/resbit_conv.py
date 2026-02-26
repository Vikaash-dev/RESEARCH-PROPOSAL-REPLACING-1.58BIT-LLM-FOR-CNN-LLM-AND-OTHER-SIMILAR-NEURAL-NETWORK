"""
ResiBitConv: Residual-Precision 1.58-bit Convolution Block.

Implements the ResiBit-YOLO architecture from the research paper.
Three parallel streams process the same input:
  - Expert A: Ternary convolution (high-frequency texture)
  - Expert B: Ternary convolution (low-frequency shape)
  - INT8 Residual Highway: Gradient anchor preventing manifold collapse

A Group-Mix aggregation combines the streams with learnable per-channel
mixing coefficients before BatchNorm and SiLU activation.

The INT8 highway is the key architectural innovation that resolves the
gradient manifold collapse ("Muon Trap") identified in the DualExpert-Base5
experiment. It provides:
  1. Gradient anchor: Well-conditioned STE gradient (256 levels vs 3)
  2. Precision residual: Captures fine-grained info lost by ternary rounding
  3. Deployment flexibility: Only 25% of FP32 bandwidth overhead

Theoretical justification for INT8 highway (STE gradient variance):
  For n-level quantization with step delta = 2/(n-1):
    Var[quant_noise] = delta^2/12
  Ternary (n=3): Var = 1/12 ≈ 0.0833
  INT8 (n=255):  Var ≈ 5.1e-6
  Ratio: ~16,322× less variance → much smoother gradient landscape.

References:
    [1] Ma et al., "The Era of 1-bit LLMs", arXiv:2402.17764, 2024.
    [2] Esser et al., "Learned Step Size Quantization", ICLR 2020.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet.quantization import (
    quantize_ternary_ste,
    quantize_int8,
    quantize_int8_per_channel,
    quantize_activations,
)


class ResiBitConv(nn.Module):
    """ResiBit Convolution Block: Dual-Stream Ternary Experts
    + INT8 Residual Highway + Group-Mix Aggregation.

    The INT8 Residual Highway anchors the gradient manifold during
    Quantization-Aware Training (QAT), preventing the dead-zero weight
    collapse observed in pure dual-expert ternary architectures.

    Training phases (controlled externally via set_phase()):
      Phase 1 (FP32 warmup): All streams in FP32, no quantization.
      Phase 2 (INT8 freeze + ternary warmup): Highway frozen, ternary QAT
              activated with annealed step-size.
      Phase 3 (Full QAT): All streams quantized simultaneously.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel. Default: 3.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides. Default: 1.
        lambda_orth: Orthogonal regularization coefficient. Default: 1e-4.
    """

    # Training phase constants
    PHASE_FP32_WARMUP = 1
    PHASE_TERNARY_WARMUP = 2
    PHASE_FULL_QAT = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        lambda_orth: float = 1e-4,
    ):
        super().__init__()
        shape = (out_channels, in_channels, kernel_size, kernel_size)

        # --- Ternary Expert Streams ---
        self.w_a = nn.Parameter(torch.randn(*shape) * 0.02)
        self.w_b = nn.Parameter(torch.randn(*shape) * 0.02)

        # --- INT8 Residual Highway ---
        self.w_r = nn.Parameter(torch.randn(*shape) * 0.02)

        # --- Group-Mix Coefficients ---
        # Initialize with bias towards highway (logit=0.5) to ensure
        # gradient stability from epoch 0. Softmax([0, 0, 0.5]) ≈
        # [0.30, 0.30, 0.40], giving the highway 40% initial weight.
        # This follows the paper's insight that the gradient anchor
        # should have meaningful influence from the start.
        mix_init = torch.zeros(3, out_channels)
        mix_init[2, :] = 0.5  # Bias towards highway stream
        self.mix_logits = nn.Parameter(mix_init)

        self.stride = stride
        self.padding = padding
        self.lambda_orth = lambda_orth
        self.bn = nn.BatchNorm2d(out_channels)

        # Phase tracking (Phase 1 = FP32 warmup by default)
        self._phase = self.PHASE_FP32_WARMUP
        # Ternary anneal factor: 1.0 = full ternary, >1.0 = softer rounding
        self._ternary_anneal = 1.0

    def set_phase(self, phase: int, ternary_anneal: float = 1.0) -> None:
        """Set the training phase for this block.

        Args:
            phase: One of PHASE_FP32_WARMUP (1), PHASE_TERNARY_WARMUP (2),
                   or PHASE_FULL_QAT (3).
            ternary_anneal: Anneal factor for ternary quantization threshold.
                           Values > 1.0 produce softer rounding (Phase 2 start).
                           1.0 = standard absmean threshold (Phase 3).
        """
        self._phase = phase
        self._ternary_anneal = ternary_anneal
        if phase == self.PHASE_TERNARY_WARMUP:
            # Freeze INT8 highway in Phase 2
            self.w_r.requires_grad_(False)
        elif phase == self.PHASE_FULL_QAT:
            # Unfreeze for full QAT
            self.w_r.requires_grad_(True)

    def _quantize_ternary_annealed(self, W: torch.Tensor) -> torch.Tensor:
        """Ternary quantization with annealed threshold.

        During Phase 2, the threshold is scaled by self._ternary_anneal,
        producing softer rounding that gradually sharpens to standard
        absmean as the anneal factor decreases to 1.0.
        """
        alpha = W.abs().mean().clamp(min=1e-6) * self._ternary_anneal
        W_scaled = W / alpha
        W_quantized = W_scaled.round().clamp(-1, 1)
        return W_quantized - W_scaled.detach() + W_scaled

    def _mix_weights(self) -> torch.Tensor:
        """Softmax-normalized per-channel mixing coefficients."""
        return F.softmax(self.mix_logits, dim=0)  # (3, C_out)

    def orthogonal_loss(self) -> torch.Tensor:
        """Frobenius penalty to prevent expert collapse.

        Implements L_orth = lambda * ||W_A^T @ W_B||_F^2
        Only active when ternary quantization is engaged (Phases 2-3).
        """
        if self._phase == self.PHASE_FP32_WARMUP:
            return torch.tensor(0.0, device=self.w_a.device)

        wa_q = self._quantize_ternary_annealed(self.w_a)
        wb_q = self._quantize_ternary_annealed(self.w_b)
        wa_flat = wa_q.view(wa_q.size(0), -1)  # (C_out, C_in * k * k)
        wb_flat = wb_q.view(wb_q.size(0), -1)
        gram = wa_flat @ wb_flat.T  # (C_out, C_out)
        return self.lambda_orth * (gram**2).sum()

    def get_base5_fused(self) -> torch.Tensor:
        """Return fused Base-5 weight for deployment (INT8, {-2..2})."""
        with torch.no_grad():
            from bitnet.quantization import quantize_ternary

            wa_q = quantize_ternary(self.w_a).to(torch.int8)
            wb_q = quantize_ternary(self.w_b).to(torch.int8)
            return wa_q + wb_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with three parallel streams + Group-Mix aggregation.

        Behavior depends on the training phase:
          Phase 1: All streams use FP32 weights (no quantization).
          Phase 2: Ternary experts use annealed QAT; highway is frozen INT8.
          Phase 3: Full QAT - ternary experts + INT8 highway all quantized.
        """
        if self._phase == self.PHASE_FP32_WARMUP:
            # Phase 1: Pure FP32 - no quantization at all
            w_a_eff = self.w_a
            w_b_eff = self.w_b
            w_r_eff = self.w_r
        elif self._phase == self.PHASE_TERNARY_WARMUP:
            # Phase 2: Ternary experts quantized (annealed), highway frozen
            w_a_eff = self._quantize_ternary_annealed(self.w_a)
            w_b_eff = self._quantize_ternary_annealed(self.w_b)
            w_r_eff = quantize_int8_per_channel(self.w_r)
        else:
            # Phase 3: Full QAT - all quantized
            w_a_eff = quantize_ternary_ste(self.w_a)
            w_b_eff = quantize_ternary_ste(self.w_b)
            w_r_eff = quantize_int8_per_channel(self.w_r)

        # Three parallel streams
        y_a = F.conv2d(x, w_a_eff, stride=self.stride, padding=self.padding)
        y_b = F.conv2d(x, w_b_eff, stride=self.stride, padding=self.padding)
        y_r = F.conv2d(x, w_r_eff, stride=self.stride, padding=self.padding)

        # Group-Mix: learnable per-channel weighted sum
        mix = self._mix_weights()  # (3, C_out)
        alpha_a = mix[0].view(1, -1, 1, 1)
        alpha_b = mix[1].view(1, -1, 1, 1)
        alpha_r = mix[2].view(1, -1, 1, 1)

        y = alpha_a * y_a + alpha_b * y_b + alpha_r * y_r
        return F.silu(self.bn(y))

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.w_a.size(1)}, "
            f"out_channels={self.w_a.size(0)}, "
            f"kernel_size={self.w_a.size(2)}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"lambda_orth={self.lambda_orth}, "
            f"phase={self._phase}"
        )
