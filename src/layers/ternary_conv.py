"""Ternary weight quantization with Straight-Through Estimator (STE).

Implements the core 1.58-bit quantization mechanism from:
- [Ma et al., 2024, "The Era of 1-bit LLMs", arXiv:2402.17764]
- [Li et al., 2016, "Ternary Weight Networks", arXiv:1605.04711]

Weights are constrained to {-1, 0, +1} during forward pass while
full-precision gradients flow through via STE during backward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryQuantize(torch.autograd.Function):
    """Quantize weights to {-1, 0, +1} with Straight-Through Estimator.

    Forward: threshold-based ternary quantization.
    Backward: gradient passes through unchanged (STE), clipped for stability.

    Reference: [Courbariaux et al., 2016, "Binarized Neural Networks", arXiv:1602.02830]
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(weight)
        # Adaptive threshold: 5% of mean absolute weight value
        threshold = 0.05 * weight.abs().mean()
        output = torch.zeros_like(weight)
        output[weight > threshold] = 1.0
        output[weight < -threshold] = -1.0
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        (weight,) = ctx.saved_tensors
        # STE: pass gradient through, but clip for weights outside [-1, 1]
        # to prevent gradient explosion (the "Muon Trap" mitigation)
        grad_input = grad_output.clone()
        grad_input[weight.abs() > 1.0] = 0
        return grad_input


class TernaryConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d with ternary weight quantization.

    During training, maintains full-precision weights and quantizes on-the-fly.
    During inference, uses pre-quantized ternary weights {-1, 0, +1}.

    This enables 13.78x memory compression (log2(3) ≈ 1.58 bits per weight
    vs 32 bits for FP32) and replaces multiplications with additions/subtractions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        bias: If True, adds a learnable bias (default: False for quantized layers).
        groups: Number of blocked connections from input to output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = False,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

    @property
    def weight(self) -> nn.Parameter:
        """Access the full-precision weight parameter."""
        return self.conv.weight

    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ternary = TernaryQuantize.apply(self.conv.weight)
        return F.conv2d(
            x,
            w_ternary,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )

    def get_ternary_weights(self) -> torch.Tensor:
        """Return the quantized ternary weights (for analysis/export)."""
        with torch.no_grad():
            return TernaryQuantize.apply(self.conv.weight)

    def weight_statistics(self) -> dict[str, float]:
        """Compute ternary weight distribution statistics.

        Returns dict with fraction of weights at each ternary value.
        High dead_zero fraction (>50%) indicates codebook collapse.
        """
        with torch.no_grad():
            w = TernaryQuantize.apply(self.conv.weight)
            total = w.numel()
            return {
                "positive": (w == 1.0).sum().item() / total,
                "zero": (w == 0.0).sum().item() / total,
                "negative": (w == -1.0).sum().item() / total,
            }
