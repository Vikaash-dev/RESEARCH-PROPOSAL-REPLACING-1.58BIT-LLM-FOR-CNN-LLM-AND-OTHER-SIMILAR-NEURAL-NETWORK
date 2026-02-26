"""
Deployment module for shift-and-add inference with Base-5 weights.

Converts trained MoQEConv / ResiBitConv blocks into deployment-ready
representations that use only integer shift-and-add operations,
eliminating all floating-point multiplications.

The Base-5 weight values {-2, -1, 0, +1, +2} map to:
    +2: output += (input << 1)      [bit shift left + add]
    +1: output += input             [add]
     0: skip                        [no-op]
    -1: output -= input             [subtract]
    -2: output -= (input << 1)      [bit shift left + subtract]

This is the key efficiency insight: a convolution with Base-5 weights
requires at most 2 additions per weight (vs. 1 multiplication + 1
addition for standard convolution), and most hardware can perform
shifts "for free" within the ALU pipeline.

References:
    [1] Ding et al., "RepVGG: Making VGG-style ConvNets Great Again",
        CVPR 2021 — structural reparameterization for deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from bitnet.quantization import quantize_ternary
from bitnet.moqe_conv import MoQEConv
from bitnet.resbit_conv import ResiBitConv


class DeployedBase5Conv(nn.Module):
    """Deployment-ready convolution using fused Base-5 weights.

    This module holds the fused quinary weight tensor (INT8, values
    in {-2, -1, 0, 1, 2}) and performs inference using standard
    convolution (which hardware accelerators optimize internally).

    For custom hardware with shift-and-add support, the weight tensor
    can be exported directly and the inference loop replaced with
    the shift-and-add kernel.

    Args:
        weight: INT8 tensor of shape (out_ch, in_ch, kH, kW) with
                values in {-2, -1, 0, 1, 2}.
        bias: Optional bias tensor.
        stride: Convolution stride.
        padding: Convolution padding.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        # Store as buffer (not parameter — no gradient needed)
        self.register_buffer("weight", weight.to(torch.int8))
        if bias is not None:
            self.register_buffer("bias", bias.clone())
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using fused Base-5 weights.

        At deployment, the INT8 weights are cast to the input dtype
        for the convolution. On custom hardware, this would use the
        shift-and-add kernel instead.
        """
        w = self.weight.to(x.dtype)
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)

    def count_ops(self, input_shape: tuple) -> dict:
        """Count the number of shift-and-add operations.

        Returns:
            Dictionary with operation counts.
        """
        batch, in_ch, H, W = input_shape
        out_ch, in_ch_w, kH, kW = self.weight.shape
        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1

        total_weight_applications = out_ch * out_H * out_W * in_ch_w * kH * kW

        # Count by operation type
        w = self.weight
        n_zeros = (w == 0).sum().item()
        n_ones = ((w == 1) | (w == -1)).sum().item()
        n_twos = ((w == 2) | (w == -2)).sum().item()
        total_weights = w.numel()

        # Ops per spatial location
        spatial_locs = out_H * out_W * batch

        return {
            "total_weight_elements": total_weights,
            "zero_weights_pct": n_zeros / total_weights * 100,
            "add_only_pct": n_ones / total_weights * 100,
            "shift_add_pct": n_twos / total_weights * 100,
            "total_output_elements": batch * out_ch * out_H * out_W,
            "multiplications": 0,  # Zero FP multiplications!
            "additions": int((n_ones + n_twos) * spatial_locs),
            "shifts": int(n_twos * spatial_locs),
            "skipped_ops": int(n_zeros * spatial_locs),
        }

    def extra_repr(self) -> str:
        return (
            f"out_channels={self.weight.size(0)}, "
            f"in_channels={self.weight.size(1)}, "
            f"kernel_size={self.weight.size(2)}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"dtype=INT8/Base5"
        )


def convert_moqe_to_deployed(moqe: MoQEConv) -> DeployedBase5Conv:
    """Convert a trained MoQEConv block to deployment form.

    Fuses the two ternary experts into a single Base-5 weight tensor
    and wraps it in a DeployedBase5Conv module.

    Args:
        moqe: Trained MoQEConv module.

    Returns:
        DeployedBase5Conv ready for inference.
    """
    fused = moqe.get_fused_base5()
    return DeployedBase5Conv(
        weight=fused,
        bias=moqe.bias,
        stride=moqe.stride,
        padding=moqe.padding,
    )


def convert_resbit_to_deployed(resbit: ResiBitConv) -> DeployedBase5Conv:
    """Convert a trained ResiBitConv block to deployment form.

    Fuses the two ternary experts into a single Base-5 weight tensor.
    The INT8 highway weights are discarded at deployment (their
    contribution is absorbed by the Group-Mix coefficients during
    the final training epochs).

    Note: For maximum accuracy, the highway contribution should be
    distilled into the ternary experts during the final epochs of
    Phase 3 training. This is a simplification for the research
    prototype.

    Args:
        resbit: Trained ResiBitConv module.

    Returns:
        DeployedBase5Conv ready for inference.
    """
    fused = resbit.get_base5_fused()
    return DeployedBase5Conv(
        weight=fused,
        bias=None,  # ResiBit uses BN instead of bias
        stride=resbit.stride,
        padding=resbit.padding,
    )


def export_model_for_deployment(
    model: nn.Module,
) -> nn.Module:
    """Convert all quantized blocks in a model to deployment form.

    Replaces MoQEConv and ResiBitConv blocks with DeployedBase5Conv
    blocks using fused weights.

    Args:
        model: Trained model with quantized blocks.

    Returns:
        Model with all quantized blocks converted to deployment form.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, MoQEConv):
            setattr(model, name, convert_moqe_to_deployed(module))
        elif isinstance(module, ResiBitConv):
            setattr(model, name, convert_resbit_to_deployed(module))
        else:
            export_model_for_deployment(module)
    return model
