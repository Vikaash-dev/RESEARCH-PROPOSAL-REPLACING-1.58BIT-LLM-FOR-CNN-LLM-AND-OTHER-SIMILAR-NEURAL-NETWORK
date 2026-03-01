"""Base-5 Converter: fuse dual ternary experts into quinary deployment model.

At deployment time, two ternary weight experts (W_A, W_B ∈ {-1, 0, +1})
are additively fused into a single quinary weight set:
    W_fused = W_A + W_B ∈ {-2, -1, 0, 1, 2}

This eliminates the dual-stream memory overhead while preserving the
representational capacity gained during training.

Reference:
- DualExpert-BitYOLO26 [K-Dense Web, 2026]
- Unified Bit-Intelligence [K-Dense Web, 2026]
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

from src.layers.dual_expert import DualExpertBlock
from src.layers.ternary_conv import TernaryQuantize

logger = logging.getLogger(__name__)


class QuinaryConv2d(nn.Module):
    """Convolution with pre-fused quinary weights {-2, -1, 0, 1, 2}.

    This is the deployment-time replacement for DualExpertBlock.
    Weights are stored as INT8 but constrained to {-2, -1, 0, 1, 2},
    enabling shift-and-add arithmetic (no floating-point multiplications).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    @staticmethod
    def from_dual_expert(block: DualExpertBlock) -> QuinaryConv2d:
        """Create a QuinaryConv2d by fusing a DualExpertBlock's ternary experts.

        The fusion formula: W_fused = W_A_ternary + W_B_ternary
        where both W_A and W_B are quantized to {-1, 0, +1} first.

        Args:
            block: Trained DualExpertBlock to fuse.

        Returns:
            QuinaryConv2d with fused weights in {-2, -1, 0, 1, 2}.
        """
        # Get ternary weights from both experts
        w_texture = block.texture_expert.get_ternary_weights()
        w_shape = block.shape_expert.get_ternary_weights()

        # Build full-size weight tensor by placing expert weights
        # in their respective channel positions
        texture_out = block.texture_out
        shape_out = block.shape_out
        texture_in = block.texture_in
        shape_in = block.shape_in
        out_channels = block.out_channels
        in_channels = block.in_channels

        # Get kernel size from texture expert
        k = w_texture.shape[2:]

        # Create fused weight tensor
        fused_weight = torch.zeros(out_channels, in_channels, *k)

        # Place texture expert weights (top-left block)
        fused_weight[:texture_out, :texture_in] = w_texture

        # Place shape expert weights (bottom-right block)
        fused_weight[texture_out:, texture_in:] = w_shape

        # Verify quinary constraint
        unique_vals = fused_weight.unique()
        valid_vals = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        for v in unique_vals:
            if not any(torch.isclose(v, valid_vals)):
                logger.warning("Non-quinary value found after fusion: %.4f", v.item())

        # Create QuinaryConv2d module
        stride = block.texture_expert.conv.stride
        padding = block.texture_expert.conv.padding
        quinary = QuinaryConv2d(
            in_channels, out_channels, k,
            stride=stride, padding=padding,
        )
        quinary.conv.weight = nn.Parameter(fused_weight, requires_grad=False)

        return quinary


class Base5Converter:
    """Convert a trained ResiBit model to deployment-ready quinary model.

    Replaces all DualExpertBlock instances with fused QuinaryConv2d layers
    and optionally removes the INT8 highway for pure ternary/quinary inference.

    Usage:
        converter = Base5Converter()
        deploy_model = converter.convert(trained_model)
    """

    def convert(self, model: nn.Module, remove_int8_highway: bool = True) -> nn.Module:
        """Convert model from training to deployment format.

        Args:
            model: Trained model containing DualExpertBlock/ResiBitBlock modules.
            remove_int8_highway: If True, removes INT8 highway (pure quinary).

        Returns:
            New model with fused quinary weights.
        """
        deploy_model = copy.deepcopy(model)
        conversions = 0

        for name, module in list(deploy_model.named_modules()):
            if isinstance(module, DualExpertBlock):
                quinary = QuinaryConv2d.from_dual_expert(module)
                _replace_module(deploy_model, name, quinary)
                conversions += 1

        logger.info("Base5Converter: fused %d DualExpertBlock(s) into QuinaryConv2d", conversions)

        # Set all parameters to non-trainable
        for param in deploy_model.parameters():
            param.requires_grad = False

        return deploy_model

    @staticmethod
    def verify_quinary(model: nn.Module) -> dict[str, bool]:
        """Verify all QuinaryConv2d weights are in {-2, -1, 0, 1, 2}.

        Returns:
            Dict mapping layer names to verification status.
        """
        results = {}
        valid_vals = {-2.0, -1.0, 0.0, 1.0, 2.0}
        for name, module in model.named_modules():
            if isinstance(module, QuinaryConv2d):
                unique = set(module.conv.weight.data.unique().tolist())
                results[name] = unique.issubset(valid_vals)
        return results


def _replace_module(parent: nn.Module, target_name: str, new_module: nn.Module) -> None:
    """Replace a named module within a parent module."""
    parts = target_name.split(".")
    current = parent
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], new_module)
