"""
Utility functions for model surgery and compression analysis.

Provides tools for:
- Replacing standard layers (nn.Linear, nn.Conv2d) with quantized versions
- Computing compression ratios
- Analyzing weight distributions for ternary networks
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from bitnet.bitlinear import BitLinear
from bitnet.moqe_conv import MoQEConv
from bitnet.resbit_conv import ResiBitConv


def replace_linear_with_bitlinear(
    model: nn.Module,
    exclude_names: Optional[list[str]] = None,
) -> nn.Module:
    """Replace all nn.Linear layers with BitLinear layers.

    Performs model surgery to replace standard linear layers with 1.58-bit
    ternary linear layers.

    Args:
        model: The model to modify (modified in-place).
        exclude_names: List of layer name patterns to exclude from replacement.

    Returns:
        The modified model.
    """
    exclude_names = exclude_names or []

    for name, module in list(model.named_children()):
        if any(ex in name for ex in exclude_names):
            continue

        if isinstance(module, nn.Linear):
            bit_linear = BitLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
            # Copy existing weights as initialization
            with torch.no_grad():
                bit_linear.weight.copy_(module.weight)
                if module.bias is not None and bit_linear.bias is not None:
                    bit_linear.bias.copy_(module.bias)
            setattr(model, name, bit_linear)
        else:
            replace_linear_with_bitlinear(module, exclude_names)

    return model


def replace_conv2d_with_moqe(
    model: nn.Module,
    exclude_names: Optional[list[str]] = None,
    min_channels: int = 8,
) -> nn.Module:
    """Replace nn.Conv2d layers with MoQEConv (Dual-Expert Base-5) blocks.

    Implements the sensitivity-aware layer replacement from the paper:
    layers with very few output channels or matching exclude patterns
    are preserved in full precision.

    Args:
        model: The model to modify (modified in-place).
        exclude_names: List of layer name patterns to exclude.
        min_channels: Minimum output channels required for replacement.

    Returns:
        The modified model.
    """
    exclude_names = exclude_names or []

    for name, module in list(model.named_children()):
        if any(ex in name for ex in exclude_names):
            continue

        if isinstance(module, nn.Conv2d):
            # Skip layers with too few channels (detection head outputs)
            if module.out_channels < min_channels:
                continue
            # Only replace square kernels for now
            k = module.kernel_size
            if isinstance(k, tuple):
                if k[0] != k[1]:
                    continue
                k = k[0]

            s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            p = module.padding[0] if isinstance(module.padding, tuple) else module.padding

            moqe = MoQEConv(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=k,
                stride=s,
                padding=p,
                bias=module.bias is not None,
            )
            # Initialize experts from the original weights
            # Add small perturbation to Expert B to break symmetry,
            # helping orthogonal regularization separate the experts
            with torch.no_grad():
                moqe.w_expert_A.copy_(module.weight)
                moqe.w_expert_B.copy_(module.weight)
                moqe.w_expert_B.add_(torch.randn_like(module.weight) * 0.001)
                if module.bias is not None and moqe.bias is not None:
                    moqe.bias.copy_(module.bias)
            setattr(model, name, moqe)
        else:
            replace_conv2d_with_moqe(module, exclude_names, min_channels)

    return model


def replace_conv2d_with_resbit(
    model: nn.Module,
    exclude_names: Optional[list[str]] = None,
    min_channels: int = 8,
) -> nn.Module:
    """Replace nn.Conv2d layers with ResiBitConv blocks.

    The ResiBit architecture adds an INT8 Residual Highway alongside the
    dual ternary experts to prevent gradient manifold collapse during QAT.

    Args:
        model: The model to modify (modified in-place).
        exclude_names: List of layer name patterns to exclude.
        min_channels: Minimum output channels required for replacement.

    Returns:
        The modified model.
    """
    exclude_names = exclude_names or []

    for name, module in list(model.named_children()):
        if any(ex in name for ex in exclude_names):
            continue

        if isinstance(module, nn.Conv2d):
            if module.out_channels < min_channels:
                continue
            k = module.kernel_size
            if isinstance(k, tuple):
                if k[0] != k[1]:
                    continue
                k = k[0]

            s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            p = module.padding[0] if isinstance(module.padding, tuple) else module.padding

            resbit = ResiBitConv(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=k,
                stride=s,
                padding=p,
            )
            # Initialize all three streams from the original weights
            # Add small perturbation to Expert B to break symmetry
            with torch.no_grad():
                resbit.w_a.copy_(module.weight)
                resbit.w_b.copy_(module.weight)
                resbit.w_b.add_(torch.randn_like(module.weight) * 0.001)
                resbit.w_r.copy_(module.weight)
            setattr(model, name, resbit)
        else:
            replace_conv2d_with_resbit(module, exclude_names, min_channels)

    return model


def compute_compression_ratio(
    num_weights: int,
    method: str = "base5",
) -> dict[str, float]:
    """Compute theoretical compression ratios for different quantization methods.

    Based on Table 1 from the DualExpert-BitYOLO26 paper.

    Args:
        num_weights: Total number of weights to compress.
        method: Quantization method. One of 'ternary', 'base5', 'int8', 'int4'.

    Returns:
        Dictionary with compression metrics.
    """
    fp32_bits = 32
    fp32_storage_mb = num_weights * fp32_bits / 8 / (1024**2)

    methods = {
        "ternary": {"bits_per_weight": math.log2(3), "effective_bits": math.log2(3)},
        "base5": {"bits_per_weight": math.log2(5), "effective_bits": math.log2(5)},
        "int8": {"bits_per_weight": 8.0, "effective_bits": 8.0},
        "int4": {"bits_per_weight": 4.0, "effective_bits": 4.0},
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

    info = methods[method]
    compressed_storage_mb = num_weights * info["bits_per_weight"] / 8 / (1024**2)
    compression_ratio = fp32_bits / info["effective_bits"]

    return {
        "method": method,
        "num_weights": num_weights,
        "bits_per_weight": info["bits_per_weight"],
        "effective_bits": info["effective_bits"],
        "fp32_storage_mb": fp32_storage_mb,
        "compressed_storage_mb": compressed_storage_mb,
        "compression_ratio": compression_ratio,
    }


def compute_weight_distribution(W: torch.Tensor) -> dict[str, float]:
    """Analyze the ternary weight distribution of a quantized tensor.

    Computes the fraction of weights in each ternary state {-1, 0, +1}.
    A healthy ternary distribution has roughly equal fractions in all three
    states (~33% each). The dead-weight collapse failure mode shows >59%
    zeros.

    Args:
        W: A ternary weight tensor with values in {-1, 0, +1}.

    Returns:
        Dictionary with fraction of weights in each state.
    """
    total = W.numel()
    neg_frac = (W == -1).sum().item() / total
    zero_frac = (W == 0).sum().item() / total
    pos_frac = (W == 1).sum().item() / total

    return {
        "fraction_negative": neg_frac,
        "fraction_zero": zero_frac,
        "fraction_positive": pos_frac,
        "total_weights": total,
        "is_healthy": zero_frac < 0.5,  # >50% zeros indicates collapse
    }
