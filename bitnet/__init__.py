"""
BitNet 1.58-bit Implementation Package

Implements the core components from two research papers:
1. DualExpert-BitYOLO26: Dual-expert ternary reparameterisation with Base-5 fusion
2. ResiBit-YOLO: Dual-Stream Ternary Experts + INT8 Residual Highway

Key components:
- Ternary quantization (absmean): weights → {-1, 0, +1}
- BitLinear: 1.58-bit linear layer replacement
- MoQEConv: Mixture-of-Quantised-Experts convolution block
- ResiBitConv: ResiBit convolution block with INT8 Residual Highway
- Model surgery utilities for replacing standard layers
"""

from bitnet.quantization import (
    quantize_ternary,
    quantize_ternary_ste,
    quantize_int8,
    quantize_int8_per_channel,
    quantize_activations,
    compute_ste_gradient_variance_ratio,
    compute_cosine_similarity,
    compute_weight_entropy,
    compute_effective_rank,
)
from bitnet.bitlinear import BitLinear
from bitnet.moqe_conv import MoQEConv
from bitnet.resbit_conv import ResiBitConv
from bitnet.training import (
    ThreePhaseScheduler,
    TrainingConfig,
    PhaseConfig,
    compute_gradient_snr,
)
from bitnet.utils import (
    replace_linear_with_bitlinear,
    replace_conv2d_with_moqe,
    replace_conv2d_with_resbit,
    compute_compression_ratio,
    compute_weight_distribution,
    compute_quantization_mse,
    compute_mix_diversification_loss,
)

__version__ = "0.1.0"

__all__ = [
    "quantize_ternary",
    "quantize_ternary_ste",
    "quantize_int8",
    "quantize_int8_per_channel",
    "quantize_activations",
    "compute_ste_gradient_variance_ratio",
    "compute_cosine_similarity",
    "compute_weight_entropy",
    "compute_effective_rank",
    "BitLinear",
    "MoQEConv",
    "ResiBitConv",
    "ThreePhaseScheduler",
    "TrainingConfig",
    "PhaseConfig",
    "compute_gradient_snr",
    "replace_linear_with_bitlinear",
    "replace_conv2d_with_moqe",
    "replace_conv2d_with_resbit",
    "compute_compression_ratio",
    "compute_weight_distribution",
    "compute_quantization_mse",
    "compute_mix_diversification_loss",
]
