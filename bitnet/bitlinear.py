"""
BitLinear: 1.58-bit linear layer replacement.

Replaces standard nn.Linear with a ternary weight linear layer that uses
absmean quantization for weights and absmax quantization for activations,
following the BitNet b1.58 methodology (Ma et al., 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet.quantization import quantize_ternary_ste, quantize_activations


class BitLinear(nn.Module):
    """1.58-bit linear layer using ternary weights {-1, 0, +1}.

    During training, maintains full-precision shadow weights and applies
    ternary quantization via STE in the forward pass. During inference,
    weights can be quantized to ternary for deployment.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.
        activation_bits: Number of bits for activation quantization. Default: 8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits

        # Full-precision shadow weights (quantized in forward pass via STE)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.rms_norm = nn.RMSNorm(in_features)

    def get_ternary_weight(self) -> torch.Tensor:
        """Return quantized ternary weight for deployment."""
        with torch.no_grad():
            alpha = self.weight.abs().mean().clamp(min=1e-6)
            return (self.weight / alpha).round().clamp(-1, 1).to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm on input activations
        x_norm = self.rms_norm(x)

        # Quantize activations
        x_quant = quantize_activations(x_norm, bits=self.activation_bits)

        # Quantize weights with STE
        w_quant = quantize_ternary_ste(self.weight)

        return F.linear(x_quant, w_quant, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"activation_bits={self.activation_bits}"
        )
