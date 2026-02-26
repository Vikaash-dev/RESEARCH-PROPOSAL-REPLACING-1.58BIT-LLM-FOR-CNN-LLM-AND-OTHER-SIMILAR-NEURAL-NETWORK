"""
MoQEConv: Mixture-of-Quantised-Experts Convolution Block.

Implements the DualExpert-BitYOLO26 architecture from the research paper.
Two independent ternary expert branches (WA, WB ∈ {-1, 0, +1}) are trained
concurrently and their weights are algebraically fused into a single quinary
({-2, -1, 0, 1, 2}) Base-5 convolution prior to deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet.quantization import quantize_ternary_ste


class MoQEConv(nn.Module):
    """Mixture-of-Quantised-Experts Convolution Block.

    Two independent ternary expert convolutions process the same input in
    parallel. Their outputs are summed, which is equivalent to a single
    convolution with the fused Base-5 weight WB5 = W_A_hat + W_B_hat.

    An orthogonal regularization term prevents expert collapse (both experts
    learning identical features).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel. Default: 3.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides. Default: 1.
        bias: If True, adds a learnable bias. Default: False.
        lambda_orth: Orthogonal regularization coefficient. Default: 1e-4.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        lambda_orth: float = 1e-4,
    ):
        super().__init__()
        shape = (out_channels, in_channels, kernel_size, kernel_size)

        # Two independent sets of full-precision shadow weights
        self.w_expert_A = nn.Parameter(torch.randn(*shape) * 0.02)
        self.w_expert_B = nn.Parameter(torch.randn(*shape) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.stride = stride
        self.padding = padding
        self.lambda_orth = lambda_orth
        self._orth_loss = torch.tensor(0.0)
        self._cached_wa_q = None
        self._cached_wb_q = None

    def get_fused_base5(self) -> torch.Tensor:
        """Return fused Base-5 weight for deployment.

        Fuses the two ternary experts element-wise:
            WB5 = W_A_hat + W_B_hat ∈ {-2, -1, 0, 1, 2}

        Returns:
            INT8 tensor with values in {-2, -1, 0, 1, 2}.
        """
        with torch.no_grad():
            from bitnet.quantization import quantize_ternary

            wa_q = quantize_ternary(self.w_expert_A).to(torch.int8)
            wb_q = quantize_ternary(self.w_expert_B).to(torch.int8)
            return wa_q + wb_q

    def orthogonal_loss(self) -> torch.Tensor:
        """Compute orthogonal regularization term to prevent expert collapse.

        Implements Equation (5): L_orth = lambda * ||W_A^T @ W_B||_F^2

        Uses pre-computed quantized weights if available (set during forward()),
        otherwise quantizes on the fly.
        """
        wa_q = self._cached_wa_q if self._cached_wa_q is not None else quantize_ternary_ste(self.w_expert_A)
        wb_q = self._cached_wb_q if self._cached_wb_q is not None else quantize_ternary_ste(self.w_expert_B)
        # Flatten to 2D: (out_channels, in_channels * k * k)
        wa_flat = wa_q.view(wa_q.size(0), -1)
        wb_flat = wb_q.view(wb_q.size(0), -1)
        gram = wa_flat @ wb_flat.T  # (out_channels, out_channels)
        return self.lambda_orth * (gram**2).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize once and cache for reuse by orthogonal_loss()
        wa_q = quantize_ternary_ste(self.w_expert_A)
        wb_q = quantize_ternary_ste(self.w_expert_B)
        self._cached_wa_q = wa_q
        self._cached_wb_q = wb_q

        y_a = F.conv2d(x, wa_q, self.bias, stride=self.stride, padding=self.padding)
        y_b = F.conv2d(x, wb_q, None, stride=self.stride, padding=self.padding)

        # Compute orthogonal loss using cached quantized weights
        self._orth_loss = self.orthogonal_loss()

        return y_a + y_b

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.w_expert_A.size(1)}, "
            f"out_channels={self.w_expert_A.size(0)}, "
            f"kernel_size={self.w_expert_A.size(2)}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"lambda_orth={self.lambda_orth}"
        )
