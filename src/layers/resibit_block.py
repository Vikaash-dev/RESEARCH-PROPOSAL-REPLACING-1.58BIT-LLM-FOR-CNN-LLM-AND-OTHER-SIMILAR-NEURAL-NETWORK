"""ResiBit Block: Dual-Expert Ternary + INT8 Residual Highway.

The core architectural innovation from:
- ResiBit-YOLO [K-Dense Web, 2026]
- Spectral-ResiBit YOLO [K-Dense Web, 2026]
- Unified Bit-Intelligence [K-Dense Web, 2026]

Combines dual ternary experts with an INT8 residual highway to prevent
gradient manifold collapse during quantization-aware training. The INT8
path provides a stable gradient channel that anchors the training dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.layers.dual_expert import DualExpertBlock
from src.layers.group_mix import GroupMix


class ResiBitBlock(nn.Module):
    """ResiBit block: dual ternary experts + INT8 residual highway.

    Architecture:
        Input ──┬── DualExpertBlock (ternary) ──┐
                │                                ├── GroupMix ── Output
                └── INT8 Highway (standard conv) ┘

    The INT8 highway provides gradient stability during QAT. Its contribution
    is scaled by a factor controlled by the PrecisionFunnel, which progressively
    reduces the INT8 path weight during training.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        scs_ratio: Texture/shape channel split ratio for DualExpertBlock.
        int8_width_ratio: Fraction of channels for INT8 highway (default: 0.25).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        scs_ratio: float = 0.5,
        int8_width_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int8_scale = 1.0  # Controlled by PrecisionFunnel

        # Dual-Expert ternary path
        self.dual_expert = DualExpertBlock(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, scs_ratio=scs_ratio,
        )

        # INT8 residual highway (standard Conv2d, quantized to INT8 at deployment)
        self.int8_highway = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Group-Mix aggregation: merges ternary expert output + INT8 highway
        self.group_mix = GroupMix(out_channels, num_streams=2)

        # Activation after mixing
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through ResiBit block.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Tuple of (output, texture_features, shape_features).
            - output: (B, C_out, H, W) — fused block output.
            - texture_features: For spectral orthogonality loss computation.
            - shape_features: For spectral orthogonality loss computation.
        """
        # Dual-Expert ternary path
        expert_out, texture_feat, shape_feat = self.dual_expert(x)

        # INT8 residual highway (scaled by PrecisionFunnel)
        int8_out = self.int8_highway(x) * self.int8_scale

        # Group-Mix aggregation
        mixed = self.group_mix(expert_out, int8_out)
        output = self.act(mixed)

        return output, texture_feat, shape_feat

    def set_int8_scale(self, scale: float) -> None:
        """Set the INT8 highway scaling factor (called by PrecisionFunnel).

        Args:
            scale: Value in [0.0, 1.0]. 1.0 = full INT8 highway,
                   0.0 = pure ternary (INT8 disabled).
        """
        self.int8_scale = max(0.0, min(1.0, scale))

    def weight_statistics(self) -> dict:
        """Aggregate weight statistics from all sub-components."""
        return {
            "dual_expert": self.dual_expert.weight_statistics(),
            "int8_scale": self.int8_scale,
        }
