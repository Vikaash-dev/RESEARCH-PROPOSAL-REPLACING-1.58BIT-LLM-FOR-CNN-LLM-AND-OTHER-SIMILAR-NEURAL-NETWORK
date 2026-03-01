"""Dual-Expert Block with Static Channel Splitting (SCS).

Implements the core dual-expert architecture from:
- NeuroBit-SCS [K-Dense Web, 2026]
- DualExpert-BitYOLO26 [K-Dense Web, 2026]

Splits input channels into texture (high-frequency) and shape (low-frequency)
pathways, each processed by an independent ternary convolution expert.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.layers.ternary_conv import TernaryConv2d


class DualExpertBlock(nn.Module):
    """Dual-Expert ternary convolution block with Static Channel Splitting.

    Splits input channels into two groups:
    - Texture Expert: processes high-frequency detail channels (ternary weights)
    - Shape Expert: processes low-frequency structure channels (ternary weights)

    The split is static (fixed at construction time) to avoid the memory
    bandwidth overhead of dynamic Mixture-of-Experts routing.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        scs_ratio: Fraction of channels allocated to texture expert (default: 0.5).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        scs_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < scs_ratio < 1.0:
            raise ValueError(f"scs_ratio must be in (0, 1), got {scs_ratio}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scs_ratio = scs_ratio

        # Compute channel split sizes
        self.texture_in = int(in_channels * scs_ratio)
        self.shape_in = in_channels - self.texture_in
        self.texture_out = int(out_channels * scs_ratio)
        self.shape_out = out_channels - self.texture_out

        # Ensure at least 1 channel per expert
        if self.texture_in < 1 or self.shape_in < 1:
            raise ValueError(
                f"Channel split results in empty expert: "
                f"texture_in={self.texture_in}, shape_in={self.shape_in}"
            )
        if self.texture_out < 1 or self.shape_out < 1:
            raise ValueError(
                f"Channel split results in empty expert output: "
                f"texture_out={self.texture_out}, shape_out={self.shape_out}"
            )

        # Dual ternary experts
        self.texture_expert = TernaryConv2d(
            self.texture_in, self.texture_out, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.shape_expert = TernaryConv2d(
            self.shape_in, self.shape_out, kernel_size,
            stride=stride, padding=padding, bias=False,
        )

        # Batch normalization per expert (stabilizes ternary training)
        self.texture_bn = nn.BatchNorm2d(self.texture_out)
        self.shape_bn = nn.BatchNorm2d(self.shape_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with static channel splitting.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Tuple of (merged_output, texture_features, shape_features).
            - merged_output: (B, C_out, H, W) — concatenated expert outputs.
            - texture_features: (B, texture_out, H, W) — for spectral loss.
            - shape_features: (B, shape_out, H, W) — for spectral loss.
        """
        # Static Channel Split
        x_texture = x[:, :self.texture_in, :, :]
        x_shape = x[:, self.texture_in:, :, :]

        # Expert processing
        texture_out = self.texture_bn(self.texture_expert(x_texture))
        shape_out = self.shape_bn(self.shape_expert(x_shape))

        # Merge via concatenation
        merged = torch.cat([texture_out, shape_out], dim=1)

        return merged, texture_out, shape_out

    def weight_statistics(self) -> dict[str, dict[str, float]]:
        """Return ternary weight statistics for both experts."""
        return {
            "texture_expert": self.texture_expert.weight_statistics(),
            "shape_expert": self.shape_expert.weight_statistics(),
        }
