"""Group-Mix aggregation module for fusing multi-stream outputs.

Combines outputs from dual ternary experts and INT8 residual highway
using learned channel-wise mixing weights.

Reference: ResiBit-YOLO [K-Dense Web, 2026] — Group-Mix operation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GroupMix(nn.Module):
    """Channel-wise learned aggregation of multiple feature streams.

    Fuses N input streams of shape (B, C, H, W) into a single output
    using a lightweight 1x1 convolution after concatenation.

    Args:
        channels: Number of channels per stream.
        num_streams: Number of input streams to fuse (default: 3).
    """

    def __init__(self, channels: int, num_streams: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.num_streams = num_streams
        # 1x1 conv to mix concatenated streams back to original channel count
        self.mix = nn.Conv2d(channels * num_streams, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        """Fuse multiple feature streams.

        Args:
            *streams: Variable number of tensors, each (B, C, H, W).

        Returns:
            Fused tensor of shape (B, C, H, W).
        """
        if len(streams) != self.num_streams:
            raise ValueError(
                f"Expected {self.num_streams} streams, got {len(streams)}"
            )
        concatenated = torch.cat(streams, dim=1)
        return self.bn(self.mix(concatenated))
