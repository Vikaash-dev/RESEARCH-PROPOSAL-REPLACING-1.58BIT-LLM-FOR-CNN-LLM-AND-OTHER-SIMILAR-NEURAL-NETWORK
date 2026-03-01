"""ResiBit-YOLO: Full model definition using ResiBit blocks.

A demonstration model that stacks ResiBit blocks in a simplified
YOLO-like backbone for proof-of-concept validation.

This is not a full YOLO implementation but shows how ResiBit blocks
integrate into a detection-style architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.layers.resibit_block import ResiBitBlock
from src.losses.spectral_ortho import SpectralOrthogonalityLoss


class ResiBitBackbone(nn.Module):
    """Simplified backbone using stacked ResiBit blocks.

    Architecture: Input → Stem → [ResiBitBlock × N] → Output

    This serves as a proof-of-concept backbone. For production YOLO integration,
    replace the ultralytics Conv blocks with ResiBitBlocks.

    Args:
        in_channels: Input image channels (default: 3 for RGB).
        base_channels: Base channel width (default: 32).
        num_blocks: Number of ResiBit blocks to stack (default: 4).
        num_classes: Number of output classes.
        scs_ratio: Static Channel Splitting ratio.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_blocks: int = 4,
        num_classes: int = 80,
        scs_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Stem: standard FP32 convolution (not quantized)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )

        # Build ResiBit blocks with progressive channel expansion
        blocks = []
        channels = base_channels
        for i in range(num_blocks):
            out_channels = channels * 2 if i < num_blocks - 1 else channels
            blocks.append(
                ResiBitBlock(
                    channels, out_channels,
                    kernel_size=3, stride=2 if i < num_blocks - 1 else 1,
                    padding=1, scs_ratio=scs_ratio,
                )
            )
            channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        # Classification head (simplified — real YOLO has detection head)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through backbone.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Tuple of (logits, expert_features):
            - logits: (B, num_classes) class predictions.
            - expert_features: List of (texture, shape) feature pairs per block,
              used for spectral orthogonality loss.
        """
        x = self.stem(x)

        expert_features = []
        for block in self.blocks:
            x, texture, shape = block(x)
            expert_features.append((texture, shape))

        pooled = self.pool(x).flatten(1)
        logits = self.classifier(pooled)

        return logits, expert_features

    def compute_spectral_loss(
        self,
        expert_features: list[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: SpectralOrthogonalityLoss,
    ) -> torch.Tensor:
        """Compute total spectral orthogonality loss across all blocks.

        Args:
            expert_features: List of (texture, shape) pairs from forward().
            loss_fn: SpectralOrthogonalityLoss instance.

        Returns:
            Mean spectral orthogonality loss across all blocks.
        """
        losses = []
        for texture, shape in expert_features:
            losses.append(loss_fn(texture, shape))
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
