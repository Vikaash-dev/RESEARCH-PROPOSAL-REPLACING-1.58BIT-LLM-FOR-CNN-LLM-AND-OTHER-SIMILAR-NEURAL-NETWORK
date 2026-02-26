"""Spectral Orthogonality Loss for expert specialization.

Enforces that texture and shape experts learn complementary
representations by minimizing their cosine similarity in the
frequency domain (via 2D FFT).

Reference:
- NeuroBit-SCS [K-Dense Web, 2026]
- Spectral-ResiBit YOLO [K-Dense Web, 2026]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralOrthogonalityLoss(nn.Module):
    """FFT-based spectral orthogonality regularizer.

    Computes 2D FFT of texture and shape feature maps, then minimizes
    their cosine similarity in the frequency domain. This forces each
    expert to specialize in non-overlapping frequency bands.

    A loss value of 0.0 means the experts are perfectly orthogonal in
    frequency space. Higher values indicate redundant overlap.

    Args:
        eps: Small constant for numerical stability in cosine similarity.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self, texture_features: torch.Tensor, shape_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute spectral orthogonality loss between two feature streams.

        Args:
            texture_features: (B, C_t, H, W) — texture expert activations.
            shape_features: (B, C_s, H, W) — shape expert activations.

        Returns:
            Scalar loss value (lower = more orthogonal).
        """
        # 2D FFT of each feature stream (magnitude spectrum)
        # Use rfft2 for real-valued inputs (more efficient)
        texture_fft = torch.fft.rfft2(texture_features).abs()
        shape_fft = torch.fft.rfft2(shape_features).abs()

        # Flatten spatial+freq dimensions for cosine similarity
        # Shape: (B, C_t, H*W_freq) and (B, C_s, H*W_freq)
        t_flat = texture_fft.flatten(start_dim=2)
        s_flat = shape_fft.flatten(start_dim=2)

        # Global average over channels to get single spectral vector per sample
        # Shape: (B, H*W_freq)
        t_avg = t_flat.mean(dim=1)
        s_avg = s_flat.mean(dim=1)

        # Cosine similarity between spectral representations
        dot = (t_avg * s_avg).sum(dim=1)
        t_norm = t_avg.norm(dim=1).clamp(min=self.eps)
        s_norm = s_avg.norm(dim=1).clamp(min=self.eps)
        cosine_sim = dot / (t_norm * s_norm)

        # Loss = mean absolute cosine similarity across batch
        # Target: 0.0 (orthogonal spectra)
        return cosine_sim.abs().mean()
