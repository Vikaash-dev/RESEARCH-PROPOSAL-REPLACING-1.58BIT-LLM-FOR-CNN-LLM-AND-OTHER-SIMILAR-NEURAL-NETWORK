"""Unit tests for SpectralOrthogonalityLoss."""

import torch
import pytest

from src.losses.spectral_ortho import SpectralOrthogonalityLoss


class TestSpectralOrthogonalityLoss:
    """Tests for the FFT-based spectral orthogonality regularizer."""

    def test_output_is_scalar(self):
        """Loss returns a scalar tensor."""
        loss_fn = SpectralOrthogonalityLoss()
        t = torch.randn(2, 8, 4, 4)
        s = torch.randn(2, 8, 4, 4)
        loss = loss_fn(t, s)
        assert loss.ndim == 0  # scalar

    def test_loss_non_negative(self):
        """Loss value is non-negative (absolute cosine similarity)."""
        loss_fn = SpectralOrthogonalityLoss()
        t = torch.randn(4, 16, 8, 8)
        s = torch.randn(4, 16, 8, 8)
        loss = loss_fn(t, s)
        assert loss.item() >= 0.0

    def test_identical_features_high_loss(self):
        """Identical features should produce high loss (not orthogonal)."""
        loss_fn = SpectralOrthogonalityLoss()
        features = torch.randn(2, 8, 4, 4)
        loss = loss_fn(features, features)
        # Cosine similarity of identical vectors = 1.0
        assert loss.item() > 0.5

    def test_gradient_exists(self):
        """Loss produces non-zero gradients for backpropagation."""
        loss_fn = SpectralOrthogonalityLoss()
        t = torch.randn(2, 8, 4, 4, requires_grad=True)
        s = torch.randn(2, 8, 4, 4, requires_grad=True)
        loss = loss_fn(t, s)
        loss.backward()
        assert t.grad is not None
        assert s.grad is not None
        assert t.grad.abs().sum() > 0
        assert s.grad.abs().sum() > 0

    def test_different_channel_counts(self):
        """Works with different channel counts for texture and shape."""
        loss_fn = SpectralOrthogonalityLoss()
        t = torch.randn(2, 12, 4, 4)
        s = torch.randn(2, 4, 4, 4)
        loss = loss_fn(t, s)
        assert loss.ndim == 0

    def test_loss_bounded_by_one(self):
        """Loss is bounded by [0, 1] since it's absolute cosine similarity."""
        loss_fn = SpectralOrthogonalityLoss()
        t = torch.randn(4, 8, 8, 8)
        s = torch.randn(4, 8, 8, 8)
        loss = loss_fn(t, s)
        assert 0.0 <= loss.item() <= 1.0 + 1e-6
