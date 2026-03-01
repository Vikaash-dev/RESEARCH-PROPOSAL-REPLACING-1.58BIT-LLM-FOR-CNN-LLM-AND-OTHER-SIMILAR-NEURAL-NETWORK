"""Unit tests for PrecisionFunnel, GradientMonitor, and WeightDistributionMonitor."""

import torch
import torch.nn as nn
import pytest
import math

from src.training.precision_funnel import PrecisionFunnel
from src.training.monitors import (
    GradientMonitor,
    WeightDistributionMonitor,
    TrainingDivergenceError,
)
from src.layers.resibit_block import ResiBitBlock


class TestPrecisionFunnel:
    """Tests for the INT8 highway scheduling mechanism."""

    def test_before_start_returns_one(self):
        """Scale is 1.0 before start_epoch."""
        funnel = PrecisionFunnel(start_epoch=50, end_epoch=250)
        assert funnel.get_scale(0) == 1.0
        assert funnel.get_scale(49) == 1.0

    def test_after_end_returns_min(self):
        """Scale is min_scale after end_epoch."""
        funnel = PrecisionFunnel(start_epoch=50, end_epoch=250, min_scale=0.0)
        assert funnel.get_scale(250) == 0.0
        assert funnel.get_scale(300) == 0.0

    def test_linear_midpoint(self):
        """Linear schedule returns 0.5 at midpoint."""
        funnel = PrecisionFunnel(
            start_epoch=0, end_epoch=100, min_scale=0.0, schedule="linear"
        )
        assert abs(funnel.get_scale(50) - 0.5) < 1e-6

    def test_cosine_start_and_end(self):
        """Cosine schedule starts at 1.0 and ends at min_scale."""
        funnel = PrecisionFunnel(
            start_epoch=0, end_epoch=100, min_scale=0.0, schedule="cosine"
        )
        assert abs(funnel.get_scale(0) - 1.0) < 1e-6
        assert abs(funnel.get_scale(100) - 0.0) < 1e-6

    def test_cosine_midpoint(self):
        """Cosine schedule returns 0.5 at midpoint."""
        funnel = PrecisionFunnel(
            start_epoch=0, end_epoch=100, min_scale=0.0, schedule="cosine"
        )
        assert abs(funnel.get_scale(50) - 0.5) < 1e-6

    def test_step_schedule(self):
        """Step schedule decreases in discrete steps."""
        funnel = PrecisionFunnel(
            start_epoch=0, end_epoch=100, min_scale=0.0, schedule="step"
        )
        # At 0% progress: 1.0
        assert funnel.get_scale(0) == 1.0
        # At 25%+ progress: should decrease by one step
        scale_25 = funnel.get_scale(25)
        assert scale_25 < 1.0

    def test_invalid_epochs_raises(self):
        """start_epoch >= end_epoch raises ValueError."""
        with pytest.raises(ValueError):
            PrecisionFunnel(start_epoch=100, end_epoch=50)

    def test_invalid_min_scale_raises(self):
        """min_scale outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            PrecisionFunnel(start_epoch=0, end_epoch=100, min_scale=-0.1)

    def test_unknown_schedule_raises(self):
        """Unknown schedule type raises ValueError."""
        funnel = PrecisionFunnel(start_epoch=0, end_epoch=100, schedule="unknown")
        with pytest.raises(ValueError, match="Unknown schedule"):
            funnel.get_scale(50)

    def test_apply_to_model(self):
        """apply_to_model sets INT8 scale on all ResiBitBlocks."""
        model = nn.Sequential(
            ResiBitBlock(16, 16, kernel_size=3, padding=1),
            ResiBitBlock(16, 16, kernel_size=3, padding=1),
        )
        funnel = PrecisionFunnel(start_epoch=0, end_epoch=100, schedule="linear")
        scale = funnel.apply_to_model(model, epoch=50)
        assert abs(scale - 0.5) < 1e-6
        for module in model.modules():
            if isinstance(module, ResiBitBlock):
                assert abs(module.int8_scale - 0.5) < 1e-6


class TestGradientMonitor:
    """Tests for gradient health monitoring."""

    def _make_model_with_gradient(self, grad_scale=1.0):
        """Helper: create a small model and run backward to populate gradients."""
        model = nn.Linear(4, 4)
        x = torch.randn(1, 4)
        loss = (model(x) * grad_scale).sum()
        loss.backward()
        return model

    def test_normal_gradient(self):
        """Normal gradients pass without error."""
        monitor = GradientMonitor(threshold=50.0, divergence_threshold=100.0)
        model = self._make_model_with_gradient(grad_scale=1.0)
        snapshot = monitor.check(model, step=0)
        assert snapshot.total_norm > 0
        assert not snapshot.has_nan
        assert not snapshot.has_inf

    def test_divergence_raises(self):
        """Large gradients raise TrainingDivergenceError."""
        monitor = GradientMonitor(threshold=1.0, divergence_threshold=2.0)
        model = self._make_model_with_gradient(grad_scale=1000.0)
        with pytest.raises(TrainingDivergenceError):
            monitor.check(model, step=0)

    def test_history_tracked(self):
        """Snapshots are stored in history."""
        monitor = GradientMonitor()
        model = self._make_model_with_gradient()
        monitor.check(model, step=0)
        monitor.check(model, step=1)
        assert len(monitor.history) == 2
        assert monitor.history[0].step == 0
        assert monitor.history[1].step == 1


class TestWeightDistributionMonitor:
    """Tests for ternary weight distribution monitoring."""

    def test_reports_ternary_stats(self):
        """Monitor reports valid distribution statistics."""
        from src.layers.ternary_conv import TernaryConv2d

        model = nn.Sequential(
            TernaryConv2d(8, 16, kernel_size=3, padding=1),
            TernaryConv2d(16, 16, kernel_size=3, padding=1),
        )
        monitor = WeightDistributionMonitor()
        snapshot = monitor.check(model, step=0)
        total = snapshot.dead_zero_pct + snapshot.positive_pct + snapshot.negative_pct
        assert abs(total - 1.0) < 0.01

    def test_empty_model(self):
        """Model with no TernaryConv2d returns zero stats."""
        model = nn.Linear(4, 4)
        monitor = WeightDistributionMonitor()
        snapshot = monitor.check(model, step=0)
        assert snapshot.dead_zero_pct == 0.0
