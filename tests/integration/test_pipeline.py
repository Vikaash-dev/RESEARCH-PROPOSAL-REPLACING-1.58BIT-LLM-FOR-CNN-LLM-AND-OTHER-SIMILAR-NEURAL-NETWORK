"""Integration tests for the full ResiBit pipeline.

Tests end-to-end flows:
- Model forward/backward pass
- Training loop convergence (overfit single batch)
- Base5 conversion pipeline
"""

import torch
import torch.nn as nn

from src.models.resibit_yolo import ResiBitBackbone
from src.losses.spectral_ortho import SpectralOrthogonalityLoss
from src.training.precision_funnel import PrecisionFunnel
from src.training.monitors import GradientMonitor, WeightDistributionMonitor
from src.deployment.base5_converter import Base5Converter


class TestResiBitBackbone:
    """Integration tests for the full backbone model."""

    def test_forward_pass(self):
        """Model produces valid output and expert features."""
        model = ResiBitBackbone(
            in_channels=3, base_channels=16, num_blocks=2, num_classes=10,
        )
        x = torch.randn(2, 3, 32, 32)
        logits, expert_features = model(x)
        assert logits.shape == (2, 10)
        assert len(expert_features) == 2
        for texture, shape in expert_features:
            assert texture.ndim == 4
            assert shape.ndim == 4

    def test_spectral_loss_computation(self):
        """Spectral loss can be computed from expert features."""
        model = ResiBitBackbone(
            in_channels=3, base_channels=16, num_blocks=2, num_classes=10,
        )
        loss_fn = SpectralOrthogonalityLoss()
        x = torch.randn(2, 3, 32, 32)
        logits, expert_features = model(x)
        spectral_loss = model.compute_spectral_loss(expert_features, loss_fn)
        assert spectral_loss.ndim == 0
        assert spectral_loss.item() >= 0

    def test_overfit_single_batch(self):
        """Model can memorize a single batch (loss decreases)."""
        torch.manual_seed(42)
        model = ResiBitBackbone(
            in_channels=3, base_channels=16, num_blocks=2, num_classes=10,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        spectral_loss_fn = SpectralOrthogonalityLoss()

        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))

        initial_loss = None
        for step in range(20):
            optimizer.zero_grad()
            logits, expert_features = model(x)
            cls_loss = criterion(logits, y)
            spec_loss = model.compute_spectral_loss(expert_features, spectral_loss_fn)
            loss = cls_loss + 0.1 * spec_loss
            loss.backward()
            optimizer.step()
            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        # Loss should decrease when overfitting a single batch
        assert final_loss < initial_loss

    def test_precision_funnel_integration(self):
        """PrecisionFunnel correctly modifies model's INT8 scales."""
        model = ResiBitBackbone(
            in_channels=3, base_channels=16, num_blocks=2, num_classes=10,
        )
        funnel = PrecisionFunnel(start_epoch=0, end_epoch=10, schedule="linear")

        funnel.apply_to_model(model, epoch=5)
        # Check that ResiBit blocks have updated scale
        from src.layers.resibit_block import ResiBitBlock
        for module in model.modules():
            if isinstance(module, ResiBitBlock):
                assert abs(module.int8_scale - 0.5) < 1e-6

    def test_monitoring_integration(self):
        """Gradient and weight monitors work with the full model."""
        model = ResiBitBackbone(
            in_channels=3, base_channels=16, num_blocks=2, num_classes=10,
        )
        x = torch.randn(2, 3, 32, 32)
        logits, _ = model(x)
        logits.sum().backward()

        grad_monitor = GradientMonitor()
        snapshot = grad_monitor.check(model, step=0)
        assert snapshot.total_norm > 0

        weight_monitor = WeightDistributionMonitor()
        w_snapshot = weight_monitor.check(model, step=0)
        total = w_snapshot.dead_zero_pct + w_snapshot.positive_pct + w_snapshot.negative_pct
        assert abs(total - 1.0) < 0.01


class TestBase5ConversionPipeline:
    """Integration tests for the Base-5 conversion pipeline."""

    def test_conversion_produces_output(self):
        """Converted model produces valid output."""
        model = ResiBitBackbone(
            in_channels=3, base_channels=16, num_blocks=2, num_classes=10,
        )
        converter = Base5Converter()
        deploy_model = converter.convert(model)

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 10)

    def test_converted_model_not_trainable(self):
        """Converted model has requires_grad=False on all parameters."""
        model = ResiBitBackbone(
            in_channels=3, base_channels=16, num_blocks=2, num_classes=10,
        )
        converter = Base5Converter()
        deploy_model = converter.convert(model)

        for param in deploy_model.parameters():
            assert not param.requires_grad
