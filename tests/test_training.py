"""
Tests for the Three-Phase Training Protocol.

Validates the structured training curriculum from the ResiBit-YOLO paper:
- Phase transitions occur at correct epoch boundaries
- Learning rate updates happen on phase change
- Ternary anneal factor interpolates correctly over Phase 2
- Gradient clipping is active only when configured
- Collapse detection identifies dead-weight state
- Orthogonal loss collection works across multiple blocks
"""

import torch
import torch.nn as nn

from bitnet.resbit_conv import ResiBitConv
from bitnet.moqe_conv import MoQEConv
from bitnet.training import (
    ThreePhaseScheduler,
    TrainingConfig,
    PhaseConfig,
    compute_gradient_snr,
)


def _make_resbit_model(num_blocks: int = 3) -> nn.Module:
    """Create a simple model with multiple ResiBit blocks."""
    layers = []
    channels = 16
    for _ in range(num_blocks):
        layers.append(ResiBitConv(channels, channels, kernel_size=3, padding=1))
    return nn.Sequential(*layers)


def _make_moqe_model(num_blocks: int = 3) -> nn.Module:
    """Create a simple model with multiple MoQE blocks."""
    layers = []
    channels = 16
    for _ in range(num_blocks):
        layers.append(MoQEConv(channels, channels, kernel_size=3, padding=1))
    return nn.Sequential(*layers)


class TestPhaseTransitions:
    """Tests for phase boundary detection and transitions."""

    def test_phase1_at_epoch_0(self):
        """Epoch 0 should be Phase 1."""
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer)

        phase = scheduler.step_epoch(0)
        assert phase == 1

    def test_phase2_after_phase1(self):
        """Phase 2 starts at epoch = phase1.num_epochs."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=5, learning_rate=1e-3),
            phase2=PhaseConfig(2, num_epochs=10, learning_rate=1e-3, grad_clip_norm=1.0),
            phase3=PhaseConfig(3, num_epochs=20, learning_rate=1e-4),
        )
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        assert scheduler.step_epoch(4) == 1  # Last epoch of Phase 1
        assert scheduler.step_epoch(5) == 2  # First epoch of Phase 2

    def test_phase3_after_phase2(self):
        """Phase 3 starts at epoch = phase1 + phase2 epochs."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=5, learning_rate=1e-3),
            phase2=PhaseConfig(2, num_epochs=10, learning_rate=1e-3),
            phase3=PhaseConfig(3, num_epochs=20, learning_rate=1e-4),
        )
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        assert scheduler.step_epoch(14) == 2  # Last epoch of Phase 2
        assert scheduler.step_epoch(15) == 3  # First epoch of Phase 3

    def test_total_epochs(self):
        """Total epochs should be sum of all phase epochs."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=10, learning_rate=1e-3),
            phase2=PhaseConfig(2, num_epochs=15, learning_rate=1e-3),
            phase3=PhaseConfig(3, num_epochs=300, learning_rate=1e-4),
        )
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        assert scheduler.total_epochs == 325


class TestLearningRate:
    """Tests for learning rate management across phases."""

    def test_lr_changes_on_phase_transition(self):
        """LR should update when transitioning between phases."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=2, learning_rate=1e-3),
            phase2=PhaseConfig(2, num_epochs=2, learning_rate=5e-4),
            phase3=PhaseConfig(3, num_epochs=2, learning_rate=1e-4),
        )
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        scheduler.step_epoch(0)  # Phase 1
        assert optimizer.param_groups[0]["lr"] == 1e-3

        scheduler.step_epoch(2)  # Phase 2
        assert optimizer.param_groups[0]["lr"] == 5e-4

        scheduler.step_epoch(4)  # Phase 3
        assert optimizer.param_groups[0]["lr"] == 1e-4


class TestTernaryAnneal:
    """Tests for the ternary quantization anneal schedule."""

    def test_anneal_at_phase2_start(self):
        """At Phase 2 start, anneal factor should be anneal_start."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=5, learning_rate=1e-3),
            phase2=PhaseConfig(
                2, num_epochs=10, learning_rate=1e-3,
                ternary_anneal_start=3.0, ternary_anneal_end=1.0,
            ),
            phase3=PhaseConfig(3, num_epochs=20, learning_rate=1e-4),
        )
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        scheduler.step_epoch(5)  # First epoch of Phase 2
        anneal = scheduler._compute_ternary_anneal(5)
        assert abs(anneal - 3.0) < 0.01

    def test_anneal_at_phase2_end(self):
        """At Phase 2 end, anneal factor should be anneal_end."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=5, learning_rate=1e-3),
            phase2=PhaseConfig(
                2, num_epochs=10, learning_rate=1e-3,
                ternary_anneal_start=3.0, ternary_anneal_end=1.0,
            ),
            phase3=PhaseConfig(3, num_epochs=20, learning_rate=1e-4),
        )
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        anneal = scheduler._compute_ternary_anneal(14)  # Last epoch of Phase 2
        assert abs(anneal - 1.0) < 0.01

    def test_anneal_linear_interpolation(self):
        """Anneal factor should interpolate linearly across Phase 2."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=0, learning_rate=1e-3),
            phase2=PhaseConfig(
                2, num_epochs=11, learning_rate=1e-3,
                ternary_anneal_start=3.0, ternary_anneal_end=1.0,
            ),
            phase3=PhaseConfig(3, num_epochs=20, learning_rate=1e-4),
        )
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        # Midpoint of Phase 2
        anneal_mid = scheduler._compute_ternary_anneal(5)
        assert abs(anneal_mid - 2.0) < 0.1, (
            f"Expected anneal ≈ 2.0 at midpoint, got {anneal_mid}"
        )

    def test_anneal_1_outside_phase2(self):
        """Anneal factor should be 1.0 in Phase 1 and Phase 3."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=5, learning_rate=1e-3),
            phase2=PhaseConfig(
                2, num_epochs=10, learning_rate=1e-3,
                ternary_anneal_start=3.0, ternary_anneal_end=1.0,
            ),
            phase3=PhaseConfig(3, num_epochs=20, learning_rate=1e-4),
        )
        model = _make_resbit_model()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        assert scheduler._compute_ternary_anneal(0) == 1.0  # Phase 1
        assert scheduler._compute_ternary_anneal(20) == 1.0  # Phase 3


class TestGradientClipping:
    """Tests for gradient clipping behavior."""

    def test_no_clip_in_phase1(self):
        """Phase 1 (default config) has no gradient clipping."""
        model = _make_resbit_model(1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer)

        scheduler.step_epoch(0)
        x = torch.randn(1, 16, 4, 4)
        y = model(x)
        y.sum().backward()

        result = scheduler.clip_gradients()
        assert result is None  # No clipping in Phase 1

    def test_clip_in_phase2(self):
        """Phase 2 should apply gradient clipping."""
        config = TrainingConfig(
            phase1=PhaseConfig(1, num_epochs=1, learning_rate=1e-3),
            phase2=PhaseConfig(2, num_epochs=5, learning_rate=1e-3, grad_clip_norm=1.0),
            phase3=PhaseConfig(3, num_epochs=5, learning_rate=1e-4),
        )
        model = _make_resbit_model(1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer, config)

        scheduler.step_epoch(1)  # Phase 2
        x = torch.randn(1, 16, 4, 4)
        y = model(x)
        y.sum().backward()

        result = scheduler.clip_gradients()
        assert result is not None  # Should return gradient norm


class TestCollapseDetection:
    """Tests for the dead-weight collapse detection mechanism."""

    def test_no_collapse_on_fresh_model(self):
        """A freshly initialized model should not show collapse."""
        model = _make_resbit_model(2)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer)

        result = scheduler.detect_collapse()
        assert not result["collapsed"], (
            f"Fresh model detected as collapsed with zero_frac={result['max_zero_fraction']:.4f}"
        )

    def test_collapse_detected_with_zero_weights(self):
        """Model with >55% zero weights should be detected as collapsed."""
        model = _make_resbit_model(1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer)

        # Force weights to mostly zeros (simulating dead-weight collapse)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, ResiBitConv):
                    m.w_a.fill_(0.0)  # All zeros → quantizes to all zeros
                    m.w_b.fill_(0.0)

        result = scheduler.detect_collapse()
        assert result["collapsed"], "Should detect collapse with all-zero weights"

    def test_diagnostics_structure(self):
        """Diagnostics should contain all expected fields."""
        model = _make_resbit_model(1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer)
        scheduler.step_epoch(0)

        # Run forward + backward to populate gradients
        x = torch.randn(1, 16, 4, 4)
        y = model(x)
        y.sum().backward()

        diag = scheduler.get_diagnostics()
        assert "epoch" in diag
        assert "phase" in diag
        assert "ternary_anneal" in diag
        assert "collapse_detected" in diag
        assert "max_zero_fraction" in diag


class TestOrthogonalLossCollection:
    """Tests for collecting orthogonal loss across multiple blocks."""

    def test_orth_loss_from_resbit_model(self):
        """Should collect orth loss from all ResiBit blocks."""
        model = _make_resbit_model(3)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer)

        # Phase 2 (orth loss active)
        scheduler.step_epoch(10)
        total_loss = scheduler.collect_orthogonal_loss()
        assert total_loss.item() >= 0

    def test_orth_loss_from_moqe_model(self):
        """Should collect orth loss from MoQE blocks too."""
        model = _make_moqe_model(3)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ThreePhaseScheduler(model, optimizer)

        total_loss = scheduler.collect_orthogonal_loss()
        assert total_loss.item() >= 0


class TestGradientSNR:
    """Tests for gradient Signal-to-Noise Ratio computation."""

    def test_snr_returns_dict(self):
        """Should return a dictionary of layer names to SNR values."""
        model = _make_resbit_model(1)
        model[0].set_phase(ResiBitConv.PHASE_FULL_QAT)

        x = torch.randn(2, 16, 4, 4)
        y = model(x)
        y.sum().backward()

        snr = compute_gradient_snr(model)
        assert isinstance(snr, dict)
        assert len(snr) > 0

    def test_snr_positive_values(self):
        """SNR values should be positive (non-negative)."""
        model = _make_resbit_model(1)
        model[0].set_phase(ResiBitConv.PHASE_FULL_QAT)

        x = torch.randn(2, 16, 4, 4)
        y = model(x)
        y.sum().backward()

        snr = compute_gradient_snr(model)
        for name, value in snr.items():
            assert value >= 0, f"Negative SNR for {name}: {value}"
