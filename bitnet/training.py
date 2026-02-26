"""
Three-Phase Training Protocol for ResiBit-YOLO.

Implements the structured training curriculum that prevents the "Muon Trap"
(gradient manifold collapse) identified in the DualExpert-Base5 experiment.

Phase 1 (FP32 Warmup):
    All streams train in full FP32 precision. No quantization is applied.
    This allows the optimizer's momentum buffers to accumulate a stable
    gradient basis before any quantization discontinuities are introduced.

Phase 2 (INT8 Highway Freeze + Ternary Warmup):
    The INT8 highway weights are frozen and quantized to INT8. Ternary
    quantization is applied to expert weights with a linearly annealed
    step-size: the absmean threshold starts at 3*alpha and reduces to alpha.
    Gradient clipping with global norm threshold 1.0 is applied.

Phase 3 (Full ResiBit QAT):
    All streams undergo their respective quantization. Ternary experts use
    standard absmean QAT; highway uses INT8. Orthogonal regularization and
    Group-Mix adaptation are active throughout.

The key insight: by decoupling the phases, the gradient manifold never
experiences the simultaneous, abrupt introduction of ternary quantization
hyperplanes across all layers that shatters it in the naive approach.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from bitnet.resbit_conv import ResiBitConv
from bitnet.moqe_conv import MoQEConv


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""

    phase_number: int
    num_epochs: int
    learning_rate: float
    grad_clip_norm: Optional[float] = None
    ternary_anneal_start: float = 1.0  # Only used in Phase 2
    ternary_anneal_end: float = 1.0
    description: str = ""


@dataclass
class TrainingConfig:
    """Complete three-phase training configuration.

    Default values follow the paper's Table 5 hyperparameters.
    """

    phase1: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        phase_number=1,
        num_epochs=10,
        learning_rate=1e-3,
        description="FP32 Warmup - no quantization, build gradient basis",
    ))
    phase2: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        phase_number=2,
        num_epochs=15,
        learning_rate=1e-3,
        grad_clip_norm=1.0,
        ternary_anneal_start=3.0,  # Start at 3*alpha (softer rounding)
        ternary_anneal_end=1.0,    # End at alpha (standard absmean)
        description="INT8 freeze + gradual ternary QAT with annealed threshold",
    ))
    phase3: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        phase_number=3,
        num_epochs=300,
        learning_rate=1e-4,
        grad_clip_norm=1.0,
        description="Full QAT - all streams quantized simultaneously",
    ))
    lambda_orth: float = 1e-4
    collapse_threshold: float = 0.55  # Zero fraction above this = collapse


class ThreePhaseScheduler:
    """Orchestrates the three-phase training protocol for ResiBit models.

    Manages phase transitions, learning rate scheduling, gradient clipping,
    ternary anneal factors, and collapse detection.

    Usage:
        scheduler = ThreePhaseScheduler(model, optimizer, config)
        for epoch in range(total_epochs):
            scheduler.step_epoch(epoch)
            for batch in dataloader:
                loss = model(batch)
                orth_loss = scheduler.collect_orthogonal_loss()
                total_loss = loss + orth_loss
                total_loss.backward()
                scheduler.clip_gradients()
                optimizer.step()
            diagnostics = scheduler.get_diagnostics()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or TrainingConfig()
        self._current_phase = 0
        self._current_epoch = 0
        self._phase_boundaries = self._compute_phase_boundaries()

    def _compute_phase_boundaries(self) -> list[tuple[int, int]]:
        """Compute (start_epoch, end_epoch) for each phase."""
        p1_end = self.config.phase1.num_epochs
        p2_end = p1_end + self.config.phase2.num_epochs
        p3_end = p2_end + self.config.phase3.num_epochs
        return [(0, p1_end), (p1_end, p2_end), (p2_end, p3_end)]

    @property
    def total_epochs(self) -> int:
        """Total number of training epochs across all phases."""
        return sum(
            p.num_epochs
            for p in [self.config.phase1, self.config.phase2, self.config.phase3]
        )

    @property
    def current_phase(self) -> int:
        """Current training phase (1, 2, or 3)."""
        return self._current_phase

    def _get_resbit_modules(self) -> list[ResiBitConv]:
        """Find all ResiBitConv modules in the model."""
        return [m for m in self.model.modules() if isinstance(m, ResiBitConv)]

    def _get_moqe_modules(self) -> list[MoQEConv]:
        """Find all MoQEConv modules in the model."""
        return [m for m in self.model.modules() if isinstance(m, MoQEConv)]

    def _compute_ternary_anneal(self, epoch: int) -> float:
        """Compute the ternary anneal factor for the current epoch.

        Linear interpolation from anneal_start to anneal_end over Phase 2.
        """
        p2_start, p2_end = self._phase_boundaries[1]
        if epoch < p2_start or epoch >= p2_end:
            return 1.0  # No annealing outside Phase 2

        phase2_epochs = p2_end - p2_start
        epoch_in_phase = epoch - p2_start
        progress = epoch_in_phase / max(phase2_epochs - 1, 1)

        start = self.config.phase2.ternary_anneal_start
        end = self.config.phase2.ternary_anneal_end
        return start + (end - start) * progress

    def _update_learning_rate(self, phase_config: PhaseConfig) -> None:
        """Update optimizer learning rate for the current phase."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = phase_config.learning_rate

    def step_epoch(self, epoch: int) -> int:
        """Advance the scheduler to the given epoch.

        Sets the training phase, updates learning rates, configures
        quantization parameters, and returns the current phase number.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Current phase number (1, 2, or 3).
        """
        self._current_epoch = epoch

        # Determine which phase we're in
        if epoch < self._phase_boundaries[0][1]:
            phase = 1
            phase_config = self.config.phase1
        elif epoch < self._phase_boundaries[1][1]:
            phase = 2
            phase_config = self.config.phase2
        else:
            phase = 3
            phase_config = self.config.phase3

        phase_changed = phase != self._current_phase
        self._current_phase = phase

        # Update learning rate on phase change
        if phase_changed:
            self._update_learning_rate(phase_config)

        # Compute ternary anneal factor
        anneal = self._compute_ternary_anneal(epoch)

        # Configure all ResiBit modules
        for module in self._get_resbit_modules():
            module.set_phase(phase, ternary_anneal=anneal)

        return phase

    def clip_gradients(self) -> Optional[float]:
        """Apply gradient clipping if configured for the current phase.

        Returns:
            The total gradient norm, or None if clipping is not active.
        """
        phase_config = [
            self.config.phase1, self.config.phase2, self.config.phase3
        ][self._current_phase - 1]

        if phase_config.grad_clip_norm is not None:
            return torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), phase_config.grad_clip_norm
            ).item()
        return None

    def collect_orthogonal_loss(self) -> torch.Tensor:
        """Sum orthogonal regularization losses from all quantized blocks.

        Returns:
            Total orthogonal loss term to add to the training loss.
        """
        total = torch.tensor(0.0)
        device = None

        for module in self._get_resbit_modules():
            loss = module.orthogonal_loss()
            if device is None and loss.device.type != "cpu":
                device = loss.device
            total = total.to(loss.device) + loss

        for module in self._get_moqe_modules():
            loss = module.orthogonal_loss()
            if device is None and loss.device.type != "cpu":
                device = loss.device
            total = total.to(loss.device) + loss

        return total

    def detect_collapse(self) -> dict[str, object]:
        """Check all ternary experts for dead-weight collapse.

        The "Muon Trap" causes >59% of weights to converge to zero.
        This method detects that condition and returns diagnostic info.

        Returns:
            Dictionary with collapse detection results:
            - 'collapsed': True if any layer shows dead-weight collapse
            - 'max_zero_fraction': Maximum zero fraction across all layers
            - 'layer_diagnostics': Per-layer zero fraction details
        """
        threshold = self.config.collapse_threshold
        layer_diags = []
        max_zero_frac = 0.0
        any_collapsed = False

        for name, module in self.model.named_modules():
            if isinstance(module, ResiBitConv):
                for expert_name, param in [("w_a", module.w_a), ("w_b", module.w_b)]:
                    with torch.no_grad():
                        alpha = param.abs().mean().clamp(min=1e-6)
                        ternary = (param / alpha).round().clamp(-1, 1)
                        zero_frac = (ternary == 0).float().mean().item()
                        max_zero_frac = max(max_zero_frac, zero_frac)
                        collapsed = zero_frac > threshold
                        any_collapsed = any_collapsed or collapsed
                        layer_diags.append({
                            "layer": f"{name}.{expert_name}",
                            "zero_fraction": zero_frac,
                            "collapsed": collapsed,
                        })

            elif isinstance(module, MoQEConv):
                for expert_name, param in [
                    ("w_expert_A", module.w_expert_A),
                    ("w_expert_B", module.w_expert_B),
                ]:
                    with torch.no_grad():
                        alpha = param.abs().mean().clamp(min=1e-6)
                        ternary = (param / alpha).round().clamp(-1, 1)
                        zero_frac = (ternary == 0).float().mean().item()
                        max_zero_frac = max(max_zero_frac, zero_frac)
                        collapsed = zero_frac > threshold
                        any_collapsed = any_collapsed or collapsed
                        layer_diags.append({
                            "layer": f"{name}.{expert_name}",
                            "zero_fraction": zero_frac,
                            "collapsed": collapsed,
                        })

        return {
            "collapsed": any_collapsed,
            "max_zero_fraction": max_zero_frac,
            "layer_diagnostics": layer_diags,
        }

    def get_diagnostics(self) -> dict[str, object]:
        """Comprehensive training diagnostics for the current state.

        Returns gradient norms, weight distributions, phase info, and
        collapse detection results.
        """
        grad_norms = {}
        weight_stats = {}
        mix_coefficients = {}

        for name, module in self.model.named_modules():
            if isinstance(module, ResiBitConv):
                # Gradient norms for each stream
                for pname, param in [
                    ("w_a", module.w_a),
                    ("w_b", module.w_b),
                    ("w_r", module.w_r),
                ]:
                    if param.grad is not None:
                        grad_norms[f"{name}.{pname}"] = param.grad.norm().item()

                # Mix coefficients
                mix = module._mix_weights().detach()
                mix_coefficients[name] = {
                    "alpha_a": mix[0].mean().item(),
                    "alpha_b": mix[1].mean().item(),
                    "alpha_r": mix[2].mean().item(),
                }

                # Weight statistics
                with torch.no_grad():
                    for pname, param in [("w_a", module.w_a), ("w_b", module.w_b)]:
                        alpha = param.abs().mean().clamp(min=1e-6)
                        ternary = (param / alpha).round().clamp(-1, 1)
                        weight_stats[f"{name}.{pname}"] = {
                            "mean": param.mean().item(),
                            "std": param.std().item(),
                            "zero_fraction": (ternary == 0).float().mean().item(),
                        }

        collapse = self.detect_collapse()

        return {
            "epoch": self._current_epoch,
            "phase": self._current_phase,
            "ternary_anneal": self._compute_ternary_anneal(self._current_epoch),
            "grad_norms": grad_norms,
            "weight_stats": weight_stats,
            "mix_coefficients": mix_coefficients,
            "collapse_detected": collapse["collapsed"],
            "max_zero_fraction": collapse["max_zero_fraction"],
        }


def compute_gradient_snr(model: nn.Module) -> dict[str, float]:
    """Compute gradient Signal-to-Noise Ratio for quantized layers.

    The gradient SNR is defined as:
        SNR = ||grad_signal||_2 / ||grad_noise||_2

    where grad_signal is the mean gradient and grad_noise is the
    deviation from the mean. Low SNR indicates the gradient is dominated
    by quantization noise, which precedes manifold collapse.

    Args:
        model: Model with gradients computed.

    Returns:
        Dictionary mapping layer names to gradient SNR values.
    """
    snr_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.numel() > 1:
            grad = param.grad
            signal = grad.mean()
            noise = (grad - signal).norm()
            signal_norm = signal.abs()
            if noise > 1e-10:
                snr_dict[name] = (signal_norm / noise).item()
    return snr_dict
