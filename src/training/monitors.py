"""Training monitors for gradient health and weight distribution.

Implements the monitoring infrastructure described in:
- ResiBit-YOLO [K-Dense Web, 2026] — gradient manifold collapse detection
- Unified Bit-Intelligence [K-Dense Web, 2026] — dead-zero analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.layers.ternary_conv import TernaryConv2d

logger = logging.getLogger(__name__)


class TrainingDivergenceError(Exception):
    """Raised when gradient norms exceed safety threshold."""

    pass


@dataclass
class GradientSnapshot:
    """Snapshot of gradient health metrics at a single step."""

    step: int
    total_norm: float
    max_layer_norm: float
    max_layer_name: str
    has_nan: bool
    has_inf: bool


class GradientMonitor:
    """Monitor gradient norms to detect the Muon Trap failure mode.

    The "Muon Trap" occurs when the interaction between the optimizer's
    Newton-Schulz orthogonalization and the STE's discontinuous gradients
    causes gradient explosion, driving >60% of weights to dead-zero.

    Args:
        threshold: Gradient norm above which a warning is raised.
        divergence_threshold: Norm above which TrainingDivergenceError is raised.
    """

    def __init__(
        self,
        threshold: float = 50.0,
        divergence_threshold: float = 100.0,
    ) -> None:
        self.threshold = threshold
        self.divergence_threshold = divergence_threshold
        self.history: list[GradientSnapshot] = []

    def check(self, model: nn.Module, step: int) -> GradientSnapshot:
        """Check gradient health across all model parameters.

        Args:
            model: The model to inspect (after backward pass).
            step: Current training step.

        Returns:
            GradientSnapshot with current metrics.

        Raises:
            TrainingDivergenceError: If gradient norm exceeds divergence_threshold.
        """
        total_norm = 0.0
        max_layer_norm = 0.0
        max_layer_name = ""
        has_nan = False
        has_inf = False

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.data
            layer_norm = grad.norm(2).item()
            total_norm += layer_norm ** 2

            if grad.isnan().any():
                has_nan = True
                logger.warning("NaN gradient detected in %s at step %d", name, step)

            if grad.isinf().any():
                has_inf = True
                logger.warning("Inf gradient detected in %s at step %d", name, step)

            if layer_norm > max_layer_norm:
                max_layer_norm = layer_norm
                max_layer_name = name

        total_norm = total_norm ** 0.5

        snapshot = GradientSnapshot(
            step=step,
            total_norm=total_norm,
            max_layer_norm=max_layer_norm,
            max_layer_name=max_layer_name,
            has_nan=has_nan,
            has_inf=has_inf,
        )
        self.history.append(snapshot)

        if total_norm > self.divergence_threshold:
            msg = (
                f"Gradient divergence at step {step}: "
                f"norm={total_norm:.2f} > {self.divergence_threshold}. "
                f"Worst layer: {max_layer_name} (norm={max_layer_norm:.2f})"
            )
            logger.error(msg)
            raise TrainingDivergenceError(msg)

        if total_norm > self.threshold:
            logger.warning(
                "Gradient norm %.2f > threshold %.2f at step %d (layer: %s)",
                total_norm,
                self.threshold,
                step,
                max_layer_name,
            )

        return snapshot


@dataclass
class WeightSnapshot:
    """Snapshot of ternary weight distribution at a single step."""

    step: int
    dead_zero_pct: float
    positive_pct: float
    negative_pct: float
    layer_stats: dict[str, dict[str, float]] = field(default_factory=dict)


class WeightDistributionMonitor:
    """Monitor ternary weight distributions to detect codebook collapse.

    Dead-zero collapse occurs when >50% of ternary weights are driven
    to zero during training, destroying the network's representational
    capacity. This is a key failure mode identified in ResiBit-YOLO.

    Args:
        collapse_threshold: Dead-zero fraction above which a warning is raised.
    """

    def __init__(self, collapse_threshold: float = 0.5) -> None:
        self.collapse_threshold = collapse_threshold
        self.history: list[WeightSnapshot] = []

    def check(self, model: nn.Module, step: int) -> WeightSnapshot:
        """Analyse ternary weight distributions across all TernaryConv2d layers.

        Args:
            model: The model to inspect.
            step: Current training step.

        Returns:
            WeightSnapshot with distribution metrics.
        """
        total_params = 0
        total_zeros = 0
        total_pos = 0
        total_neg = 0
        layer_stats = {}

        for name, module in model.named_modules():
            if isinstance(module, TernaryConv2d):
                stats = module.weight_statistics()
                layer_stats[name] = stats
                n = module.weight.numel()
                total_params += n
                total_zeros += int(stats["zero"] * n)
                total_pos += int(stats["positive"] * n)
                total_neg += int(stats["negative"] * n)

        if total_params == 0:
            dead_zero = 0.0
            pos_pct = 0.0
            neg_pct = 0.0
        else:
            dead_zero = total_zeros / total_params
            pos_pct = total_pos / total_params
            neg_pct = total_neg / total_params

        snapshot = WeightSnapshot(
            step=step,
            dead_zero_pct=dead_zero,
            positive_pct=pos_pct,
            negative_pct=neg_pct,
            layer_stats=layer_stats,
        )
        self.history.append(snapshot)

        if dead_zero > self.collapse_threshold:
            logger.warning(
                "Dead-zero collapse detected at step %d: %.1f%% zeros "
                "(threshold: %.1f%%)",
                step,
                dead_zero * 100,
                self.collapse_threshold * 100,
            )

        return snapshot
