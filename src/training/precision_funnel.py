"""Precision Funnel: progressive INT8 highway reduction schedule.

During training, the INT8 residual highway starts at full strength
(scale=1.0) and is progressively reduced toward 0.0, forcing the
ternary experts to take over representation capacity.

Reference:
- Unified Bit-Intelligence [K-Dense Web, 2026] — Precision Funnel concept
- ResiBit-YOLO [K-Dense Web, 2026] — INT8 highway architecture
"""

from __future__ import annotations

import math

from src.layers.resibit_block import ResiBitBlock


class PrecisionFunnel:
    """Training schedule that progressively reduces INT8 highway contribution.

    Supports multiple schedule types:
    - linear: Linear decay from 1.0 to min_scale
    - cosine: Cosine annealing (smoother transition)
    - step: Discrete step reduction at specified epochs

    Args:
        start_epoch: Epoch at which to begin reducing INT8 scale.
        end_epoch: Epoch at which INT8 scale reaches min_scale.
        min_scale: Minimum INT8 highway scale (default: 0.0 = pure ternary).
        schedule: Type of decay schedule ('linear', 'cosine', 'step').
    """

    def __init__(
        self,
        start_epoch: int = 50,
        end_epoch: int = 250,
        min_scale: float = 0.0,
        schedule: str = "cosine",
    ) -> None:
        if start_epoch >= end_epoch:
            raise ValueError(f"start_epoch ({start_epoch}) must be < end_epoch ({end_epoch})")
        if not 0.0 <= min_scale <= 1.0:
            raise ValueError(f"min_scale must be in [0, 1], got {min_scale}")

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.min_scale = min_scale
        self.schedule = schedule

    def get_scale(self, epoch: int) -> float:
        """Compute INT8 highway scale for the given epoch.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Scale factor in [min_scale, 1.0].
        """
        if epoch < self.start_epoch:
            return 1.0
        if epoch >= self.end_epoch:
            return self.min_scale

        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

        if self.schedule == "linear":
            return 1.0 - progress * (1.0 - self.min_scale)
        elif self.schedule == "cosine":
            # Cosine annealing: smoother transition
            return self.min_scale + (1.0 - self.min_scale) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        elif self.schedule == "step":
            # Step decay at 25%, 50%, 75% progress
            steps = [0.25, 0.5, 0.75, 1.0]
            scale = 1.0
            step_size = (1.0 - self.min_scale) / len(steps)
            for s in steps:
                if progress >= s:
                    scale -= step_size
            return max(scale, self.min_scale)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def apply_to_model(self, model: object, epoch: int) -> float:
        """Apply current scale to all ResiBitBlocks in a model.

        Args:
            model: nn.Module containing ResiBitBlock instances.
            epoch: Current training epoch.

        Returns:
            The scale value that was applied.
        """
        scale = self.get_scale(epoch)
        for module in _iter_modules(model):
            if isinstance(module, ResiBitBlock):
                module.set_int8_scale(scale)
        return scale


def _iter_modules(model: object):
    """Iterate over all modules in a model (handles nn.Module)."""
    if hasattr(model, "modules"):
        yield from model.modules()
