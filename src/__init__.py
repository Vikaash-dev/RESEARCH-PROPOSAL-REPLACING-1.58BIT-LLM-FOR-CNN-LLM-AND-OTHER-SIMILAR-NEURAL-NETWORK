"""ResiBit-YOLO: Residual-Precision 1.58-bit Object Detection.

Implements the core components from the research proposals:
- TernaryConv2d with Straight-Through Estimator (STE)
- DualExpertBlock with Static Channel Splitting (SCS)
- ResiBitBlock with INT8 Residual Highway
- SpectralOrthogonalityLoss for expert specialization
- PrecisionFunnel training schedule
- Base5Converter for deployment fusion
"""

from src.layers.ternary_conv import TernaryConv2d, TernaryQuantize
from src.layers.dual_expert import DualExpertBlock
from src.layers.resibit_block import ResiBitBlock
from src.losses.spectral_ortho import SpectralOrthogonalityLoss
from src.training.precision_funnel import PrecisionFunnel
from src.deployment.base5_converter import Base5Converter

__version__ = "0.1.0"

__all__ = [
    "TernaryConv2d",
    "TernaryQuantize",
    "DualExpertBlock",
    "ResiBitBlock",
    "SpectralOrthogonalityLoss",
    "PrecisionFunnel",
    "Base5Converter",
]
