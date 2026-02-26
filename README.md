# BitNet 1.58-bit Implementation: DualExpert-BitYOLO26 & ResiBit-YOLO

Implementation of two research papers on replacing standard neural network weights with 1.58-bit ternary quantization ({-1, 0, +1}) for CNNs, LLMs, and edge deployment:

1. **DualExpert-BitYOLO26**: Dual-expert ternary reparameterization with Base-5 fusion
2. **ResiBit-YOLO**: Residual-Precision 1.58-bit detection with INT8 gradient highway

## Core Innovation

Standard BitNet b1.58 quantization (weights → {-1, 0, +1}) achieves remarkable compression for large language models but **fails catastrophically** for compact CNNs like YOLO. This implementation provides two solutions:

| Architecture | Compression | Training Stability | Key Feature |
|---|---|---|---|
| Single Ternary | 20.19× | ❌ Collapses | Baseline 1.58-bit |
| **DualExpert Base-5** | **13.78×** | ⚠️ Requires warmup | Two ternary experts → quinary {-2..+2} |
| **ResiBit** | **~10×** | ✅ Stable | Dual ternary + INT8 highway + Group-Mix |

## Architecture Overview

### Base-5 Reparameterization (DualExpert-BitYOLO26)

Two independent ternary experts (WA, WB ∈ {-1, 0, +1}) are trained in parallel and fused before deployment:

```
WB5 = WA_hat + WB_hat ∈ {-2, -1, 0, 1, 2}  (quinary)
```

This achieves **2.32 effective bits** from two 1.58-bit streams — a 46.5% capacity increase — while requiring only a single memory load at inference via shift-and-add arithmetic (no floating-point multiplications).

### ResiBit Block (ResiBit-YOLO)

Three parallel streams with learnable Group-Mix aggregation:

```
Y = BN(SiLU(αA·Conv(X; WA) + αB·Conv(X; WB) + αR·Conv(X; WR)))
```

- **Expert A** (ternary): High-frequency texture features
- **Expert B** (ternary): Low-frequency shape features
- **INT8 Highway** (WR): Gradient anchor preventing manifold collapse

### Three-Phase Training Protocol

The structured training curriculum prevents the "Muon Trap" (gradient manifold collapse):

| Phase | Epochs | Description |
|---|---|---|
| Phase 1 | 1–10 | FP32 warmup — build optimizer momentum basis |
| Phase 2 | 11–25 | INT8 highway frozen, ternary QAT with annealed threshold (3α → α) |
| Phase 3 | 26+ | Full QAT — all streams quantized, gradient clipping active |

## Installation

```bash
pip install torch numpy pytest
```

## Quick Start

```python
import torch
from bitnet import ResiBitConv, MoQEConv, BitLinear
from bitnet import ThreePhaseScheduler, TrainingConfig

# --- 1.58-bit Linear Layer (for transformers/LLMs) ---
linear = BitLinear(in_features=768, out_features=256)
x = torch.randn(4, 768)
y = linear(x)  # Uses ternary weights + activation quantization

# --- Dual-Expert Conv Block (for CNNs) ---
moqe = MoQEConv(in_channels=32, out_channels=64, kernel_size=3, padding=1)
x = torch.randn(1, 32, 16, 16)
y = moqe(x)  # Two ternary experts, sum outputs

# Deploy: fuse to Base-5 quinary weights
fused_weights = moqe.get_fused_base5()  # {-2, -1, 0, 1, 2} INT8

# --- ResiBit Block (for stable QAT) ---
resbit = ResiBitConv(in_channels=32, out_channels=64, kernel_size=3, padding=1)
resbit.set_phase(ResiBitConv.PHASE_FULL_QAT)  # Enable quantization
y = resbit(x)  # Three streams: Expert A + Expert B + INT8 Highway
```

### Model Surgery

Replace standard layers in any existing model:

```python
from bitnet import replace_conv2d_with_resbit, replace_linear_with_bitlinear
import torchvision.models as models

model = models.resnet18()
model = replace_conv2d_with_resbit(model, exclude_names=["conv1"])  # Keep first layer FP32
model = replace_linear_with_bitlinear(model, exclude_names=["fc"])  # Keep classifier FP32
```

### Three-Phase Training

```python
from bitnet import ThreePhaseScheduler, TrainingConfig

model = ...  # Model with ResiBitConv blocks
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ThreePhaseScheduler(model, optimizer)

for epoch in range(scheduler.total_epochs):
    phase = scheduler.step_epoch(epoch)

    for x, y in dataloader:
        logits = model(x)
        loss = criterion(logits, y)
        orth_loss = scheduler.collect_orthogonal_loss()
        total_loss = loss + orth_loss

        total_loss.backward()
        scheduler.clip_gradients()
        optimizer.step()
        optimizer.zero_grad()

    # Monitor for gradient collapse
    diagnostics = scheduler.get_diagnostics()
    if diagnostics["collapse_detected"]:
        print(f"WARNING: Dead-weight collapse detected at epoch {epoch}")
```

## Running Tests

```bash
python -m pytest tests/ -v
```

All 103 tests validate theoretical claims from the papers:
- Ternary quantization correctness and STE gradient flow
- Base-5 fusion produces exactly quinary values
- Convolution linearity: Conv(X;WA) + Conv(X;WB) = Conv(X;WA+WB)
- 13.78× compression ratio matches paper Table 1
- INT8 highway gradient anchor prevents manifold collapse
- Three-phase training protocol phase transitions
- Dead-weight collapse detection

## Running Benchmarks

```bash
python benchmark.py
```

## Project Structure

```
bitnet/
├── __init__.py          # Package exports
├── quantization.py      # Ternary, INT8, and activation quantization with STE
├── bitlinear.py         # 1.58-bit linear layer (BitNet b1.58 for transformers)
├── moqe_conv.py         # Dual-Expert Base-5 convolution (DualExpert-BitYOLO26)
├── resbit_conv.py       # ResiBit convolution with INT8 highway (ResiBit-YOLO)
├── training.py          # Three-phase training protocol and collapse detection
└── utils.py             # Model surgery, compression analysis, weight distribution
tests/
├── test_quantization.py # Quantization correctness and information theory
├── test_bitlinear.py    # BitLinear layer tests
├── test_moqe_conv.py    # MoQE dual-expert and Base-5 fusion tests
├── test_resbit_conv.py  # ResiBit three-stream and phase behavior tests
├── test_training.py     # Three-phase scheduler and collapse detection tests
└── test_utils.py        # Model surgery and compression ratio tests
benchmark.py             # Complete benchmark demonstrating all paper claims
```

## Key Theoretical Results

### Compression Ratios (Paper Table 1)

| Method | Bits/Weight | Compression vs FP32 |
|---|---|---|
| FP32 (baseline) | 32.000 | 1.00× |
| INT8 | 8.000 | 4.00× |
| INT4 | 4.000 | 8.00× |
| **DualExpert Base-5** | **2.322** | **13.78×** |
| Single Ternary | 1.585 | 20.19× |

### The Muon Trap

Abrupt ternary QAT causes gradient manifold collapse due to the interaction between:
1. **Muon optimizer's Newton-Schulz orthogonalization**: Amplifies dominant gradient direction
2. **STE gradient discontinuity**: Introduces systematic quantization noise

The result: >59% of weights collapse to zero, mAP50 → 0.0.

**Solution**: The INT8 Residual Highway provides 256 quantization levels (vs ternary's 3), producing a well-conditioned STE gradient that anchors the gradient manifold during QAT.

## References

- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (2024)
- Liu et al., "Bi-Real Net: Enhancing the Performance of 1-bit CNNs" (2018)
- Ding et al., "RepVGG: Making VGG-style ConvNets Great Again" (2021)
- Kosson & Jaggi, "Muon Optimizer" (2024)
- Wang et al., "BitNet b1.58 Reloaded" (2025)

## License

Research implementation for academic use.
