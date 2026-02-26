# Research Proposal: Replacing 1.58-Bit LLM Techniques for CNN, LLM and Other Neural Networks

> **Status:** These are untested, brainstormed approaches being explored.

## Overview

This repository collects research proposals investigating extreme low-bit (1.58-bit / ternary) quantization for compact neural network architectures—particularly real-time object detectors such as YOLO. While Large Language Models (LLMs) can absorb quantization noise thanks to massive parameter redundancy, thin convolutional networks suffer from gradient manifold collapse, codebook instability, and catastrophic accuracy loss under sub-2-bit quantization. The papers in this repository propose novel architectures and training strategies that aim to close this gap.

### Core Problem — The Capacity-Stability Dilemma

Ternary weights ({−1, 0, +1}) offer a theoretical **13.78×** memory reduction versus 32-bit baselines and enable shift-and-add inference with **no floating-point multiplications**. However, directly quantizing lightweight detectors leads to:

- **Gradient manifold shattering** — training gradients lose their geometric structure.
- **Codebook collapse** — effective weight diversity shrinks below what the task requires.
- **Catastrophic mAP degradation** — detection accuracy drops to near zero.

The proposals below each attack these problems from different angles.

## Repository Contents

| File | Paper / Document | Pages |
|------|-----------------|-------|
| `manuscript.pdf` | Spectral-ResiBit YOLO (IEEE TPAMI preprint format) | 4 |
| `75164103-a63b-451f-9e00-625d47c02dea.pdf` | NeuroBit-SCS | 3 |
| `8a1d8042-3b95-497f-9dda-fd0c9b9feb5b.pdf` | Unified Bit-Intelligence | 3 |
| `8b2d7b18-be2a-42ab-908d-62720fd29691.pdf` | ResiBit-YOLO (full investigation) | 28 |
| `c0f315e2-829b-40dc-b069-0b21bedc015c.pdf` | DualExpert-BitYOLO26 | 24 |
| `Document.pdf` | Real-Time Road Anomaly Detection System (Bharat AI SoC Challenge) | 26 |

## Paper Summaries

### 1. Spectral-ResiBit YOLO (`manuscript.pdf`)

*Spectral-ResiBit YOLO: A Frequency-Aware, Latent-Shared 1.58-bit Detector for Edge Intelligence*

Proposes a frequency-aware architecture that resolves the representational deficit of sub-2-bit quantization through three synergistic components: spectral decomposition, residual bit-precision anchoring, and latent-shared feature manifolds. Formatted as an IEEE TPAMI preprint.

### 2. NeuroBit-SCS (`75164103-a63b-451f-9e00-625d47c02dea.pdf`)

*NeuroBit-SCS: Decoupling Texture and Shape in 1.58-Bit YOLO Architectures via Static Channel Splitting*

Introduces **Static Channel Splitting (SCS)** and **INT8 Residual Anchoring (RPA)** to explicitly decouple high-frequency texture features from low-frequency shape representations using a dual-expert ternary block and a Spectral Orthogonality loss.

### 3. Unified Bit-Intelligence (`8a1d8042-3b95-497f-9dda-fd0c9b9feb5b.pdf`)

*Unified Bit-Intelligence: Resolving the Capacity-Stability Dilemma in Sub-2-Bit Edge Detectors*

Presents a unified framework that transitions from the instability of naïve ternary quantization (the "Muon Trap") to a stable, high-capacity approach. Uses a **Precision Funnel** to anchor the gradient manifold and a **Latent-Shared Manifold (LSM)** to cross-pollinate features between dual ternary experts.

### 4. ResiBit-YOLO (`8b2d7b18-be2a-42ab-908d-62720fd29691.pdf`)

*ResiBit-YOLO: Residual-Precision 1.58-bit Object Detection with Dual-Stream Ternary Experts*

The most detailed investigation (28 pages). Documents a problem-solution study of gradient manifold collapse in extreme quantization for edge vision. Introduces the **ResiBit block**: Dual-Stream Ternary Experts (W_A, W_B ∈ {−1, 0, +1}) fused with an INT8 Residual Highway and aggregated by a Group-Mix operation. Includes experiments showing complete gradient manifold collapse (mAP50 = 0.0) under naïve approaches and the recovery enabled by residual precision.

### 5. DualExpert-BitYOLO26 (`c0f315e2-829b-40dc-b069-0b21bedc015c.pdf`)

*DualExpert-BitYOLO26: Restoring Representational Capacity in Extreme Quantisation via Ternary Reparameterisation for Edge-Efficient Real-Time Object Detection*

Trains two parallel ternary weight experts (W_A, W_B ∈ {−1, 0, +1}) and fuses them via additive reparameterisation into a single quinary ({−2, −1, 0, 1, 2}) convolutional layer before deployment. Achieves a 13.78× memory reduction while operating exclusively via shift-and-add arithmetic.

### 6. Real-Time Road Anomaly Detection (`Document.pdf`)

*Real-Time Road Anomaly Detection System — Bharat AI SoC Challenge*

An edge AI application targeting Raspberry Pi that processes dashcam footage in real-time to detect and log road anomalies such as potholes and unexpected obstacles. This document relates to a practical deployment scenario that could benefit from the extreme quantization techniques explored in the other proposals.

## Key Technical Concepts

| Concept | Description |
|---------|-------------|
| **1.58-bit / Ternary Weights** | Weights constrained to {−1, 0, +1}, enabling ~13.78× compression vs. FP32 |
| **Dual-Stream Ternary Experts** | Two parallel ternary convolution streams whose outputs are fused |
| **INT8 Residual Highway** | A higher-precision residual path that anchors gradient stability during QAT |
| **Precision Funnel** | Progressive precision reduction schedule during training |
| **Spectral Orthogonality Loss** | Regularizer that keeps texture and shape feature subspaces orthogonal |
| **Static Channel Splitting (SCS)** | Fixed partitioning of channels into texture vs. shape pathways |
| **Latent-Shared Manifold (LSM)** | Shared feature space enabling knowledge transfer between expert streams |
| **Quantization-Aware Training (QAT)** | Training with simulated quantization to learn quantization-robust weights |

## Implementation: ResiBit-YOLO Python Package

The `src/` directory contains a working PyTorch implementation of the core components described in the research papers. All components are tested (62 tests, 100% pass rate).

### Quick Start

```bash
pip install torch>=2.1.0 pytest
python -m pytest tests/ -v          # Run all tests
```

### Core Layers (`src/layers/`)

| Module | Class | Description |
|--------|-------|-------------|
| `ternary_conv.py` | `TernaryQuantize` | Autograd function: weights → {−1, 0, +1} with STE gradient |
| `ternary_conv.py` | `TernaryConv2d` | Drop-in `nn.Conv2d` replacement with ternary quantization |
| `dual_expert.py` | `DualExpertBlock` | Static Channel Splitting into Texture + Shape ternary experts |
| `resibit_block.py` | `ResiBitBlock` | Dual experts + INT8 Residual Highway + GroupMix aggregation |
| `group_mix.py` | `GroupMix` | Learned channel-wise fusion of multi-stream features |

### Loss Functions (`src/losses/`)

| Module | Class | Description |
|--------|-------|-------------|
| `spectral_ortho.py` | `SpectralOrthogonalityLoss` | FFT-based regularizer enforcing expert frequency specialization |

### Training Utilities (`src/training/`)

| Module | Class | Description |
|--------|-------|-------------|
| `precision_funnel.py` | `PrecisionFunnel` | Schedule to progressively reduce INT8 highway (cosine/linear/step) |
| `monitors.py` | `GradientMonitor` | Detects gradient explosion / Muon Trap failure mode |
| `monitors.py` | `WeightDistributionMonitor` | Detects dead-zero codebook collapse in ternary weights |

### Deployment (`src/deployment/`)

| Module | Class | Description |
|--------|-------|-------------|
| `base5_converter.py` | `Base5Converter` | Fuses dual ternary experts into quinary {−2,−1,0,1,2} weights |
| `base5_converter.py` | `QuinaryConv2d` | Deployment-time fused convolution layer |

### Model (`src/models/`)

| Module | Class | Description |
|--------|-------|-------------|
| `resibit_yolo.py` | `ResiBitBackbone` | Proof-of-concept backbone stacking ResiBit blocks |

### Usage Example

```python
import torch
from src.layers.resibit_block import ResiBitBlock
from src.losses.spectral_ortho import SpectralOrthogonalityLoss
from src.training.precision_funnel import PrecisionFunnel

# Create a ResiBit block
block = ResiBitBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1)

# Forward pass returns output + expert features (for spectral loss)
x = torch.randn(1, 32, 16, 16)
output, texture_feat, shape_feat = block(x)

# Compute spectral orthogonality loss
spectral_loss = SpectralOrthogonalityLoss()(texture_feat, shape_feat)

# PrecisionFunnel: progressively disable INT8 highway during training
funnel = PrecisionFunnel(start_epoch=50, end_epoch=250, schedule="cosine")
scale = funnel.get_scale(epoch=150)  # Returns ~0.5
block.set_int8_scale(scale)
```

## ADA-7 Development Framework

This project follows the **Advanced Development Assistant (ADA-7)** methodology — a 7-stage, evidence-based development process that blends academic research with industry best practices. All knowledge gathered during research analysis is tracked in text files for transparency and reproducibility.

### Knowledge Base (`knowledge/`)

| File | Description |
|------|-------------|
| [`arxiv-analysis.txt`](knowledge/arxiv-analysis.txt) | Analysis of arXiv papers on ternary quantization, binary NNs, and QAT |
| [`github-repos-analysis.txt`](knowledge/github-repos-analysis.txt) | Analysis of relevant GitHub repositories (BitNet, ultralytics, ncnn, etc.) |
| [`cross-analysis.txt`](knowledge/cross-analysis.txt) | Cross-reference: paper concepts ↔ existing implementations, gap analysis |

### Development Stages (`ada7-framework/`)

| Stage | Document | Focus |
|-------|----------|-------|
| 1 | [`stage1-requirements-analysis.txt`](ada7-framework/stage1-requirements-analysis.txt) | User stories, competitive intelligence, SMART requirements |
| 2 | [`stage2-architecture-design.txt`](ada7-framework/stage2-architecture-design.txt) | 3 architecture options with academic validation and decision matrix |
| 3 | [`stage3-component-design.txt`](ada7-framework/stage3-component-design.txt) | Module breakdown, technology stack, dependency graph |
| 4 | [`stage4-implementation-strategy.txt`](ada7-framework/stage4-implementation-strategy.txt) | Phased plan (MoSCoW), CI/CD, code templates |
| 5 | [`stage5-testing-framework.txt`](ada7-framework/stage5-testing-framework.txt) | Test pyramid, quality gates, failure response protocol |
| 6 | [`stage6-deployment-infrastructure.txt`](ada7-framework/stage6-deployment-infrastructure.txt) | Environment strategy, Docker, Raspberry Pi deployment, monitoring |
| 7 | [`stage7-maintenance-evolution.txt`](ada7-framework/stage7-maintenance-evolution.txt) | Operational metrics, evolution roadmap, incident playbooks |

## License

This repository does not currently specify a license. All rights are reserved by the authors unless otherwise stated.
