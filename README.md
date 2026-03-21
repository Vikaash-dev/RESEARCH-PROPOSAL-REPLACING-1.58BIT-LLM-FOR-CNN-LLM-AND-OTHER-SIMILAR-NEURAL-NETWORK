# RESEARCH-PROPOSAL-REPLACING-1.58BIT-LLM-FOR-CNN-LLM-AND-OTHER-SIMILAR-NEURAL-NETWORK

> ⚠️ **DISCLAIMER: All files in this repository are UNVALIDATED. The research proposals, documents, and manuscripts have not been peer-reviewed, experimentally verified, or formally validated. All experimental results, architectural claims, benchmark numbers, and theoretical analyses are preliminary, unverified, and should be treated as speculative. Do not cite, reproduce, or rely on this material for any purpose without independent verification.**

---

## Table of Contents

1. [Overview and Research Context](#1-overview-and-research-context)
2. [Repository Contents and File Map](#2-repository-contents-and-file-map)
3. [Research Progression: The Paper Series](#3-research-progression-the-paper-series)
   - [Paper 1 — DualExpert-BitYOLO26](#31-paper-1--dualexpert-bityolo26)
   - [Paper 2 — NeuroBit-SCS](#32-paper-2--neurobit-scs)
   - [Paper 3 — ResiBit-YOLO](#33-paper-3--resibit-yolo)
   - [Paper 4 — Unified Bit-Intelligence](#34-paper-4--unified-bit-intelligence)
   - [Paper 5 — Spectral-ResiBit YOLO (Final Publication)](#35-paper-5--spectral-resibit-yolo-final-publication)
   - [Document 6 — Real-Time Road Anomaly Detection System](#36-document-6--real-time-road-anomaly-detection-system)
4. [Core Problem: The Capacity-Stability Dilemma](#4-core-problem-the-capacity-stability-dilemma)
5. [Key Technical Concepts Explained](#5-key-technical-concepts-explained)
6. [Proposed Architectural Solutions](#6-proposed-architectural-solutions)
7. [Experimental Results Summary](#7-experimental-results-summary)
8. [Cross-Reference with Related and Foundational Works](#8-cross-reference-with-related-and-foundational-works)
9. [Critical Analysis and Open Questions](#9-critical-analysis-and-open-questions)
10. [Unvalidated Research Disclaimer](#10-unvalidated-research-disclaimer)
11. [Empirical Validation Plan](#11-empirical-validation-plan)

---

## 1. Overview and Research Context

This repository contains a series of research manuscripts and one supporting application document. The primary research theme is the application of **1.58-bit (ternary) neural network quantization** — a technique proven highly effective for Large Language Models (LLMs) — to compact **Convolutional Neural Networks (CNNs)**, specifically the **YOLO family of real-time object detectors** targeting **edge hardware** (ARM-based platforms such as the Raspberry Pi 5).

### The Core Scientific Question

> *Can the ternary weight paradigm ({-1, 0, +1}), which has achieved near-lossless compression in LLMs with billions of parameters, be applied to CNNs with only 2–3 million parameters without catastrophic accuracy loss?*

The papers collectively answer: **No, not directly — but a modified architecture can bridge the gap.** The series progresses from identifying the failure (complete accuracy collapse), diagnosing its root cause (the "Muon Trap"), and iteratively designing an architecture that resolves it.

### Why This Question Matters

The **BitNet b1.58** paradigm (Ma et al., 2024) demonstrated that LLMs can achieve near-parity with full-precision models using only ternary weights. This produces two game-changing properties:

1. **Memory compression**: ~20x smaller weight storage vs. FP32, ~4x vs. INT8
2. **Multiplication elimination**: Every multiply-accumulate is replaced by integer addition/subtraction or a no-op, cutting energy consumption dramatically on edge hardware

If this paradigm could be transferred to vision models, it would enable truly multiplication-free object detection on sub-watt edge platforms. The research in this repository represents a structured investigation of whether, how, and under what architectural conditions this is feasible.

---

## 2. Repository Contents and File Map

| File | Title | Type | Date | Status |
|------|-------|------|------|--------|
| `c0f315e2-829b-40dc-b069-0b21bedc015c.pdf` | **DualExpert-BitYOLO26**: Restoring Representational Capacity in Extreme Quantization via Ternary Reparameterization | Research Paper (Paper 1 in series) | Feb 2026 | ⚠️ Unvalidated |
| `f0d4ebf0-a0b4-4b62-9f54-527d87e3e6bb.pdf` | **DualExpert-BitYOLO26** *(duplicate)* | Research Paper (identical to above) | Feb 2026 | ⚠️ Unvalidated |
| `75164103-a63b-451f-9e00-625d47c02dea.pdf` | **NeuroBit-SCS**: Decoupling Texture and Shape in 1.58-Bit YOLO Architectures via Static Channel Splitting | Research Paper (Paper 2 in series) | Feb 21, 2026 | ⚠️ Unvalidated |
| `8b2d7b18-be2a-42ab-908d-62720fd29691.pdf` | **ResiBit-YOLO**: Residual-Precision 1.58-bit Object Detection with Dual-Stream Ternary Experts | Research Paper (Paper 3 in series) | Feb 2026 | ⚠️ Unvalidated |
| `8a1d8042-3b95-497f-9dda-fd0c9b9feb5b.pdf` | **Unified Bit-Intelligence**: Resolving the Capacity-Stability Dilemma in Sub-2-Bit Edge Detectors | Research Manifesto (Paper 4 in series) | Feb 24, 2026 | ⚠️ Unvalidated |
| `manuscript.pdf` | **Spectral-ResiBit YOLO**: A Frequency-Aware, Latent-Shared 1.58-bit Detector for Edge Intelligence | IEEE TPAMI Preprint (Paper 5 — final) | 2026 | ⚠️ Unvalidated |
| `0db44a32-c492-48b6-a239-cb8528f337f0.pdf` | **Spectral-ResiBit YOLO** *(duplicate of manuscript.pdf)* | IEEE TPAMI Preprint (identical to above) | 2026 | ⚠️ Unvalidated |
| `Document.pdf` | **Real-Time Road Anomaly Detection System** (Bharat AI SoC Challenge, IIITDM Kurnool) | Application Document (separate project) | 2026 | ⚠️ Unvalidated |

> **Note on duplicates**: `c0f315e2` and `f0d4ebf0` contain identical content (DualExpert-BitYOLO26). Likewise, `manuscript.pdf` and `0db44a32` are the same Spectral-ResiBit YOLO manuscript. These may represent version history or distribution copies.

---

## 3. Research Progression: The Paper Series

The five primary research papers form a clearly structured **Problem → Investigation → Discovery → Solution → Unification** narrative, spanning roughly a two-week period in February 2026. All papers are produced under the "K-Dense Web" name.

```
[Paper 1] DualExpert-BitYOLO26         <- Initial architectural concept + Base-5 theory
       |
       v
[Paper 2] NeuroBit-SCS                 <- Adding channel splitting + spectral loss
       |
       v
[Paper 3] ResiBit-YOLO                 <- Forensic audit: failure confirmed -> INT8 fix proposed
       |
       v
[Paper 4] Unified Bit-Intelligence     <- Synthesis: LSM + Spectral + ResiBit in one manifesto
       |
       v
[Paper 5] Spectral-ResiBit YOLO        <- Final IEEE-format paper: empirical results on COCO
```

---

### 3.1 Paper 1 — DualExpert-BitYOLO26

**Full title**: *DualExpert-BitYOLO26: Restoring Representational Capacity in Extreme Quantisation via Ternary Reparameterisation for Edge-Efficient Real-Time Object Detection*

**Files**: `c0f315e2-829b-40dc-b069-0b21bedc015c.pdf`, `f0d4ebf0-a0b4-4b62-9f54-527d87e3e6bb.pdf`

#### What it proposes

This paper introduces the foundational concept of the **Dual-Expert Base-5 architecture**. The central insight is:

- A single ternary weight matrix W in {-1, 0, +1} encodes only log2(3) ≈ 1.585 bits/parameter.
- If you train **two independent ternary experts** (W_A and W_B) and **fuse them by element-wise addition**, the resulting quinary weight W_fused = W_A + W_B in {-2, -1, 0, +1, +2} encodes log2(5) ≈ 2.322 bits/parameter.
- This recovers ~47% additional representational capacity (1.585 → 2.322 bits) without incurring dual memory bandwidth at inference time, because the fused weight is a single matrix stored in base-5 encoding.

This approach is positioned within the framework of **Mixture of Quantization Experts (MoQE)**, drawing inspiration from sparse Mixture-of-Experts (Shazeer et al., 2017) but applying the concept to weight quantization rather than dynamic routing.

#### Key technical details

- **Base model**: YOLOv26n — a DFL-free, NMS-free YOLO variant optimized for edge integer arithmetic
- **Layer replacement**: 113 of 126 convolutional layers replaced with `Base5_YOLO_Conv` modules (13 high-sensitivity layers preserved in FP32 via MSE sensitivity analysis)
- **Compression claim**: 13.78x weight storage reduction vs. FP32 — derived from 32 / log2(5) = 32 / 2.322 = 13.78
- **Training protocol**: QAT with warmup-based mitigation to avoid gradient divergence with the MuSGD optimizer
- **Hardware deployment**: CUDA Pack-Store-Load-Unpack-Compute (PSLUC) kernel for base-3 to base-2 transcoding

#### Two formal hypotheses tested

| Hypothesis | Claim | Outcome |
|-----------|-------|---------|
| **H1 (Efficiency)** | 13.78x weight compression achievable | Confirmed — compression is information-theoretically valid |
| **H2 (Stability)** | 5-epoch QAT sufficient to adapt YOLO26n | Falsified — complete mAP collapse (mAP50: 0.6437 → 0.0000) |

#### Critical weakness identified

The MuSGD optimizer (a variant of the Muon optimizer using Newton-Schulz momentum orthogonalization) causes **gradient divergence** when combined with the discontinuous Straight-Through Estimator (STE) gradient. The paper prescribes a warmup protocol but does not experimentally validate it in this document.

---

### 3.2 Paper 2 — NeuroBit-SCS

**Full title**: *NeuroBit-SCS: Decoupling Texture and Shape in 1.58-Bit YOLO Architectures via Static Channel Splitting*

**File**: `75164103-a63b-451f-9e00-625d47c02dea.pdf`

#### What it proposes

This paper introduces two key components that become central to all subsequent work:

**1. Static Channel Splitting (SCS)**

Rather than using dynamic routing (as in standard MoE), the input feature map X (C x H x W) is **statically partitioned** into two disjoint groups at the channel dimension:
- X_A (first C/2 channels) → processed by the "Texture Expert" (high-frequency specialist)
- X_B (last C/2 channels) → processed by the "Shape Expert" (low-frequency specialist)

This eliminates the routing overhead of dynamic MoE designs while preserving the dual-expert capacity benefit. SCS also enables parallel execution on ARM NEON SIMD hardware without synchronization barriers.

**2. Spectral Orthogonality Loss via FFT**

To prevent the two experts from learning **redundant features** (a key risk when partitioning channels arbitrarily), a frequency-domain penalty is applied:

```
L_spec = sum_{u,v} |FFT(Y_A)_{u,v} * conj(FFT(Y_B)_{u,v})|
```

where FFT denotes the 2D Discrete Fourier Transform. By minimizing the spectral overlap between expert outputs, the loss **forces Expert A to dominate high-frequency rings** (textural edges, fine spatial details) and **Expert B to dominate low-frequency components** (global shape, geometry). This is analogous to the frequency decomposition used in wavelets and multi-scale signal processing.

**3. INT8 Residual Anchoring (RPA)**

A lightweight INT8-precision bypass path runs parallel to the dual ternary experts. This "signal anchor" preserves high-fidelity spatial information that would otherwise be destroyed by ternary rounding, and is critical for small-object bounding box regression.

#### Key findings

| Model | Weights | mAP50-95 | Inference Speed (Pi 5) |
|-------|---------|-----------|------------------------|
| YOLO26-FP16 | FP16 | 37.4% | 1.0x |
| YOLO26-INT8 | INT8 | 36.8% | 1.8x |
| BitNet-YOLO | 1.58-bit | 21.2% | 2.1x |
| **NeuroBit-SCS** | **Base-5** | **36.2%** | **2.4x** |

The paper also identifies and resolves the **"Packing Fallacy"**: naive bit-packing of ternary weights into 2-bit values (fitting 4 weights per byte) incurs a 20–30% throughput penalty on ARM NEON due to masking/shifting instructions. NeuroBit-SCS instead uses **4-bit nibble alignment** with SIMD-optimized channel-wise interleaving, eliminating this overhead.

> ⚠️ **Unvalidated note**: The 98% FP16 accuracy and 2.4x speedup claims are stated without a reproducible experimental pipeline. No code, dataset split, or hardware benchmark protocol is provided to validate these figures.

---

### 3.3 Paper 3 — ResiBit-YOLO

**Full title**: *ResiBit-YOLO: Residual-Precision 1.58-bit Object Detection with Dual-Stream Ternary Experts — A Problem-Solution Investigation of Gradient Manifold Collapse in Extreme Quantization for Edge Vision*

**File**: `8b2d7b18-be2a-42ab-908d-62720fd29691.pdf`

#### Structure

This is the most experimental and forensically detailed paper in the series. It is structured as a **Problem-Solution Narrative** in three parts:

- **Part I (Investigation)**: Full description of the Dual-Expert Base-5 experiment on YOLOv26n
- **Part II (Discovery)**: Quantitative results and root-cause failure analysis
- **Part III (Solution)**: The ResiBit-YOLO architecture specification

#### Experimental Setup (Part I)

- **Base model**: YOLOv26n pre-trained (mAP50 = 0.6437, mAP50-95 = 0.4788)
- **Calibration dataset**: COCO128
- **Layer replacement**: 113/126 layers → `Base5_YOLO_Conv` (sensitivity analysis: top 10% MSE + out_channels==4 layers preserved)
- **QAT**: 5 epochs, AdamW optimizer, batch 4, image size 640, CPU execution, **no warmup** (quantization active from epoch 1)
- **Ternary quantization function**:
  ```
  W_hat = clip(round(W / alpha), -1, 1)    where alpha = mean(|W_ij|)
  ```
- **Gradient backpropagation via STE**: dW_hat/dW ≈ 1 if |W| <= 1, else 0

#### Model surgery details

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total convolutional layers | 126 | Full YOLOv26n backbone |
| Average quantization MSE | 0.005639 | Low mean sensitivity |
| Maximum quantization MSE | 0.079618 | High outlier sensitivity |
| MSE exclusion threshold | 0.007575 | Top 10% by MSE |
| Layers excluded (MSE) | 13 | High-sensitivity preserved |
| Layers excluded (ch=4) | 6 | Narrow-channel preserved |
| Layers replaced (Base-5) | 113 | 89.7% replacement rate |
| Layers preserved (FP32) | 13 | 10.3% of layers |

#### Critical Results (Part II)

**H1 Confirmed — Efficiency**:

| Representation | Bits/weight | Storage (MB) | Compression |
|----------------|-------------|--------------|-------------|
| FP32 (baseline) | 32.000 | 9.714 | 1.00x |
| INT8 | 8.000 | 2.429 | 4.00x |
| Single ternary (1.58-bit) | 1.585 | 0.481 | 20.19x |
| **Dual Base-5 (fused)** | **2.322** | **0.705** | **13.78x** |

**H2 Falsified — Stability (Complete Failure)**:

| Model | mAP50 | mAP50-95 |
|-------|-------|-----------|
| FP32 Baseline | 0.6437 | 0.4788 |
| QAT Fine-tuned (5 epochs) | **0.0000** | **0.0000** |

Dead weight analysis: **59.3% of Expert A** and **60.5% of Expert B** weights converged to zero within 5 epochs — the "Dead Neuron" state.

#### Root Cause Analysis: The Muon Trap

The paper identifies a **two-factor failure mechanism**:

**Factor 1 — STE gradient noise**: The STE approximates the gradient through the non-differentiable ternary rounding function as an identity (i.e., "pretend the quantizer doesn't exist"). This introduces systematic **gradient mismatch** between the true loss surface and the estimated gradient. The larger the weight magnitudes, the larger the discrepancy.

**Factor 2 — Newton-Schulz momentum orthogonalization (Muon optimizer)**: The Muon optimizer iteratively applies:
```
M_{k+1} = (3/2) * M_k - (1/2) * M_k * (M_k^T * M_k)
```
to maintain orthogonality in the momentum matrix. In FP32 training, this prevents gradient collapse by keeping weight updates diverse. However, when the input gradients are noisy (from STE), Newton-Schulz **amplifies** the noise by projecting it into a high-energy singular subspace that is incoherent with the feature hierarchy. The result: every gradient step drives weights toward zero — the global attractor of the AbsMean quantization function — until the network is completely dead.

The paper terms this the **"Muon Trap"**: Muon's stabilizing mechanism for FP32 becomes a catastrophic amplifier under ternary QAT.

#### The ResiBit Solution (Part III)

The ResiBit block adds a **lightweight INT8 Residual Highway** parallel to the dual ternary streams in every replaced convolutional block:

```
Y = GroupMix( W_A * X  +  W_B * X  +  W_INT8 * X )
              ^ternary    ^ternary    ^INT8 precision anchor
```

**Three-phase training protocol**:
1. **Phase 1 (FP32 warmup)**: Train the entire network in full precision for N epochs to establish a stable feature hierarchy
2. **Phase 2 (Gradual QAT)**: Apply ternary quantization to the backbone layers only, with the INT8 highway carrying the gradient
3. **Phase 3 (Full deployment)**: Freeze the INT8 highway; fuse W_A + W_B into the Base-5 quinary matrix for inference

**Why the INT8 highway works**: During backpropagation, the INT8 highway provides a **continuous, high-precision gradient path** that is not subjected to STE noise. This prevents the Muon optimizer from amplifying only the noisy STE gradients; instead, the optimizer receives a blend of noisy ternary gradients and precise INT8 gradients, maintaining manifold stability.

> ⚠️ **Unvalidated note**: The three-phase training protocol and ResiBit architecture are **proposed but not experimentally validated** within this paper. The paper ends at the architectural specification stage.

---

### 3.4 Paper 4 — Unified Bit-Intelligence

**Full title**: *Unified Bit-Intelligence: Resolving the Capacity-Stability Dilemma in Sub-2-Bit Edge Detectors*

**File**: `8a1d8042-3b95-497f-9dda-fd0c9b9feb5b.pdf`

#### Role in the Series

This is the **synthesis and manifesto paper**. It integrates all prior building blocks — Base-5 fusion (Paper 1), SCS + Spectral Loss (Paper 2), INT8 Residual Highway (Paper 3) — into the **Universal Spectral-ResiBit (USR) Cell**, which becomes the core architectural unit of the final Spectral-ResiBit YOLO paper.

#### The Latent-Shared Manifold (LSM)

The LSM is this paper's key new contribution over Paper 3. It resolves a subtle **incompatibility between SCS and Base-5 Fusion**:

- **SCS** partitions input channels into disjoint groups: Expert A sees only X_A, Expert B sees only X_B. They cannot share information.
- **Base-5 Fusion** requires that W_A and W_B process the **same input** to produce a meaningful fused weight W_A + W_B.

If experts process different inputs (SCS), the algebraic fusion W_fused = W_A + W_B cannot be interpreted as a convolution over a single shared input — the two outputs represent information from **different parts** of the feature space and cannot be simply added.

The LSM resolves this by introducing a **three-way partition**:
```
C channels --> [X_A (private)] + [X_B (private)] + [X_shared (shared)]
```

The formulation is:
```
Y = Mix( W_A(X_A union X_shared)  +  W_B(X_B union X_shared) )
```

In the shared subspace C_shared, both W_A and W_B process the same input. Their weights on this subspace can be fused as W_fused = W_{A,shared} + W_{B,shared} in {-2, -1, 0, +1, +2}, restoring the Base-5 efficiency advantage while allowing private channels to carry expert-specialized information. This is analogous to the "shared parameters" concept in Multi-Task Learning, but applied at the quantization level.

#### Empirical Benchmarks (First quantitative claims in the series)

| Metric | FP32 Baseline | Naive Ternary | USR-YOLO |
|--------|--------------|---------------|---------|
| mAP50 (COCO) | 0.6437 | 0.0000 | **0.6051** |
| Weight Size (MB) | 7.6 | 0.38 | 0.55 |
| Compression | 1.0x | 20.2x | **13.78x** |
| Latency (Pi 5) | 42ms | 12ms | **18ms** |

**Hardware-Software Co-Design: PSLUC on ARM NEON**

The paper presents a specific bit-serial execution strategy for ARM NEON:
- **Base-3 to Base-5 packing**: 5 ternary weights packed into one 8-bit byte (3^5 = 243 < 2^8 = 256), achieving **1.6 effective bits/weight** in storage
- **Bit-serial computation**: W = +1 → `vadd.i8`; W = -1 → `vsub.i8`; W = 0 → skip. No multiplier instructions used.
- Claimed: **2.4x speedup over optimized INT8 kernels** on Raspberry Pi 5

> ⚠️ **Unvalidated note**: The 0.6051 mAP50 and 2.4x speedup figures are presented without a reproducible experimental protocol.

---

### 3.5 Paper 5 — Spectral-ResiBit YOLO (Final Publication)

**Full title**: *Spectral-ResiBit YOLO: A Frequency-Aware, Latent-Shared 1.58-bit Detector for Edge Intelligence*

**Files**: `manuscript.pdf`, `0db44a32-c492-48b6-a239-cb8528f337f0.pdf`

**Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) Preprint

#### Role in the Series

This is the **final, publication-format version** of the complete architecture. It synthesizes all prior work into a single, coherent IEEE-format paper with a full experimental evaluation section.

#### Architecture: The Three Pillars

**Pillar 1 — The USR Cell (Latent-Shared Manifold)**

The Universal Spectral-ResiBit Cell integrates:
- Dual-stream ternary experts (W_A, W_B in {-1, 0, +1})
- Shared channel subspace for Base-5 fusion compatibility
- Grouped-channel shuffle (Mix) for feature aggregation

```
Y = Mix( W_A(X_{C_A} union X_{C_S})  +  W_B(X_{C_B} union X_{C_S}) )
```

where the shared subspace C_S enables W_shared = W_{A,S} + W_{B,S} in {-2, -1, 0, +1, +2}.

**Pillar 2 — FFT-based Spectral Orthogonality**

```
L_spec = lambda * sum_{u,v} |FFT(W_A(X))_{u,v} * conj(FFT(W_B(X))_{u,v})|
```

This frequency-domain penalty forces:
- **Expert A** → Texture Specialist (high-frequency rings: edges, fine details)
- **Expert B** → Shape Specialist (low-frequency geometric bases: object boundaries, global structure)

**Pillar 3 — INT8 Residual Highways (Residual-Precision Anchoring, RPA)**

Every ternary block is bypassed by a lightweight INT8 residual path (the "Precision Funnel"). During backpropagation, this highway provides a **continuous gradient mantissa** that prevents the Newton-Schulz routine from amplifying STE noise into a singular incoherent subspace.

#### Forensic Audit Results (Reproduced from Paper 3)

- FP32 baseline mAP50: **0.6437**
- BitNet-style ternary QAT (no fix): mAP50 collapsed to **0.0000** within 5 epochs
- Dead Neuron ratio: **>60%** of weights converged to zero

#### Performance Recovery on COCO128

| Model | Precision | mAP50 | Memory (MB) |
|-------|-----------|-------|-------------|
| YOLO26-Baseline | FP32 | 0.6437 | 21.4 |
| BitNet-YOLO | 1.58-bit | 0.0000 | 1.1 |
| ResiBit-YOLO (v1) | Mixed | 0.5120 | 6.8 |
| **Spectral-ResiBit (USR)** | **USR Cell** | **0.6051** | **1.6** |

**Ablation study — contribution of each component**:

| RPA | USR | Spec-Loss | mAP50 | Recovery |
|-----|-----|-----------|-------|---------|
| — | — | — | 0.0000 | 0% |
| Yes | — | — | 0.4812 | 75% |
| Yes | Yes | — | 0.5723 | 89% |
| Yes | Yes | Yes | **0.6051** | **94%** |

Key insight from ablation: **RPA alone recovers 75% of accuracy loss** — confirming it is the most critical component. USR Cell adds 14 percentage points, and Spectral Loss adds the final 5 percentage points.

#### Hardware Manifesto: PSLUC on ARM NEON

The paper introduces the **Pack-Store-Load-Unpack-Compute (PSLUC)** paradigm:
- Avoids the "Packing Fallacy": 4-ternary-per-byte (2-bit packing) requires shift/mask operations costing ~20% throughput on Raspberry Pi 5
- Instead: **4-bit nibble alignment** with direct 128-bit NEON register loads (zero unpacking overhead)
- Bit-serial execution: W=+1 → `vadd.i8`; W=-1 → `vsub.i8`; W=0 → no-op
- Claimed energy reduction: **3.5x** vs. floating-point on ARM edge nodes

> ⚠️ **Unvalidated note**: The PSLUC energy and throughput claims lack a physical measurement methodology. No power-meter measurements are referenced. Hardware speedup claims are theoretical.

---

### 3.6 Document 6 — Real-Time Road Anomaly Detection System

**Full title**: *Real-Time Road Anomaly Detection System — Bharat AI SoC Challenge, Problem Statement 3*

**File**: `Document.pdf`

**Institution**: Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kurnool

**Team**: Akula Gayatri, Pithana Omkara Sri Harsha, Kanamarlapudi Jyoshikaa (Mentor: Dr. Eswaramoorthy KV)

#### What this document is

This is **separate from the main K-Dense Web paper series** and represents a **student competition entry** for the Bharat AI SoC (System-on-Chip) Challenge, hosted in partnership with ARM. It is included in this repository as a **practical application case study** showing how current-generation YOLO models are deployed on Raspberry Pi edge hardware — which is directly relevant to the goal of the primary research series (demonstrating that quantized YOLO detectors work at the extreme edge).

#### Technical Summary

The system implements **dual-model edge inference** on a Raspberry Pi 4:

- **Pothole Detection Pipeline**: Custom-trained YOLOv11n model exported to ONNX format for inference via ONNX Runtime. Detects potholes and calculates approximate diameter from bounding box pixel dimensions.
- **Obstacle Detection Pipeline**: YOLOv11n in PyTorch format trained on the COCO dataset. Detects and classifies obstacles (humans, animals, vehicles).
- **Motion Tracking**: Vehicle motion classified across multiple frames (stationary vs. moving) via centroid tracking.
- **Logging**: CSV output with timestamped detections, bounding box coordinates, confidence scores, and pothole diameter estimates.

**Target Performance**: >=5 FPS on Raspberry Pi 4 (ARM Cortex-A72) with <200ms end-to-end latency

**Processing pipeline parameters**:
- Input: 640x480 at 30fps (dashcam)
- Frame skip: process every 6th frame (targeting 5 FPS throughput)
- Model input: resized/normalized to 640x640
- NMS IoU threshold: 0.4
- Memory budget: <1 GB

#### Relationship to the Main Paper Series

This document demonstrates the **practical deployment context** for the ternary quantization research:
- It confirms that even modern FP16/INT8 YOLO models require significant optimization tricks to achieve 5 FPS on Raspberry Pi 4
- The challenge of running two simultaneous YOLO models at real-time speeds on ARM hardware directly motivates the need for 13.78x memory compression and multiplication-free inference
- It serves as a real-world baseline: if the Spectral-ResiBit YOLO architecture were validated, it could replace both YOLOv11n models here with a single ternary-weight model using a fraction of the memory and power

> ⚠️ **Unvalidated note**: Performance figures (FPS, accuracy rates) are self-reported by the competition team without independent verification.

---

## 4. Core Problem: The Capacity-Stability Dilemma

The central research challenge is formally named the **"Capacity-Stability Dilemma"** in the series. Understanding it requires understanding why ternary quantization behaves differently in LLMs vs. CNNs.

### Why Ternary Quantization Works in LLMs

In a large transformer (e.g., 70B parameters with 8,192 channels/layer):
- Each layer has **thousands of weights per output feature**, so individual weight imprecision averages out through the collective vote of millions of weights
- The model has **enormous redundancy**: adjacent weights carry overlapping information, so losing a few to the zero attractor does not matter
- Quantization noise is a small perturbation on a robust, high-dimensional manifold

### Why Ternary Quantization Fails in Thin CNNs

In YOLOv26n (2.4M parameters, 32–256 channels/layer):
- Each layer has only **32–256 weights per output feature** — there is almost no redundancy
- The model encodes **dense spatial information**: every weight must faithfully represent a specific spatial frequency component
- Quantization noise is not a small perturbation — it is a **structural shattering** of the representation

Formally, the series defines the dilemma as:
- **Capacity problem**: Ternary weights cannot represent the high-frequency spatial gradients required for bounding box regression in dense prediction
- **Stability problem**: The Muon optimizer + STE interaction drives >60% of weights to zero during QAT, destroying detection capability entirely

### The Solution Philosophy

Rather than accepting the dilemma as fundamental, the papers argue that it can be dissolved through **manifold anchoring**: if the gradient manifold is prevented from shattering (via INT8 residual highways), and if the representational capacity of the ternary subspace is doubled (via dual-expert Base-5 fusion), then the ternary paradigm can be made to work for thin CNNs.

The BitNet b1.58 Reloaded analysis (Nielsen et al., 2024) provides independent external support for this framing: it explicitly noted that 1.58-bit performance degradation increases substantially for models below approximately 3 billion parameters, with smaller models experiencing 5–10% degradation on NLP tasks. For a 2.4M-parameter object detector, the degradation is not 5–10% — it is, as the experiment confirms, complete.

---

## 5. Key Technical Concepts Explained

### 5.1 Ternary Quantization and 1.58 Bits

A **ternary weight** W in {-1, 0, +1} requires log2(3) ≈ **1.585 bits** of information to specify (since there are 3 possible values). The "1.58-bit" naming convention comes directly from this information-theoretic quantity.

The **quantization function** used throughout the series is:
```
W_hat = clip( round(W / alpha), -1, 1 )
alpha = mean(|W|)   (AbsMean scaling)
```

This per-tensor absolute-mean scaling ensures that the threshold between 0 and ±1 adapts to the weight distribution, minimizing the fraction of weights forced to zero.

### 5.2 The Straight-Through Estimator (STE)

Since the `round()` function is non-differentiable, standard backpropagation cannot be applied directly. The **STE** approximates the gradient as:
```
dW_hat/dW ≈ 1  if |W| <= 1,  else 0
```

This is the "pretend the quantizer doesn't exist" trick. It introduces a gradient error of magnitude proportional to the distance between the true weight and its quantized value. In thin networks with small redundancy, this error accumulates dangerously.

### 5.3 The Muon Optimizer

The **Muon optimizer** (Jordan et al., 2024) applies Newton-Schulz iterations to the momentum matrix to maintain approximate orthogonality:
```
M_{k+1} = (3/2) * M_k - (1/2) * M_k * (M_k^T * M_k)
```

In FP32 training, this diversifies gradient directions and prevents feature collapse. Under ternary QAT, however, the noisy STE gradients flood the top singular subspace of M_k, and the Newton-Schulz routine amplifies this noise by projecting all gradient energy into a single incoherent direction. The series calls this the **"Muon Trap"**.

### 5.4 Base-5 Reparameterization

If W_A, W_B in {-1, 0, +1}, then their element-wise sum:
```
W_fused = W_A + W_B  in  {-2, -1, 0, +1, +2}
```
is a **quinary (base-5)** weight. By the linearity of convolution:
```
W_A * X + W_B * X = (W_A + W_B) * X = W_fused * X
```
So at **inference time**, both expert branches are **fused into a single pass** — there is no dual memory load. The fused weight maps to shift-and-add instructions:
- W = +2 → left shift by 1 (`vshl.i8`)
- W = +1 → integer add (`vadd.i8`)
- W = 0 → skip (implicit sparsity, no operation)
- W = -1 → integer subtract (`vsub.i8`)
- W = -2 → negate then shift

### 5.5 Spectral Orthogonality via FFT

The **Spectral Orthogonality Loss** leverages the fact that the 2D Discrete Fourier Transform (2D-DFT) decomposes any feature map into its constituent spatial frequencies. By penalizing the **element-wise product** of the frequency spectra of both expert outputs:
```
L_spec = lambda * sum_{u,v} |FFT(Y_A)_{u,v} * conj(FFT(Y_B)_{u,v})|
```
the loss ensures that when Expert A dominates a frequency bin (u, v), Expert B is suppressed there, and vice versa. This is mathematically equivalent to maximizing the orthogonality of the two experts' output representations in the frequency domain — a form of **mutual information minimization** in the Fourier basis.

### 5.6 Gradient Manifold Shattering

The "gradient manifold" refers to the high-dimensional surface in weight space along which training updates travel. In healthy FP32 training, this manifold is smooth and well-structured, with each layer encoding coherent spatial features. Under abrupt ternary QAT without stabilization, the manifold **shatters**: gradients become incoherent, pointing in contradictory directions at each update step, until the optimizer finds the trivial minimum (all weights = 0, which satisfies the ternary constraint W_hat = 0 whenever |W| < alpha).

---

## 6. Proposed Architectural Solutions

### Architecture Evolution Across the Series

```
DualExpert-BitYOLO26 (Paper 1):
  X -> [W_A * X] + [W_B * X]  -> W_fused * X     (Base-5 fusion, unstable)

NeuroBit-SCS (Paper 2):
  X -> {X_A -> W_A * X_A, X_B -> W_B * X_B} + RPA     (SCS + residual anchor)
  + Spectral Orthogonality Loss penalizing FFT(Y_A) * FFT(Y_B)

ResiBit-YOLO (Paper 3):
  X -> [W_A * X] + [W_B * X] + [W_INT8 * X]  -> GroupMix -> Y
  (dual ternary + INT8 residual, trained in 3 phases)

Unified Bit-Intelligence / USR Cell (Paper 4 & 5):
  Partition X into [X_A (private)] + [X_B (private)] + [X_shared (shared)]
  Expert A processes (X_A union X_shared) with ternary weights W_A
  Expert B processes (X_B union X_shared) with ternary weights W_B
  Shared weights fused: W_shared = W_{A,shared} + W_{B,shared} (Base-5)
  + Spectral Orthogonality Loss (Expert A = high-freq, Expert B = low-freq)
  + INT8 Precision Funnel (RPA) in parallel
  + GroupMix aggregation
  -> Y
```

### The USR Cell: A Unified Solution

The final USR (Universal Spectral-ResiBit) Cell combines all three pillars:
- The LSM enables Base-5 compression in the shared subspace while allowing SCS-style specialization in private channels
- Spectral Orthogonality ensures experts specialize in different frequency ranges, preventing feature redundancy and maximizing effective bit depth
- The INT8 Residual Highway prevents the Muon Trap by providing a clean gradient path during QAT

The USR Cell is designed as a **drop-in replacement** for any standard convolutional block, requiring only modification of the weight tensors and the addition of the RPA path — no changes to the surrounding architecture.

---

## 7. Experimental Results Summary

> ⚠️ All results below are **self-reported and unvalidated**. No independent reproduction has been performed.

### Primary Benchmark (COCO128 — reported in Spectral-ResiBit YOLO / Unified Bit-Intelligence)

| Model | mAP50 | Compression | Memory (MB) | Latency (Pi 5) |
|-------|-------|-------------|-------------|----------------|
| YOLO26 FP32 baseline | 0.6437 | 1x | 21.4 | 42 ms |
| BitNet-YOLO (naive ternary) | 0.0000 | ~20x | 1.1 | 12 ms |
| ResiBit-YOLO v1 (RPA only) | 0.5120 | — | 6.8 | — |
| **Spectral-ResiBit YOLO (USR)** | **0.6051** | **13.78x** | **1.6** | **18 ms** |

### NeuroBit-SCS (COCO2017 Val — reported separately)

| Model | mAP50-95 | Speed (Pi 5) |
|-------|-----------|-------------|
| YOLO26-FP16 | 37.4% | 1.0x |
| YOLO26-INT8 | 36.8% | 1.8x |
| BitNet-YOLO | 21.2% | 2.1x |
| **NeuroBit-SCS** | **36.2%** | **2.4x** |

### Ablation Study (Spectral-ResiBit YOLO)

| RPA | USR Cell | Spectral Loss | mAP50 | % FP32 Recovered |
|-----|---------|---------------|-------|------------------|
| No | No | No | 0.0000 | 0% |
| Yes | No | No | 0.4812 | 75% |
| Yes | Yes | No | 0.5723 | 89% |
| Yes | Yes | Yes | **0.6051** | **94%** |

### Key Metrics Summary

- **Accuracy recovery**: 94% of FP32 mAP50 (0.6051 vs. 0.6437)
- **Weight compression**: 13.78x vs. FP32
- **Inference speedup**: 2.33x vs. FP32 latency (42ms → 18ms on Pi 5)
- **Dead neuron reduction**: From >60% (naive ternary) to operational model
- **Memory footprint**: 1.6 MB vs. 21.4 MB (FP32) — fits in L2 cache of most edge SoCs

---

## 8. Cross-Reference with Related and Foundational Works

### Foundational Papers (Directly Cited in the Series)

| Paper | Citation | Key Relevance |
|-------|----------|---------------|
| **BitNet b1.58** (Ma et al., 2024) | arXiv:2402.17764 | The direct origin of the 1.58-bit paradigm. Introduced ternary weights {-1,0,+1} with AbsMean quantization for LLMs. Demonstrated Pareto-optimal efficiency vs. accuracy. Every paper in this series takes BitNet b1.58 as its starting point and asks: "does this work for CNNs?" |
| **Muon Optimizer** (Jordan et al., 2024) | kellerjordan.github.io/posts/muon | The optimizer whose Newton-Schulz orthogonalization creates the "Muon Trap" under STE. Used as YOLO26's default optimizer (MuSGD variant). Central to the failure analysis in Papers 1, 3, 4, and 5. |
| **StableQAT** (Chen et al., 2026) | arXiv:2601.19320 | Addresses stable quantization-aware training at ultra-low bitwidths. Directly relevant to the STE gradient noise problem identified throughout the series. |
| **Sparse MoE** (Shazeer et al., 2017) | ICLR 2017 | Foundational Mixture-of-Experts paper with sparsely-gated routing. Motivates the dual-expert design in DualExpert-BitYOLO26. SCS is presented as a hardware-friendly static alternative to dynamic routing. |
| **Bi-Real Net** (Liu et al., 2018) | ECCV 2018 | Key precursor for binary CNNs with residual shortcuts. Introduced the concept of a "real-valued shortcut" to preserve information flow past binary layers — directly analogous to the INT8 Residual Highway (RPA) in this series. Achieved 56.4% top-1 accuracy on ImageNet with an 18-layer binary architecture vs. 44.1% for XNOR-Net. |
| **YOLO26** (Ultralytics, 2025) | arXiv:2509.25164 | The base detector used throughout. Key features exploited: DFL-free head (simpler loss, fewer parameters), NMS-free one-to-one assignment (integer-friendly), MuSGD optimizer, quantization-friendly architecture choices. |

### Related Works (Not Directly Cited but Highly Relevant)

| Paper / Work | Relevance to This Series |
|-------------|--------------------------|
| **BitNet** (Wang et al., 2023) | Original 1-bit LLM paper by Microsoft Research. Predecessor to BitNet b1.58. Used 1-bit {-1,+1} weights. Established the BitNet architecture that b1.58 extended to ternary. |
| **XNOR-Net** (Rastegari et al., 2016) | Foundational binary CNN paper. Pure 1-bit weights with channel-wise scaling. Only 44.1% ImageNet top-1 accuracy, highlighting the accuracy gap that Bi-Real Net and this series try to close. |
| **Ternary Weight Networks (TWN)** (Li et al., 2016) | Early ternary weight networks using {-Delta, 0, +Delta} with learned threshold Delta per layer. Predecessor to the AbsMean quantization used in BitNet b1.58. Showed ternary quantization could approach FP32 accuracy for AlexNet. |
| **Trained Ternary Quantization (TTQ)** (Zhu et al., 2017) | Introduced two separate positive/negative scaling factors for ternary weights (more flexible than TWN). Shows ternary CNNs can approach FP32 accuracy on ImageNet with sufficient model width — supporting the "thin model problem" argument. |
| **DoReFa-Net** (Zhou et al., 2016) | Quantizes weights, activations, and gradients simultaneously to low bits. Demonstrates that gradient quantization noise is the dominant challenge at ultra-low precision — directly relevant to the STE problem analyzed in this series. |
| **QMoE** (Frantar & Alistarh, 2023) | Sub-1-bit compression of trillion-parameter MoE models via token-wise routing with expert-differentiated quantization granularity. Directly related to the MoQE framing in DualExpert-BitYOLO26, but targets the much larger-model regime. |
| **LLM.int8()** (Dettmers et al., 2022) | INT8 inference for LLMs with mixed-precision for outlier channels (some channels kept in FP16). Parallels the INT8 residual highway concept: high-precision pathway for the most sensitive computations in the network. |
| **Post-Training Quantization Survey** (Gholami et al., 2022) | Comprehensive review of quantization methods. Confirms that QAT with careful warmup is essential below 4-bit precision — directly validating the three-phase training protocol proposed in ResiBit-YOLO. |
| **BitNet b1.58 Reloaded** (Nielsen et al., 2024) | Analysis noting 1.58-bit performance degradation increases substantially for models below ~3B parameters, with 5–10% degradation on NLP tasks. Provides external validation for the "thin model problem" that is the core challenge of this series. |
| **Bitnet.cpp** (Wang et al., 2025) | Practical deployment of BitNet inference achieving 2–4x CPU speedup on ARM/x86 vs. llama.cpp. Validates that the PSLUC paradigm proposed in the series can be realized in practice. |
| **MobileNet** (Howard et al., 2017) | Depthwise separable convolutions for efficient CNNs. Conceptually parallel to SCS: factorizes convolution operations to reduce computation. The channel-splitting intuition in SCS echoes MobileNet's spatial/channel factorization. |
| **Automated Fine-Grained MoE Quantization** (Anonymous, 2025) | Dual-granularity bit allocation for MoE models: outlier channels get higher precision, uniform channels get lower precision. Achieves <1% accuracy loss on Mixtral-8x7B at 3–4-bit. Related to the mixed-precision (ternary + INT8 residual) strategy in ResiBit-YOLO. |

### Research Positioning Diagram

```
BINARY CNNs (XNOR-Net, 2016)
  --> Accuracy gap too large for dense prediction
  --> Bi-Real Net (ECCV 2018): real-valued residual alongside binary computation
        ^
        | analogous design principle
        v
INT8 Residual Highway (RPA) in ResiBit-YOLO / Spectral-ResiBit YOLO

TERNARY LLMs
  --> BitNet (1-bit, 2023)
  --> BitNet b1.58 (ternary, 2024) -- confirmed Pareto-optimal for LLMs
        ^
        | this series asks: apply to CNNs?
        v
  DualExpert-BitYOLO26 -> NeuroBit-SCS -> ResiBit-YOLO -> Spectral-ResiBit YOLO

MIXTURE OF EXPERTS (Shazeer et al., 2017)
  --> Dynamic routing MoE (high accuracy, high routing overhead)
  --> Static Channel Splitting (SCS) = zero-overhead static MoE
  --> Mixture of Quantization Experts (MoQE) framing in this series

QAT STABILITY RESEARCH
  --> DoReFa-Net: gradient quantization is the bottleneck
  --> StableQAT: targeted solutions for ultra-low-bit QAT
  --> Muon Optimizer + STE = Muon Trap (identified in this series)
        |
        v
  INT8 residual highway as the Muon Trap mitigation strategy
```

---

## 9. Critical Analysis and Open Questions

### Strengths of the Research

1. **Clear problem identification**: The Muon Trap failure mode is a genuine and previously undocumented interaction between a modern optimizer and ternary QAT. The forensic approach (running the failed experiment first, then analyzing the dead weights) is methodologically sound and honest about failures.

2. **Information-theoretic grounding**: The 13.78x compression claim and the Base-5 precision analysis are mathematically rigorous. These do not depend on experimental outcomes and can be independently verified.

3. **Modular architectural evolution**: Each paper adds exactly one component and justifies it against the prior failure mode. The ablation study in Spectral-ResiBit YOLO cleanly attributes accuracy recovery to individual components with clear marginal contributions.

4. **Hardware specificity**: The PSLUC ARM NEON analysis goes beyond typical "X faster" claims by identifying specific SIMD instruction-level execution for each weight value. The Packing Fallacy analysis is a genuine hardware engineering insight with a concrete alternative.

5. **Bi-Real Net connection**: Explicitly connecting the INT8 highway to Bi-Real Net's real-valued residual concept situates this work correctly in the literature and strengthens the theoretical grounding with an established precedent.

6. **Transparent failure**: ResiBit-YOLO is unusual in the literature for providing a complete forensic account of an experiment that **failed**, including dead-weight percentage distributions per expert. This level of negative-result transparency is valuable for the field.

### Weaknesses and Open Questions

1. **COCO128 is not a standard benchmark**: COCO128 contains only 128 training images. Standard YOLO benchmarking uses the full COCO val2017 (5,000 images). Results on COCO128 are not comparable to published YOLO benchmark numbers and may not generalize to the full distribution.

2. **No code or weights released**: No PyTorch implementation, training script, dataset splits, or model weights are provided. The three-phase training protocol is described but cannot be reproduced from the papers alone. This makes independent validation impossible.

3. **NeuroBit-SCS vs. Spectral-ResiBit inconsistency**: NeuroBit-SCS claims 98% of FP16 accuracy and 2.4x speedup on COCO2017 Val, while Spectral-ResiBit shows 94% of FP32 mAP and 2.33x on COCO128. These use different datasets (COCO2017 Val vs. COCO128) and different baselines (FP16 vs. FP32), making cross-paper comparison impossible and masking potential inconsistencies in the reported numbers.

4. **Missing INT8 baseline in the main comparison table**: No INT8 QAT YOLO baseline is provided alongside the Spectral-ResiBit results. INT8 is the practical industry-standard comparison point, and the omission makes it hard to evaluate whether 13.78x compression is worth the accuracy trade-off relative to simply using INT8 (which achieves 4x compression with near-lossless accuracy).

5. **LSM partition ratio unspecified**: What fraction of channels should be in the shared subspace vs. private? This hyperparameter likely dominates the accuracy-compression tradeoff, but no sensitivity analysis is presented.

6. **Activation quantization is unaddressed**: BitNet b1.58 quantizes activations to INT8 alongside ternary weights. The papers focus exclusively on **weight quantization** and appear to leave activations at full precision. True multiplication-free inference requires both weight and activation quantization — without activation quantization, the claimed arithmetic savings may not fully materialize in practice.

7. **Energy consumption is asserted, not measured**: The 3.5x energy reduction on ARM is a theoretical claim based on eliminating multiplications, not a physical measurement. Power consumption is affected by many factors (memory bandwidth, cache behavior, clock gating, pipeline stalls) that are not modeled. ARM NEON vadd/vsub instructions have non-trivial power costs that may not match the theoretical model.

8. **Spectral separation maintenance during training is unverified**: The spectral orthogonality loss is a training-time regularizer. There is no experiment showing how the spectral separation coefficient evolves during training, whether it saturates, or whether the experts converge to the claimed frequency specialization in practice.

9. **K-Dense Web generation flags**: The explicit "Generated using K-Dense Web (k-dense.ai)" footer on multiple manuscripts raises questions about the original authorship and verification of experimental data. It is unclear whether the described experiments were physically executed or whether some results were generated synthetically.

10. **No ablation on phase durations**: The three-phase training protocol (warmup → QAT → freeze) is proposed without any experiment showing the sensitivity of final accuracy to phase lengths. How many warmup epochs are needed? When should quantization be gradually introduced?

---

## 10. Unvalidated Research Disclaimer

> ⚠️ **All files in this repository are UNVALIDATED.**

The research proposals, manuscripts, preprints, and application documents in this repository have **not** been:
- Peer-reviewed by independent experts in the field
- Experimentally verified by any third party
- Validated against standard benchmarks using publicly reproducible code
- Published in or accepted by any peer-reviewed venue

### Specific Concerns

- All experimental results (mAP scores, speedup ratios, compression factors, energy figures) are **self-reported** by the authors
- Multiple documents are explicitly marked **"Generated using K-Dense Web (k-dense.ai)"**, indicating AI-assisted or AI-generated content — the authenticity of experimental data should be independently verified before reliance
- No code, training scripts, dataset splits, model weights, or hardware measurement logs are released to enable reproduction
- Benchmark conditions (COCO128 vs. COCO val2017, FP16 vs. FP32 baselines, hardware measurement methodology) are inconsistent across papers
- Quantitative claims for nominally the same architecture produce different numbers in different papers without explanation

### Guidance for Readers

- Use these materials for **background reading and idea generation only**
- **Do not cite** these results as established findings in derivative work
- **Do not make engineering or deployment decisions** based on the accuracy or performance numbers herein without independent experimental validation
- Treat all architectural proposals as **preliminary hypotheses** requiring rigorous experimental confirmation
- The mathematical analyses (Base-5 compression derivation, spectral orthogonality formulation, PSLUC instruction mapping) can be independently verified and are the most reliably informative parts of these documents

---

## 11. Empirical Validation Plan

The manuscripts here remain speculative until they are backed by reproducible experiments. The following protocol defines a minimal, evidence-focused pathway to validate (or falsify) the Spectral-ResiBit YOLO claims through empirical testing.

### 11.1 Experimental Goals
- **Accuracy**: Demonstrate that Base-5 dual experts + INT8 residual highways + spectral orthogonality recover mAP (mean Average Precision) within ±1 absolute percentage point (e.g., 50% → 49–51%) of the FP32 baseline on COCO val2017.
- **Efficiency**: Show ≥3× end-to-end latency speedup (or proportional energy-per-frame reduction) on Raspberry Pi 5/4 versus FP32, with multiplication-free execution verified in operator profiling.
- **Stability**: Confirm that spectral separation and sparsity metrics converge consistently across seeds.

### 11.2 Reproducible Setup
- **Codebase**: Implement Spectral-ResiBit YOLO atop a public YOLO variant (e.g., YOLOv8n or YOLOv5n) in PyTorch 2.x with quantization-aware training; publish all training/inference scripts and configs.
- **Datasets**: COCO 2017 train/val with exact split hashes; document any synthetic augmentations used.
- **Hardware**:
  - Training: 1× NVIDIA 3090/A100-class GPU (record driver/CUDA/cuDNN versions).
  - Edge: Raspberry Pi 5 (preferred) or Pi 4 with NEON; optional Jetson Orin Nano for comparison.
- **Environment**: Pin Python, PyTorch, ONNX Runtime, and quantization toolkit versions; release `requirements.txt` or container/Dockerfile.
- **Seeds**: Run ≥3 seeds; report mean ± std for all metrics.

### 11.3 Training and Ablation Matrix
Train with the same data and schedule (warmup → QAT → freeze) and publish checkpoints for:
1. FP32 baseline.
2. INT8 baseline (standard PTQ/QAT) without dual experts.
3. Ternary-only Base-3 (no INT8 residuals, no spectral loss).
4. Base-5 dual experts (ternary fusion) without residual highways.
5. Base-5 + INT8 residual highways (ResiBit) without spectral loss.
6. Full Spectral-ResiBit (Base-5 + INT8 residuals + spectral orthogonality loss).

Log for every run:
- mAP50 and mAP50-95 on COCO val2017 and per-class AP for S/M/L objects
- Weight sparsity, spectral separation coefficient, and loss curves over epochs
- Training throughput and GPU memory usage

### 11.4 Inference Benchmarks (Edge)
- Export ONNX and TFLite int8/ternary models with calibration sets.
- Measure end-to-end **latency/FPS** on Raspberry Pi 5/4 at 640×640; report median and P95 over ≥500 frames.
- Measure **power** with an inline USB power meter; report average/peak watts and energy per frame.
- Profile operator breakdown to confirm the fraction of ops running multiplication-free (shift/add/skip).

### 11.5 Success / Fail Criteria
- Accuracy gap ≤1% absolute mAP50-95 versus FP32; any drop beyond this is a failure of parity.
- Latency/energy improvement ≥3× versus FP32 at identical resolution and batch size.
- Spectral separation coefficient (mean cross-spectrum correlation; see spectral orthogonality loss in §5.5) remains >0.3 after convergence; this is an initial target to keep expert frequency content meaningfully distinct and should be tuned once empirical distributions are observed. mAP standard deviation across seeds stays <0.5 percentage points.

### 11.6 Artifacts to Release
- Training/eval scripts, configs, fixed seeds, and logged metrics (TensorBoard/CSV).
- Checkpoints for all ablations plus exported ONNX/TFLite artifacts.
- Exact benchmark commands and raw power/latency logs.
- A short validation report summarizing results, deviations, and known issues.

### 11.7 Sandbox Mathematical and Empirical Checks (executed here)
The following quick checks were executed in this sandbox to provide minimal evidence and guardrails. They **do not** replace full training/benchmark validation above.

| Check | Method | Result |
|-------|--------|--------|
| Information-theoretic bits | log2(3), log2(5) | log2(3) = 1.584963, log2(5) = 2.321928 |
| Compression vs. FP32 | 32 / log2(5) | 13.7816× smaller weight storage than FP32 (theoretical) |
| Base-5 fusion linearity | Pure-Python conv sanity test (seed=0) comparing `(W_A + W_B) * X` vs. `W_A*X + W_B*X` | max difference = 0 using integer arithmetic; confirms linearity of fused convolution but does **not** validate end-to-end accuracy, training, or quantization noise behavior (floating-point implementations should expect small epsilons). |

**Limitations**: No training/inference code exists in this repository, so dataset-level accuracy and hardware latency/energy measurements could not be executed here. The full validation plan above remains required to substantiate the manuscript claims.
