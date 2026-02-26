"""
Quantization functions for 1.58-bit (ternary) and INT8 networks.

Implements the absmean ternary quantization from BitNet b1.58 (Ma et al., 2024)
and the INT8 residual quantization from ResiBit-YOLO.
"""

import torch


def quantize_ternary(W: torch.Tensor) -> torch.Tensor:
    """Absmean ternary quantization to {-1, 0, +1}.

    Implements Equation (2) from the papers:
        alpha = mean(|W|)
        W_hat = clip(round(W / alpha), -1, 1)

    Args:
        W: Weight tensor of any shape.

    Returns:
        Quantized tensor with values in {-1, 0, +1}.
    """
    alpha = W.abs().mean().clamp(min=1e-6)
    return (W / alpha).round().clamp(-1, 1)


def quantize_ternary_ste(W: torch.Tensor) -> torch.Tensor:
    """Absmean ternary quantization with Straight-Through Estimator (STE).

    Uses the STE trick to allow gradient flow through the non-differentiable
    rounding operation: detach the rounding, pass gradient through the
    scaled input.

    Implements Equations (1)-(2) from the papers:
        Forward: W_hat = clip(round(W / alpha), -1, 1)
        Backward: dW_hat/dW ≈ 1_{|W| <= 1}  (STE)

    Args:
        W: Weight tensor of any shape.

    Returns:
        Quantized tensor with values in {-1, 0, +1}, with STE gradients.
    """
    alpha = W.abs().mean().clamp(min=1e-6)
    W_scaled = W / alpha
    # STE: detach the rounding, pass gradient through W_scaled
    W_quantized = W_scaled.round().clamp(-1, 1)
    return W_quantized - W_scaled.detach() + W_scaled


def quantize_int8(W: torch.Tensor) -> torch.Tensor:
    """Symmetric INT8 quantization using absmax scaling.

    Maps weights to the range [-127, 127] using symmetric quantization.

    Args:
        W: Weight tensor of any shape.

    Returns:
        Quantized tensor with values in [-127, 127] (as float for training).
    """
    scale = W.abs().max().clamp(min=1e-6) / 127.0
    W_quantized = (W / scale).round().clamp(-127, 127)
    # STE for training
    return W_quantized - (W / scale).detach() + (W / scale)


def quantize_activations(X: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Absmax activation quantization to b-bit integers.

    Implements the activation quantization from BitNet b1.58:
        Q_b = max_val
        X_hat = clip(round(X * Q_b / gamma), -Q_b, Q_b)
    where gamma = max(|X|).

    Args:
        X: Activation tensor of any shape.
        bits: Number of bits for quantization (default: 8).

    Returns:
        Quantized activation tensor.
    """
    Q_b = 2 ** (bits - 1) - 1  # e.g. 127 for 8-bit
    gamma = X.abs().max().clamp(min=1e-6)
    X_scaled = X * Q_b / gamma
    X_quantized = X_scaled.round().clamp(-Q_b, Q_b)
    # STE for training
    return X_quantized - X_scaled.detach() + X_scaled
