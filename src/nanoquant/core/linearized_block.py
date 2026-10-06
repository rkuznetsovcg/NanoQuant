# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single-block linear replacement pilot for decoder-only models."""

import math
from typing import Optional

import torch
import torch.nn as nn


LINEARIZED_BLOCK_INDICES_KEY = "__nanoquant_linearized_block_indices__"


class LinearizedDecoderBlock(nn.Module):
    """Replace a decoder block by one fitted affine map over hidden states."""

    def __init__(self, hidden_size: int, device=None, dtype=None):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.linear.requires_grad_(False)

    def forward(self, hidden_states, *args, **kwargs):
        # The approximation has no attention or recurrent state, so cache and
        # positional arguments accepted by the original decoder block are unused.
        # Qwen3.5's decoder layers return a tensor, so preserve that contract.
        return self.linear(hidden_states)


@torch.no_grad()
def fit_linearized_decoder_block(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_tokens: int = 16384,
    ridge: float = 1e-4,
    chunk_tokens: int = 1024,
    device: Optional[torch.device] = None,
):
    """Fit ``LinearizedDecoderBlock`` using streaming ridge least squares.

    Inputs are the current (already compressed-prefix) block activations and
    targets are the full-precision block outputs captured by NanoQuant. Only
    sufficient statistics and a bounded token sample reside on the GPU.
    """
    if inputs.shape != targets.shape or inputs.ndim < 2:
        raise ValueError("Linear block fitting expects matching [..., hidden] input and target tensors")
    if max_tokens < 2 or chunk_tokens < 1:
        raise ValueError("linearize_max_tokens must be >= 2 and linearize_chunk_tokens must be >= 1")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("linearize_ridge must be finite and nonnegative")

    hidden_size = inputs.shape[-1]
    total_tokens = inputs.numel() // hidden_size
    sample_count = min(total_tokens, max_tokens)
    if sample_count < 2:
        raise ValueError("At least two calibration tokens are required to fit a linearized block")

    # Stride across the cached windows instead of materializing a large index
    # tensor or copying all calibration activations to the accelerator.
    stride = max(1, math.ceil(total_tokens / sample_count))
    input_rows = inputs.detach().reshape(-1, hidden_size)[::stride][:sample_count]
    target_rows = targets.detach().reshape(-1, hidden_size)[::stride][:sample_count]
    sample_count = input_rows.shape[0]
    work_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sum_x = torch.zeros(hidden_size, dtype=torch.float32, device=work_device)
    sum_r = torch.zeros_like(sum_x)
    for start in range(0, sample_count, chunk_tokens):
        end = min(start + chunk_tokens, sample_count)
        x = input_rows[start:end].to(device=work_device, dtype=torch.float32, non_blocking=True)
        y = target_rows[start:end].to(device=work_device, dtype=torch.float32, non_blocking=True)
        sum_x.add_(x.sum(dim=0))
        sum_r.add_((y - x).sum(dim=0))
    mean_x = sum_x / sample_count
    mean_r = sum_r / sample_count

    covariance_xx = torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=work_device)
    covariance_xr = torch.zeros_like(covariance_xx)
    centered_residual_sq = torch.zeros((), dtype=torch.float32, device=work_device)
    target_sq = torch.zeros_like(centered_residual_sq)
    for start in range(0, sample_count, chunk_tokens):
        end = min(start + chunk_tokens, sample_count)
        x = input_rows[start:end].to(device=work_device, dtype=torch.float32, non_blocking=True)
        y = target_rows[start:end].to(device=work_device, dtype=torch.float32, non_blocking=True)
        x_centered = x - mean_x
        residual_centered = (y - x) - mean_r
        covariance_xx.add_(x_centered.T @ x_centered)
        covariance_xr.add_(x_centered.T @ residual_centered)
        centered_residual_sq.add_(residual_centered.square().sum())
        target_sq.add_(y.square().sum())

    covariance_xx = (covariance_xx + covariance_xx.T) * 0.5
    diagonal_scale = (covariance_xx.diagonal().mean() / sample_count).clamp_min(torch.finfo(torch.float32).eps)
    regularization = max(ridge, 1e-7) * diagonal_scale
    chol = None
    ridge_multiplier = 1
    for _ in range(5):
        system = covariance_xx.clone()
        system.diagonal().add_(regularization * sample_count * ridge_multiplier)
        candidate_chol, info = torch.linalg.cholesky_ex(system)
        if int(info.max().item()) == 0:
            chol = candidate_chol
            break
        ridge_multiplier *= 10
    if chol is None:
        raise RuntimeError("Ridge system for the linearized block was not positive definite")
    residual_map = torch.cholesky_solve(covariance_xr, chol)

    # x + x @ residual_map + bias is represented as a single nn.Linear.
    fitted_bias = mean_r - mean_x @ residual_map
    linear = LinearizedDecoderBlock(hidden_size, device=work_device, dtype=inputs.dtype)
    linear.linear.weight.copy_(residual_map.T.to(dtype=inputs.dtype))
    linear.linear.weight.diagonal().add_(1.0)
    linear.linear.bias.copy_(fitted_bias.to(dtype=inputs.dtype))

    stored_residual_map = linear.linear.weight.float().T.contiguous()
    stored_residual_map.diagonal().sub_(1.0)
    stored_bias = linear.linear.bias.float()
    mean_error = mean_r - mean_x @ stored_residual_map - stored_bias
    fitted_sse = centered_residual_sq - 2 * (stored_residual_map * covariance_xr).sum()
    fitted_sse += (stored_residual_map * (covariance_xx @ stored_residual_map)).sum()
    fitted_sse += sample_count * mean_error.square().sum()
    relative_rmse = (fitted_sse.clamp_min(0) / target_sq.clamp_min(1e-12)).sqrt().item()

    diagnostics = {
        "tokens": sample_count,
        "relative_rmse": relative_rmse,
        "ridge": ridge,
        "parameters": hidden_size * hidden_size + hidden_size,
    }
    return linear, diagnostics
