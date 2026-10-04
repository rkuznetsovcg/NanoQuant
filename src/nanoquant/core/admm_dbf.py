# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
ADMM from DBF. We remove support for sparsity, but the ADMM logic is kept.
https://github.com/usamec/double_binary/blob/master/raw_stuff/compress/Compress-Llama-2-7B-tar15.ipynb
"""

import torch


@torch.no_grad()
def power_iteration(A, num_iters=5, initial_v=None):
    # Reuse the previous right singular vector when successive matrices are
    # nearby; otherwise start from a random vector on the appropriate device.
    n = A.shape[1]
    if initial_v is None or initial_v.numel() != n:
        v = torch.randn(n, dtype=A.dtype, device=A.device)
    else:
        v = initial_v.to(device=A.device, dtype=A.dtype)
    v = torch.nn.functional.normalize(v, dim=0, eps=1e-12)

    for _ in range(num_iters):
        # Epsilon-safe normalization avoids a device-to-host sync for a
        # Python zero check on every power-iteration step.
        u = torch.nn.functional.normalize(torch.mv(A, v), dim=0, eps=1e-12)
        v = torch.nn.functional.normalize(torch.mv(A.mT, u), dim=0, eps=1e-12)

    # Estimate the dominant singular value as ||A*v||
    u_unnorm = torch.mv(A, v)
    sigma = torch.norm(u_unnorm)
    u = torch.nn.functional.normalize(u_unnorm, dim=0, eps=1e-12)
    return u, sigma, v


@torch.no_grad()
def svd_abs(W, num_iters=5, initial_v=None, return_state=False):
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=num_iters, initial_v=initial_v)
    apx = s * torch.outer(u, v)
    result = apx * Sg
    return (result, v) if return_state else result


@torch.no_grad()
def svd_abs2(W):
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=5)
    return u * s, Sg, v


def _admm_solve_step(X, Y, Z, U, rho_start, reg=3e-2, inner_iters=3, warm_start_iters=2):
    """
    ADMM solver that mimics the original `find_other2` logic.
    It uses `rho_start` for the first step and a fixed `rho=1` for subsequent steps.
    """
    orig_dtype = X.dtype
    X, Y, Z, U = (t.to(torch.float32) for t in (X, Y, Z, U))

    XX = X.T.matmul(X)
    XX.diagonal().add_(XX.diagonal().mean() * reg)
    XY = X.T.matmul(Y)

    # Factor both positive-definite systems once. Cholesky solves avoid
    # materializing dense inverses and are more stable for ill-conditioned X.
    identity = torch.eye(XX.shape[1], dtype=XX.dtype, device=XX.device)
    L_start, start_info = torch.linalg.cholesky_ex(XX + identity * rho_start, check_errors=not X.is_cuda)
    L_fixed, fixed_info = torch.linalg.cholesky_ex(XX + identity, check_errors=not X.is_cuda)
    if X.is_cuda:
        torch._assert_async(start_info.eq(0), "DBF initial system matrix is not positive-definite")
        torch._assert_async(fixed_info.eq(0), "DBF fixed system matrix is not positive-definite")

    # First step uses rho_start
    Factor = torch.cholesky_solve(XY + rho_start * (Z - U), L_start, upper=False)

    # Use a full initial power iteration, then warm-start nearby projections
    # with the previous right singular vector.
    warm_v = None
    project_iters = 5
    for _ in range(inner_iters - 1):
        Z, warm_v = svd_abs(Factor + U, num_iters=project_iters, initial_v=warm_v, return_state=True)
        project_iters = min(5, int(warm_start_iters)) if warm_start_iters > 0 else 5
        U = U + (Factor - Z)
        Factor = torch.cholesky_solve(XY + (Z - U), L_fixed, upper=False)

    Z, _warm_v = svd_abs(Factor + U, num_iters=project_iters, initial_v=warm_v, return_state=True)
    U = U + (Factor - Z)

    Z, U, Factor = (t.to(orig_dtype) for t in (Z, U, Factor))
    return Z, U, Factor


def factorize_admm_dbf(
    W,
    i_norm,
    o_norm,
    mid_rank,
    iters=260,
    is_transpose=False,
    eps=1e-8,
    reg=3e-2,
    use_latent=False,
    warm_start_iters=2,
    early_stop=True,
    min_outer_iters=120,
    check_interval=10,
    convergence_tolerance=2e-3,
    stable_sign_tolerance=1e-3,
    rho_stop_threshold=0.85,
    compute_diagnostic=True,
):
    """
    Decomposes the weight matrix W into two binary matrices A and B using ADMM.
    Assumes W has the shape (out_features, in_features).
    """
    if is_transpose:
        # For layers like fc2/down_proj where in > out, process the transpose
        results = factorize_admm_dbf(
            W.T,
            o_norm,
            i_norm,
            mid_rank,
            iters,
            is_transpose=False,
            eps=eps,
            reg=reg,
            use_latent=use_latent,
            warm_start_iters=warm_start_iters,
            early_stop=early_stop,
            min_outer_iters=min_outer_iters,
            check_interval=check_interval,
            convergence_tolerance=convergence_tolerance,
            stable_sign_tolerance=stable_sign_tolerance,
            rho_stop_threshold=rho_stop_threshold,
            compute_diagnostic=compute_diagnostic,
        )
        # Return A, B in the correct order and restore the original weight shape
        transposed_results = {
            "A": results["B"],
            "B": results["A"],
            "A_latent": results["B_latent"],
            "B_latent": results["A_latent"],
            # Swap pre and post for the transpose
            "scale_pre": results["scale_post"],
            "scale_mid": results["scale_mid"],
            "scale_post": results["scale_pre"],
        }
        if compute_diagnostic:
            transposed_results["W_final"] = results["W_final"].T
        return transposed_results

    device = W.device
    out_features, in_features = W.shape

    # Re-scale norms by a heuristic factor (128) to compensate for the division by
    # n_samples in calibration. This restores the magnitude range the ADMM solver
    # expects, preventing numerical instability/underflow.
    norm_i = (i_norm).sqrt().clamp(eps)
    norm_o = (o_norm).sqrt().clamp(eps).unsqueeze(1)
    W_norm = W * norm_i * norm_o

    Az = torch.randn((out_features, mid_rank), device=device)
    Au = torch.zeros_like(Az)
    Bz = torch.randn((mid_rank, in_features), device=device)
    Bu = torch.zeros_like(Bz)

    check_interval = max(1, int(check_interval))
    stop_min_iters = min(max(1, int(min_outer_iters)), max(1, iters))
    stable_checks = 0
    if early_stop and iters > 0:
        A_check = Az.clone()
        B_check = Bz.clone()

    for itt in range(iters):
        # Calculate rho_start, which changes over iterations
        rho_start = min(1.0, itt / (iters - 3))**3 if iters > 3 else 1.0

        # Update A (W.T = B.T @ A.T)
        # The asymmetric scaling is kept as it is part of the original's design
        mid_norm_b = Bz.norm(dim=1).clamp(eps)
        X_A = Bz.T / mid_norm_b
        Az_T, Au_T, Als_T = _admm_solve_step(
            X_A, W_norm.T, Az.T, Au.T, rho_start, reg=reg, warm_start_iters=warm_start_iters
        )
        Az, Au, Als = Az_T.T, Au_T.T, Als_T.T

        # Update B (W = A @ B)
        mid_norm_a = Az.norm(dim=0).clamp(eps)
        X_B = Az / mid_norm_a
        Bz, Bu, Bls = _admm_solve_step(
            X_B, W_norm, Bz, Bu, rho_start, reg=reg, warm_start_iters=warm_start_iters
        )

        if (early_stop and itt + 1 >= stop_min_iters and (itt + 1) % check_interval == 0
                and rho_start >= rho_stop_threshold):
            primal_a = torch.linalg.vector_norm(Als - Az) / torch.linalg.vector_norm(Als).clamp_min(eps)
            primal_b = torch.linalg.vector_norm(Bls - Bz) / torch.linalg.vector_norm(Bls).clamp_min(eps)
            dual_a = torch.linalg.vector_norm(rho_start * (Az - A_check)) / \
                     torch.linalg.vector_norm(rho_start * A_check).clamp_min(eps)
            dual_b = torch.linalg.vector_norm(rho_start * (Bz - B_check)) / \
                     torch.linalg.vector_norm(rho_start * B_check).clamp_min(eps)
            sign_change_a = (Az.sign() != A_check.sign()).float().mean()
            sign_change_b = (Bz.sign() != B_check.sign()).float().mean()
            converged = bool(torch.maximum(torch.maximum(primal_a, primal_b),
                                           torch.maximum(dual_a, dual_b)) <= convergence_tolerance)
            signs_stable = bool(torch.maximum(sign_change_a, sign_change_b) <= stable_sign_tolerance)
            stable_checks = stable_checks + 1 if converged and signs_stable else 0
            A_check.copy_(Az)
            B_check.copy_(Bz)
            if stable_checks >= 2:
                break

    # --- 1. Final Scaling and Normalization ---
    A_final_unbalanced = Az / norm_o
    B_final_unbalanced = Bz / norm_i

    A_latent_unbalanced = (Als + Au) / norm_o
    B_latent_unbalanced = (Bls + Bu) / norm_i

    # --- 2. Final Norm Balancing ---
    # Balance the norms of A and B for numerical stability
    norm_A = A_final_unbalanced.norm().clamp(eps)
    norm_B = B_final_unbalanced.norm().clamp(eps)
    balance_factor = (norm_B / norm_A).sqrt()

    A_final = A_final_unbalanced * balance_factor
    B_final = B_final_unbalanced / balance_factor

    A_latent = A_latent_unbalanced * balance_factor
    B_latent = B_latent_unbalanced / balance_factor

    # The mid_scale compensates for the scaling applied to B's input (Az)
    final_mid_scale_factor = Az.norm(dim=0).clamp(eps)
    mid_scale = 1 / final_mid_scale_factor

    if compute_diagnostic:
        W_final = (A_final * mid_scale).matmul(B_final)

    # --- Extracting the 3 scales for NanoQuantLinear ---
    # Calculate scales based on mean magnitudes
    A = A_final.T
    B = B_final

    u1, b1, v1 = svd_abs2(B.float())
    u2, b2, v2 = svd_abs2(A.float())

    scale_pre = v1
    scale_mid = u1 * mid_scale * u2
    scale_post = v2

    result = {
        "A": b2,  # (mid, out)
        "B": b1,  # (mid, in)
        "A_latent": b2 if not use_latent else A_latent.T,
        "B_latent": b1 if not use_latent else B_latent,
        "scale_pre": scale_pre,
        "scale_mid": scale_mid,
        "scale_post": scale_post,
    }
    if compute_diagnostic:
        result["W_final"] = W_final
    return result
