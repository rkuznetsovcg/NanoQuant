"""Bounded ADMM rank probes with content-addressed results and bit parity."""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .admm_nq import factorize_admm_nanoquant
from .resume import _atomic_save, content_key
from ..utils.utils import find_layers, rank_allocation_importance, set_seed


def packed_bits(module, rank, num_scales=2):
    a, b = module.in_features, module.out_features
    # The repository packer pads EACH ROW to a 32-bit word.
    return 32*(rank*((a+31)//32) + b*((rank+31)//32)) + 16*(a+b+(rank if num_scales == 3 else 0))


def fit_error_curve(probes):
    """Positive monotone power curve; noisy/worsening probes keep the proxy."""
    if len(probes) < 2 or not all(math.isfinite(error) and error > 0 for _, error in probes):
        return None
    probes = sorted(probes)
    if any(right[1] >= left[1] for left, right in zip(probes, probes[1:])):
        return None
    x = [math.log(rank) for rank, _ in probes]
    y = [math.log(error) for _, error in probes]
    x_mean, y_mean = sum(x)/len(x), sum(y)/len(y)
    variance = sum((value-x_mean)**2 for value in x)
    if variance <= 0:
        return None
    slope = sum((a-x_mean)*(b-y_mean) for a, b in zip(x, y)) / variance
    if slope >= -1e-3:
        return None
    eta = min(-slope, 4.0)
    return math.exp(y_mean + eta*x_mean), eta


def allocate_measured_ranks(modules, initial, curves, num_scales=2):
    """Reallocate inside this block, never increase its initial packed budget.

    Unprobed matrices keep their assigned ranks. Limit measured changes to the
    probed interval, so the short fits cannot extrapolate to rank 32 or the full
    width of a large layer.
    """
    import heapq
    result = dict(initial)
    keys = sorted(curves)
    budget = sum(packed_bits(modules[key], initial[key], num_scales) for key in keys)
    for key in keys:
        result[key] = curves[key][2]
    used = sum(packed_bits(modules[key], result[key], num_scales) for key in keys)
    candidates = []

    def push(key):
        level, eta, _lower, upper = curves[key]
        rank = result[key]
        next_rank = min(rank+32, upper)
        if next_rank <= rank:
            return
        cost = packed_bits(modules[key], next_rank, num_scales)-packed_bits(modules[key], rank, num_scales)
        gain = level*(rank**(-eta)-next_rank**(-eta))
        heapq.heappush(candidates, (-gain/cost, key, next_rank, cost))

    for key in keys:
        push(key)
    while candidates:
        _score, key, next_rank, cost = heapq.heappop(candidates)
        if used+cost > budget:
            continue
        used += cost
        result[key] = next_rank
        push(key)
    return result


@torch.no_grad()
def refine_block_ranks(block, names, block_index, ranks, config):
    limit = int(config.get("rank_probe_candidates", 0))
    if limit <= 0:
        return
    if config["admm_type"] != "nanoquant":
        raise ValueError("Measured rank probes currently require nanoquant ADMM")
    iterations = int(config.get("rank_probe_iters", 50))
    if iterations < 1:
        raise ValueError("rank_probe_iters must be positive")
    modules = {name: module for name, module in find_layers(block).items() if name in names}
    initial = {name: ranks[f"{block_index}.{name}"] for name in modules}
    # Choose candidates with the largest proxy marginal return. The default
    # cheap allocator still covers all other weights.
    allocation = config.get("rank_allocation", "sensitivity")
    if allocation == "uniform":
        # Preserve the historical probe-candidate ordering for uniform ranks.
        allocation = "sensitivity"
    priorities = {}
    for name, module in modules.items():
        if allocation == "kronq_trace":
            importance = rank_allocation_importance(module, allocation)
        else:
            # Keep the existing control profile's GPU-side screening path
            # unchanged; only the explicit KronQ variant uses trace proxies.
            total = torch.zeros((), device=module.weight.device)
            for start in range(0, module.out_features, 256):
                tile = module.weight[start:start+256].float()
                total.add_((tile.square()*module.i_norm.float()[None, :]
                            * module.o_norm[start:start+256].float()[:, None]).sum())
            importance = total.item()
        priorities[name] = importance / ((initial[name]+32)*32*(module.in_features+module.out_features))
    candidates = sorted(modules, key=lambda name: (-priorities[name], name))[:limit]
    cache_dir = Path(config["resume_dir"]) / "rank-probes" if config.get("resume_dir") else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    curves = {}
    for name in candidates:
        module = modules[name]
        rank = initial[name]
        cap = min(module.in_features, module.out_features)
        probe_ranks = sorted({max(min(32, cap), min(cap, 32*int(rank*multiplier/32)))
                              for multiplier in (0.6, 1.0, 1.4)} | {rank})
        settings = {"version": 1, "iterations": iterations, "inner": config["admm_inner_iters"],
                    "warm": config.get("admm_warm_start_iters", 2), "reg": config["admm_reg"],
                    "scheduler": config["admm_penalty_scheduler"], "seed": config["seed"],
                    "ranks": probe_ranks, "metric": "initial-diagonal", "dtype": str(module.weight.dtype)}
        path = cache_dir / (content_key((module.weight, module.i_norm, module.o_norm), settings)+".pt") if cache_dir else None
        if path is not None and path.exists():
            probes = torch.load(path, weights_only=True)
        else:
            probes = []
            for probe_rank in probe_ranks:
                set_seed(config["seed"])
                factors = factorize_admm_nanoquant(
                    module.weight.detach(), module.i_norm, module.o_norm, probe_rank,
                    outer_iters=iterations, inner_iters=config["admm_inner_iters"], reg=config["admm_reg"],
                    is_transpose=module.out_features < module.in_features,
                    rho_scheduler=config["admm_penalty_scheduler"],
                    warm_start_iters=config.get("admm_warm_start_iters", 2), early_stop=False,
                    compute_diagnostic=False)
                # Score the deployed signs and scales, not the continuous ADMM
                # proxy. Row tiles avoid a dense reconstructed weight cache.
                V = factors["B"].sign()
                V = torch.where(V == 0, torch.ones_like(V), V)
                error = torch.zeros((), device=module.weight.device)
                for start in range(0, module.out_features, 256):
                    end = min(start+256, module.out_features)
                    U = factors["A"][:, start:end].mT.sign()
                    U = torch.where(U == 0, torch.ones_like(U), U)
                    prediction = (F.linear(U.float(), V.float().mT)
                                  * factors["scale_pre"].to(torch.bfloat16).float()
                                  * factors["scale_post"][:, start:end].to(torch.bfloat16).float().mT)
                    residual = (module.weight[start:end].float()-prediction.float())
                    error.add_((residual.square()*module.i_norm.float()[None, :]
                                * module.o_norm[start:end].float()[:, None]).sum())
                probes.append((probe_rank, error.item()))
                del factors, V
            if path is not None:
                _atomic_save(probes, path)
        fit = fit_error_curve(probes)
        if fit is not None:
            curves[name] = (*fit, min(probe_ranks), max(probe_ranks))
    allocated = allocate_measured_ranks(modules, initial, curves)
    for name, rank in allocated.items():
        ranks[f"{block_index}.{name}"] = rank
    print(f"\tMeasured rank curves: {len(curves)}/{len(candidates)} usable; packed block budget preserved")
