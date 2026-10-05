"""Sequential reconstruction and bounded, current-input curvature capture."""
import math
from collections import OrderedDict

import torch

from .importance import _get_robust_batch_tau, PERCENTILE


_INPUT_GROUPS = {
    "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
    "gate_proj": "gate_up", "up_proj": "gate_up",
    "in_proj_qkv": "gdn_input", "in_proj_z": "gdn_input",
    "in_proj_a": "gdn_input", "in_proj_b": "gdn_input",
}


def shared_input_key(name):
    parent, _, leaf = name.rpartition(".")
    # Only known decoder branches: TuneFP updates Linear weights, while their
    # input norms and the block inputs are frozen. Completed upstream parents
    # are frozen/cached by the reconstruction scheduler.
    if parent in {"self_attn", "linear_attn", "mlp"} and leaf in _INPUT_GROUPS:
        return parent, _INPUT_GROUPS[leaf]
    return None


def reconstruction_groups(names, schedule="sequential"):
    if schedule not in {"sequential", "shared_input", "parent"}:
        raise ValueError("tune_schedule must be sequential, shared_input or parent")
    groups = OrderedDict()
    # Match dependencies in the forward graph: all attention input projections
    # precede its output projection. Older selection lists placed k after o.
    names = list(names)
    for parent in ("self_attn", "linear_attn"):
        indices = [index for index, name in enumerate(names) if name.startswith(parent+".")]
        if indices:
            projections = [names[index] for index in indices]
            projections.sort(key=lambda name: name.rsplit(".", 1)[-1] in {"o_proj", "out_proj"})
            for index, name in zip(indices, projections):
                names[index] = name
    for name in names:
        parent, _, leaf = name.rpartition(".")
        key = name if schedule == "sequential" else parent or name
        if schedule == "shared_input":
            key = (parent, _INPUT_GROUPS.get(leaf, leaf))
        groups.setdefault(key, []).append(name)
    return list(groups.values())


class _InputsCaptured(Exception):
    pass


@torch.no_grad()
def refresh_input_stats(block, linears, inputs, kwargs, batch_size, shrinkage=0.4,
                        strategy="online", correlation_block_size=0):
    """Measure after TuneFP; exit at the last required linear's pre-hook.

    Only identical input tensor views share accumulation. No output GEMM or
    downstream block forward is needed once all requested inputs are seen.
    Correlations use a fresh, clipped input second moment; output Fisher stays
    diagonal. This is a bounded one-sided adaptation, not a KL-Shampoo fit.
    """
    from .compress_block import _batch_kwargs
    from .curvature import BlockInputMetric

    if not 0 <= shrinkage <= 1:
        raise ValueError("calib_shrinkage must lie in [0, 1]")
    if batch_size < 1 or inputs.shape[0] < 1:
        raise ValueError("Fresh statistics require samples and a positive batch size")
    if correlation_block_size not in {0, 128, 256}:
        raise ValueError("correlation_block_size must be 0, 128 or 256")
    if strategy not in {"online", "two_phase", "dbf", "none"}:
        raise ValueError("Unknown calibration strategy")
    device = next(block.parameters()).device
    records, layer_records, handles = {}, {}, []
    seen, views, view_owners = set(), {}, {}
    fixed_tau = {}

    def run_pass(profiling=False):
        def hook(module, args, call_kwargs):
            x = args[0] if args else call_kwargs.get("input")
            if not isinstance(x, torch.Tensor):
                raise RuntimeError("Missing linear input during fresh statistics")
            # Include shape, strides and offset: different views of one storage
            # must not accidentally share their statistics.
            key = (x.data_ptr(), tuple(x.shape), tuple(x.stride()))
            if key in views:
                record = views[key]
            else:
                flat = x.detach().reshape(-1, x.shape[-1]).float()
                norms = flat.norm(dim=1, keepdim=True)
                tau = _get_robust_batch_tau(norms, PERCENTILE)
                if profiling:
                    previous = fixed_tau.get(module)
                    fixed_tau[module] = tau if previous is None else torch.maximum(previous, tau)
                    record = None
                else:
                    record = records.get(module)
                    if record is None:
                        record = {"sum": torch.zeros(flat.shape[-1], device=device), "count": 0,
                                  "tau": None, "cov": []}
                        if correlation_block_size:
                            for start in range(0, flat.shape[-1], correlation_block_size):
                                size = min(correlation_block_size, flat.shape[-1]-start)
                                record["cov"].append(torch.zeros(size, size, device=device))
                        records[module] = record
                    if strategy in {"dbf", "none"}:
                        clipped = flat
                    else:
                        if strategy == "two_phase":
                            # Shared views need their own fixed profiling entry.
                            gmax = fixed_tau[module]
                        else:
                            old = record["tau"]
                            gmax = tau if old is None else torch.maximum(old, tau)
                            if old is not None:
                                correction = torch.where(tau > old, (tau / (old + 1e-8)).square(),
                                                         torch.ones_like(tau))
                                record["sum"].mul_(correction)
                                for covariance in record["cov"]:
                                    covariance.mul_(correction)
                            record["tau"] = gmax
                        clipped = flat * (gmax / (norms + 1e-8)).clamp(max=1)
                    record["sum"].add_(clipped.square().sum(0))
                    record["count"] += flat.shape[0]
                    for index, covariance in enumerate(record["cov"]):
                        start = index * correlation_block_size
                        tile = clipped[:, start:start + covariance.shape[0]]
                        covariance.addmm_(tile.mT, tile)
                views[key] = record
                view_owners[key] = module
            if profiling:
                fixed_tau[module] = fixed_tau[view_owners[key]]
            layer_records[module] = key if profiling else record
            seen.add(module)
            if len(seen) == len(linears):
                raise _InputsCaptured

        try:
            handles.extend(module.register_forward_pre_hook(hook, with_kwargs=True) for module in linears)
            for start in range(0, inputs.shape[0], batch_size):
                end = min(start + batch_size, inputs.shape[0])
                seen.clear()
                views.clear()
                view_owners.clear()
                block._nanoquant_active_indices = list(range(start, end))
                try:
                    block(inputs[start:end].to(device), **_batch_kwargs(kwargs, end-start, device))
                except _InputsCaptured:
                    pass
                if len(seen) != len(linears):
                    raise RuntimeError("A requested projection did not execute during fresh input capture")
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
            block._nanoquant_active_indices = None

    if strategy == "two_phase":
        run_pass(profiling=True)
        layer_records.clear()
    run_pass()
    finalized = {}
    for module in linears:
        record = layer_records[module]
        key = id(record)
        if key not in finalized:
            normalization = inputs.shape[0] if strategy == "dbf" else 1
            diagonal = record["sum"] / record["count"] * normalization
            if strategy == "none":
                diagonal = torch.ones_like(diagonal)
            diagonal = (1-shrinkage)*diagonal + shrinkage*diagonal.mean()
            metric = None
            if record["cov"]:
                covariance = [value / record["count"] * normalization for value in record["cov"]]
                # Shrink raw covariance before congruence transport; retain
                # its diagonal as the scale map, then temper its spectrum.
                mean = record["sum"].sum() / (record["count"] * diagonal.numel()) * normalization
                for value in covariance:
                    value.mul_(1-shrinkage)
                    value.diagonal().add_(shrinkage * mean)
                metric = BlockInputMetric.from_covariance(covariance, diagonal, exponent=0.5)
            finalized[key] = (diagonal, metric)
        module.i_norm, module._nanoquant_input_metric = finalized[key]


def validate_reconstruction_config(config):
    reconstruction_groups([], config.get("tune_schedule", "sequential"))
    if config.get("rank_allocation", "uniform") not in {"sensitivity", "kronq_trace", "uniform"}:
        raise ValueError("rank_allocation must be sensitivity, kronq_trace or uniform")
    if config.get("rank_budget", "nominal") not in {"nominal", "uniform"}:
        raise ValueError("rank_budget must be nominal or uniform")
    min_uniform_ratio = float(config.get("rank_allocation_min_uniform_ratio", 0.75))
    max_uniform_ratio = float(config.get("rank_allocation_max_uniform_ratio", 1.25))
    if (not math.isfinite(min_uniform_ratio) or not math.isfinite(max_uniform_ratio)
            or not 0 < min_uniform_ratio <= 1 <= max_uniform_ratio):
        raise ValueError("Adaptive rank ratios must satisfy 0 < min <= 1 <= max")
    size = config.get("correlation_block_size", 0)
    if size not in {0, 128, 256}:
        raise ValueError("correlation_block_size must be 0, 128 or 256")
    if size and (not config.get("refresh_input_stats", True) or config.get("admm_type", "nanoquant") != "nanoquant"):
        raise ValueError("Correlations require fresh inputs and nanoquant ADMM")
    if config.get("rank_probe_candidates", 0) < 0 or config.get("rank_probe_iters", 50) < 1:
        raise ValueError("Rank probes need a nonnegative candidate count and positive iterations")
    if config.get("rank_probe_candidates", 0) and config.get("admm_type", "nanoquant") != "nanoquant":
        raise ValueError("Measured rank probes require nanoquant ADMM")
    if config.get("nonfact_plateau_tolerance", 0) < 0:
        raise ValueError("Plateau tolerance cannot be negative")
    if config.get("nonfact_plateau_min_epochs", 3) < 1 or config.get("nonfact_plateau_patience", 2) < 1:
        raise ValueError("Plateau stopping needs positive minimum epochs and patience")


def stage_quant_config(config, names, hidden_size):
    """Large-model epoch caps from ShamAN-Q, adapted to Gated DeltaNet stages.

    Keep every inter-projection TuneFP round. Only early stages receive smaller
    epoch budgets. Explicit parent/shared-input schedules retain their requested
    uniform epoch budgets for a clean comparison with the old implementation.
    """
    if (not config.get("layer_epoch_schedule", True) or hidden_size < 4096
            or config.get("tune_schedule", "sequential") != "sequential"):
        return config
    caps = {"q_proj": 2, "k_proj": 2, "v_proj": 4, "o_proj": 4, "out_proj": 4,
            "in_proj_qkv": 2, "in_proj_z": 4, "in_proj_a": 4, "in_proj_b": 4,
            "gate_proj": 6, "up_proj": 8, "down_proj": 8}
    cap = max(caps.get(name.rsplit(".", 1)[-1], 8) for name in names)
    result = dict(config)
    for key in ("nonfact_epochs", "fact_epochs"):
        result[key] = min(int(config[key]), cap)
    return result
