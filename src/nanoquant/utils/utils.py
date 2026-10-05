# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import gc
import heapq
import inspect
import math
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


QWEN3_5_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}
QWEN_MODEL_TYPES = {"qwen3"} | QWEN3_5_MODEL_TYPES


def set_seed(seed, use_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if use_deterministic:
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def cleanup_memory(verbose=False, release_cuda_cache=True) -> None:
    """Collect Python garbage and optionally return cached CUDA memory."""
    caller_name = ""
    if verbose:
        try:
            caller_name = f" (from {inspect.stack()[1].function})"
        except (ValueError, KeyError):
            pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem() if verbose else 0

    del_vars = [k for k in list(globals().keys()) if k.startswith("_tmp_")]
    for k in del_vars:
        globals().pop(k, None)
    gc.collect()

    if torch.cuda.is_available():
        if release_cuda_cache:
            # https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/33
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        if verbose:
            memory_after = total_reserved_mem()
            print(f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GiB"
                  f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GiB)")


def extract_hidden_states(output):
    """Accept decoder blocks returning a tensor, a tuple, or a ModelOutput."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output[0]


def find_layers(module, layers=None, name=''):
    """
    Recursively finds all instances of specified layers in a module.
    """
    if layers is None:
        layers = [nn.Linear]

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def get_layers_to_factorize(model_type: str) -> list[str]:
    """Returns a list of sublayers to factorize based on the model architecture."""
    if (
        model_type in ["llama", "mistral", "mixtral", "mobilellm"]
        or model_type in QWEN_MODEL_TYPES
        or model_type.startswith("gemma")
    ):
        ret = [
            'self_attn.q_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'self_attn.k_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
            'mlp.down_proj',
        ]
        if model_type in QWEN3_5_MODEL_TYPES:
            # Qwen3.5/Qwen3.8 use Gated DeltaNet in most decoder blocks and
            # regular self-attention in every fourth block. Keep both attention
            # variants before the MLP so their frozen outputs can be reused
            # while the downstream MLP is tuned.
            linear_attention = [
                'linear_attn.in_proj_qkv',
                'linear_attn.in_proj_z',
                'linear_attn.in_proj_b',
                'linear_attn.in_proj_a',
                'linear_attn.out_proj',
            ]
            ret = ret[:4] + linear_attention + ret[4:]
    elif model_type == "opt":
        ret = [
            'self_attn.q_proj',
            'self_attn.v_proj',
            'self_attn.out_proj',
            'self_attn.k_proj',
            'fc1',
            'fc2',
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return ret


def get_decoder_layers(model):
    """
    Returns the list of decoder layers based on the model architecture.
    """
    model_type = model.config.model_type
    if model_type in QWEN3_5_MODEL_TYPES:
        # Full multimodal checkpoints keep the decoder under language_model;
        # AutoModelForCausalLM exposes the text-only decoder directly here.
        model_body = getattr(model, "model", model)
        language_model = getattr(model_body, "language_model", model_body)
        layers = getattr(language_model, "layers", None)
        if layers is None:
            raise AttributeError(f"Could not find Qwen3.5 text decoder layers for '{model_type}'.")
        return layers
    if model_type in ["llama", "mistral", "mixtral", "mobilellm", "qwen3"] or model_type.startswith("gemma"):
        return model.model.layers
    elif model_type == "opt":
        return model.model.decoder.layers
    elif model_type == "gpt2":
        return model.transformer.h
    raise AttributeError(f"Could not find decoder layers for model architecture '{model_type}'.")


def get_decoder_layer_cls_name(model: nn.Module) -> List[str]:
    """Helper to get the class name of the decoder blocks (to prevent accelerate from splitting blocks)."""
    try:
        layers = get_decoder_layers(model)
        if layers:
            return [layers[0].__class__.__name__]
    except AttributeError:
        pass
    return []


def estimate_weight_storage(model, planned_layers, num_scales=2):
    """Estimate weight bytes from shapes without reading or copying tensors.

    planned_layers contains (linear_module, rank) pairs before replacement.
    Metadata, activation caches, gradients, optimizer states and kernel padding
    are excluded. Unchanged tied parameters are counted once.
    """
    selected_modules = {id(module) for module, _rank in planned_layers}
    remaining_parameters = {}
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if id(module) in selected_modules and name == "weight":
                continue
            remaining_parameters[id(parameter)] = parameter
    remaining_bytes = sum(parameter.numel() * parameter.element_size()
                          for parameter in remaining_parameters.values())
    original_elements = sum(parameter.numel() for parameter in model.parameters())
    selected_elements = 0
    packed_bytes = 0
    unpacked_bytes = 0
    for module, rank in planned_layers:
        a, b = module.in_features, module.out_features
        selected_elements += a * b
        scale_elements = a + b + (rank if num_scales == 3 else 0)
        packed_bytes += 4*rank*((a+31)//32) + 4*b*((rank+31)//32) + 2*scale_elements
        unpacked_bytes += 2 * (rank * (a + b) + scale_elements)
    return {
        "selected_bpw": 8 * packed_bytes / max(1, selected_elements),
        "whole_model_bpw": 8 * (remaining_bytes + packed_bytes) / max(1, original_elements),
        "remaining_weight_bytes": remaining_bytes,
        "packed_weight_bytes": remaining_bytes + packed_bytes,
        "unpacked_weight_bytes": remaining_bytes + unpacked_bytes,
    }


def rank_allocation_importance(module, rank_allocation="sensitivity"):
    """Return the layer score used by the rank allocator.

    ``kronq_trace`` adapts KronQ's joint input/output Hessian-trace score using
    NanoQuant's clipped diagonal statistics. It does not apply KronQ's BiIP
    transform or full Hessian solver. Missing statistics use identity-Hessian
    traces, so this path never needs to scan the weights.
    """
    a, b = module.in_features, module.out_features
    input_importance = getattr(module, "i_norm", None)
    output_importance = getattr(module, "o_norm", None)

    if rank_allocation == "kronq_trace":
        input_trace = (float(input_importance.detach().float().sum().item())
                       if input_importance is not None else float(a))
        output_trace = (float(output_importance.detach().float().sum().item())
                        if output_importance is not None else float(b))
        importance = input_trace * output_trace
        if not math.isfinite(importance) or importance < 0:
            raise ValueError("KronQ trace rank importance must be finite and nonnegative")
        return importance

    if rank_allocation != "sensitivity":
        raise ValueError(
            f"Unsupported rank importance strategy {rank_allocation!r}; "
            "expected 'sensitivity' or 'kronq_trace'."
        )

    if input_importance is None or output_importance is None:
        return float(module.weight.float().square().sum().item())

    input_importance = input_importance.detach().float().cpu()
    output_importance = output_importance.detach().float().cpu()
    weighted_energy = 0.0
    for row_start in range(0, b, 256):
        row_end = min(row_start + 256, b)
        tile = module.weight[row_start:row_end].detach().float().cpu()
        weighted_energy += (
            tile.square()
            * input_importance.unsqueeze(0)
            * output_importance[row_start:row_end].unsqueeze(1)
        ).sum().item()
    return max(weighted_energy, 0.0)


def uniform_rank_for_shape(a, b, bits, num_scales=2):
    """Return the existing analytical rank baseline for one linear layer."""
    cap = min(a, b)
    min_rank = min(32, cap)
    if cap <= 0 or bits is None or a * b == 0:
        return min_rank

    total_budget_bits = a * b * bits
    parameter_sum = a + b
    if num_scales == 3:
        raw_rank = (total_budget_bits - 16 * parameter_sum) / (parameter_sum + 16)
    else:
        raw_rank = (total_budget_bits / parameter_sum) - 16
    rank = (int(raw_rank) // 32) * 32
    if rank <= 0:
        rank = min_rank
    return min(cap, max(rank, min_rank))


def rank_allocation_bounds(uniform_rank, cap, min_ratio=0.75, max_ratio=1.25):
    """Bound adaptive ranks around the analytical uniform rank in 32-wide steps."""
    lower = max(min(32, cap), math.ceil(uniform_rank * min_ratio / 32) * 32)
    upper = min(cap, math.floor(uniform_rank * max_ratio / 32) * 32)
    # Keep the baseline reachable when a small matrix or 32-rank granularity
    # would otherwise round one side of the interval past it.
    return min(lower, uniform_rank), max(upper, uniform_rank)


def calculate_ranks(model, layers_to_analyze, quant_config):
    """
    Unified entry point for bit allocation.
    """
    # Use a stable, model-wide uniform baseline unless a caller opts into an
    # adaptive allocator. Adaptive scores may reorder budget within this band,
    # but cannot starve late layers or saturate a few early ones.
    rank_allocation = quant_config.get("rank_allocation", "uniform")
    if rank_allocation not in {"sensitivity", "kronq_trace", "uniform"}:
        raise ValueError(
            f"Unsupported rank_allocation={rank_allocation!r}; "
            "expected 'sensitivity', 'kronq_trace' or 'uniform'."
        )

    budget_mode = quant_config.get("rank_budget", "nominal")
    if budget_mode not in {"nominal", "uniform"}:
        raise ValueError("rank_budget must be nominal or uniform")
    num_scales = 3 if quant_config['admm_type'] == 'dbf' else 2
    bits = quant_config['bits']
    print(f"Rank calculation: Bits = ({bits:.2f}), Scales: {num_scales}, Allocation: {rank_allocation}")
    min_ratio = float(quant_config.get("rank_allocation_min_uniform_ratio", 0.75))
    max_ratio = float(quant_config.get("rank_allocation_max_uniform_ratio", 1.25))
    if rank_allocation != "uniform":
        print(f"Adaptive rank bounds: {min_ratio:.2f}–{max_ratio:.2f}× uniform rank per linear")
    ranks = {}
    specs = []
    storage_layers = []
    for i, layer in enumerate(get_decoder_layers(model)):
        subset = find_layers(layer)
        for name in layers_to_analyze:
            if name in subset:
                lx = subset[name]
                a, b = lx.in_features, lx.out_features
                final_rank = uniform_rank_for_shape(a, b, bits, num_scales)
                key = f"{i}.{name}"
                storage_layers.append((key, lx))
                if rank_allocation == "uniform":
                    ranks[key] = final_rank
                    continue

                min_rank, max_rank = rank_allocation_bounds(
                    final_rank, min(a, b), min_ratio, max_ratio)
                specs.append({
                    "key": key,
                    "a": a,
                    "b": b,
                    "rank": min_rank,
                    "max_rank": max_rank,
                    "uniform_rank": final_rank,
                    "importance": rank_allocation_importance(lx, rank_allocation),
                })

    if specs:
        # Redistribute the same selected-linear weight budget in 32-rank
        # increments. More sensitive layers receive rank first; the diminishing
        # score prevents one large layer from consuming the whole budget.
        target_budget = sum(item["a"] * item["b"] * bits for item in specs)

        def storage_cost(item, rank):
            scale_elements = item["a"] + item["b"] + (rank if num_scales == 3 else 0)
            return 32*(rank*((item["a"]+31)//32) + item["b"]*((rank+31)//32)) + 16*scale_elements

        if budget_mode == "uniform":
            target_budget = sum(storage_cost(item, item["uniform_rank"]) for item in specs)
        used_budget = sum(storage_cost(item, item["rank"]) for item in specs)
        if used_budget > target_budget:
            raise ValueError("Bit budget cannot fit minimum ranks and scale vectors")
        remaining_budget = max(0.0, target_budget - used_budget)
        # Marginal utility decreases monotonically as a layer receives rank,
        # so a max-heap avoids rescanning every selected matrix for each rank
        # increment (O(steps * layers) -> O((steps + layers) log layers)).
        candidates = []

        def push_next(item_index):
            item = specs[item_index]
            step = min(32, item["max_rank"] - item["rank"])
            if step <= 0:
                return
            delta_cost = storage_cost(item, item["rank"] + step) - storage_cost(item, item["rank"])
            score = item["importance"] / ((item["rank"] + step) * delta_cost)
            heapq.heappush(candidates, (-score, item_index, step, delta_cost))

        for item_index in range(len(specs)):
            push_next(item_index)

        while candidates and remaining_budget > 0:
            _negative_score, item_index, step, delta_cost = heapq.heappop(candidates)
            if delta_cost > remaining_budget:
                continue
            item = specs[item_index]
            item["rank"] += step
            remaining_budget -= delta_cost
            push_next(item_index)
        for item in specs:
            ranks[item["key"]] = item["rank"]

    storage = estimate_weight_storage(model, [(module, ranks[key]) for key, module in storage_layers], num_scales)
    print(
        f"Weight storage estimate: selected linears {storage['selected_bpw']:.3f} bpw; "
        f"whole model {storage['whole_model_bpw']:.3f} bpw; "
        f"packed {storage['packed_weight_bytes'] / 2**30:.2f} GiB; "
        f"with BF16 factors {storage['unpacked_weight_bytes'] / 2**30:.2f} GiB; "
        f"unchanged weights {storage['remaining_weight_bytes'] / 2**30:.2f} GiB. "
        "Excludes activations, optimizer state, metadata and kernel padding."
    )
    return ranks
