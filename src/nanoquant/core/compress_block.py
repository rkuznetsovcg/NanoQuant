# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import time

import torch
import torch.nn as nn
from ..optimi import AdamW
from .admm_dbf import factorize_admm_dbf
from .admm_nq import factorize_admm_nanoquant
from ..modules.linear import NanoQuantLinear
from ..utils.utils import extract_hidden_states, find_layers, set_seed


@torch.jit.script
def fused_weighted_mse(pred, tgt, importance):
    return ((pred.float() - tgt.float()).square() * importance).sum()


def get_param_group_config(target_module, binary_lr=1e-5, scale_lr=1e-5, bias_lr=1e-5):
    """
    Get the parameter group config for the optimizer.
    """
    # create param groups
    groups = {'binary': [], 'scale': [], 'bias': []}
    # collect params
    for module in target_module.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            # get tag
            tag = getattr(param, 'optim_group', None)
            # fallback if no tag (for bias)
            if tag is None:
                if param.ndim == 1 and 'bias' in name:
                    tag = 'bias'
                else:
                    continue
            if tag in groups:
                groups[tag].append(param)
    # collect and return param groups with respective lr
    configs = []
    for key, lr in zip(groups.keys(), [binary_lr, scale_lr, bias_lr]):
        if groups[key]:
            configs.append({'params': groups[key], 'lr': lr})
    return configs


def _batch_kwargs(kwargs, batch_size, device=None, pin_memory=False, host_refs=None):
    """Broadcast singleton kwargs and move this microbatch's tensors to device."""
    def expand(value):
        if isinstance(value, torch.Tensor):
            if value.ndim and value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, *value.shape[1:])
            if (pin_memory and device is not None and torch.device(device).type == "cuda"
                    and value.device.type == "cpu"):
                if not value.is_pinned() or not value.is_contiguous():
                    # Singleton kwargs may have been expanded with stride-zero
                    # views above. pin_memory() rejects tensors with overlapping
                    # storage, so materialize a contiguous host batch first.
                    value = value.contiguous().pin_memory()
                    if host_refs is not None:
                        host_refs.append(value)
            return value.to(device=device, non_blocking=True) if device is not None else value
        elif isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        elif isinstance(value, tuple):
            return tuple(expand(item) for item in value)
        elif isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(kwargs)


class _CudaMicrobatchPrefetcher:
    """Stage one host microbatch while the current GPU batch is computing."""

    def __init__(self, inputs, targets, indices, batch_size, kwargs, device="cuda"):
        self.inputs = inputs
        self.targets = targets
        self.indices = indices
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.device = torch.device(device)
        self.offset = 0
        self.pending = None
        self.inflight_host_refs = []
        self.stream = torch.cuda.Stream(device=self.device)

    @property
    def num_batches(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def reset(self, indices):
        self.indices = indices
        self.offset = 0

    def prefetch(self):
        if self.pending is not None or self.offset >= len(self.indices):
            return
        self.inflight_host_refs = [
            (event, refs) for event, refs in self.inflight_host_refs if not event.query()
        ]
        end = min(self.offset + self.batch_size, len(self.indices))
        sample_indices = self.indices[self.offset:end]
        self.offset = end
        host_inputs = self.inputs[sample_indices].pin_memory()
        host_targets = self.targets[sample_indices].pin_memory()
        host_refs = [host_inputs, host_targets]
        with torch.cuda.stream(self.stream):
            batch_inputs = host_inputs.to(self.device, non_blocking=True)
            batch_targets = host_targets.to(self.device, non_blocking=True)
            batch_kwargs = _batch_kwargs(
                self.kwargs,
                len(sample_indices),
                self.device,
                pin_memory=True,
                host_refs=host_refs,
            )
            copy_done = torch.cuda.Event()
            copy_done.record(self.stream)
        self.pending = (sample_indices.tolist(), batch_inputs, batch_targets, batch_kwargs, copy_done)
        self.inflight_host_refs.append((copy_done, host_refs))

    def next(self):
        if self.pending is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(device=self.device)
        sample_indices, batch_inputs, batch_targets, batch_kwargs, copy_done = self.pending
        current_stream.wait_event(copy_done)

        def record(value):
            if isinstance(value, torch.Tensor):
                if value.device.type == "cuda":
                    value.record_stream(current_stream)
            elif isinstance(value, dict):
                for item in value.values():
                    record(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    record(item)

        record(batch_kwargs)
        batch_inputs.record_stream(current_stream)
        batch_targets.record_stream(current_stream)
        self.pending = None
        return sample_indices, batch_inputs, batch_targets, batch_kwargs

    def release(self):
        # Synchronize only the prefetch stream before dropping pinned source
        # buffers; this does not wait for the model's current compute stream.
        self.stream.synchronize()
        self.inflight_host_refs.clear()


@torch.enable_grad()
def tune_nonfact(block, block_inputs, block_target_outputs, importance, kwargs, quant_config,
                 excluded_module_prefixes=()):
    # set random seed
    set_seed(quant_config['seed'])
    # get hyperparams
    device = "cuda"
    numel = block_target_outputs.numel()
    batch_size = quant_config['nonfact_batch_size']
    epochs = quant_config['nonfact_epochs']
    num_samples = quant_config['num_calib_samples']
    if batch_size < 1:
        raise ValueError("nonfact_batch_size must be at least 1")
    total_steps = math.ceil(num_samples / batch_size) * epochs
    lr = quant_config['nonfact_lr']
    # collect linear weight parameters
    params = []
    excluded_module_prefixes = tuple(excluded_module_prefixes)
    for module_name, module in block.named_modules():
        if any(module_name == prefix or module_name.startswith(prefix + ".")
               for prefix in excluded_module_prefixes):
            continue
        if isinstance(module, nn.Linear):
            module.weight.requires_grad = True
            params.append(module.weight)
    assert len(params) > 0, "No linear layers found in the block"
    # get optimizer and lr_scheduler
    optimizer = AdamW(params, lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4 * lr)
    prefetcher = _CudaMicrobatchPrefetcher(
        block_inputs, block_target_outputs, torch.arange(num_samples), batch_size, kwargs, device
    )
    plateau_tolerance = float(quant_config.get('nonfact_plateau_tolerance', 0.0))
    plateau_min_epochs = max(1, int(quant_config.get('nonfact_plateau_min_epochs', 3)))
    plateau_patience = max(1, int(quant_config.get('nonfact_plateau_patience', 2)))
    if plateau_tolerance < 0:
        raise ValueError("nonfact_plateau_tolerance cannot be negative")
    previous_loss = None
    plateau_epochs = 0
    # optimization loop
    for epoch in range(epochs):
        data_idx = torch.randperm(num_samples, device="cpu", dtype=torch.long)
        epoch_loss = torch.zeros(1, device=device)
        prefetcher.reset(data_idx)
        prefetcher.prefetch()
        for _ in range(prefetcher.num_batches):
            sample_indices, batch_inputs, batch_targets, batch_kwargs = prefetcher.next()
            block._nanoquant_active_indices = sample_indices
            # One batched loss produces the same summed gradient as accumulating
            # the former per-sample backwards before each optimizer step.
            y = extract_hidden_states(block(batch_inputs, **batch_kwargs))
            loss = fused_weighted_mse(y, batch_targets, importance)
            prefetcher.prefetch()
            # backprop
            (loss / batch_size).backward()
            # gradient update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            # update epoch loss
            epoch_loss += loss.detach()
        prefetcher.release()
        # log loss
        current_loss = (epoch_loss / numel).item()
        print(f"\t\t(Epoch {epoch+1:02d}/{epochs:02d}) Block Loss: {current_loss:.4e}")
        if previous_loss is not None and plateau_tolerance > 0 and epoch+1 >= plateau_min_epochs:
            improvement = (previous_loss-current_loss) / max(abs(previous_loss), 1e-12)
            plateau_epochs = plateau_epochs+1 if 0 <= improvement < plateau_tolerance else 0
            if plateau_epochs >= plateau_patience:
                print(f"\t\tTuneFP plateau: stopped after {epoch+1} epochs")
                break
        previous_loss = current_loss

    for p in params:
        p.requires_grad = False
    block.zero_grad(set_to_none=True)
    del params, optimizer


@torch.no_grad()
def factorize_and_replace(layer, name, rank, quant_config):
    """
    Factorizes and replaces a submodule with a quantized version (NanoQuantLinear).
    """
    set_seed(quant_config['seed'])
    lx_orig = find_layers(layer)[name]
    # ADMM treats W as read-only, so keep a detached view instead of cloning
    # the full matrix several times before factorization.
    weight_for_factorization = lx_orig.weight.detach()
    new_module = lx_orig
    device = "cuda"

    # --- 1. Iterative Factorization and Module Conversion ---
    W_res = weight_for_factorization

    admm_time = time.time()
    compute_diagnostic = bool(quant_config.get('log_reconstruction_error', False))
    # Select factorization function based on type
    is_transpose = W_res.shape[0] < W_res.shape[1]
    if quant_config['admm_type'] == 'dbf':
        factor_results = factorize_admm_dbf(W_res.to(device), lx_orig.i_norm.to(device), lx_orig.o_norm.to(device),
                                            mid_rank=rank, iters=quant_config['admm_outer_iters'],
                                            is_transpose=is_transpose,
                                            reg=quant_config.get('admm_reg', 3e-2),
                                            warm_start_iters=quant_config.get('admm_warm_start_iters', 2),
                                            early_stop=quant_config.get('admm_early_stop', True),
                                            min_outer_iters=quant_config.get('admm_min_outer_iters', 120),
                                            check_interval=quant_config.get('admm_check_interval', 10),
                                            convergence_tolerance=quant_config.get('admm_convergence_tolerance', 2e-3),
                                            stable_sign_tolerance=quant_config.get('admm_stable_sign_tolerance', 1e-3),
                                            rho_stop_threshold=quant_config.get('admm_rho_stop_threshold', 0.85),
                                            compute_diagnostic=compute_diagnostic)
    elif quant_config['admm_type'] == 'nanoquant':
        factor_results = factorize_admm_nanoquant(
            W_res.to(device), lx_orig.i_norm.to(device), lx_orig.o_norm.to(device), mid_rank=rank,
            outer_iters=quant_config['admm_outer_iters'], inner_iters=quant_config['admm_inner_iters'],
            reg=quant_config.get('admm_reg', 3e-2),
            is_transpose=is_transpose, rho_scheduler=quant_config['admm_penalty_scheduler'],
            print_admm_steps=quant_config['admm_print_steps'],
            warm_start_iters=quant_config.get('admm_warm_start_iters', 2),
            early_stop=quant_config.get('admm_early_stop', True),
            min_outer_iters=quant_config.get('admm_min_outer_iters', 120),
            check_interval=quant_config.get('admm_check_interval', 10),
            convergence_tolerance=quant_config.get('admm_convergence_tolerance', 2e-3),
            stable_sign_tolerance=quant_config.get('admm_stable_sign_tolerance', 1e-3),
            rho_stop_threshold=quant_config.get('admm_rho_stop_threshold', 0.85),
            compute_diagnostic=compute_diagnostic,
            input_metric=getattr(lx_orig, "_nanoquant_input_metric", None))
    else:
        raise ValueError("admm_type must be nanoquant or dbf")
    if hasattr(lx_orig, "_nanoquant_input_metric"):
        del lx_orig._nanoquant_input_metric
    admm_time = time.time() - admm_time

    # Dense reconstruction is only built for this optional diagnostic. Move
    # it on the weight device for comparison and release it
    # before converting the module.
    if compute_diagnostic:
        reconstructed_weight = factor_results.pop("W_final").to(weight_for_factorization.device)
        squared_error = (reconstructed_weight.float() - weight_for_factorization.float()).square().sum()
        diagnostic_sums = torch.stack((
            squared_error,
            weight_for_factorization.float().square().sum(),
        )).tolist()
        recon_error_raw, original_norm_sq = diagnostic_sums
        per_el_error = recon_error_raw / W_res.numel()
        if original_norm_sq > 0:
            normalized_error = recon_error_raw / original_norm_sq
            print(
                f"\t\tADMM weight recon error: raw={recon_error_raw:.4f}, norm={normalized_error:.4f}, per_el={per_el_error:.4e}, ADMM time={admm_time:.2f}s"
            )
        del reconstructed_weight

    # Assemble the factors consumed by NanoQuantLinear.
    final_factor_results = argparse.Namespace(**factor_results)

    # Replace module class and convert
    do_tuning = quant_config['tune_fact']
    new_module.__class__ = NanoQuantLinear
    new_module.__quant_convert__(do_train=do_tuning, rank=rank, factor_results=final_factor_results)

    # --- 2. Finalization ---
    if not do_tuning and new_module.bias is not None and hasattr(lx_orig, 'bias') and lx_orig.bias is not None:
        new_module.bias.data.copy_(lx_orig.bias.data)

    del W_res, weight_for_factorization, lx_orig

    return new_module, final_factor_results


@torch.enable_grad()
def tune_fact(block, target_linears, block_inputs, block_target_outputs, importance, kwargs, quant_config):
    # set random seed
    set_seed(quant_config['seed'])
    # get hyperparams
    device = "cuda"
    numel = block_target_outputs.numel()
    batch_size = quant_config['fact_batch_size']
    epochs = quant_config['fact_epochs']
    num_samples = quant_config['num_calib_samples']
    if batch_size < 1:
        raise ValueError("fact_batch_size must be at least 1")
    total_steps = math.ceil(num_samples / batch_size) * epochs
    binary_lr = quant_config['fact_binary_lr']
    scale_lr = quant_config['fact_scale_lr']
    bias_lr = quant_config['fact_bias_lr']

    if not isinstance(target_linears, (tuple, list)):
        target_linears = [target_linears]
    # The final MLP projection is followed only by the block residual add.
    # Its output scale has a closed-form least-squares update for the same
    # block reconstruction loss, so exclude it from Adam and accumulate the
    # sufficient statistics during the normal training forwards.
    analytic_scale_targets = [
        module for module in target_linears
        if getattr(module, "_nanoquant_layer_path", "").endswith("mlp.down_proj")
        and hasattr(module, "scale_post")
    ]
    analytic_captures = {}
    analytic_handles = []
    named_modules = dict(block.named_modules())

    def first_tensor(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        return None

    for module in analytic_scale_targets:
        layer_path = module._nanoquant_layer_path
        mlp_module = named_modules.get(layer_path.rpartition(".")[0])
        if mlp_module is None:
            continue
        module.scale_post.requires_grad_(False)
        analytic_captures[module] = {"projection": None, "mlp": None}
        analytic_handles.append(module.register_forward_hook(
            lambda _mod, _inputs, output, target=module: analytic_captures[target].__setitem__(
                "projection", first_tensor(output).detach() if first_tensor(output) is not None else None)))
        analytic_handles.append(mlp_module.register_forward_hook(
            lambda _mod, _inputs, output, target=module: analytic_captures[target].__setitem__(
                "mlp", first_tensor(output).detach() if first_tensor(output) is not None else None)))

    def accumulate_analytic_scale_stats(numerators, denominators, block_output, target_output):
        for module, capture in analytic_captures.items():
            projection_output = capture["projection"]
            mlp_output = capture["mlp"]
            if projection_output is None or mlp_output is None:
                continue
            scale = module.scale_post.detach().float().reshape(1, 1, -1)
            signed_floor = torch.where(scale >= 0, torch.full_like(scale, 1e-8), torch.full_like(scale, -1e-8))
            safe_scale = torch.where(scale.abs() >= 1e-8, scale, signed_floor)
            bias = module.bias.detach().float().reshape(1, 1, -1) if module.bias is not None else 0.0
            core_output = (projection_output.float() - bias) / safe_scale
            desired_branch = target_output.float() - block_output.detach().float() + mlp_output.float()
            desired_core = desired_branch - bias
            numerators[module].add_((core_output * desired_core).sum(dim=(0, 1)).view_as(module.scale_post))
            denominators[module].add_(core_output.square().sum(dim=(0, 1)).view_as(module.scale_post))

    # get optimizer and lr_scheduler
    param_config = get_param_group_config(block, binary_lr=binary_lr, scale_lr=scale_lr, bias_lr=bias_lr)
    optimizer = AdamW(param_config, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4 * scale_lr)
    prefetcher = _CudaMicrobatchPrefetcher(
        block_inputs, block_target_outputs, torch.arange(num_samples), batch_size, kwargs, device
    )
    # optimization loop
    for epoch in range(epochs):
        data_idx = torch.randperm(num_samples, device="cpu", dtype=torch.long)
        epoch_loss = torch.zeros(1, device=device)
        analytic_numerators = {
            module: torch.zeros_like(module.scale_post, dtype=torch.float32) for module in analytic_captures
        }
        analytic_denominators = {
            module: torch.zeros_like(module.scale_post, dtype=torch.float32) for module in analytic_captures
        }
        prefetcher.reset(data_idx)
        prefetcher.prefetch()
        for _ in range(prefetcher.num_batches):
            sample_indices, batch_inputs, batch_targets, batch_kwargs = prefetcher.next()
            block._nanoquant_active_indices = sample_indices
            y = extract_hidden_states(block(batch_inputs, **batch_kwargs))
            loss = fused_weighted_mse(y, batch_targets, importance)

            accumulate_analytic_scale_stats(
                analytic_numerators,
                analytic_denominators,
                y,
                batch_targets,
            )
            prefetcher.prefetch()
            # backprop
            (loss / batch_size).backward()
            # gradient update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            # update epoch loss
            epoch_loss += loss.detach()
        prefetcher.release()
        # log loss
        print(f"\t\t(Epoch {epoch+1:02d}/{epochs:02d}) Block Loss: {(epoch_loss / numel).item():.4e}")

        for module in analytic_captures:
            denominator = analytic_denominators[module]
            updated_scale = analytic_numerators[module] / denominator.clamp_min(1e-12)
            module.scale_post.data.copy_(torch.where(denominator > 1e-12, updated_scale, module.scale_post.data.float()))

    if analytic_captures:
        # Recompute the closed-form optimum once at the final factor values;
        # the per-epoch updates above already used sufficient statistics from
        # the regular training forwards.
        final_numerators = {module: torch.zeros_like(module.scale_post, dtype=torch.float32)
                            for module in analytic_captures}
        final_denominators = {module: torch.zeros_like(module.scale_post, dtype=torch.float32)
                              for module in analytic_captures}
        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                indices = list(range(start, end))
                block._nanoquant_active_indices = indices
                device_indices = torch.tensor(indices, dtype=torch.long)
                batch_inputs = block_inputs[device_indices].to(device, non_blocking=True)
                batch_targets = block_target_outputs[device_indices].to(device, non_blocking=True)
                final_output = extract_hidden_states(block(batch_inputs, **_batch_kwargs(kwargs, end - start, device)))
                accumulate_analytic_scale_stats(
                    final_numerators,
                    final_denominators,
                    final_output,
                    batch_targets,
                )
        for module in analytic_captures:
            denominator = final_denominators[module]
            updated_scale = final_numerators[module] / denominator.clamp_min(1e-12)
            module.scale_post.data.copy_(torch.where(denominator > 1e-12, updated_scale, module.scale_post.data.float()))

    for handle in analytic_handles:
        handle.remove()
    # Harden every factorized layer that participated in this grouped update.
    for target_linear in target_linears:
        target_linear.finalize()
    del param_config, optimizer
