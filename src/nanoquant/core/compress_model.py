# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import random
import json
import fnmatch
from pathlib import Path

import torch
from ..optimi import AdamW
from ..core.compress_block import (factorize_and_replace, tune_fact, tune_nonfact)
from .linearized_block import fit_linearized_decoder_block
from ..modules.linear import NanoQuantLinear
from .reconstruction_plan import reconstruction_groups, refresh_input_stats, shared_input_key, validate_reconstruction_config, stage_quant_config
from .resume import BlockCheckpoint, _atomic_save, content_key
from .rank_probe import refine_block_ranks
from ..utils.eval_utils import evaluate_ppl_after_block
from ..utils.load_utils import cache_inputs_and_kwargs, load_model
from ..utils.utils import (calculate_ranks, cleanup_memory, extract_hidden_states, find_layers, get_decoder_layers,
                           get_layers_to_factorize, QWEN3_5_MODEL_TYPES, set_seed)
from tqdm import tqdm, trange


def _kwargs_for_batch(kwargs, batch_size, device=None):
    """Expand cached singleton kwargs and move the current batch to device."""
    def expand(value):
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, *value.shape[1:])
            return value.to(device=device, non_blocking=True) if device is not None else value
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(expand(item) for item in value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(kwargs)


def _tree_to_cpu(value):
    """Copy each captured batch tensor once, before splitting it into views."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    return value


def _tree_cpu_sample(value, sample_index, batch_size):
    """Take sample views from an already transferred host output tree."""
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == batch_size:
            return value[sample_index:sample_index + 1]
        return value
    if isinstance(value, tuple):
        return tuple(_tree_cpu_sample(item, sample_index, batch_size) for item in value)
    if isinstance(value, list):
        return [_tree_cpu_sample(item, sample_index, batch_size) for item in value]
    if isinstance(value, dict):
        return {key: _tree_cpu_sample(item, sample_index, batch_size) for key, item in value.items()}
    return value


def _tree_stack_cached(values, device):
    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim and first.shape[0] == 1:
            # Assemble on the host so one tensor crosses PCIe per microbatch.
            return torch.cat(values, dim=0).to(device=device, non_blocking=True)
        return first.to(device=device, non_blocking=True)
    if isinstance(first, tuple):
        return tuple(_tree_stack_cached([value[i] for value in values], device) for i in range(len(first)))
    if isinstance(first, list):
        return [_tree_stack_cached([value[i] for value in values], device) for i in range(len(first))]
    if isinstance(first, dict):
        return {key: _tree_stack_cached([value[key] for value in values], device) for key in first}
    return first


class _GroupOutputCaptured(Exception):
    """Stop a cache-only block forward immediately after the requested group."""


def _capture_and_patch_group(block, group_module, block_inputs, kwargs, batch_size, active_patches):
    """Cache a frozen group's outputs and replace its forward during later tuning."""
    cached = [None] * block_inputs.shape[0]
    active_indices = []

    def capture(_module, _inputs, output):
        host_output = _tree_to_cpu(output)
        for offset, sample_index in enumerate(active_indices):
            cached[sample_index] = _tree_cpu_sample(host_output, offset, len(active_indices))
        raise _GroupOutputCaptured

    handle = group_module.register_forward_hook(capture)
    block_device = next(block.parameters()).device
    try:
        with torch.no_grad():
            for start in range(0, block_inputs.shape[0], batch_size):
                end = min(start + batch_size, block_inputs.shape[0])
                active_indices[:] = list(range(start, end))
                block._nanoquant_active_indices = active_indices.copy()
                batch_inputs = block_inputs[start:end].to(block_device, non_blocking=True)
                try:
                    block(batch_inputs, **_kwargs_for_batch(kwargs, end - start, block_device))
                except _GroupOutputCaptured:
                    pass
    finally:
        handle.remove()

    if any(value is None for value in cached):
        raise RuntimeError("Could not cache every sample output for a frozen NanoQuant block group")

    original_forward = group_module.forward

    def cached_forward(*args, **call_kwargs):
        indices = getattr(block, "_nanoquant_active_indices", None)
        if indices is None:
            return original_forward(*args, **call_kwargs)
        hidden_states = args[0] if args and isinstance(args[0], torch.Tensor) else call_kwargs.get("hidden_states")
        device = hidden_states.device if isinstance(hidden_states, torch.Tensor) else block_device
        return _tree_stack_cached([cached[index] for index in indices], device)

    group_module.forward = cached_forward
    active_patches.append((group_module, original_forward))


class _ChunkedLinearKDLoss(torch.autograd.Function):
    """Full-vocabulary teacher cross-entropy without materializing logits."""

    @staticmethod
    def _logits(hidden, weight, bias, start, end, logit_scale, softcap):
        bias_chunk = bias[start:end] if bias is not None else None
        logits = torch.nn.functional.linear(hidden, weight[start:end], bias_chunk).float() * logit_scale
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    @staticmethod
    def forward(ctx, student_hidden, teacher_hidden, weight, bias, mask, temperature, chunk_size, logit_scale,
                softcap):
        teacher_hidden = teacher_hidden.to(device=student_hidden.device, dtype=student_hidden.dtype, non_blocking=True)
        vocab_size = weight.shape[0]
        student_lse = None
        teacher_lse = None
        cross_entropy = None
        # Merge normalized chunk losses online. This needs one projection of
        # each hidden state per vocabulary tile instead of a separate pass for
        # the normalizers followed by a second projection pass for the loss.
        # Using local CE plus log-normalizer increments avoids cancellation
        # between two large terms (global logsumexp - expected student logit).
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            student_piece = _ChunkedLinearKDLoss._logits(
                student_hidden, weight, bias, start, end, logit_scale, softcap) / temperature
            teacher_piece = _ChunkedLinearKDLoss._logits(
                teacher_hidden, weight, bias, start, end, logit_scale, softcap) / temperature
            student_part = torch.logsumexp(student_piece, dim=-1)
            teacher_part = torch.logsumexp(teacher_piece, dim=-1)
            teacher_probability = (teacher_piece - teacher_part.unsqueeze(-1)).exp()
            chunk_ce = (teacher_probability * (student_part.unsqueeze(-1) - student_piece)).sum(dim=-1)
            if student_lse is None:
                student_lse, teacher_lse, cross_entropy = student_part, teacher_part, chunk_ce
            else:
                merged_student = torch.logaddexp(student_lse, student_part)
                merged_teacher = torch.logaddexp(teacher_lse, teacher_part)
                old_mass = (teacher_lse - merged_teacher).exp()
                chunk_mass = (teacher_part - merged_teacher).exp()
                cross_entropy = (
                    old_mass * (cross_entropy + (merged_student - student_lse))
                    + chunk_mass * (chunk_ce + (merged_student - student_part))
                )
                student_lse, teacher_lse = merged_student, merged_teacher

        flat_mask = mask.reshape(-1).to(device=student_hidden.device, dtype=torch.float32)
        valid_tokens = flat_mask.sum().clamp_min(1.0)
        loss = (cross_entropy.reshape(-1) * flat_mask).sum() / valid_tokens

        ctx.temperature = float(temperature)
        ctx.chunk_size = int(chunk_size)
        ctx.logit_scale = float(logit_scale)
        ctx.softcap = float(softcap)
        ctx.save_for_backward(student_hidden, teacher_hidden, weight, bias, flat_mask, student_lse, teacher_lse,
                              valid_tokens)
        return loss * (temperature ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        (student_hidden, teacher_hidden, weight, bias, flat_mask, student_lse, teacher_lse,
         valid_tokens) = ctx.saved_tensors
        vocab_size = weight.shape[0]
        grad_hidden = torch.zeros_like(student_hidden, dtype=torch.float32)
        flat_grad = grad_hidden.reshape(-1, grad_hidden.shape[-1])
        token_scale = (flat_mask * grad_output.float() * ctx.temperature / valid_tokens).unsqueeze(-1)

        for start in range(0, vocab_size, ctx.chunk_size):
            end = min(start + ctx.chunk_size, vocab_size)
            student_piece = _ChunkedLinearKDLoss._logits(
                student_hidden, weight, bias, start, end, ctx.logit_scale, ctx.softcap) / ctx.temperature
            teacher_piece = _ChunkedLinearKDLoss._logits(
                teacher_hidden, weight, bias, start, end, ctx.logit_scale, ctx.softcap) / ctx.temperature
            student_probability = (student_piece - student_lse.unsqueeze(-1)).exp()
            teacher_probability = (teacher_piece - teacher_lse.unsqueeze(-1)).exp()
            grad_logits = (student_probability - teacher_probability).reshape(-1, end - start)

            if ctx.softcap > 0:
                # Recover tanh(raw / softcap) from the already-computed output,
                # avoiding a second lm_head GEMM for the derivative.
                softcap_output = student_piece * ctx.temperature
                softcap_derivative = 1.0 - (softcap_output / ctx.softcap).square()
                grad_logits = grad_logits * softcap_derivative.reshape_as(grad_logits)
            grad_logits = grad_logits * token_scale * ctx.logit_scale
            flat_grad.addmm_(grad_logits, weight[start:end].float())

        return grad_hidden.to(student_hidden.dtype), None, None, None, None, None, None, None, None


@torch.no_grad()
def compress_block_recon(model, fp_model, dataloader, quant_config):
    """
    Compresses a model using a functional, sequential tune-then-factorize approach.
    """
    validate_reconstruction_config(quant_config)
    # set seed
    set_seed(quant_config['seed'])
    # get device
    dev = "cuda"
    # adjust model configs
    model.cpu()
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = False
    # get relevant blocks/layers
    q_blocks = get_decoder_layers(model)
    linearize_index = quant_config.get("linearize_block_index")
    if linearize_index is not None:
        linearize_index = int(linearize_index)
        if model.config.model_type not in QWEN3_5_MODEL_TYPES:
            raise ValueError("The linearized-block pilot currently supports Qwen3.5/Qwen3.8 decoder blocks only")
        if linearize_index >= len(q_blocks):
            raise ValueError(f"linearize_block_index={linearize_index} is outside the {len(q_blocks)} decoder blocks")
    fp_blocks = get_decoder_layers(fp_model) if fp_model is not None else q_blocks
    reference_model = fp_model if fp_model is not None else model
    if fp_model is not None:
        fp_model.gradient_checkpointing_disable()
        fp_model.eval()
        fp_model.config.use_cache = False
    layers_to_factorize = get_layers_to_factorize(model.config.model_type)
    checkpoint = BlockCheckpoint(model, dataloader, quant_config, fp_model)
    snapshot = checkpoint.restore(model)
    if snapshot is None:
        admm_ranks = calculate_ranks(model, layers_to_factorize, quant_config)
        original_inputs, kwargs = cache_inputs_and_kwargs(reference_model, dataloader, dev)
        kwargs = _tree_to_cpu(kwargs)
        kwargs['use_cache'] = False
        if 'past_key_value' in kwargs:
            kwargs['past_key_value'] = None
        compressed_inputs = original_inputs.clone().detach().cpu()
        start_block = 0
        checkpoint.initialize(model, admm_ranks, original_inputs, compressed_inputs, kwargs)
    else:
        admm_ranks = snapshot['ranks']
        original_inputs, compressed_inputs = snapshot['original_inputs'], snapshot['compressed_inputs']
        kwargs, start_block = snapshot['kwargs'], snapshot['completed_blocks']
        del snapshot
    cleanup_memory()
    if quant_config.get('correlation_block_size', 0) and quant_config['admm_type'] != 'nanoquant':
        raise ValueError("Input correlations require nanoquant ADMM")
    if quant_config.get('correlation_block_size', 0) and not quant_config.get('refresh_input_stats', True):
        raise ValueError("Input correlations require fresh input statistics")

    for i in trange(start_block, len(q_blocks), desc="Compressing Layers"):
        # Capture this block's reference outputs before changing its weights.
        q_block = q_blocks[i].to(dev)
        fp_block = fp_blocks[i].to(dev)
        # The block optimizers explicitly enable only the parameters they
        # update. Freeze norms and other unused parameters up front so their
        # gradients are not built or retained by block-level autograd.
        for parameter in q_block.parameters():
            parameter.requires_grad_(False)
        # Run reference and compressed block forwards in microbatches to reduce
        # Python and host-device transfer overhead.
        with torch.no_grad():
            # Keep the full sample cache in host memory; only the active
            # reference/propagation microbatch is transferred to the GPU.
            target_outputs = torch.empty_like(original_inputs)
            io_batch_size = int(quant_config.get('block_io_batch_size', 1))
            if io_batch_size < 1:
                raise ValueError("block_io_batch_size must be at least 1")
            for start in range(0, quant_config['num_calib_samples'], io_batch_size):
                end = min(start + io_batch_size, quant_config['num_calib_samples'])
                batch_input = original_inputs[start:end].to(dev, non_blocking=True)
                batch_kwargs = _kwargs_for_batch(kwargs, end - start, dev)
                batch_output = extract_hidden_states(fp_block(batch_input, **batch_kwargs))
                target_outputs[start:end] = batch_output.cpu().detach()
        # get qblock inputs
        # Tuning only reads this cache. It is updated after all groups finish.
        tuning_inputs = compressed_inputs
        if i == linearize_index:
            block_kind = "Gated DeltaNet" if hasattr(q_block, "linear_attn") else (
                "full attention" if hasattr(q_block, "self_attn") else type(q_block).__name__
            )
            linearized_block, diagnostics = fit_linearized_decoder_block(
                tuning_inputs,
                target_outputs,
                max_tokens=int(quant_config.get("linearize_max_tokens", 16384)),
                ridge=float(quant_config.get("linearize_ridge", 1e-4)),
                chunk_tokens=int(quant_config.get("linearize_chunk_tokens", 1024)),
                device=dev,
            )
            print(
                f"Linearized-block pilot: replacing block {i}/{len(q_blocks)-1} ({block_kind}) with one affine map; "
                f"fit tokens={diagnostics['tokens']}, calibration relative RMSE={diagnostics['relative_rmse']:.5f}, "
                f"parameters={diagnostics['parameters']:,}. Skipping TuneFP and ADMM for this block."
            )
            q_block = linearized_block
            q_blocks[i] = q_block
            original_inputs = target_outputs
            with torch.no_grad():
                for start in range(0, quant_config['num_calib_samples'], io_batch_size):
                    end = min(start + io_batch_size, quant_config['num_calib_samples'])
                    batch_input = compressed_inputs[start:end].to(dev, non_blocking=True)
                    batch_kwargs = _kwargs_for_batch(kwargs, end - start, dev)
                    batch_output = extract_hidden_states(q_block(batch_input, **batch_kwargs))
                    compressed_inputs[start:end] = batch_output.cpu().detach()
            if fp_model is not None:
                fp_blocks[i] = fp_block.cpu()
            q_blocks[i] = q_block.cpu()
            checkpoint.save(q_blocks[i], i + 1, admm_ranks, original_inputs, compressed_inputs, kwargs)

            del q_block, fp_block, target_outputs, tuning_inputs, batch_input, batch_output, batch_kwargs
            cleanup_memory()

            if quant_config.get('eval_after_each_block', False):
                test_ppl = evaluate_ppl_after_block(model, model_name=quant_config['model_id'], dev=dev)
                print(f"\t\tBlock {i}: Test Data PPL        = {test_ppl:.3f}")
            continue

        # get all linear layers
        sublayers = find_layers(q_block)
        # get importance
        # Try to get importance from common layer names, fall back to uniform
        importance_layer = sublayers.get('mlp.down_proj', sublayers.get('fc2', None))
        if importance_layer is None:
            # Fallback to uniform importance if expected layer not found
            importance = torch.ones(model.config.hidden_size, device=dev)
        elif not hasattr(importance_layer, 'o_norm'):
            # Fallback if o_norm attribute missing
            importance = torch.ones(model.config.hidden_size, device=dev)
        else:
            importance = importance_layer.o_norm.to(dev)
        names = [name for name in layers_to_factorize if name in sublayers]
        group_items = reconstruction_groups(names, quant_config.get('tune_schedule', 'sequential'))
        refine_block_ranks(q_block, names, i, admm_ranks, quant_config)
        active_patches = []
        named_modules = dict(q_block.named_modules())
        completed_groups = []
        remaining = set(names)
        input_stats_cache = {}
        try:
            for group_index, group_layers in enumerate(group_items):
                group_name = ", ".join(group_layers)
                text_config = getattr(model.config, 'text_config', model.config)
                stage_config = stage_quant_config(quant_config, group_layers, text_config.hidden_size)
                if quant_config['tune_nonfact']:
                    print(f"\t(1/3) Block {i+1}/{len(q_blocks)}, {group_name} | Tuning Non-Factorized Weights...")
                    tune_nonfact(q_block, tuning_inputs, target_outputs, importance, kwargs, stage_config,
                                 excluded_module_prefixes=completed_groups)

                if quant_config.get('refresh_input_stats', True):
                    patterns = tuple(part.strip() for part in quant_config.get('correlation_layers', '').split(',')
                                     if part.strip())
                    correlated = [name for name in group_layers if any(fnmatch.fnmatchcase(name, pattern)
                                                                      for pattern in patterns)]
                    pending = []
                    for name in group_layers:
                        key = shared_input_key(name)
                        cached = input_stats_cache.get(key) if key is not None else None
                        needs_metric = bool(quant_config.get('correlation_block_size', 0) and name in correlated)
                        if cached is None or (needs_metric and cached[1] is None):
                            pending.append(name)
                        else:
                            sublayers[name].i_norm = cached[0]
                            sublayers[name]._nanoquant_input_metric = cached[1] if needs_metric else None
                    if pending:
                        correlation_size = quant_config.get('correlation_block_size', 0) if any(
                            name in correlated for name in pending) else 0
                        refresh_input_stats(q_block, [sublayers[name] for name in pending], tuning_inputs,
                                            kwargs, io_batch_size, quant_config['calib_shrinkage'],
                                            quant_config['calib_strategy'], correlation_size)
                        for name in pending:
                            module = sublayers[name]
                            key = shared_input_key(name)
                            if key is not None:
                                input_stats_cache[key] = (module.i_norm, module._nanoquant_input_metric)
                            if name not in correlated:
                                module._nanoquant_input_metric = None
                    # Release expensive channel-block eigenstates as soon as all
                    # members consuming this exact input have been initialized.
                    future = remaining.difference(group_layers)
                    for key in list(input_stats_cache):
                        if not any(shared_input_key(name) == key for name in future):
                            del input_stats_cache[key]

                group_linears = []
                for name in group_layers:
                    print(f"\t(2/3) Block {i+1}/{len(q_blocks)}, {name} | Initialization via ADMM...")
                    curr_rank = admm_ranks.get(f"{i}.{name}")
                    nano_linear, final_factor_results = factorize_and_replace(q_block, name, curr_rank, quant_config)
                    nano_linear._nanoquant_layer_path = name
                    group_linears.append(nano_linear)
                    del final_factor_results
                if quant_config['tune_fact']:
                    print(f"\t(3/3) Block {i+1}/{len(q_blocks)}, {group_name} | Tuning Factorized Weights...")
                    tune_fact(q_block, group_linears, tuning_inputs, target_outputs, importance, kwargs, stage_config)
                remaining.difference_update(group_layers)
                # Cache a parent only when ALL its selected projections are done;
                # caching it after q_proj would invalidate k/v/o inputs and gradients.
                for parent in dict.fromkeys(name.rpartition('.')[0] for name in group_layers):
                    if (parent and parent not in completed_groups
                            and not any(name.startswith(parent+'.') for name in remaining)):
                        completed_groups.append(parent)
                        group_module = named_modules.get(parent)
                        if remaining and group_module is not None:
                            _capture_and_patch_group(q_block, group_module, tuning_inputs, kwargs,
                                                     io_batch_size, active_patches)

        finally:
            q_block._nanoquant_active_indices = None
            for patched_module, original_forward in reversed(active_patches):
                patched_module.forward = original_forward
        if fp_model is not None:
            fp_blocks[i] = fp_block.cpu()
        # fp_blocks[i+1] input = fp_blocks[i] output
        original_inputs = target_outputs

        # use qblock[i] outputs for qblocks[i+1] inputs
        with torch.no_grad():
            for start in range(0, quant_config['num_calib_samples'], io_batch_size):
                end = min(start + io_batch_size, quant_config['num_calib_samples'])
                batch_input = compressed_inputs[start:end].to(dev, non_blocking=True)
                batch_kwargs = _kwargs_for_batch(kwargs, end - start, dev)
                batch_output = extract_hidden_states(q_block(batch_input, **batch_kwargs))
                compressed_inputs[start:end] = batch_output.cpu().detach()
        q_blocks[i] = q_block.cpu()
        checkpoint.save(q_blocks[i], i+1, admm_ranks, original_inputs, compressed_inputs, kwargs)

        del q_block, fp_block, target_outputs, tuning_inputs, sublayers, importance, batch_input, batch_output, active_patches
        cleanup_memory()

        if quant_config.get('eval_after_each_block', False):
            test_ppl = evaluate_ppl_after_block(model, model_name=quant_config['model_id'], dev=dev)
            print(f"\t\tBlock {i}: Test Data PPL        = {test_ppl:.3f}")

    return model


def compress_model_recon(model, fp_model, dataloader, quant_config, dev="cuda"):
    """
    Use knowledge distillation to globally tune scales.

    Passing fp_model=None loads a temporary teacher owned by this function,
    whose weights are released immediately after hidden-state caching.
    """
    @torch.no_grad()
    def _compute_teacher_hidden_cache(fp_model, dataloader, num_samples, batch_size, dev="cuda"):
        """Cache final teacher hidden states; logits are regenerated in chunks."""
        if hasattr(fp_model, "config"):
            fp_model.config.use_cache = False
        fp_model.eval().to(dev)
        teacher_hidden_cache = {}
        selected = dataloader[:num_samples].to(device=dev, non_blocking=True)
        for start in tqdm(range(0, num_samples, batch_size), desc="Caching teacher hidden states"):
            end = min(start + batch_size, num_samples)
            outputs = fp_model.model(selected[start:end])
            hidden = extract_hidden_states(outputs)
            host_hidden = hidden.detach().cpu()
            for offset, sample_index in enumerate(range(start, end)):
                teacher_hidden_cache[sample_index] = host_hidden[offset:offset + 1]

        return teacher_hidden_cache

    # set seed
    set_seed(quant_config['seed'])
    model.cpu()

    available_samples = len(dataloader)
    if available_samples < 1:
        raise ValueError("Model-level KD requires at least one calibration sample")
    model_kd_num_samples = min(
        available_samples, max(1, int(quant_config.get('model_kd_num_samples', available_samples)))
    )
    model_kd_batch_size = max(1, int(quant_config.get('model_kd_batch_size', 1)))

    owns_teacher = fp_model is None
    teacher_cache_path = None
    if owns_teacher and quant_config.get('resume_dir'):
        directory = Path(quant_config['resume_dir'])
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['completed_blocks'] == len(get_decoder_layers(model)):
                head_tensors = [model.lm_head.weight]
                if getattr(model.lm_head, 'bias', None) is not None:
                    head_tensors.append(model.lm_head.bias)
                key = content_key([dataloader[:model_kd_num_samples], *head_tensors],
                                  {'version': 1, 'signature': manifest['signature'],
                                   'batch_size': model_kd_batch_size, 'device': dev})
                teacher_cache_path = directory / ('teacher-hidden-'+key+'.pt')
    if teacher_cache_path is not None and teacher_cache_path.exists():
        teacher_hidden_cache = torch.load(teacher_cache_path, map_location='cpu', weights_only=True)
        print("Reusing committed teacher hidden states before scale KD")
    else:
        if owns_teacher:
            fp_model = load_model(
                quant_config['model_id'], quant_config['seqlen'],
                device_map=quant_config.get('device_map', 'cpu'),
                require_fast_linear_attention=quant_config.get('require_fast_linear_attention', False),
                attn_implementation=quant_config.get('attn_implementation', 'auto'),
                revision=quant_config.get('model_revision'))
        fp_model.eval()
        teacher_hidden_cache = _compute_teacher_hidden_cache(
            fp_model=fp_model, dataloader=dataloader, num_samples=model_kd_num_samples,
            batch_size=model_kd_batch_size, dev=dev)
        if teacher_cache_path is not None:
            _atomic_save(teacher_hidden_cache, teacher_cache_path)
        if not owns_teacher:
            fp_model.cpu()
        del fp_model
        cleanup_memory(verbose=True)
    # Teacher loading/caching can consume RNG even in eval mode. A cache hit
    # must lead to the same KD shuffle/checkpoint random sequence as a miss.
    set_seed(quant_config['seed'])

    # Identify Pad Token for Masking
    pad_token_id = -100
    if hasattr(model, "config"):
        model.config.use_cache = False  # Disable KV cache for training
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is not None:
            pad_token_id = model.config.pad_token_id

    model.train()
    # Enable Gradient Checkpointing to save VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
    # Re-entrant checkpointing needs its input activation to require gradients
    # when the base weights are frozen and only internal scales are trainable.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(dev)

    # Global KD updates only NanoQuant scales. Freeze embeddings, norms,
    # binary factors, and all other weights to avoid computing their gradients.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    params_to_tune = []
    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            module.do_train = True
            for name, param in module.named_parameters():
                if 'scale' in name:
                    param.requires_grad = True
                    params_to_tune.append(param)

    print(f"Total number of scale parameters to tune: {len(params_to_tune)}")
    if not params_to_tune:
        print("No scales found to tune. Returning original model.")
        model.eval()
        return model

    if quant_config.get("model_kd_pack_factors", False):
        for module in model.modules():
            if isinstance(module, NanoQuantLinear):
                module.begin_kd_packed_factors()

    optimizer = AdamW(params_to_tune, lr=quant_config['model_kd_lr'])
    epochs = quant_config["model_kd_epochs"]
    total_steps = epochs * ((model_kd_num_samples + model_kd_batch_size - 1) // model_kd_batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Prepare data indices and pre-load if needed
    data_indices = list(range(model_kd_num_samples))
    dataloader = dataloader[:model_kd_num_samples].to(device=dev, non_blocking=True)

    # -------------------------------------------
    # 3) KD-tuning loop (student model)
    # -------------------------------------------
    with torch.enable_grad():
        step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            random.shuffle(data_indices)
            total_train_loss = torch.zeros(1, device=dev)

            for start in range(0, model_kd_num_samples, model_kd_batch_size):
                batch_indices = data_indices[start:start + model_kd_batch_size]
                batch = dataloader[batch_indices]

                # Mask Generation
                if pad_token_id != -100:
                    mask = (batch != pad_token_id).int().to(dev)
                else:
                    mask = torch.ones_like(batch).int().to(dev)

                # The custom loss streams vocabulary chunks in both forward and
                # backward, so neither full logits nor their autograd graph is
                # materialized. Only hidden states and one vocab tile are live.
                student_outputs = model.model(batch)
                student_hidden = extract_hidden_states(student_outputs)
                teacher_hidden = torch.cat([teacher_hidden_cache[idx] for idx in batch_indices], dim=0)
                logit_scale = float(getattr(model.config, "logit_scale", 1.0))
                softcap = float(getattr(model.config, "final_logit_softcapping", 0.0) or 0.0)
                vocab_chunk_size = int(quant_config.get("model_kd_vocab_chunk_size", 4096))
                if vocab_chunk_size < 1:
                    raise ValueError("model_kd_vocab_chunk_size must be at least 1")
                loss = _ChunkedLinearKDLoss.apply(
                    student_hidden,
                    teacher_hidden,
                    model.lm_head.weight,
                    getattr(model.lm_head, "bias", None),
                    mask,
                    1.0,
                    vocab_chunk_size,
                    logit_scale,
                    softcap,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1

                total_train_loss += loss.detach()

            steps_per_epoch = (model_kd_num_samples + model_kd_batch_size - 1) // model_kd_batch_size
            avg_train = total_train_loss / steps_per_epoch
            print(f"Epoch {epoch} - Loss: {avg_train.item():.4f}")

    # -------------------------------------------
    # 4) Cleanup
    # -------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    del params_to_tune, optimizer, scheduler, dataloader, teacher_hidden_cache
    cleanup_memory(verbose=True)

    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            module.end_kd_packed_factors()
            module.do_train = False

    model.eval()
    return model
