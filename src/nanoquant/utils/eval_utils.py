# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path

import torch.nn as nn
import torch
import numpy as np
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm


def _kld_cache_manifest(testenc, nsamples, seqlen, stride, vocab_size, metadata):
    input_ids = testenc.detach().to(device="cpu", dtype=torch.long)
    used_ids = input_ids[:, :nsamples * seqlen].contiguous()
    positions_per_window = len(range(0, seqlen - 1, stride))
    return {
        "schema": 1,
        "metric": "full_vocab_kl_sampled_positions",
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "context_length": int(seqlen),
        "sample_stride": int(stride),
        "windows": int(nsamples),
        "predicted_positions": int(nsamples * positions_per_window),
        "positions_per_window": int(positions_per_window),
        "vocab_size": int(vocab_size),
        "input_ids_sha256": hashlib.sha256(used_ids.numpy().tobytes()).hexdigest(),
        **(metadata or {}),
    }


def _open_kld_reference_cache(cache_dir, manifest, mode):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    log_probs_path = cache_dir / "bf16_log_probs.npy"
    shape = (manifest["predicted_positions"], manifest["vocab_size"])

    existing_is_valid = False
    if manifest_path.is_file() and log_probs_path.is_file():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_array = np.load(log_probs_path, mmap_mode="r")
            existing_is_valid = existing_manifest == manifest and existing_array.shape == shape
        except (OSError, ValueError, json.JSONDecodeError):
            existing_is_valid = False

    if mode == "candidate":
        if not existing_is_valid:
            raise RuntimeError(
                "BF16 KL reference cache is missing or does not match this evaluation. "
                "Re-run the paired evaluation so BF16 can rebuild it."
            )
        return np.load(log_probs_path, mmap_mode="r"), True

    if existing_is_valid:
        return np.load(log_probs_path, mmap_mode="r+"), True

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_probs = np.lib.format.open_memmap(
        log_probs_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    return log_probs, False


@torch.no_grad()
def evaluate_ppl(
    model,
    testenc,
    dev,
    dataset_name,
    args=None,
    verbose=True,
    kld_role=None,
    kld_cache_dir=None,
    kld_sample_stride=64,
    kld_metadata=None,
    kld_result=None,
):
    """
    Core helper function for calculating Perplexity (PPL).
    This function contains the actual PPL calculation logic and is called by other evaluation functions.
    """
    model.eval()
    if args is None:
        model.to(dev)
    else:
        if not args.model_offload:
            model.to(dev)

    if hasattr(testenc, 'input_ids'):
        testenc = testenc.input_ids
    seqlen = getattr(model, 'seqlen', model.config.max_position_embeddings)
    print(f"Using sequence length: {seqlen} (model max: {model.config.max_position_embeddings})")
    nsamples = testenc.numel() // seqlen

    # bos token for gemma3
    use_bos_stride = "gemma" in model.config.model_type.lower()
    bos_tensor = None
    effective_seqlen = seqlen
    nll_seqlen = seqlen - 1
    if use_bos_stride:
        effective_seqlen -= 1  # Reserve one position for BOS token
        nll_seqlen += 1
        bos_tensor = torch.tensor([[model.generation_config.bos_token_id]], device=model.device)
        print("Inject bos_token_id for Gemma model")

    if verbose:
        print(f'Evaluating perplexity on {dataset_name} - num_samples={nsamples}, seqlen={seqlen}')

    if nsamples == 0:
        if verbose:
            print(f"Not enough data for PPL evaluation on {dataset_name} with seqlen {seqlen}. Skipping.")
        return None

    if kld_role not in (None, "reference", "candidate"):
        raise ValueError("kld_role must be 'reference', 'candidate', or None")
    if kld_role is not None:
        if kld_cache_dir is None:
            raise ValueError("kld_cache_dir is required when computing paired KL")
        if kld_sample_stride < 1:
            raise ValueError("kld_sample_stride must be at least 1")
        vocab_size = getattr(model.config, "vocab_size", None)
        if vocab_size is None and hasattr(model.config, "text_config"):
            vocab_size = getattr(model.config.text_config, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("Could not determine model vocabulary size for KL evaluation")
        kld_manifest = _kld_cache_manifest(
            testenc, nsamples, seqlen, kld_sample_stride, vocab_size, kld_metadata,
        )
        kld_cache, kld_cache_hit = _open_kld_reference_cache(
            kld_cache_dir, kld_manifest, kld_role,
        )
        kld_per_position = []
        kld_positions_per_window = kld_manifest["positions_per_window"]
    else:
        kld_cache = None
        kld_cache_hit = False
        kld_per_position = None
        kld_positions_per_window = 0

    nlls = []
    # Create a custom progress bar to show cumulative PPL
    if verbose:
        pbar = tqdm(range(nsamples), desc=f"Evaluating PPL for {dataset_name} (PPL: N/A)", disable=not verbose)
    else:
        pbar = range(nsamples)

    for i in pbar:
        i0 = i * effective_seqlen
        i1 = (i + 1) * effective_seqlen
        batch = testenc[:, i0:i1].to(dev)

        if use_bos_stride:
            batch = torch.cat([bos_tensor, batch], dim=1)

        outputs = model(batch, use_cache=False)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if kld_role == "candidate" or (kld_role == "reference" and not kld_cache_hit):
            sampled_positions = torch.arange(
                0, shift_logits.shape[1], kld_sample_stride, device=shift_logits.device,
            )
            sampled_logits = shift_logits[0].index_select(0, sampled_positions).float()
            sampled_log_probs = torch.nn.functional.log_softmax(sampled_logits, dim=-1)
            if sampled_log_probs.shape != (
                kld_positions_per_window, kld_manifest["vocab_size"],
            ):
                raise RuntimeError(
                    "Sampled model logits do not match the KL cache shape: "
                    f"{tuple(sampled_log_probs.shape)} vs "
                    f"({kld_positions_per_window}, {kld_manifest['vocab_size']})"
                )
            row_start = i * kld_positions_per_window
            row_end = row_start + kld_positions_per_window

            if kld_role == "reference":
                kld_cache[row_start:row_end] = sampled_log_probs.cpu().numpy()
            else:
                reference_log_probs = torch.from_numpy(
                    np.array(kld_cache[row_start:row_end], copy=True)
                ).to(device=sampled_log_probs.device)
                per_position_kl = (
                    reference_log_probs.exp() * (reference_log_probs - sampled_log_probs)
                ).sum(dim=-1).clamp_min_(0.0)
                kld_per_position.append(per_position_kl.cpu().numpy())

        neg_log_likelihood = loss.float() * nll_seqlen

        nlls.append(neg_log_likelihood)

        # Update progress bar with current PPL
        if verbose and len(nlls) > 0:
            current_ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * nll_seqlen))
            pbar.set_description(f"Evaluating PPL for {dataset_name} (PPL: {current_ppl.item():.4f})")

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * nll_seqlen))

    if verbose:
        print(f"Perplexity on {dataset_name}: {ppl.item():.4f}")

    if kld_role == "reference" and not kld_cache_hit:
        kld_cache.flush()
        manifest_path = Path(kld_cache_dir) / "manifest.json"
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(kld_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)

    if kld_role == "candidate":
        if not kld_per_position:
            raise RuntimeError("KL evaluation produced no sampled token positions")
        all_positions = np.concatenate(kld_per_position)
        metric = {
            "direction": "BF16 || NanoQuant",
            "variant": "full-vocabulary KL on a systematic position sample",
            "mean_nats_per_token": float(all_positions.mean()),
            "median_nats_per_token": float(np.median(all_positions)),
            "p90_nats_per_token": float(np.percentile(all_positions, 90)),
            "p99_nats_per_token": float(np.percentile(all_positions, 99)),
            "predicted_positions": int(all_positions.size),
            "context_length": int(seqlen),
            "position_stride": int(kld_sample_stride),
            "lower_is_closer_to_bf16": True,
        }
        if kld_result is not None:
            kld_result.update(metric)

    return ppl.item()


@torch.no_grad()
def evaluate_ppl_on_windows(model, dataset, dev, dataset_name, verbose=True):
    """Evaluate PPL on independent token windows without joining their boundaries."""
    model.eval().to(dev)
    seqlen = getattr(model, 'seqlen', model.config.max_position_embeddings)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    total_nll = torch.zeros((), dtype=torch.float64, device=dev)
    total_tokens = 0

    if verbose:
        print(f"Evaluating PPL on {dataset_name}: {len(dataset)} windows, max seqlen={seqlen}")
        pbar = tqdm(range(len(dataset)), desc=f"Evaluating PPL for {dataset_name}")
    else:
        pbar = range(len(dataset))

    for index in pbar:
        row = dataset[index]
        input_ids = torch.as_tensor(row["input_ids"], dtype=torch.long)
        if input_ids.ndim != 1:
            raise ValueError(f"Expected 1D input_ids in window {index}; got shape {tuple(input_ids.shape)}")
        input_ids = input_ids[:seqlen]
        if input_ids.numel() < 2:
            continue
        input_ids = input_ids.unsqueeze(0).to(dev)

        attention_mask = row.get("attention_mask")
        if attention_mask is not None:
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)[:input_ids.shape[1]]
            attention_mask = attention_mask.unsqueeze(0).to(dev)
            if attention_mask[:, 1:].sum().item() == 0:
                continue
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            outputs = model(input_ids, use_cache=False)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        token_nll = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        if attention_mask is not None:
            loss_mask = attention_mask[:, 1:].reshape(-1).to(token_nll.dtype)
            total_nll += (token_nll * loss_mask).sum().to(torch.float64)
            total_tokens += int(loss_mask.sum().item())
        else:
            total_nll += token_nll.sum().to(torch.float64)
            total_tokens += shift_labels.numel()

        if verbose and (index + 1) % 8 == 0 and total_tokens:
            pbar.set_postfix(ppl=f"{torch.exp(total_nll / total_tokens).item():.4f}")

    if total_tokens == 0:
        raise ValueError(f"No valid next-token labels found in {dataset_name}")

    ppl = torch.exp(total_nll / total_tokens).item()
    if verbose:
        print(f"Perplexity on {dataset_name}: {ppl:.4f} ({total_tokens} predicted tokens)")
    return ppl, total_tokens


@torch.no_grad()
def evaluate_ppl_after_block(model, model_name, dev, get_test_ppl=True):
    """
    Function to evaluate PPL after block-wise processing during the compression stage.
    It internally calls the core `evaluate_ppl` function.
    """
    test_ppl = None

    # Evaluate PPL on the test dataset
    if get_test_ppl:
        from ..utils.data_utils import get_test_loaders
        _, test_loader = get_test_loaders("wikitext2", model_name=model_name, seqlen=model.seqlen)
        test_ppl = evaluate_ppl(model, test_loader, dev, "wikitext2", None, verbose=False)

    return test_ppl


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    tasks_str,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    args=None,
    calibration_dataset_path=None,
    kld_role=None,
    kld_cache_dir=None,
    kld_sample_stride=64,
    kld_metadata=None,
):
    """
    Main function to comprehensively evaluate a final model on PPL and/or zero-shot tasks.
    """
    results = {}
    device = next(model.parameters()).device
    model.eval()

    # Perplexity Evaluation
    if eval_ppl:
        datasets = [ds.strip() for ds in eval_ppl.split(',') if ds.strip()]
        for dataset in datasets:
            try:
                if dataset.lower() in {"calibration", "calib"}:
                    if not calibration_dataset_path:
                        raise ValueError("PPL task 'calibration' requires a calibration dataset path")
                    from datasets import load_from_disk

                    calibration_windows = load_from_disk(calibration_dataset_path)
                    ppl_result, predicted_tokens = evaluate_ppl_on_windows(
                        model,
                        calibration_windows,
                        device,
                        "calibration set (in-sample)",
                        verbose=True,
                    )
                    results["calibration_in_sample"] = {
                        "ppl": ppl_result,
                        "windows": len(calibration_windows),
                        "predicted_tokens": predicted_tokens,
                    }
                    continue

                from ..utils.data_utils import get_test_loaders
                _, testloader = get_test_loaders(
                    dataset,
                    model_name=model.config._name_or_path,
                    seqlen=model.seqlen,
                    tokenizer=tokenizer,
                )
                kld_result = {}
                apply_kld = dataset.lower() == "wikitext2" and kld_role is not None
                ppl_result = evaluate_ppl(
                    model,
                    testloader,
                    device,
                    dataset,
                    args,
                    verbose=True,
                    kld_role=kld_role if apply_kld else None,
                    kld_cache_dir=kld_cache_dir if apply_kld else None,
                    kld_sample_stride=kld_sample_stride,
                    kld_metadata=kld_metadata,
                    kld_result=kld_result,
                )
                if ppl_result is not None:
                    results[dataset] = {"ppl": ppl_result}
                if kld_result:
                    results["wikitext2_kld_full_vocab"] = kld_result
            except Exception as e:
                print(f"Failed to evaluate PPL on dataset {dataset}: {e}")
                if dataset.lower() == "wikitext2" and kld_role is not None:
                    raise
                continue

    # Zero-shot Task Evaluation
    if tasks_str:
        task_names = [task for task in tasks_str.split(',') if task.strip()]
        if task_names:
            print(f"[INFO] Starting zero-shot evaluation for tasks: {', '.join(task_names)}")

            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
            )
            harness_results = simple_evaluate(
                model=lm,
                tasks=task_names,
                num_fewshot=num_fewshot,
                limit=None if limit == -1 else limit,
                log_samples=False,
            )
            results.update(harness_results["results"])

            msg = f"Zero-shot tasks results: {json.dumps(harness_results['results'], indent=2)}"
            print(msg)

    return results
