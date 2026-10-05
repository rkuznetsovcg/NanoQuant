# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json

import torch.nn as nn
import torch
from lm_eval.evaluator import simple_evaluate
from tqdm import tqdm


@torch.no_grad()
def evaluate_ppl(model, testenc, dev, dataset_name, args=None, verbose=True):
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

        neg_log_likelihood = loss.float() * nll_seqlen

        nlls.append(neg_log_likelihood)

        # Update progress bar with current PPL
        if verbose and len(nlls) > 0:
            current_ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * nll_seqlen))
            pbar.set_description(f"Evaluating PPL for {dataset_name} (PPL: {current_ppl.item():.4f})")

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * nll_seqlen))

    if verbose:
        print(f"Perplexity on {dataset_name}: {ppl.item():.4f}")

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
                _, testloader = get_test_loaders(dataset, model_name=model.config._name_or_path, seqlen=model.seqlen)
                ppl_result = evaluate_ppl(model, testloader, device, dataset, args, verbose=True)
                if ppl_result is not None:
                    results[dataset] = {"ppl": ppl_result}
            except Exception as e:
                print(f"Failed to evaluate PPL on dataset {dataset}: {e}")
                continue

    # Zero-shot Task Evaluation
    if tasks_str:
        task_names = [task for task in tasks_str.split(',') if task.strip()]
        if task_names:
            print(f"[INFO] Starting zero-shot evaluation for tasks: {', '.join(task_names)}")

            harness_results = simple_evaluate(
                model="hf",
                model_args={
                    "pretrained": model,
                    "tokenizer": tokenizer,
                },
                tasks=task_names,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=str(device),
                limit=None if limit == -1 else limit,
            )
            results.update(harness_results["results"])

            msg = f"Zero-shot tasks results: {json.dumps(harness_results['results'], indent=2)}"
            print(msg)

    return results
