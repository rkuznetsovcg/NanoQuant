# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run a paired BF16-versus-NanoQuant evaluation on identical inputs."""

import argparse
import datetime
import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from numbers import Real
from statistics import mean

import torch
from transformers import AutoConfig

from .utils.eval_utils import evaluate_model
from .utils.load_utils import load_compressed_model, load_model, load_tokenizer
from .utils.utils import cleanup_memory, set_seed


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _primary_accuracy(task_result: dict):
    for key in ("acc_norm", "acc"):
        value = task_result.get(key)
        if isinstance(value, Real):
            return key, float(value)
    return None, None


def _validate_results(
    results: dict,
    ppl_datasets: list[str],
    task_names: list[str],
    label: str,
    require_kld: bool = False,
) -> None:
    missing = []
    for dataset in ppl_datasets:
        key = "calibration_in_sample" if dataset.lower() in {"calibration", "calib"} else dataset
        if key not in results or not isinstance(results[key], dict) or "ppl" not in results[key]:
            missing.append(f"PPL:{key}")
    for task in task_names:
        task_result = results.get(task)
        if not isinstance(task_result, dict) or _primary_accuracy(task_result)[1] is None:
            missing.append(f"zero-shot:{task}")
    if require_kld and not isinstance(results.get("wikitext2_kld_full_vocab"), dict):
        missing.append("KL:WikiText-2 full vocabulary")
    if missing:
        raise RuntimeError(f"{label} evaluation did not produce required metrics: {', '.join(missing)}")


def _evaluate_one(label, model_factory, tokenizer_factory, protocol, output_path, state, kld_cache_dir):
    print(f"\n===== {label}: загрузка модели =====", flush=True)
    model = model_factory()
    try:
        tokenizer = tokenizer_factory()
        model.config.use_cache = False
        model.seqlen = protocol["sequence_length"]
        model.eval()
        print(
            f"===== {label}: PPL {protocol['ppl_datasets']} и задачи {protocol['zero_shot_tasks']} =====",
            flush=True,
        )
        set_seed(protocol["seed"])
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            tasks_str=",".join(protocol["zero_shot_tasks"]),
            eval_ppl=",".join(protocol["ppl_datasets"]),
            num_fewshot=protocol["num_fewshot"],
            limit=protocol["limit"],
            batch_size=protocol["batch_size"],
            calibration_dataset_path=protocol["calibration_dataset"],
            kld_role="reference" if label == "bf16" else "candidate",
            kld_cache_dir=str(kld_cache_dir),
            kld_sample_stride=protocol["kl_divergence"]["position_stride"],
            kld_metadata={
                "base_model_id": protocol["model_id"],
                "base_model_revision": protocol["model_revision"],
            },
        )
        _validate_results(
            results,
            protocol["ppl_datasets"],
            protocol["zero_shot_tasks"],
            label,
            require_kld=label == "nanoquant",
        )
        state["results"][label] = results
        _write_json(output_path, state)
        print(f"===== {label}: оценка завершена =====", flush=True)
        return results
    finally:
        del model
        gc.collect()
        cleanup_memory()


def _build_comparison(results: dict, task_names: list[str]) -> dict:
    bf16 = results["bf16"]
    quantized = results["nanoquant"]
    ppl_comparison = {}
    for key in ("wikitext2", "calibration_in_sample"):
        if key not in bf16 or key not in quantized:
            continue
        bf16_ppl = float(bf16[key]["ppl"])
        quantized_ppl = float(quantized[key]["ppl"])
        ppl_comparison[key] = {
            "bf16": bf16_ppl,
            "nanoquant": quantized_ppl,
            "delta_nanoquant_minus_bf16": quantized_ppl - bf16_ppl,
            "relative_change_percent": (quantized_ppl / bf16_ppl - 1.0) * 100.0,
            "direction": "lower is better",
        }

    task_comparison = {}
    deltas = []
    for task in task_names:
        metric, bf16_score = _primary_accuracy(bf16[task])
        quant_metric, quantized_score = _primary_accuracy(quantized[task])
        if metric != quant_metric:
            raise RuntimeError(f"Primary accuracy metric changed between models for {task}: {metric} vs {quant_metric}")
        delta_pp = (quantized_score - bf16_score) * 100.0
        deltas.append(delta_pp)
        task_comparison[task] = {
            "metric": metric,
            "bf16": bf16_score,
            "nanoquant": quantized_score,
            "delta_percentage_points": delta_pp,
            "direction": "higher is better",
        }

    return {
        "perplexity": ppl_comparison,
        "kl_divergence": {
            "wikitext2": {
                **quantized["wikitext2_kld_full_vocab"],
                "dataset": "WikiText-2 raw test",
                "reference_model": "BF16",
                "candidate_model": "NanoQuant",
            },
        },
        "zero_shot": task_comparison,
        "mean_zero_shot_delta_percentage_points": mean(deltas) if deltas else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="NanoQuant JSON profile with eval settings")
    parser.add_argument("--checkpoint", required=True, help="Completed NanoQuant .pt checkpoint")
    parser.add_argument("--run-metadata", required=True, help="run_complete.json from quantization")
    parser.add_argument("--output", required=True, help="Paired comparison JSON output")
    parser.add_argument("--revision", default=None, help="Pinned Hugging Face model commit")
    parser.add_argument("--revision-source", default="provided", help="How the model revision was selected")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    metadata_path = Path(args.run_metadata).resolve()
    output_path = Path(args.output).resolve()
    if not config_path.is_file() or not checkpoint_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("The evaluation config, checkpoint, and run metadata must all exist")

    profile = json.loads(config_path.read_text(encoding="utf-8"))
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_id = profile["model_id"]
    if run_metadata.get("model_id") != model_id:
        raise ValueError(f"Run metadata is for {run_metadata.get('model_id')}, config requests {model_id}")
    expected_size = int(run_metadata.get("checkpoint_size_bytes", 0))
    if expected_size and checkpoint_path.stat().st_size != expected_size:
        raise ValueError(
            f"Checkpoint size mismatch: found {checkpoint_path.stat().st_size}, expected {expected_size} bytes"
        )

    recorded_revision = run_metadata.get("model_revision")
    requested_revision = args.revision or recorded_revision
    source_config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, revision=requested_revision,
    )
    source_revision = getattr(source_config, "_commit_hash", None) or requested_revision
    if not source_revision:
        raise RuntimeError(f"Could not resolve a pinned Hugging Face revision for {model_id}")
    if recorded_revision and source_revision != recorded_revision:
        raise ValueError(
            f"Base model revision mismatch: quantization used {recorded_revision}, evaluation resolved {source_revision}"
        )

    calibration_path = Path(profile["calib_dataset"]).resolve()
    if not calibration_path.is_dir():
        raise FileNotFoundError(f"Calibration dataset not found: {calibration_path}")
    ppl_datasets = [item.strip() for item in profile.get("ppl_task", "wikitext2,calibration").split(",") if item.strip()]
    if not any(dataset.lower() == "wikitext2" for dataset in ppl_datasets):
        ppl_datasets.insert(0, "wikitext2")
    task_names = [
        item.strip()
        for item in profile.get(
            "zeroshot_task", "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge",
        ).split(",")
        if item.strip()
    ]
    if not ppl_datasets or not task_names:
        raise ValueError("Paired comparison requires PPL datasets and zero-shot tasks")

    protocol = {
        "model_id": model_id,
        "model_revision": source_revision,
        "model_revision_source": "run metadata" if recorded_revision else args.revision_source,
        "revision_recorded_during_quantization": bool(recorded_revision),
        "sequence_length": int(profile.get("seqlen", 2048)),
        "dtype": "bfloat16",
        "attention_backend": profile.get("attn_implementation", "auto"),
        "ppl_datasets": ppl_datasets,
        "kl_divergence": {
            "metric": "full_vocab_next_token_kl",
            "direction": "KL(p_BF16 || p_NanoQuant)",
            "dataset": "WikiText-2 raw test",
            "context_length": int(profile.get("seqlen", 2048)),
            "position_stride": 64,
            "reported_unit": "nats_per_token",
        },
        "zero_shot_tasks": task_names,
        "num_fewshot": int(profile.get("num_fewshot", 0)),
        "limit": int(profile.get("limit", -1)),
        "batch_size": profile.get("batch_size", "auto"),
        "calibration_dataset": str(calibration_path),
        "seed": int(profile.get("seed", 0)),
        "quantized_checkpoint": str(checkpoint_path),
        "quantized_checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "quantized_checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "nanoquant_commit": run_metadata.get("nanoquant_commit"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "protocol": protocol,
        "results": {},
    }
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("protocol") == protocol:
            state["results"] = previous.get("results", {})
            print("Совпадающий предыдущий результат найден; уже готовая модель будет пропущена.", flush=True)

    cache_key = hashlib.sha256(str(output_path).encode("utf-8")).hexdigest()[:20]
    kld_cache_dir = Path(tempfile.gettempdir()) / "nanoquant-paired-kld" / cache_key
    bf16_kld_manifest = kld_cache_dir / "manifest.json"
    bf16_kld_values = kld_cache_dir / "bf16_log_probs.npy"
    if (
        "bf16" in state["results"]
        and "nanoquant" not in state["results"]
        and not (bf16_kld_manifest.is_file() and bf16_kld_values.is_file())
    ):
        state["results"].pop("bf16")
        print("Локальный KL-кэш BF16 отсутствует; пересчитаю BF16 перед NanoQuant.", flush=True)

    required_fast_attention = bool(profile.get("require_fast_linear_attention", False))
    backend = protocol["attention_backend"]

    def tokenizer_factory():
        return load_tokenizer(model_id, revision=source_revision)

    def evaluate_bf16():
        init_functions = ("kaiming_uniform_", "uniform_", "normal_")
        original_initializers = {name: getattr(torch.nn.init, name) for name in init_functions}
        try:
            return load_model(
                model_id,
                seqlen=protocol["sequence_length"],
                device_map="cuda",
                require_fast_linear_attention=required_fast_attention,
                attn_implementation=backend,
                revision=source_revision,
                print_model=False,
            )
        finally:
            for name, function in original_initializers.items():
                setattr(torch.nn.init, name, function)

    def evaluate_nanoquant():
        model = load_compressed_model(
            model_name_or_path=model_id,
            checkpoint_path=str(checkpoint_path),
            seqlen=protocol["sequence_length"],
            device="cuda",
            has_mid_scale=(profile.get("admm_type", "nanoquant") == "dbf"),
            dtype=torch.bfloat16,
            attn_implementation=backend,
            revision=source_revision,
        )
        return model.cuda()

    if "bf16" not in state["results"]:
        _evaluate_one("bf16", evaluate_bf16, tokenizer_factory, protocol, output_path, state, kld_cache_dir)
    else:
        _validate_results(state["results"]["bf16"], ppl_datasets, task_names, "bf16 cached")
        print("===== BF16: использую сохранённые результаты =====", flush=True)

    if "nanoquant" not in state["results"]:
        _evaluate_one("nanoquant", evaluate_nanoquant, tokenizer_factory, protocol, output_path, state, kld_cache_dir)
    else:
        _validate_results(
            state["results"]["nanoquant"], ppl_datasets, task_names, "nanoquant cached", require_kld=True,
        )
        print("===== NanoQuant: использую сохранённые результаты =====", flush=True)

    state["comparison"] = _build_comparison(state["results"], task_names)
    state["status"] = "complete"
    state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_json(output_path, state)
    shutil.rmtree(kld_cache_dir, ignore_errors=True)

    print("\n===== BF16 → NanoQuant: разницы =====", flush=True)
    for dataset, values in state["comparison"]["perplexity"].items():
        print(
            f"{dataset}: {values['bf16']:.4f} → {values['nanoquant']:.4f} PPL "
            f"({values['delta_nanoquant_minus_bf16']:+.4f}; {values['relative_change_percent']:+.2f}%)",
            flush=True,
        )
    for dataset, values in state["comparison"]["kl_divergence"].items():
        print(
            f"KL {dataset} ({values['direction']}): {values['mean_nats_per_token']:.6g} nats/token "
            f"(median {values['median_nats_per_token']:.6g}; "
            f"p90 {values['p90_nats_per_token']:.6g}; "
            f"n={values['predicted_positions']})",
            flush=True,
        )
    for task, values in state["comparison"]["zero_shot"].items():
        print(
            f"{task} ({values['metric']}): {values['bf16']:.4f} → {values['nanoquant']:.4f} "
            f"({values['delta_percentage_points']:+.2f} pp)",
            flush=True,
        )
    print(
        "Средняя разница zero-shot: "
        f"{state['comparison']['mean_zero_shot_delta_percentage_points']:+.2f} pp",
        flush=True,
    )
    print(f"Полный парный отчёт: {output_path}", flush=True)


if __name__ == "__main__":
    main()
