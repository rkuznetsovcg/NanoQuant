# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run a paired BF16-versus-NanoQuant evaluation on identical inputs."""

import argparse
import datetime
import gc
import hashlib
import json
import math
import re
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from numbers import Real
from statistics import mean

import torch
from transformers import AutoConfig

from .checkpoint_audit import inspect_checkpoint
from .utils.eval_utils import evaluate_model
from .utils.load_utils import load_compressed_model, load_model, load_tokenizer
from .utils.utils import cleanup_memory, set_seed


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n",
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
    for metric_name in ("acc_norm", "acc"):
        for key, value in task_result.items():
            # Recent lm-eval versions qualify metric keys with the aggregation,
            # e.g. "acc_norm,none"; older versions may return the bare name.
            if (key == metric_name or key.startswith(metric_name + ",")) and isinstance(value, Real):
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
    invalid = []
    for dataset in ppl_datasets:
        key = "calibration_in_sample" if dataset.lower() in {"calibration", "calib"} else dataset
        if key not in results or not isinstance(results[key], dict) or "ppl" not in results[key]:
            missing.append(f"PPL:{key}")
        else:
            value = results[key]["ppl"]
            if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                invalid.append(f"PPL:{key}")
    for task in task_names:
        task_result = results.get(task)
        if not isinstance(task_result, dict) or _primary_accuracy(task_result)[1] is None:
            missing.append(f"zero-shot:{task}")
        else:
            value = _primary_accuracy(task_result)[1]
            if not math.isfinite(value) or not 0 <= value <= 1:
                invalid.append(f"zero-shot:{task}")
    if require_kld and not isinstance(results.get("wikitext2_kld_full_vocab"), dict):
        missing.append("KL:WikiText-2 full vocabulary")
    elif require_kld:
        kl = results["wikitext2_kld_full_vocab"]
        for key in ("mean_nats_per_token", "median_nats_per_token", "p90_nats_per_token", "p99_nats_per_token"):
            value = kl.get(key)
            if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                invalid.append(f"KL:{key}")
        if not isinstance(kl.get("predicted_positions"), int) or kl["predicted_positions"] < 1:
            invalid.append("KL:predicted_positions")
    if missing:
        raise RuntimeError(f"{label} evaluation did not produce required metrics: {', '.join(missing)}")
    if invalid:
        raise RuntimeError(f"{label} metrics must be finite and in range: {', '.join(invalid)}")


def _recover_bf16_results_from_log(
    log_path: Path,
    ppl_datasets: list[str],
    expected_revision: str,
    expected_protocol: dict | None = None,
) -> dict | None:
    """Recover completed baseline metrics when an older evaluator failed after logging them."""
    if not log_path or not log_path.is_file():
        return None
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    bf16_start = log_text.find("===== bf16: загрузка модели =====")
    if bf16_start < 0:
        return None
    if expected_protocol is not None:
        protocol_line = re.search(r"^NANOQUANT_EVAL_PROTOCOL (.+)$", log_text[:bf16_start], re.MULTILINE)
        if protocol_line is None:
            return None
        try:
            if json.loads(protocol_line.group(1)) != expected_protocol:
                return None
        except json.JSONDecodeError:
            return None
    nanoquant_start = log_text.find("===== nanoquant:", bf16_start + 1)
    bf16_log = log_text[bf16_start:nanoquant_start] if nanoquant_start >= 0 else log_text[bf16_start:]
    revision_match = re.search(r"Ревизия модели:\s*([0-9a-f]{40,64})", log_text[:bf16_start])
    if not revision_match or revision_match.group(1) != expected_revision:
        return None

    recovered = {}
    for dataset in ppl_datasets:
        if dataset.lower() in {"calibration", "calib"}:
            match = re.search(
                r"Perplexity on calibration set \(in-sample\):\s*([0-9.eE+-]+)\s*"
                r"\((\d+) predicted tokens\)",
                bf16_log,
            )
            windows_match = re.search(
                r"Evaluating PPL on calibration set \(in-sample\):\s*(\d+) windows",
                bf16_log,
            )
            if not match or not windows_match:
                return None
            recovered["calibration_in_sample"] = {
                "ppl": float(match.group(1)),
                "windows": int(windows_match.group(1)),
                "predicted_tokens": int(match.group(2)),
            }
            continue

        match = re.search(
            rf"Perplexity on {re.escape(dataset)}:\s*([0-9.eE+-]+)",
            bf16_log,
        )
        if not match:
            return None
        recovered[dataset] = {"ppl": float(match.group(1))}

    marker = "Zero-shot tasks results:"
    marker_index = bf16_log.rfind(marker)
    if marker_index < 0:
        return None
    json_start = bf16_log.find("{", marker_index + len(marker))
    if json_start < 0:
        return None
    try:
        task_results, _ = json.JSONDecoder().raw_decode(bf16_log[json_start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(task_results, dict):
        return None
    recovered.update(task_results)
    return recovered


def _evaluate_one(
    label,
    model_factory,
    tokenizer_factory,
    protocol,
    output_path,
    state,
    kld_cache_dir,
    kld_only=False,
):
    print(f"\n===== {label}: загрузка модели =====", flush=True)
    model = None
    try:
        model = model_factory()
        tokenizer = tokenizer_factory()
        model.config.use_cache = False
        model.seqlen = protocol["sequence_length"]
        model.eval()
        print(
            f"===== {label}: PPL "
            f"{['wikitext2'] if kld_only else protocol['ppl_datasets']} и задачи "
            f"{[] if kld_only else protocol['zero_shot_tasks']} =====",
            flush=True,
        )
        set_seed(protocol["seed"])
        evaluated_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            tasks_str="" if kld_only else ",".join(protocol["zero_shot_tasks"]),
            eval_ppl="wikitext2" if kld_only else ",".join(protocol["ppl_datasets"]),
            num_fewshot=protocol["num_fewshot"],
            limit=protocol["limit"],
            batch_size=protocol["batch_size"],
            calibration_dataset_path=protocol["calibration_dataset"],
            seed=protocol["seed"],
            kld_role="reference" if label == "bf16" else "candidate",
            kld_cache_dir=str(kld_cache_dir),
            kld_sample_stride=protocol["kl_divergence"]["position_stride"],
            kld_metadata={
                "base_model_id": protocol["model_id"],
                "base_model_revision": protocol["model_revision"],
                "evaluation_code_sha256": protocol["evaluation_code_sha256"],
                "runtime_versions": protocol["runtime_versions"],
                "attention_backend": protocol["attention_backend"],
            },
        )
        results = dict(state["results"].get(label, {})) if kld_only else {}
        results.update(evaluated_results)
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
    except Exception as error:
        state.update({"status": "failed", "failed_model": label,
                      "error": f"{type(error).__name__}: {error}"})
        _write_json(output_path, state)
        raise
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
    parser.add_argument("--fresh", action="store_true", help="Evaluate both models again; ignore previous metrics and KL cache")
    parser.add_argument(
        "--resume-log",
        default=None,
        help="Previous comparison log from which a completed BF16 evaluation may be recovered",
    )
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

    evaluation_code = hashlib.sha256()
    for relative_path in ("compare_eval.py", "checkpoint_audit.py", "core/linearized_block.py",
                          "utils/eval_utils.py", "utils/load_utils.py", "utils/utils.py", "modules/linear.py"):
        evaluation_code.update((Path(__file__).parent/relative_path).read_bytes())
    calibration_digest = hashlib.sha256()
    for data_path in sorted(calibration_path.rglob("*")):
        if data_path.is_file():
            calibration_digest.update(str(data_path.relative_to(calibration_path)).encode())
            with data_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    calibration_digest.update(chunk)
    protocol = {
        "evaluation_schema": 2,
        "evaluation_code_sha256": evaluation_code.hexdigest(),
        "runtime_versions": {package: version(package) for package in ("torch", "transformers", "lm_eval", "datasets")},
        "calibration_sha256": calibration_digest.hexdigest(),
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
    print(f"NANOQUANT_EVAL_PROTOCOL {json.dumps(protocol, ensure_ascii=False, sort_keys=True)}", flush=True)
    print(f"Ревизия модели: {source_revision}", flush=True)
    state = {
        "status": "running",
        "protocol": protocol,
        "results": {},
    }
    if output_path.is_file() and not args.fresh:
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("protocol") == protocol:
            state["results"] = previous.get("results", {})
            print("Совпадающий предыдущий результат найден; уже готовая модель будет пропущена.", flush=True)

    if "bf16" not in state["results"] and args.resume_log and not args.fresh:
        recovered = _recover_bf16_results_from_log(
            Path(args.resume_log), ppl_datasets, expected_revision=source_revision, expected_protocol=protocol,
        )
        if recovered is not None:
            try:
                _validate_results(recovered, ppl_datasets, task_names, "BF16 recovered from log")
            except RuntimeError as error:
                print(f"Не удалось безопасно восстановить BF16 из лога: {error}", flush=True)
            else:
                state["results"]["bf16"] = recovered
                _write_json(output_path, state)
                print(
                    "Восстановил законченные BF16-метрики из предыдущего лога; "
                    "zero-shot повторно запускать не нужно.",
                    flush=True,
                )

    cache_key = hashlib.sha256(str(output_path).encode("utf-8")).hexdigest()[:20]
    kld_cache_dir = Path(tempfile.gettempdir()) / "nanoquant-paired-kld" / cache_key
    if args.fresh:
        shutil.rmtree(kld_cache_dir, ignore_errors=True)
        print("Свежий прогон: обе модели и KL будут пересчитаны.", flush=True)
    state["checkpoint_audit"] = inspect_checkpoint(checkpoint_path, source_config, float(profile.get("bits", 0.55)))
    _write_json(output_path, state)
    if state["checkpoint_audit"]["errors"]:
        state["status"] = "invalid_checkpoint"
        _write_json(output_path, state)
        raise RuntimeError(f"Checkpoint has invalid weights. Details saved in {output_path}")
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
        if "nanoquant" not in state["results"]:
            print("BF16-метрики уже готовы; проверяю или восстанавливаю KL-кэш коротким WikiText-проходом.", flush=True)
            _evaluate_one(
                "bf16", evaluate_bf16, tokenizer_factory, protocol, output_path, state, kld_cache_dir,
                kld_only=True,
            )
        else:
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
