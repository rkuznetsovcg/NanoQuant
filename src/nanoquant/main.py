# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for NanoQuant model compression and evaluation.

Usage:
    python -m nanoquant.main --model_id meta-llama/Llama-2-7b-hf --qmodel_path output.pt
    nanoquant --model_id meta-llama/Llama-2-7b-hf --qmodel_path output.pt
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

try:
    from loguru import logger
except ImportError:
    import logging as logger

import torch
from transformers import HfArgumentParser

from .modules.hub import NanoQuantConfigDataclass, NanoQuantModel
from .utils.eval_utils import evaluate_model
from .utils.load_utils import load_tokenizer
from .utils.utils import cleanup_memory


@dataclass
class ModelArguments:
    model_id: str = field(
        default="Qwen/Qwen3-4B-Base",
        metadata={"help": "Model identifier or local path"},
    )
    seqlen: int = field(default=2048, metadata={"help": "Sequence length"})
    qmodel_path: Optional[str] = field(default=None, metadata={"help": "Path to save/load quantized model checkpoint"})
    from_hub: bool = field(default=False, metadata={"help": "Load pre-quantized model from HuggingFace Hub"})
    hub_model_id: Optional[str] = field(default=None,
                                        metadata={"help": "HuggingFace Hub model ID (defaults to model_id)"})
    device_map: str = field(
        default="cpu",
        metadata={"help": "Device map for model loading ('cpu' or 'auto')"},
    )
    block_io_batch_size: int = field(default=4, metadata={"help": "Microbatch size for block reference/propagation passes"})
    require_fast_linear_attention: bool = field(
        default=False,
        metadata={"help": "Require importable FLA/causal-conv1d kernels for Qwen3.5/3.8 before loading weights"},
    )
    attn_implementation: str = field(
        default="auto",
        metadata={"help": "Full-attention backend (BF16 forward/backward; FA4 is explicit)",
                  "choices": ["auto", "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4"]},
    )
    model_revision: Optional[str] = field(default=None, metadata={"help": "Pinned Hugging Face model revision"})


@dataclass
class QuantArguments:
    bits: float = field(default=1.0, metadata={"help": "Target quantization bits"})
    rank_allocation: str = field(
        default="sensitivity",
        metadata={"help": "Rank budget allocation", "choices": ["sensitivity", "kronq_trace", "uniform"]},
    )
    seed: int = field(default=0, metadata={"help": "Random seed"})
    num_calib_samples: int = field(default=128, metadata={"help": "Number of calibration samples"})
    calib_dataset: str = field(default="wikitext2", metadata={"help": "Calibration dataset"})
    calib_shrinkage: float = field(default=0.4, metadata={"help": "Calibration shrinkage factor"})
    calib_strategy: str = field(
        default="online",
        metadata={
            "help": "Calibration strategy",
            "choices": ["online", "two_phase", "dbf", "none"],
        },
    )


@dataclass
class TuneArguments:
    layer_epoch_schedule: bool = field(default=True, metadata={"help": "Use per-projection 2/4/6/8 epoch caps on large models"})
    tune_schedule: str = field(default="sequential", metadata={"help": 'TuneFP schedule: sequential (quality default), shared_input or parent'})
    refresh_input_stats: bool = field(default=True, metadata={"help": 'Refresh clipped diagonal inputs after each TuneFP'})
    resume_dir: str = field(default="", metadata={"help": 'Local directory for atomic block resume and cached rank probes'})
    rank_budget: str = field(default="nominal", metadata={"help": 'Rank budget: nominal bits or parity with rounded uniform ranks'})
    rank_probe_candidates: int = field(default=0, metadata={"help": 'Maximum measured rank candidates per block; 0 keeps the cheap proxy'})
    rank_probe_iters: int = field(default=50, metadata={"help": 'ADMM iterations per rank probe'})
    correlation_block_size: int = field(default=0, metadata={"help": 'Input covariance channel blocks: 0, 128 or 256'})
    correlation_layers: str = field(default="mlp.down_proj", metadata={"help": 'Comma-separated layer globs eligible for bounded correlations'})
    nonfact_plateau_tolerance: float = field(default=0.0, metadata={"help": 'Optional relative epoch plateau tolerance; 0 disables stopping'})
    nonfact_plateau_min_epochs: int = field(default=3, metadata={"help": 'Minimum epochs before TuneFP plateau stopping'})
    nonfact_plateau_patience: int = field(default=2, metadata={"help": 'Consecutive plateau epochs required to stop TuneFP'})

    tune_nonfact: bool = field(default=True, metadata={"help": "Tune non-factorized layers"})
    nonfact_lr: float = field(default=1e-4, metadata={"help": "LR for non-factorized binary parameters"})
    nonfact_batch_size: int = field(default=4, metadata={"help": "Batch size for non-factorized tuning"})
    nonfact_epochs: int = field(default=8, metadata={"help": "Epochs for non-factorized tuning"})
    admm_type: str = field(
        default="nanoquant",
        metadata={
            "help": "ADMM type",
            "choices": ["nanoquant", "dbf"]
        },
    )
    admm_outer_iters: int = field(default=400, metadata={"help": "ADMM outer iterations"})
    admm_inner_iters: int = field(default=5, metadata={"help": "ADMM inner iterations"})
    admm_reg: float = field(default=3e-2, metadata={"help": "ADMM regularization strength"})
    admm_early_stop: bool = field(default=True, metadata={"help": "Stop ADMM after repeated residual and sign convergence"})
    admm_min_outer_iters: int = field(default=120, metadata={"help": "Minimum ADMM iterations before convergence checks"})
    admm_check_interval: int = field(default=10, metadata={"help": "ADMM convergence check interval"})
    admm_convergence_tolerance: float = field(default=2e-3, metadata={"help": "Relative ADMM residual convergence tolerance"})
    admm_stable_sign_tolerance: float = field(default=1e-3, metadata={"help": "Maximum sign-change fraction at convergence checks"})
    admm_rho_stop_threshold: float = field(default=0.85, metadata={"help": "Minimum ADMM rho before early stopping is allowed"})
    admm_penalty_scheduler: str = field(
        default="linear",
        metadata={
            "help": "ADMM penalty scheduler",
            "choices": ["linear", "cubic", "logistic", "exp_decay", "exp_growth"],
        },
    )
    admm_print_steps: bool = field(default=False, metadata={"help": "Print ADMM optimization steps"})
    tune_fact: bool = field(default=True, metadata={"help": "Tune factorized layers"})
    fact_binary_lr: float = field(default=1e-5, metadata={"help": "LR for factorized binary parameters"})
    fact_scale_lr: float = field(default=1e-5, metadata={"help": "LR for factorized scale parameters"})
    fact_bias_lr: float = field(default=1e-5, metadata={"help": "LR for factorized bias parameters"})
    fact_batch_size: int = field(default=1, metadata={"help": "Batch size for factorized tuning"})
    fact_epochs: int = field(default=8, metadata={"help": "Epochs for factorized tuning"})
    tune_model: bool = field(default=True, metadata={"help": "Perform model-level KD tuning"})
    model_kd_lr: float = field(default=1e-5, metadata={"help": "LR for model knowledge distillation"})
    model_kd_batch_size: int = field(default=1, metadata={"help": "Batch size for model KD"})
    model_kd_num_samples: int = field(default=64, metadata={"help": "Calibration samples used only for model-level KD"})
    model_kd_vocab_chunk_size: int = field(default=4096, metadata={"help": "Vocabulary tile size for memory-bounded KD"})
    model_kd_pack_factors: bool = field(default=False, metadata={"help": "Store frozen binary factors in packed form during KD (saves VRAM, adds per-forward unpacking)"})
    admm_warm_start_iters: int = field(default=2, metadata={"help": "Power iterations when reusing the prior ADMM singular vector"})
    log_reconstruction_error: bool = field(default=False, metadata={"help": "Materialize dense reconstructed weights for ADMM diagnostics"})


@dataclass
class EvalArguments:
    ppl_task: str = field(
        default="",
        metadata={"help": "Perplexity dataset(s), comma-separated. Leave empty to skip."},
    )
    zeroshot_task: str = field(
        default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge",
        metadata={"help": "Zero-shot tasks, comma-separated. Leave empty to skip."},
    )
    batch_size: str = field(default="auto", metadata={"help": "Batch size for zero-shot evaluation (auto = automatic)"})
    num_fewshot: int = field(default=0, metadata={"help": "Few-shot examples for zero-shot tasks"})
    limit: int = field(default=-1, metadata={"help": "Sample limit for zero-shot (-1 = all)"})
    eval_after_each_block: bool = field(
        default=False,
        metadata={"help": "Run full WikiText2 PPL after every compressed block (diagnostic; adds substantial runtime)"},
    )


def init_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    if hasattr(logger, "remove"):
        try:
            logger.remove()
        except ValueError:
            pass
        logger.add(
            sys.stderr,
            level=log_level,
            format=
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.add(log_file, level="DEBUG", rotation="10 MB")
    else:
        logger.basicConfig(level=getattr(logger, log_level, logger.INFO))


def main():
    parser = HfArgumentParser((ModelArguments, QuantArguments, TuneArguments, EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, quant_args, tune_args, eval_args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).resolve()))
    else:
        model_args, quant_args, tune_args, eval_args = parser.parse_args_into_dataclasses()

    init_logging()

    # Merge into NanoQuantConfigDataclass
    quant_config = NanoQuantConfigDataclass(
        model_id=model_args.model_id,
        model_revision=model_args.model_revision,
        bits=quant_args.bits,
        rank_allocation=quant_args.rank_allocation,
        seed=quant_args.seed,
        num_calib_samples=quant_args.num_calib_samples,
        calib_dataset=quant_args.calib_dataset,
        calib_shrinkage=quant_args.calib_shrinkage,
        calib_strategy=quant_args.calib_strategy,
        seqlen=model_args.seqlen,
        device_map=model_args.device_map,
        block_io_batch_size=model_args.block_io_batch_size,
        require_fast_linear_attention=model_args.require_fast_linear_attention,
        attn_implementation=model_args.attn_implementation,
        tune_schedule=tune_args.tune_schedule,
        layer_epoch_schedule=tune_args.layer_epoch_schedule,
        refresh_input_stats=tune_args.refresh_input_stats,
        resume_dir=tune_args.resume_dir,
        rank_budget=tune_args.rank_budget,
        rank_probe_candidates=tune_args.rank_probe_candidates,
        rank_probe_iters=tune_args.rank_probe_iters,
        correlation_block_size=tune_args.correlation_block_size,
        correlation_layers=tune_args.correlation_layers,
        nonfact_plateau_tolerance=tune_args.nonfact_plateau_tolerance,
        nonfact_plateau_min_epochs=tune_args.nonfact_plateau_min_epochs,
        nonfact_plateau_patience=tune_args.nonfact_plateau_patience,
        tune_nonfact=tune_args.tune_nonfact,
        nonfact_lr=tune_args.nonfact_lr,
        nonfact_batch_size=tune_args.nonfact_batch_size,
        nonfact_epochs=tune_args.nonfact_epochs,
        admm_type=tune_args.admm_type,
        admm_outer_iters=tune_args.admm_outer_iters,
        admm_inner_iters=tune_args.admm_inner_iters,
        admm_warm_start_iters=tune_args.admm_warm_start_iters,
        admm_reg=tune_args.admm_reg,
        admm_early_stop=tune_args.admm_early_stop,
        admm_min_outer_iters=tune_args.admm_min_outer_iters,
        admm_check_interval=tune_args.admm_check_interval,
        admm_convergence_tolerance=tune_args.admm_convergence_tolerance,
        admm_stable_sign_tolerance=tune_args.admm_stable_sign_tolerance,
        admm_rho_stop_threshold=tune_args.admm_rho_stop_threshold,
        admm_penalty_scheduler=tune_args.admm_penalty_scheduler,
        admm_print_steps=tune_args.admm_print_steps,
        log_reconstruction_error=tune_args.log_reconstruction_error,
        tune_fact=tune_args.tune_fact,
        fact_binary_lr=tune_args.fact_binary_lr,
        fact_scale_lr=tune_args.fact_scale_lr,
        fact_bias_lr=tune_args.fact_bias_lr,
        fact_batch_size=tune_args.fact_batch_size,
        fact_epochs=tune_args.fact_epochs,
        tune_model=tune_args.tune_model,
        model_kd_lr=tune_args.model_kd_lr,
        model_kd_batch_size=tune_args.model_kd_batch_size,
        model_kd_num_samples=tune_args.model_kd_num_samples,
        model_kd_vocab_chunk_size=tune_args.model_kd_vocab_chunk_size,
        model_kd_pack_factors=tune_args.model_kd_pack_factors,
        eval_after_each_block=eval_args.eval_after_each_block,
    )

    if model_args.from_hub:
        hub_id = model_args.hub_model_id or model_args.model_id
        logger.info(f"Loading pre-quantized model from Hub: {hub_id}")
        nanoquant_model = NanoQuantModel.from_pretrained(
            hub_id, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=model_args.attn_implementation,
            model_revision=model_args.model_revision,
        )
        loaded_from_hub = True
    else:
        logger.info(f"Quantizing model: {model_args.model_id}")
        nanoquant_model = NanoQuantModel.from_pretrained_quantize(
            model_id=model_args.model_id,
            qmodel_path=model_args.qmodel_path,
            quant_config=quant_config,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        loaded_from_hub = False

    model = nanoquant_model.model
    cleanup_memory()

    if model_args.qmodel_path and not os.path.exists(model_args.qmodel_path) and not loaded_from_hub:
        nanoquant_model._save_checkpoint(model, model_args.qmodel_path)
        logger.info(f"Saved quantized model to {model_args.qmodel_path}")

    model.eval()
    if not loaded_from_hub:
        model = model.cuda()

    tokenizer = load_tokenizer(model_args.model_id, revision=model_args.model_revision)

    try:
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            tasks_str=eval_args.zeroshot_task,
            eval_ppl=eval_args.ppl_task,
            num_fewshot=eval_args.num_fewshot,
            limit=eval_args.limit,
            batch_size="auto" if eval_args.batch_size is None else eval_args.batch_size,
            calibration_dataset_path=quant_config.calib_dataset,
        )
        logger.info(f"Results:\n{json.dumps(results, indent=2)}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise RuntimeError(f"Evaluation failed: {e}") from e


if __name__ == "__main__":
    main()
