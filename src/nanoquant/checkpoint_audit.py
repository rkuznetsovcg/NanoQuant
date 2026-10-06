"""Inspect an existing checkpoint and optionally evaluate WikiText without requantizing."""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .core.linearized_block import LINEARIZED_BLOCK_INDICES_KEY, LinearizedDecoderBlock
from .utils.utils import get_decoder_layers


def _shape(state, key, errors):
    packed_key = key + "_packed"
    shape_key = key + "_shape"
    if key in state and packed_key in state:
        errors.append(f"Both packed and dense factor present: {key}")
    if packed_key in state:
        if shape_key not in state:
            errors.append(f"Missing shape: {shape_key}")
            return None
        shape_tensor = state[shape_key]
        if (not isinstance(shape_tensor, torch.Tensor) or shape_tensor.ndim != 1
                or shape_tensor.numel() != 2 or shape_tensor.dtype not in (torch.int32, torch.int64)):
            errors.append(f"Invalid integer shape metadata: {shape_key}")
            return None
        shape = tuple(int(value) for value in shape_tensor.tolist())
        if min(shape) <= 0:
            errors.append(f"Invalid shape: {shape_key}={shape}")
            return None
        tensor = state[packed_key]
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32
                or tuple(tensor.shape) != (shape[0], (shape[1]+31)//32)):
            errors.append(f"Invalid int32 packed tensor or dimensions: {packed_key}")
        return shape
    tensor = state.get(key)
    if tensor is not None and (not isinstance(tensor, torch.Tensor) or tensor.ndim != 2
                               or min(tensor.shape) <= 0):
        errors.append(f"Invalid dense factor: {key}")
        return None
    return tuple(tensor.shape) if tensor is not None else None


def audit_state_dict(state, template, target_bits=0.55, progress=False):
    """Validate shapes/keys/numbers without expanding the binary factor storage."""
    errors, warnings, layers = [], [], []
    if not isinstance(state, dict):
        raise TypeError("The checkpoint must contain a state_dict dictionary")
    linearized_indices = []
    linearized_metadata = state.pop(LINEARIZED_BLOCK_INDICES_KEY, None)
    if linearized_metadata is not None:
        if (not isinstance(linearized_metadata, torch.Tensor) or linearized_metadata.ndim != 1
                or linearized_metadata.dtype not in (torch.int32, torch.int64)):
            errors.append("Invalid linearized block index metadata")
        else:
            linearized_indices = [int(index) for index in linearized_metadata.tolist()]
            if len(linearized_indices) != len(set(linearized_indices)):
                errors.append("Duplicate linearized block indices")
            if getattr(template.config, "model_type", None) not in {"qwen3_5", "qwen3_5_text"}:
                errors.append("Linearized block metadata is unsupported for this model architecture")
            else:
                blocks = get_decoder_layers(template)
                text_config = getattr(template.config, "text_config", template.config)
                for index in linearized_indices:
                    if index < 0 or index >= len(blocks):
                        errors.append(f"Linearized block index {index} is outside the model depth")
                        continue
                    blocks[index] = LinearizedDecoderBlock(
                        int(text_config.hidden_size), device="meta", dtype=torch.bfloat16,
                    )
    expected = {key: tuple(tensor.shape) for key, tensor in template.state_dict().items()}
    embedding = getattr(template, "get_input_embeddings", lambda: None)()
    embedding_key = next((f"{name}.weight" for name, module in template.named_modules() if module is embedding), None)
    consumed = set()
    for name, module in template.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        prefix = name + "."
        if not any(prefix + suffix in state for suffix in ("U", "V", "U_packed", "V_packed")):
            continue
        u_shape = _shape(state, prefix+"U", errors)
        v_shape = _shape(state, prefix+"V", errors)
        expected.pop(prefix+"weight", None)
        for factor in ("U", "V"):
            for suffix in ("", "_packed", "_shape"):
                if prefix+factor+suffix in state:
                    consumed.add(prefix+factor+suffix)
        if u_shape is None or v_shape is None:
            errors.append(f"Incomplete binary factors: {name}")
            continue
        rank = v_shape[0]
        if v_shape != (rank, module.in_features) or u_shape != (module.out_features, rank):
            errors.append(f"Factor dimensions disagree with layer: {name}, U={u_shape}, V={v_shape}")
        expected[prefix+"scale_pre"] = (1, module.in_features)
        expected[prefix+"scale_post"] = (1, module.out_features)
        if prefix+"scale_mid" in state:
            expected[prefix+"scale_mid"] = (1, rank)
        for factor in ("U", "V"):
            value = state.get(prefix+factor)
            if isinstance(value, torch.Tensor) and not torch.all((value == -1) | (value == 1)):
                errors.append(f"Nonbinary unpacked factor: {prefix+factor}")
        original_elements = module.in_features*module.out_features
        stored_bytes = sum(state[key].numel()*state[key].element_size() for key in consumed
                           if key.startswith(prefix) and not key.endswith("_shape")
                           and isinstance(state[key], torch.Tensor))
        for scale in ("scale_pre", "scale_post", "scale_mid"):
            value = state.get(prefix+scale)
            if isinstance(value, torch.Tensor):
                stored_bytes += value.numel()*value.element_size()
                if torch.count_nonzero(value).item() == 0:
                    warnings.append(f"All-zero scale: {prefix+scale}")
        channels = module.in_features+module.out_features
        raw_rank = (original_elements*target_bits-16*channels)/(channels+(16 if prefix+"scale_mid" in state else 0))
        uniform_rank = max(32, int(raw_rank//32)*32)
        uniform_rank = min(uniform_rank, module.in_features, module.out_features)
        layers.append({"name": name, "rank": rank, "uniform_rank": uniform_rank,
                       "rank_relative_to_uniform": rank/max(1, uniform_rank),
                       "original_elements": original_elements, "stored_bytes": stored_bytes,
                       "bpw": 8*stored_bytes/max(1, original_elements)})

    for key, shape in expected.items():
        if key not in state:
            tied_head = (key == "lm_head.weight" and
                         getattr(getattr(template, "config", None), "tie_word_embeddings", False) and
                         embedding_key in state)
            if not tied_head:
                errors.append(f"Missing weight or persistent buffer: {key}")
        elif not isinstance(state[key], torch.Tensor):
            errors.append(f"Nontensor checkpoint value: {key}")
        elif tuple(state[key].shape) != shape:
            errors.append(f"Wrong shape: {key}, expected {shape}, got {tuple(state[key].shape)}")
        consumed.add(key)
    errors.extend(f"Unexpected weight: {key}" for key in state.keys()-consumed)
    for index, (key, tensor) in enumerate(state.items()):
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"Nontensor checkpoint value: {key}")
            continue
        if tensor.is_floating_point():
            flat = tensor.reshape(-1)
            for start in range(0, flat.numel(), 2**20):
                if not torch.isfinite(flat[start:start+2**20]).all():
                    errors.append(f"Non-finite weight: {key}")
                    break
        if progress and (index+1) % 512 == 0:
            print(f"Checked values: {index+1}/{len(state)} tensors", flush=True)
    if not layers:
        errors.append("No NanoQuant compressed linear layers found in checkpoint")
    selected_elements = sum(layer["original_elements"] for layer in layers)
    return {"status": "invalid" if errors else "valid_format", "errors": errors, "warnings": warnings,
            "quantized_linear_count": len(layers), "layers": layers,
            "linearized_block_indices": linearized_indices,
            "selected_linears_bpw": 8*sum(layer["stored_bytes"] for layer in layers)/max(1, selected_elements),
            "tensor_storage_gib": sum(value.numel()*value.element_size() for value in state.values()
                                      if isinstance(value, torch.Tensor))/2**30,
            "quality_measured": False}


def inspect_checkpoint(checkpoint, source_config, target_bits=0.55):
    """Inspect packed storage on CPU before spending GPU time on evaluation."""
    source_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        template = AutoModelForCausalLM.from_config(source_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    print("Checking every checkpoint key, shape and floating-point weight…", flush=True)
    report = audit_state_dict(state, template, target_bits, progress=True)
    del state, template
    gc.collect()
    print(f"Checkpoint format: {report['status']}; compressed layers: {report['quantized_linear_count']}; "
          f"selected weights: {report['selected_linears_bpw']:.3f} bpw", flush=True)
    for message in report["errors"]+report["warnings"]:
        print(message, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--ppl", action="store_true", help="Evaluate only the existing checkpoint on WikiText-2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    revision = args.revision or config.get("model_revision")
    source_config = AutoConfig.from_pretrained(config["model_id"], revision=revision, trust_remote_code=True)
    report = inspect_checkpoint(args.checkpoint, source_config, float(config.get("bits", 0.55)))
    report.update({"checkpoint": str(Path(args.checkpoint).resolve()), "model_id": config["model_id"],
                   "model_revision": revision, "checkpoint_size_bytes": Path(args.checkpoint).stat().st_size})
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    if report["errors"]:
        raise RuntimeError(f"Checkpoint validation failed. Details: {output}")
    if args.ppl:
        from .utils.data_utils import get_test_loaders
        from .utils.eval_utils import evaluate_ppl
        from .utils.load_utils import load_compressed_model, load_tokenizer

        model = load_compressed_model(config["model_id"], args.checkpoint, int(config["seqlen"]), args.device,
                                      dtype=torch.bfloat16, attn_implementation="sdpa", revision=revision)
        tokenizer = load_tokenizer(config["model_id"], revision=revision)
        _, test = get_test_loaders("wikitext2", model_name=config["model_id"], tokenizer=tokenizer)
        report["wikitext2"] = {"ppl": evaluate_ppl(model, test, args.device, "wikitext2")}
        report["quality_measured"] = True
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(f"Checkpoint audit: {output}", flush=True)


if __name__ == "__main__":
    main()
