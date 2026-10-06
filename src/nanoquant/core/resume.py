"""Atomic block checkpoints, validated against actual source weights and tokens."""
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from ..modules.linear import NanoQuantLinear
from .linearized_block import LinearizedDecoderBlock
from ..utils.utils import get_decoder_layers


def update_tensor_digest(digest, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))


def content_key(tensors, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())
    for tensor in tensors:
        update_tensor_digest(digest, tensor)
    return digest.hexdigest()


def has_block_checkpoint(config):
    directory = config.get("resume_dir", "")
    return bool(directory and (Path(directory) / "manifest.json").is_file())


def _atomic_save(payload, path):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


class BlockCheckpoint:
    """Commit one block shard and one activation snapshot per completed block.

    Only the current block is mutated by block reconstruction. Save its complete
    state (including dense projections not selected for quantization); untouched
    future blocks remain identical to the validated base checkpoint. Restoring
    every committed shard reconstructs the tuned prefix without a model copy.
    """
    def __init__(self, model, tokens, config, reference_model=None):
        self.directory = Path(config["resume_dir"]) if config.get("resume_dir") else None
        self.manifest = None
        self.linearize_block_index = config.get("linearize_block_index")
        self.total_blocks = len(get_decoder_layers(model))
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        settings = {key: value for key, value in config.items()
                    if key not in {"resume_dir", "eval_after_each_block"}}
        # Keep old baseline manifests resumable when the optional pilot is off.
        # These controls had no effect in earlier versions and remain inert
        # unless a block index is explicitly selected.
        if config.get("linearize_block_index") is None:
            for key in ("linearize_block_index", "linearize_max_tokens", "linearize_chunk_tokens", "linearize_ridge"):
                settings.pop(key, None)
        settings["model_config"] = model.config.to_dict()
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())
        update_tensor_digest(digest, tokens)
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            update_tensor_digest(digest, tensor)
        if reference_model is not None and reference_model is not model:
            for name, tensor in reference_model.state_dict().items():
                digest.update(("teacher."+name).encode())
                update_tensor_digest(digest, tensor)
        self.signature = digest.hexdigest()
        path = self.directory / "manifest.json"
        if path.exists():
            self.manifest = json.loads(path.read_text())
            if self.manifest.get("version") != 1 or self.manifest.get("signature") != self.signature:
                raise ValueError("Resume checkpoint differs in source weights, calibration tokens or settings; use another resume_dir")

    def restore(self, model):
        if self.manifest is None:
            return None
        blocks = get_decoder_layers(model)
        initial = torch.load(self.directory / "initial.pt", map_location="cpu", weights_only=True)
        completed_blocks = self.manifest["completed_blocks"]
        if self.linearize_block_index is not None and int(self.linearize_block_index) < completed_blocks:
            index = int(self.linearize_block_index)
            if index >= len(blocks):
                raise ValueError(f"Resumed linearized block index {index} is outside the model depth")
            text_config = getattr(model.config, "text_config", model.config)
            parameter = next(blocks[index].parameters(), None)
            device = parameter.device if parameter is not None else torch.device("cpu")
            dtype = parameter.dtype if parameter is not None else torch.bfloat16
            blocks[index] = LinearizedDecoderBlock(int(text_config.hidden_size), device=device, dtype=dtype)
        for name, module in model.named_modules():
            if name in initial["stats"]:
                i_norm, o_norm = initial["stats"][name]
                module.register_buffer("i_norm", i_norm, persistent=False)
                module.register_buffer("o_norm", o_norm, persistent=False)
        for index in range(completed_blocks):
            payload = torch.load(self.directory / f"block-{index:04d}.pt", map_location="cpu", weights_only=True)
            modules = dict(blocks[index].named_modules())
            for name, metadata in payload["linears"].items():
                module = modules[name]
                module.__class__ = NanoQuantLinear
                module.init_for_inference(metadata["rank"], metadata["has_mid"])
            blocks[index].load_state_dict(payload["state"], strict=True)
            for parameter in blocks[index].parameters():
                parameter.requires_grad_(False)
        snapshot = torch.load(self.directory / self.manifest["snapshot"], map_location="cpu", weights_only=True)
        random.setstate(snapshot["rng"]["python"])
        numpy_state = snapshot["rng"]["numpy"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        torch.set_rng_state(snapshot["rng"]["torch"])
        if torch.cuda.is_available() and snapshot["rng"]["cuda"]:
            saved_cuda = snapshot["rng"]["cuda"]
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.set_rng_state(saved_cuda[min(device_index, len(saved_cuda)-1)], device_index)
        snapshot["ranks"] = initial["ranks"] | snapshot["ranks"]
        print(f"Resuming after {completed_blocks} completed decoder blocks")
        return snapshot

    def initialize(self, model, ranks, original_inputs, compressed_inputs, kwargs):
        if self.directory is None or self.manifest is not None:
            return
        stats = {name: (module.i_norm, module.o_norm) for name, module in model.named_modules()
                 if hasattr(module, "i_norm") and hasattr(module, "o_norm")}
        _atomic_save(_cpu_tree({"stats": stats, "ranks": ranks}), self.directory / "initial.pt")
        self.save(None, 0, ranks, original_inputs, compressed_inputs, kwargs)

    def save(self, block, completed_blocks, ranks, original_inputs, compressed_inputs, kwargs):
        if self.directory is None:
            return
        if block is not None:
            metadata = {name: {"rank": module.rank, "has_mid": hasattr(module, "scale_mid")}
                        for name, module in block.named_modules() if isinstance(module, NanoQuantLinear)}
            _atomic_save(_cpu_tree({"state": block.state_dict(), "linears": metadata}),
                         self.directory / f"block-{completed_blocks-1:04d}.pt")
        numpy_state = np.random.get_state()
        payload = {"ranks": ranks, "original_inputs": original_inputs,
                   "compressed_inputs": compressed_inputs, "kwargs": kwargs,
                   "completed_blocks": completed_blocks,
                   "rng": {"python": random.getstate(),
                           "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                           "torch": torch.get_rng_state(),
                           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        filename = "activations-" + uuid.uuid4().hex + ".pt"
        _atomic_save(_cpu_tree(payload), self.directory / filename)
        manifest = {"version": 1, "signature": self.signature, "completed_blocks": completed_blocks,
                    "snapshot": filename, "stage": "pre_kd" if completed_blocks == self.total_blocks else "reconstruction"}
        temporary = self.directory / "manifest.tmp"
        with temporary.open("w") as stream:
            json.dump(manifest, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.directory / "manifest.json")
        if self.manifest is not None:
            (self.directory / self.manifest["snapshot"]).unlink(missing_ok=True)
        self.manifest = manifest
