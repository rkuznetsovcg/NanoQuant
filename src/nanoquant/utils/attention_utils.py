"""Select training-capable full-attention kernels before loading weights."""

from importlib import import_module

import torch


ATTENTION_BACKENDS = ("auto", "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4")
_HUB_BACKENDS = {
    "flash_attention_2": "kernels-community/flash-attn2",
    # Do not use Transformers' FA3 fallback: some releases route it to vLLM.
    "flash_attention_3": "kernels-community/flash-attn3",
    "flash_attention_4": "kernels-community/flash-attn4",
}
_NATIVE_MODULES = {
    "flash_attention_2": "flash_attn",
    "flash_attention_3": "flash_attn_interface",
    "flash_attention_4": "flash_attn.cute",
}


def _preload_flash_backend(backend):
    """Import/register a kernel, without running forward, backward or a benchmark."""
    utils = import_module("transformers.utils")
    native_check = getattr(utils, f"is_flash_attn_{backend[-1]}_available", lambda: False)
    errors = []
    if native_check():
        try:
            module = import_module(_NATIVE_MODULES[backend])
            if not all(callable(getattr(module, name, None)) for name in
                       ("flash_attn_func", "flash_attn_varlen_func")):
                raise ImportError("missing regular or variable-length attention function")
            return backend
        except Exception as error:
            errors.append(f"native: {error}")
    if utils.is_kernels_available():
        try:
            integration = import_module("transformers.integrations.hub_kernels")
            repo = _HUB_BACKENDS[backend]
            module = integration.load_and_register_attn_kernel(repo)
            if not all(callable(getattr(module, name, None)) for name in
                       ("flash_attn_func", "flash_attn_varlen_func")):
                raise ImportError("missing regular or variable-length attention function")
            return repo
        except Exception as error:
            errors.append(f"Hub: {error}")
    raise RuntimeError("; ".join(errors) or "no compatible native package or Transformers kernels extra")


def resolve_attention_backend(config, requested="auto", execution_device=None, dtype=torch.bfloat16):
    """Use BF16 Flash kernels with backward; retain SDPA when auto cannot load one.

    Quantization stages on CUDA even when weights initially load to CPU. Pass
    execution_device for checkpoint inference, where an actual CPU target matters.
    FA4 is explicit because its Transformers integration is still beta.
    """
    if requested not in ATTENTION_BACKENDS:
        raise ValueError(f"attn_implementation must be one of {ATTENTION_BACKENDS}; got {requested!r}")
    if requested == "sdpa":
        return requested
    if dtype not in {torch.bfloat16, torch.float16}:
        if requested != "auto":
            raise RuntimeError(f"{requested} requires BF16/FP16 weights; got {dtype}")
        return "sdpa"
    text_config = getattr(config, "text_config", config)
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = text_config.hidden_size // text_config.num_attention_heads
    cpu_target = isinstance(execution_device, (str, torch.device)) and str(execution_device).split(":")[0] in {"cpu", "mps"}
    cuda = torch.cuda.is_available() and not getattr(torch.version, "hip", None) and not cpu_target
    if not cuda or head_dim > 256:
        if requested != "auto":
            raise RuntimeError(f"{requested} requires NVIDIA CUDA and head_dim <= 256 (got {head_dim})")
        print("Full attention: SDPA (no eligible NVIDIA CUDA target or head dimension).")
        return "sdpa"
    major, minor = torch.cuda.get_device_capability()
    dropout = float(getattr(text_config, "attention_dropout", 0.0))
    if requested == "auto":
        candidates = ["flash_attention_3", "flash_attention_2"] if major == 9 else ["flash_attention_2"] if major == 8 else []
    else:
        supported = (major >= 8 if requested == "flash_attention_2" else
                     major == 9 if requested == "flash_attention_3" else major in {9, 10, 11})
        if not supported:
            raise RuntimeError(f"{requested} is not enabled for training on CUDA capability {major}.{minor}")
        candidates = [requested]
    for backend in candidates:
        # FA3/4 interfaces have no dropout; FA2's consumer-GPU head-256
        # backward support also requires zero dropout.
        if dropout and (backend != "flash_attention_2" or
                        (head_dim > 192 and (major, minor) in {(8, 6), (8, 9)})):
            if requested != "auto":
                raise RuntimeError(f"{backend} training is not enabled with attention_dropout={dropout} on this GPU")
            continue
        try:
            selected = _preload_flash_backend(backend)
            print(f"Full attention: {selected}; {dtype} forward/backward, head_dim={head_dim}.")
            return selected
        except Exception as error:
            if requested != "auto":
                raise RuntimeError(
                    f"Cannot load {backend} before model weights: {error}. "
                    "Install `pip install -e '.[kernels]'` for compatible Hub builds, "
                    "or install the native FlashAttention package for this GPU."
                ) from error
            print(f"Full attention: {backend} unavailable ({error}).")
    print("Full attention: SDPA; PyTorch selects its available attention kernel. FA4 requires explicit selection.")
    return "sdpa"
