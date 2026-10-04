"""Backend selection checks, with imports/devices mocked; no kernel execution."""

from types import SimpleNamespace

import pytest

from nanoquant.utils import attention_utils, load_utils


@pytest.fixture
def config():
    return SimpleNamespace(model_type="qwen3_5_text", hidden_size=5120,
                           num_attention_heads=24, head_dim=256, attention_dropout=0.0)


def gpu(monkeypatch, capability=(9, 0)):
    monkeypatch.setattr(attention_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(attention_utils.torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(attention_utils.torch.version, "hip", None)


def test_sdpa_does_not_import_or_query_device(config, monkeypatch):
    monkeypatch.setattr(attention_utils, "_preload_flash_backend", lambda _: pytest.fail("unexpected import"))
    assert attention_utils.resolve_attention_backend(config, "sdpa") == "sdpa"


def test_auto_cpu_and_cpu_checkpoint_use_sdpa(config, monkeypatch):
    monkeypatch.setattr(attention_utils.torch.cuda, "is_available", lambda: False)
    assert attention_utils.resolve_attention_backend(config) == "sdpa"
    gpu(monkeypatch)
    assert attention_utils.resolve_attention_backend(config, execution_device="cpu") == "sdpa"


@pytest.mark.parametrize("capability,expected", [((8, 0), "flash_attention_2"),
                                              ((8, 9), "flash_attention_2"),
                                              ((9, 0), "flash_attention_3")])
def test_auto_selects_training_family_for_head256(config, monkeypatch, capability, expected):
    gpu(monkeypatch, capability)
    monkeypatch.setattr(attention_utils, "_preload_flash_backend", lambda backend: backend)
    assert attention_utils.resolve_attention_backend(config) == expected


def test_auto_reports_import_failure_then_falls_back(config, monkeypatch):
    gpu(monkeypatch)
    seen = []

    def unavailable(backend):
        seen.append(backend)
        raise ImportError("no build for this Torch/CUDA")

    monkeypatch.setattr(attention_utils, "_preload_flash_backend", unavailable)
    assert attention_utils.resolve_attention_backend(config) == "sdpa"
    assert seen == ["flash_attention_3", "flash_attention_2"]
    with pytest.raises(RuntimeError, match="before model weights"):
        attention_utils.resolve_attention_backend(config, "flash_attention_3")


def test_fa3_hub_loads_training_repo(config, monkeypatch):
    gpu(monkeypatch)
    seen = []
    kernel = SimpleNamespace(flash_attn_func=lambda: None, flash_attn_varlen_func=lambda: None)

    def load(repo):
        seen.append(repo)
        return kernel

    modules = {
        "transformers.utils": SimpleNamespace(is_flash_attn_3_available=lambda: False,
                                               is_kernels_available=lambda: True),
        "transformers.integrations.hub_kernels": SimpleNamespace(load_and_register_attn_kernel=load),
    }
    monkeypatch.setattr(attention_utils, "import_module", modules.__getitem__)
    assert attention_utils.resolve_attention_backend(config, "flash_attention_3") == "kernels-community/flash-attn3"
    assert seen == ["kernels-community/flash-attn3"]


def test_beta_fa4_is_explicit(config, monkeypatch):
    gpu(monkeypatch, (10, 0))
    monkeypatch.setattr(attention_utils, "_preload_flash_backend", lambda backend: backend)
    assert attention_utils.resolve_attention_backend(config) == "sdpa"
    assert attention_utils.resolve_attention_backend(config, "flash_attention_4") == "flash_attention_4"


def test_float32_checkpoint_keeps_sdpa(config, monkeypatch):
    gpu(monkeypatch)
    dtype = attention_utils.torch.float32
    assert attention_utils.resolve_attention_backend(config, dtype=dtype) == "sdpa"
    with pytest.raises(RuntimeError, match="BF16/FP16"):
        attention_utils.resolve_attention_backend(config, "flash_attention_3", dtype=dtype)


def test_rejects_sage_and_incompatible_dimensions(config, monkeypatch):
    gpu(monkeypatch)
    with pytest.raises(ValueError, match="attn_implementation"):
        attention_utils.resolve_attention_backend(config, "sageattention")
    config.head_dim = 512
    assert attention_utils.resolve_attention_backend(config) == "sdpa"
    with pytest.raises(RuntimeError, match="head_dim"):
        attention_utils.resolve_attention_backend(config, "flash_attention_3")


def test_dropout_skips_fa3_and_consumer_head256_fa2(config, monkeypatch):
    gpu(monkeypatch)
    config.attention_dropout = 0.1
    monkeypatch.setattr(attention_utils, "_preload_flash_backend", lambda backend: backend)
    assert attention_utils.resolve_attention_backend(config) == "flash_attention_2"
    gpu(monkeypatch, (8, 9))
    assert attention_utils.resolve_attention_backend(config) == "sdpa"


def test_flashqla_checks_dispatcher_without_replacing_model(config, monkeypatch):
    gpu(monkeypatch)
    config.linear_key_head_dim = config.linear_value_head_dim = 128
    monkeypatch.delenv("FLA_DISABLE_BACKEND_DISPATCH", raising=False)
    monkeypatch.delenv("FLA_FLASH_QLA", raising=False)
    monkeypatch.setattr(load_utils, "find_spec", lambda _: object())
    calls = []
    module = SimpleNamespace(FlashQLABackend=object, chunk_gated_delta_rule=lambda: None,
                             chunk_gated_delta_rule_fwd=lambda: None, chunk_gated_delta_rule_bwd=lambda: None)

    def imported(name):
        calls.append(name)
        return module

    monkeypatch.setattr(load_utils, "import_module", imported)
    load_utils._check_flashqla_backend(config)
    assert calls == ["fla.ops.gated_delta_rule.backends.flash_qla", "flash_qla"]


def test_flashqla_environment_opt_out_prevents_import(config, monkeypatch):
    gpu(monkeypatch)
    monkeypatch.setattr(load_utils, "find_spec", lambda _: object())
    monkeypatch.setenv("FLA_FLASH_QLA", "0")
    monkeypatch.setattr(load_utils, "import_module", lambda _: pytest.fail("unexpected FlashQLA import"))
    load_utils._check_flashqla_backend(config)
