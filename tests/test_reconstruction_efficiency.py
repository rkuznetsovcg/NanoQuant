"""Focused checks for cost reductions; no model downloads or CUDA required."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nanoquant.core import compress_model
from nanoquant.core.admm_dbf import _admm_solve_step
from nanoquant.core.importance import _online_clipping_hook
from nanoquant.utils.load_utils import _check_fast_linear_attention
from nanoquant.utils.utils import estimate_weight_storage, extract_hidden_states


@pytest.mark.parametrize("chunk_size", [1, 3, 11])
@pytest.mark.parametrize("temperature,logit_scale,softcap", [(1.0, 1.0, 0.0), (0.7, 1.3, 4.0)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_online_kd_matches_dense_loss_and_hidden_gradient(chunk_size, temperature, logit_scale, softcap, with_bias):
    generator = torch.Generator().manual_seed(7)
    student = torch.randn(2, 4, 5, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 4, 5, generator=generator)
    weight = torch.randn(11, 5, generator=generator)
    bias = torch.randn(11, generator=generator) if with_bias else None
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    def logits(hidden):
        result = F.linear(hidden, weight, bias).float() * logit_scale
        if softcap:
            result = softcap * torch.tanh(result / softcap)
        return result / temperature

    dense = -(logits(teacher).softmax(-1) * logits(student).log_softmax(-1)).sum(-1)
    dense = (dense * mask).sum() / mask.sum() * temperature**2
    dense_grad = torch.autograd.grad(dense, student)[0]
    online = compress_model._ChunkedLinearKDLoss.apply(
        student, teacher, weight, bias, mask, temperature, chunk_size, logit_scale, softcap
    )
    online_grad = torch.autograd.grad(online, student)[0]
    torch.testing.assert_close(online, dense, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(online_grad, dense_grad, atol=2e-5, rtol=2e-5)


def test_online_kd_projects_each_hidden_state_once_per_tile(monkeypatch):
    original = compress_model._ChunkedLinearKDLoss._logits
    calls = []

    def record(*args):
        calls.append((args[3], args[4]))
        return original(*args)

    monkeypatch.setattr(compress_model._ChunkedLinearKDLoss, "_logits", staticmethod(record))
    hidden = torch.zeros(1, 2, 4)
    weight = torch.ones(11, 4)
    loss = compress_model._ChunkedLinearKDLoss.apply(
        hidden, hidden, weight, None, torch.ones(1, 2), 1.0, 3, 1.0, 0.0
    )
    assert torch.isfinite(loss)
    assert len(calls) == 2 * math.ceil(11 / 3)


def test_group_cache_stops_before_downstream_work_and_preserves_shuffled_outputs(monkeypatch):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(4, 4)
            self.downstream = nn.Linear(4, 4)
            self.downstream_calls = 0

        def forward(self, inputs):
            output = self.attention(inputs)
            self.downstream_calls += 1
            return (self.downstream(output),)

    block = Block()
    inputs = torch.arange(60, dtype=torch.float32).reshape(5, 3, 4)
    patches = []
    compress_model._capture_and_patch_group(block, block.attention, inputs, {}, 2, patches)
    assert block.downstream_calls == 0
    assert len(patches) == 1
    indices = [4, 0, 2]
    block._nanoquant_active_indices = indices
    expected_attention = F.linear(inputs[indices], block.attention.weight, block.attention.bias)
    expected = block.downstream(expected_attention)
    actual = block(inputs[indices])[0]
    torch.testing.assert_close(actual, expected)
    # Qwen passes hidden_states by keyword. A meta tensor lets this test check
    # device routing without requiring CUDA or executing any linear operation.
    requested_devices = []
    original_stack = compress_model._tree_stack_cached

    def record_device(values, device):
        requested_devices.append(device)
        return original_stack(values, "cpu")

    monkeypatch.setattr(compress_model, "_tree_stack_cached", record_device)
    block.attention.forward(hidden_states=torch.empty(3, 3, 4, device="meta"))
    assert requested_devices[0].type == "meta"
    for module, original_forward in patches:
        module.forward = original_forward


def test_owned_teacher_is_released_before_student_moves_to_training_device(monkeypatch):
    teacher_refs = []

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.model = Body()

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)

        def forward(self, tokens):
            return (self.embedding(tokens),)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True, pad_token_id=None)
            self.model = Body()

        def to(self, *args, **kwargs):
            assert teacher_refs and teacher_refs[0]() is None
            return super().to(*args, **kwargs)

    def load_teacher(model_id, seqlen, **kwargs):
        assert model_id == "local-teacher" and seqlen == 3
        teacher = Teacher()
        teacher_refs.append(weakref.ref(teacher))
        return teacher

    monkeypatch.setattr(compress_model, "load_model", load_teacher)
    config = {"seed": 0, "model_id": "local-teacher", "seqlen": 3,
              "model_kd_num_samples": 2, "model_kd_batch_size": 2}
    student = Student()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert compress_model.compress_model_recon(student, None, tokens, config, dev="cpu") is student
    assert teacher_refs[0]() is None


def test_fast_kernel_requirement_reports_import_failure(monkeypatch):
    from nanoquant.utils import load_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def unavailable(_module_name):
        raise ImportError("kernel extension unavailable")

    monkeypatch.setattr(load_utils, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="qwen-fast"):
        _check_fast_linear_attention(SimpleNamespace(model_type="qwen3_5_text"), required=True)


def test_storage_estimate_counts_unchanged_tied_weights_once():
    model = nn.Module()
    model.embedding = nn.Embedding(3, 4)
    model.head = nn.Linear(4, 3, bias=False)
    model.head.weight = model.embedding.weight
    model.selected = nn.Linear(4, 8, bias=True)
    estimate = estimate_weight_storage(model, [(model.selected, 2)])
    remaining_bytes = (3 * 4 + 8) * 4  # float32 embedding and untouched bias
    assert estimate["remaining_weight_bytes"] == remaining_bytes
    # Row padding: V has 2 words, U has 8 words; scales occupy 24 bytes.
    assert estimate["packed_weight_bytes"] == remaining_bytes + 64
    assert estimate["unpacked_weight_bytes"] == remaining_bytes + 2 * (2 * (4 + 8) + 4 + 8)


@pytest.mark.parametrize("previous_max", [None, 2.0, 6.0])
def test_device_side_clipping_keeps_original_statistics_update(previous_max):
    inputs = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    previous = torch.tensor(previous_max) if previous_max is not None else None
    states = {"layer": {"i_norm": {"global_max": previous}}}
    stats = {"i_norm": {"layer": torch.ones(2)}}
    tau = inputs.norm(dim=1).max()
    expected = torch.ones(2)
    gmax = tau if previous is None else previous
    if previous is not None and tau > previous:
        expected *= (tau / (previous + 1e-8)).square()
        gmax = tau
    clip = (gmax / (inputs.norm(dim=1, keepdim=True) + 1e-8)).clamp(max=1.0)
    expected += (inputs * clip).square().mean(0)
    _online_clipping_hook(None, (inputs,), None, "layer", stats, states, "cpu", is_forward=True)
    torch.testing.assert_close(stats["i_norm"]["layer"], expected)
    torch.testing.assert_close(states["layer"]["i_norm"]["global_max"], gmax)


def test_dbf_solve_preserves_regularized_normal_equations():
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(13, 5, generator=generator)
    y = torch.randn(13, 7, generator=generator)
    z = torch.randn(5, 7, generator=generator)
    u = torch.randn(5, 7, generator=generator)
    rho, reg = 0.2, 0.03
    gram = x.T @ x
    system = gram + torch.eye(5) * (gram.diagonal().mean() * reg + rho)
    expected = torch.linalg.solve(system, x.T @ y + rho * (z - u))
    _z, _u, factor = _admm_solve_step(x, y, z, u, rho, reg=reg, inner_iters=1)
    torch.testing.assert_close(factor, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "model_output"])
def test_block_output_extraction_preserves_every_sample(output_kind):
    hidden = torch.arange(60).reshape(5, 3, 4)
    output = hidden
    if output_kind == "tuple":
        output = (hidden, None)
    elif output_kind == "model_output":
        output = SimpleNamespace(last_hidden_state=hidden)
    assert extract_hidden_states(output) is hidden
    assert extract_hidden_states(output).shape == (5, 3, 4)
