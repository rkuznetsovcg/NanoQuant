"""Regression coverage for the ShamAN-Q adaptations (not run during implementation)."""
import pytest
import torch
from torch import nn

from nanoquant.core.admm_nq import _admm_solve_step, factorize_admm_nanoquant
from nanoquant.core.curvature import BlockInputMetric, metric_solve_step
from nanoquant.core.rank_probe import allocate_measured_ranks, fit_error_curve, packed_bits
from nanoquant.core.reconstruction_plan import (
    reconstruction_groups, refresh_input_stats, shared_input_key, validate_reconstruction_config, stage_quant_config,
)
from nanoquant.core.resume import BlockCheckpoint, content_key, has_block_checkpoint
from nanoquant.modules.linear import NanoQuantLinear


ATTENTION = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.k_proj']
MLP = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
GDN = [f'linear_attn.{name}' for name in ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj']]


@pytest.mark.parametrize('attention,count', [(ATTENTION, 7), (GDN, 8)])
def test_sequential_schedule_retains_every_tunefp_round(attention, count):
    groups = reconstruction_groups(attention+MLP)
    assert len(groups) == count
    assert all(len(group) == 1 for group in groups)
    assert set(sum(groups, [])) == set(attention+MLP)
    assert groups[len(attention)-1][0].endswith(('o_proj', 'out_proj'))
    assert len(reconstruction_groups(attention+MLP, 'shared_input')) == 4
    assert len(reconstruction_groups(attention+MLP, 'parent')) == 2


def test_shared_input_keys_exclude_downstream_projections():
    assert shared_input_key('self_attn.q_proj') == shared_input_key('self_attn.k_proj')
    assert shared_input_key('mlp.gate_proj') == shared_input_key('mlp.up_proj')
    assert shared_input_key('mlp.down_proj') is None
    assert shared_input_key('unknown.q_proj') is None


class CaptureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.executed_after_capture = False

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(input=x)
        self.executed_after_capture = True
        return q+k


@pytest.mark.parametrize('strategy', ['online', 'two_phase', 'dbf'])
def test_fresh_inputs_share_statistics_and_stop_before_last_gemm(strategy):
    block = CaptureBlock()
    data = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).repeat(5, 1, 1)/12
    refresh_input_stats(block, [block.q_proj, block.k_proj], data, {}, 2, 0.4, strategy)
    raw = data.square().mean(dim=(0, 1))
    # With fewer than 1000 tokens the robust tau is their maximum; no clipping.
    expected = (raw*0.6+raw.mean()*0.4) * (len(data) if strategy == "dbf" else 1)
    torch.testing.assert_close(block.q_proj.i_norm, expected)
    assert block.q_proj.i_norm is block.k_proj.i_norm
    assert not block.executed_after_capture
    assert block._nanoquant_active_indices is None
    assert not block.q_proj._forward_pre_hooks
    assert not block.k_proj._forward_pre_hooks


def test_fresh_capture_removes_hooks_when_a_target_is_unreachable():
    block = CaptureBlock()
    missing = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match='did not execute'):
        refresh_input_stats(block, [missing], torch.zeros(1, 2, 4), {}, 1)
    assert not missing._forward_pre_hooks


def test_bounded_covariance_is_finite_for_zero_inputs_and_partial_blocks():
    block = CaptureBlock()
    refresh_input_stats(block, [block.q_proj], torch.zeros(2, 3, 4), {}, 1,
                        correlation_block_size=128)
    metric = block.q_proj._nanoquant_input_metric
    assert metric.blocks[0].shape == (4, 4)
    assert torch.isfinite(metric.blocks[0]).all()
    assert (metric.eigenvalues[0] > 0).all()


def test_metric_transport_tempering_preserves_trace():
    covariance = torch.tensor([[4.0, 1.0], [1.0, 1.0]])
    diagonal = covariance.diagonal()
    metric = BlockInputMetric.from_covariance([covariance], diagonal)
    transported = covariance/diagonal.sqrt()[:, None]/diagonal.sqrt()[None, :]
    torch.testing.assert_close(metric.blocks[0].trace(), transported.trace())
    assert (metric.eigenvalues[0] > 0).all()


@pytest.mark.parametrize('right_identity', [False, True])
def test_identity_curvature_reduces_to_existing_admm_stabilizer(right_identity):
    torch.manual_seed(3)
    X, Y, Z, U = torch.randn(7, 3), torch.randn(7, 4), torch.randn(3, 4), torch.randn(3, 4)
    left = BlockInputMetric.from_covariance([torch.eye(7)], torch.ones(7))
    right = BlockInputMetric.from_covariance([torch.eye(4)], torch.ones(4)) if right_identity else None
    expected = _admm_solve_step(X, Y, Z, U, 0.6, 0.03)
    actual = metric_solve_step(X, Y, Z, U, 0.6, 0.03, left, right)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_block_sylvester_satisfies_both_sided_equation():
    torch.manual_seed(4)
    X, Y, Z, U = torch.randn(6, 3), torch.randn(6, 5), torch.randn(3, 5), torch.randn(3, 5)
    covariances = [torch.tensor([[2.0, 0.4], [0.4, 1.0]]), torch.eye(3)*2]
    right = BlockInputMetric.from_covariance(covariances, torch.cat([c.diagonal() for c in covariances]))
    factor = metric_solve_step(X, Y, Z, U, 0.7, 0.03, right_metric=right)
    gram = X.mT@X
    alpha = 0.7*gram.diagonal().mean().abs()+0.03
    lhs = right.apply((gram@factor).mT).mT+alpha*factor
    rhs = right.apply((X.mT@Y).mT).mT+0.7*(Z-U)
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_input_curvature_follows_transposed_factorization():
    metric = BlockInputMetric.from_covariance([torch.eye(8)], torch.ones(8))
    result = factorize_admm_nanoquant(torch.randn(4, 8), torch.ones(8), torch.ones(4), 2,
                                     outer_iters=1, is_transpose=True, input_metric=metric,
                                     compute_diagnostic=False)
    assert result['A'].shape == (2, 4)
    assert result['B'].shape == (2, 8)
    assert torch.isfinite(result['A']).all()


def test_rank_curve_rejects_increasing_error_and_fits_actual_marginals():
    assert fit_error_curve([(32, 2.0), (64, 3.0)]) is None
    level, eta = fit_error_curve([(32, 1024/32), (64, 1024/64), (96, 1024/96)])
    assert eta == pytest.approx(1.0)
    assert level == pytest.approx(1024)


def test_measured_allocator_keeps_packed_budget_and_unprobed_ranks():
    modules = {'a': nn.Linear(256, 256), 'b': nn.Linear(256, 128), 'unprobed': nn.Linear(128, 128)}
    initial = {'a': 64, 'b': 64, 'unprobed': 64}
    curves = {'a': (10000.0, 1.0, 32, 96), 'b': (100.0, 1.0, 32, 96)}
    result = allocate_measured_ranks(modules, initial, curves)
    assert result['unprobed'] == initial['unprobed']
    assert sum(packed_bits(modules[key], result[key]) for key in modules) <= sum(
        packed_bits(modules[key], initial[key]) for key in modules)
    assert result['a'] > result['b']
    assert all(32 <= result[key] <= 96 for key in curves)


class ToyConfig:
    model_type = 'qwen3_5_text'
    def to_dict(self):
        return {'model_type': self.model_type, 'use_cache': False}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ToyConfig()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)) for _ in range(2)])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.register_buffer('i_norm', torch.ones(4), persistent=False)
                module.register_buffer('o_norm', torch.ones(4), persistent=False)


def test_resume_restores_complete_tuned_block_and_calibration_caches(tmp_path):
    torch.manual_seed(0)
    source = ToyModel()
    base = {key: value.clone() for key, value in source.state_dict().items()}
    tokens = torch.arange(12).reshape(2, 6)
    config = {'resume_dir': str(tmp_path), 'seed': 0, 'bits': 0.55}
    checkpoint = BlockCheckpoint(source, tokens, config)
    inputs = torch.randn(2, 3, 4)
    checkpoint.initialize(source, {'0.0': 2, '1.0': 2}, inputs, inputs+1, {'use_cache': False})
    block = source.model.layers[0]
    dense = block[1].weight.detach().clone()+2
    block[1].weight.data.copy_(dense)
    linear = block[0]
    linear.__class__ = NanoQuantLinear
    linear.init_for_inference(2)
    with torch.no_grad():
        linear.U.fill_(1)
        linear.V.fill_(-1)
        linear.scale_pre.fill_(0.5)
        linear.scale_post.fill_(2)
    checkpoint.save(block, 1, {'0.0': 2, '1.0': 2}, inputs, inputs+1, {'use_cache': False})
    fresh = ToyModel()
    fresh.load_state_dict(base)
    restored = BlockCheckpoint(fresh, tokens, config).restore(fresh)
    assert restored['completed_blocks'] == 1
    assert isinstance(fresh.model.layers[0][0], NanoQuantLinear)
    torch.testing.assert_close(fresh.model.layers[0][1].weight, dense)
    torch.testing.assert_close(fresh.model.layers[0][0].U, torch.ones(4, 2, dtype=torch.bfloat16))
    torch.testing.assert_close(restored['compressed_inputs'], inputs+1)
    assert 'i_norm' in fresh.model.layers[1][0]._buffers
    assert not fresh.model.layers[1][0]._buffers['i_norm'].is_cuda
    assert has_block_checkpoint(config)
    with pytest.raises(ValueError, match='differs'):
        BlockCheckpoint(fresh, tokens+1, config)


def test_content_key_changes_with_target_metric_and_settings():
    tensor = torch.arange(8, dtype=torch.bfloat16)
    original = content_key([tensor], {'rank': 32})
    assert content_key([tensor+1], {'rank': 32}) != original
    assert content_key([tensor], {'rank': 64}) != original
    assert content_key([tensor], {'rank': 32}) == original


@pytest.mark.parametrize('settings', [
    {'tune_schedule': 'invalid'}, {'correlation_block_size': 64},
    {'correlation_block_size': 128, 'refresh_input_stats': False},
    {'correlation_block_size': 128, 'admm_type': 'dbf'},
    {'rank_probe_candidates': -1}, {'rank_probe_iters': 0},
    {'rank_probe_candidates': 1, 'admm_type': 'dbf'}, {'rank_budget': 'unknown'},
    {'nonfact_plateau_patience': 0},
])
def test_invalid_recipes_fail_before_model_loading(settings):
    with pytest.raises(ValueError):
        validate_reconstruction_config(settings)


@pytest.mark.parametrize('name,cap', [('self_attn.q_proj', 2), ('self_attn.k_proj', 2),
                                     ('self_attn.v_proj', 4), ('mlp.gate_proj', 6),
                                     ('mlp.down_proj', 8), ('linear_attn.in_proj_qkv', 2)])
def test_large_model_epochs_keep_rounds_and_cap_early_stages(name, cap):
    config = {'nonfact_epochs': 8, 'fact_epochs': 4}
    stage = stage_quant_config(config, [name], 5120)
    assert stage['nonfact_epochs'] == cap
    assert stage['fact_epochs'] == min(4, cap)
    assert config['nonfact_epochs'] == 8
    assert stage_quant_config(config, [name], 2048) is config
    disabled = config | {'layer_epoch_schedule': False}
    assert stage_quant_config(disabled, [name], 5120) is disabled



def test_recursive_checkpoint_keeps_sibling_packs_and_unpacks_partial_rows():
    block = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 5))
    for module in block:
        module.__class__ = NanoQuantLinear
        module.init_for_inference(3)
        with torch.no_grad():
            module.U.fill_(1)
            module.V.fill_(-1)
            module.scale_pre.fill_(1)
            module.scale_post.fill_(1)
    state = block.state_dict()
    assert {'0.U_packed', '0.V_packed', '1.U_packed', '1.V_packed'} <= set(state)
    assert '0.U' not in state and '1.V' not in state
    restored = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 5))
    for module in restored:
        module.__class__ = NanoQuantLinear
        module.init_for_inference(3)
    restored.load_state_dict(state, strict=True)
    for actual, expected in zip(restored, block):
        torch.testing.assert_close(actual.U, expected.U)
        torch.testing.assert_close(actual.V, expected.V)
    assert packed_bits(block[0], 3) == 32*(3+7)+16*(5+7)
