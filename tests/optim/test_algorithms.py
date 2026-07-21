"""Tests for the pluggable Muon algorithm registry, NorMuon, and split_sizes.

The registry/API tests run on CPU with ``zeropower`` stubbed to identity (the
same seam existing optim tests use), so they check the framework wiring and the
algorithm math without paying for real orthogonalization. Real-kernel parity
and DTensor coverage are CUDA-gated, mirroring test_optim_foreach.py.
"""

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
from testkit import run_distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import muonium.optim.algorithms.base as algo_base
from muonium import (
    Muon,
    MuonAlgorithm,
    get_algorithm,
    register_muon_algorithm,
    registered_algorithms,
)
from muonium.optim.algorithms import split_lr_scales

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 CUDA devices"
)

# Tolerances for CUDA tests running real bf16 orthogonalization; see
# test_optim_foreach.py for the rationale.
RTOL = 1e-2
ATOL = 3e-2
MAX_MISMATCH_PCT = 5.0


def assert_close_pct(actual, expected, *, rtol, atol, max_mismatch_pct, msg=None):
    actual = actual.detach().to(torch.float32)
    expected = expected.detach().to(torch.float32)
    mismatched = ~torch.isclose(actual, expected, rtol=rtol, atol=atol)
    pct = 100.0 * mismatched.sum().item() / max(mismatched.numel(), 1)
    if pct > max_mismatch_pct:
        raise AssertionError(
            (msg or "")
            + f"\n{pct:.4f}% of elements mismatched (allowed {max_mismatch_pct:.4f}%) "
            f"at rtol={rtol}, atol={atol}"
        )


# --- registry -----------------------------------------------------------------------


@register_muon_algorithm
class _SignDescent(MuonAlgorithm):
    """Toy third-party algorithm: sign of the raw gradient, no orthogonalization."""

    name = "test-signsgd"
    options = {"scale": 1.0}

    # Lax keyword signature (everything else lands in **kwargs) is the supported
    # pattern for third-party algorithms; options always arrive as keywords.
    def update(self, param, grad, state, lr_ratio, *, lr, scale, **kwargs):  # ty: ignore[invalid-method-override]
        state["momentum_buffer"].copy_(grad)
        param.copy_(param - lr * scale * grad.sign())


def test_builtin_algorithms_are_registered():
    assert {"muon", "normuon"}.issubset(set(registered_algorithms()))


def test_get_algorithm_unknown_name_raises_with_known_names():
    with pytest.raises(ValueError, match="Unknown algorithm 'nope'") as excinfo:
        get_algorithm("nope")
    assert "muon" in str(excinfo.value)


def test_constructor_rejects_unknown_algorithm_via_group():
    p = nn.Parameter(torch.randn(2, 2))
    with pytest.raises(ValueError, match="Unknown algorithm"):
        Muon([{"params": [p], "algorithm": "rmsprop"}])


def test_register_rejects_reserved_and_duplicate_names_and_shadowed_options():
    class ReservedName(MuonAlgorithm):
        name = "adam"

        def update(self, *args, **kwargs):
            pass

    with pytest.raises(ValueError, match="reserved for the Adam path"):
        register_muon_algorithm(ReservedName)

    class DuplicateName(MuonAlgorithm):
        name = "muon"

        def update(self, *args, **kwargs):
            pass

    with pytest.raises(ValueError, match="already registered"):
        register_muon_algorithm(DuplicateName)

    class ShadowedOption(MuonAlgorithm):
        name = "test-shadowed"
        options = {"momentum": 0.5}

        def update(self, *args, **kwargs):
            pass

    with pytest.raises(ValueError, match="reserved group keys"):
        register_muon_algorithm(ShadowedOption)


def test_custom_registered_algorithm_steps_end_to_end():
    p = nn.Parameter(torch.ones(2, 3))
    grad = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 3.0, -1.0]])
    p.grad = grad.clone()
    optimizer = Muon([{"params": [p], "algorithm": "test-signsgd", "scale": 2.0}], lr=0.1)

    assert optimizer.param_groups[0]["scale"] == 2.0

    optimizer.step()

    torch.testing.assert_close(p, torch.ones(2, 3) - 0.1 * 2.0 * grad.sign())
    torch.testing.assert_close(optimizer.state[p]["momentum_buffer"], grad)


def test_custom_algorithm_option_falls_back_to_declared_default():
    p = nn.Parameter(torch.ones(2, 2))
    optimizer = Muon([{"params": [p], "algorithm": "test-signsgd"}])
    assert optimizer.param_groups[0]["scale"] == 1.0


# --- group mixing ------------------------------------------------------------------


def test_mixed_muon_normuon_and_adam_groups_step(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    muon_p = nn.Parameter(torch.randn(4, 3))
    normuon_p = nn.Parameter(torch.randn(4, 3))
    adam_p = nn.Parameter(torch.randn(5))
    optimizer = Muon(
        [
            {"params": [muon_p]},
            {"params": [normuon_p], "algorithm": "normuon", "muon_beta2": 0.9},
            {"params": [adam_p], "algorithm": "adamw", "fused": False},
        ]
    )
    before = [t.detach().clone() for t in (muon_p, normuon_p, adam_p)]
    for t in (muon_p, normuon_p, adam_p):
        t.grad = torch.randn_like(t)

    optimizer.step()

    for t, old in zip((muon_p, normuon_p, adam_p), before, strict=True):
        assert not torch.equal(t, old)
    assert "variance_neuron" not in optimizer.state[muon_p]
    assert set(optimizer.state[normuon_p]) == {
        "step",
        "momentum_buffer",
        "variance_neuron",
        "lr_ratio",
    }
    assert optimizer.state[normuon_p]["variance_neuron"].shape == (4, 1)
    assert optimizer.param_groups[1]["muon_beta2"] == 0.9


# --- NorMuon numerics ----------------------------------------------------------------


def _normuon_reference_step(p, g, buf, v, *, lr, wd, cautious, momentum, nesterov, beta2):
    """Straight-line transcription of the NorMuon math with identity zeropower.

    flatten defaults to False, so 3D tensors stay batches of matrices: all norms
    are per-matrix (keepdim over the last two dims) and lr_ratio comes from the
    matrix dims.
    """
    buf.mul_(momentum).add_(g)
    u = g + momentum * buf if nesterov else buf.clone()

    norm_u = u.norm(p=2, dim=(-2, -1), keepdim=True)
    v.lerp_((u * u).mean(dim=-1, keepdim=True), 1 - beta2)
    u = u / (v.sqrt() + 1e-8)
    u = u * (norm_u / u.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8))

    if wd != 0:
        u = u + wd * p * (u * p > 0) if cautious else u + wd * p
    lr_ratio = math.sqrt(max(1.0, p.shape[-2] / p.shape[-1]))
    p.sub_(lr_ratio * lr * u)


@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (2, 3, 4)])
def test_normuon_matches_reference_transcription(monkeypatch, shape):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(shape))
    ref_p = p.detach().clone()
    ref_buf = torch.zeros_like(ref_p)
    ref_v = torch.zeros_like(ref_p[..., :1])
    kwargs: dict[str, Any] = dict(lr=0.1, wd=0.1, momentum=0.5, nesterov=True)
    optimizer = Muon(
        [{"params": [p], "algorithm": "normuon", "muon_beta2": 0.9}], **kwargs
    )

    for step in range(3):
        grad = torch.randn(shape, generator=torch.Generator().manual_seed(100 + step))
        p.grad = grad.clone()
        optimizer.step()
        _normuon_reference_step(
            ref_p,
            grad.clone(),
            ref_buf,
            ref_v,
            lr=0.1,
            wd=0.1,
            cautious=True,
            momentum=0.5,
            nesterov=True,
            beta2=0.9,
        )

    torch.testing.assert_close(p.detach(), ref_p, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(optimizer.state[p]["variance_neuron"], ref_v, rtol=1e-5, atol=1e-7)


def test_normuon_foreach_matches_per_param_on_cpu(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    monkeypatch.setattr(
        algo_base, "foreach_zeropower", lambda gs, **_: [g.clone() for g in gs]
    )
    torch.manual_seed(1)
    shapes = [(4, 3), (4, 3), (3, 4)]
    ref_params = [nn.Parameter(torch.randn(s)) for s in shapes]
    fe_params = [nn.Parameter(r.detach().clone()) for r in ref_params]
    kwargs: dict[str, Any] = dict(lr=0.2, wd=0.1, momentum=0.5)
    ref = Muon([{"params": ref_params, "algorithm": "normuon"}], **kwargs)
    fe = Muon([{"params": fe_params, "algorithm": "normuon"}], foreach=True, **kwargs)

    for step in range(3):
        grads = [
            torch.randn(s, generator=torch.Generator().manual_seed(7 * step + i))
            for i, s in enumerate(shapes)
        ]
        for p, g in zip(ref_params, grads, strict=True):
            p.grad = g.clone()
        for p, g in zip(fe_params, grads, strict=True):
            p.grad = g.clone()
        ref.step()
        fe.step()

    for r, f in zip(ref_params, fe_params, strict=True):
        torch.testing.assert_close(f.detach(), r.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            fe.state[f]["variance_neuron"], ref.state[r]["variance_neuron"],
            rtol=1e-5, atol=1e-7,
        )


@requires_cuda
def test_normuon_foreach_matches_per_param_on_cuda_real_kernels():
    torch.manual_seed(0)
    shapes = [(32, 16), (32, 16), (16, 32)]
    gen = torch.Generator(device="cuda").manual_seed(0)
    ref_params = [
        nn.Parameter(torch.randn(s, device="cuda", generator=gen)) for s in shapes
    ]
    fe_params = [nn.Parameter(r.detach().clone()) for r in ref_params]
    kwargs: dict[str, Any] = dict(lr=0.2, wd=0.1, orthogonalization_strategy="newton_schulz")
    ref = Muon([{"params": ref_params, "algorithm": "normuon"}], foreach=False, **kwargs)
    fe = Muon([{"params": fe_params, "algorithm": "normuon"}], foreach=True, **kwargs)

    grad_gen = torch.Generator(device="cuda").manual_seed(123)
    for _ in range(3):
        grads = [torch.randn(s, device="cuda", generator=grad_gen) for s in shapes]
        for p, g in zip(ref_params, grads, strict=True):
            p.grad = g.clone()
        for p, g in zip(fe_params, grads, strict=True):
            p.grad = g.clone()
        ref.step()
        fe.step()

    for i, (r, f) in enumerate(zip(ref_params, fe_params, strict=True)):
        assert_close_pct(
            f, r, rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT,
            msg=f"NorMuon foreach diverged from reference for param {i}",
        )


# --- split_sizes ---------------------------------------------------------------------


def test_split_sizes_validation_errors():
    p2d = nn.Parameter(torch.randn(6, 4))
    p3d = nn.Parameter(torch.randn(2, 3, 4))

    with pytest.raises(ValueError, match="at least 2 block sizes"):
        Muon([{"params": [p2d], "split_sizes": (6,)}])
    with pytest.raises(ValueError, match="positive integers"):
        Muon([{"params": [p2d], "split_sizes": (4, -2)}])
    with pytest.raises(ValueError, match="only supported for 2D"):
        Muon([{"params": [p3d], "split_sizes": (1, 1)}])
    with pytest.raises(ValueError, match="must sum to dim 0"):
        Muon([{"params": [p2d], "split_sizes": (2, 2)}])


def test_split_sizes_normalizes_to_tuple():
    p = nn.Parameter(torch.randn(6, 4))
    optimizer = Muon([{"params": [p], "split_sizes": [2, 4]}])
    assert optimizer.param_groups[0]["split_sizes"] == (2, 4)


def test_split_sizes_orthogonalizes_row_blocks_independently(monkeypatch):
    # Norm-dependent fake: distinguishes per-block from whole-matrix processing.
    monkeypatch.setattr(
        algo_base, "foreach_zeropower", lambda gs, **_: [g / g.norm() for g in gs]
    )
    p = nn.Parameter(torch.ones(6, 4))
    grad = torch.randn(6, 4, generator=torch.Generator().manual_seed(3))
    p.grad = grad.clone()
    optimizer = Muon(
        [{"params": [p], "split_sizes": (2, 4)}],
        lr=0.1,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
    )

    optimizer.step()

    scales = split_lr_scales((6, 4), (2, 4))
    expected_u = torch.cat(
        [
            scale * block / block.norm()
            for block, scale in zip(grad.split([2, 4], dim=0), scales, strict=True)
        ],
        dim=0,
    )
    lr_ratio = math.sqrt(6 / 4)
    torch.testing.assert_close(
        p.detach(), torch.ones(6, 4) - lr_ratio * 0.1 * expected_u, rtol=1e-5, atol=1e-6
    )


def test_split_lr_scales_match_per_block_ratio_rule():
    scales = split_lr_scales((6, 4), (2, 4))
    full = math.sqrt(max(1.0, 6 / 4))
    assert scales == pytest.approx(
        (math.sqrt(max(1.0, 2 / 4)) / full, math.sqrt(max(1.0, 4 / 4)) / full)
    )


def test_normuon_split_sizes_normalizes_per_block(monkeypatch):
    monkeypatch.setattr(
        algo_base, "foreach_zeropower", lambda gs, **_: [g.clone() for g in gs]
    )
    torch.manual_seed(5)
    p = nn.Parameter(torch.randn(6, 4))
    ref_p = p.detach().clone()
    grad = torch.randn(6, 4)
    p.grad = grad.clone()
    optimizer = Muon(
        [{"params": [p], "algorithm": "normuon", "split_sizes": (2, 4), "muon_beta2": 0.9}],
        lr=0.1,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
    )

    optimizer.step()

    scales = split_lr_scales((6, 4), (2, 4))
    v_blocks, u_blocks = [], []
    for block, scale in zip(grad.split([2, 4], dim=0), scales, strict=True):
        u = scale * block
        norm_u = u.norm()
        v = (1 - 0.9) * (u * u).mean(dim=-1, keepdim=True)
        u = u / (v.sqrt() + 1e-8)
        u = u * (norm_u / u.norm().clamp(min=1e-8))
        u_blocks.append(u)
        v_blocks.append(v)
    expected_u = torch.cat(u_blocks, dim=0)
    lr_ratio = math.sqrt(6 / 4)

    torch.testing.assert_close(
        p.detach(), ref_p - lr_ratio * 0.1 * expected_u, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        optimizer.state[p]["variance_neuron"], torch.cat(v_blocks, dim=0),
        rtol=1e-5, atol=1e-7,
    )


# --- checkpointing -------------------------------------------------------------------


def test_normuon_state_dict_round_trip_preserves_training_continuity(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = nn.Parameter(torch.ones(4, 2))
    uninterrupted_p = nn.Parameter(p.detach().clone())
    loaded_p = nn.Parameter(p.detach().clone())
    group = lambda param: [{"params": [param], "algorithm": "normuon", "muon_beta2": 0.9}]
    kwargs: dict[str, Any] = dict(lr=0.1, wd=0.0, momentum=0.5, nesterov=False)
    uninterrupted = Muon(group(uninterrupted_p), **kwargs)
    to_save = Muon(group(p), **kwargs)
    loaded = Muon(group(loaded_p), **kwargs)

    first_grad = torch.full_like(p, 0.5)
    for param in (uninterrupted_p, p):
        param.grad = first_grad.clone()
    uninterrupted.step()
    to_save.step()
    loaded_p.data.copy_(p.data)
    loaded.load_state_dict(to_save.state_dict())

    assert loaded.param_groups[0]["algorithm"] == "normuon"
    assert loaded.param_groups[0]["muon_beta2"] == 0.9

    second_grad = torch.full_like(p, 0.25)
    for param in (uninterrupted_p, loaded_p):
        param.grad = second_grad.clone()
    uninterrupted.step()
    loaded.step()

    torch.testing.assert_close(loaded_p, uninterrupted_p)
    for key in ["step", "momentum_buffer", "variance_neuron", "lr_ratio"]:
        torch.testing.assert_close(
            loaded.state[loaded_p][key], uninterrupted.state[uninterrupted_p][key]
        )


def test_legacy_checkpoint_without_algorithm_key_loads_as_muon(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = nn.Parameter(torch.ones(4, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([p], lr=0.1, wd=0.0)
    optimizer.step()
    state_dict = optimizer.state_dict()
    for saved_group in state_dict["param_groups"]:
        saved_group.pop("algorithm", None)
        saved_group.pop("split_sizes", None)

    fresh_p = nn.Parameter(torch.ones(4, 2))
    fresh = Muon([fresh_p], lr=0.1, wd=0.0)
    fresh.load_state_dict(state_dict)

    assert fresh.param_groups[0]["algorithm"] == "muon"
    assert fresh.param_groups[0]["split_sizes"] is None
    fresh_p.grad = torch.full_like(fresh_p, 0.5)
    fresh.step()


# --- DTensor -------------------------------------------------------------------------
#
# Mirrors _muon_foreach_dtensor_worker in test_optim_foreach.py: an nccl world,
# NorMuon groups on Shard(0) DTensors, compared against a single-process run on
# the equivalent full tensors. NOTE: full_tensor() is a collective — every rank
# must call it.

DIST_SHAPES = [(8, 16), (8, 16), (16, 8)]


def _normuon_foreach_dtensor_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cuda", (world_size,))
    torch.manual_seed(0)
    fulls = [torch.randn(n * world_size, m, device="cuda") for n, m in DIST_SHAPES]
    grads = [torch.randn_like(f) for f in fulls]

    dparams = [nn.Parameter(distribute_tensor(f.clone(), mesh, [Shard(0)])) for f in fulls]
    for p, g in zip(dparams, grads, strict=True):
        p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])
    dopt = Muon([{"params": dparams, "algorithm": "normuon"}], foreach=True, lr=0.1, wd=0.0)
    dopt.step()

    rparams = [nn.Parameter(f.clone()) for f in fulls]
    for p, g in zip(rparams, grads, strict=True):
        p.grad = g.clone()
    Muon(
        [{"params": rparams, "algorithm": "normuon"}], foreach=True, lr=0.1, wd=0.0
    ).step()

    for dp, rp in zip(dparams, rparams, strict=True):
        dparam = dp.data
        variance = dopt.state[dp]["variance_neuron"]
        assert isinstance(dparam, DTensor)
        assert isinstance(variance, DTensor)
        assert variance.placements == dparam.placements
        assert variance.shape == (dparam.shape[0], 1)
        assert_close_pct(
            dparam.full_tensor(),
            rp,
            rtol=RTOL,
            atol=ATOL,
            max_mismatch_pct=MAX_MISMATCH_PCT,
        )


@requires_2_gpus
def test_normuon_foreach_dtensor_matches_single_process() -> None:
    run_distributed(
        _normuon_foreach_dtensor_worker, world_size=2, backend="nccl", device_type="cuda"
    )
