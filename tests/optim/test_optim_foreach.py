"""Equivalence tests for ``MuonForeach`` against the per-parameter ``Muon`` reference.

``MuonForeach`` overrides only ``muon()`` to batch the update with ``torch._foreach_*``
ops; every other piece (group building, state, Adam path) is inherited. So the
batched update must produce the *same* parameter trajectory as the reference
per-parameter loop in :class:`Muon` for the same config and inputs.

This pins the weight-decay semantics in particular: the foreach path must apply
``weight_decay`` (with cautious masking) into the update direction exactly as the
reference does, not a decoupled ``1 - lr*lr`` shrink that ignores ``weight_decay``
and ``cautious_wd``.

Both paths orthogonalize via the CUDA-only Triton kernel, so these are
``@requires_cuda``.
"""

from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from testkit import run_distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import dtensor_muon.optim.optim_foreach as optim_foreach_module
from dtensor_muon.optim.optim import Muon
from dtensor_muon.optim.optim_foreach import MuonForeach
from dtensor_muon.orthogonalize import OrthogonalizationStrategy

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 CUDA devices"
)


def assert_close(actual, expected, *, rtol, atol, max_mismatch_pct, msg=None):
    """float32 closeness check tolerating a small fraction of mismatched elements
    (bf16 orthogonalization produces a few near-zero outliers)."""
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


# Includes a duplicated shape so the foreach path actually batches two tensors
# into one ``foreach_zeropower`` call (the interesting case vs. the per-param ref).
SHAPES = [(32, 16), (32, 16), (16, 32), (24, 24)]

# bf16 orthogonalization accumulated over a few steps: loose elementwise budget.
# The cautious-WD mask is discrete, so a few mask bits near ``u*p == 0`` flip when
# the batched kernel differs from the per-param one at the bf16 level — hence the
# few-percent budget. The buggy ``1 - lr*lr`` decay diverges on ~all elements, well
# above this.
RTOL = 1e-2
ATOL = 3e-2
MAX_MISMATCH_PCT = 5.0


def _make_params(shapes, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, device=device, generator=gen)) for s in shapes]


def test_foreach_batch_size_chunks_same_shape_group(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        optim_foreach_module,
        "move_tensors_to_device",
        lambda tensors, _src, _dst: tensors,
    )

    def fake_foreach_muon(p, g, buf, lr_ratio, *args, **kwargs):
        calls.append((len(p), [tuple(t.shape) for t in g]))

    monkeypatch.setattr(optim_foreach_module, "_foreach_muon", fake_foreach_muon)
    params = [nn.Parameter(torch.ones(2, 2)) for _ in range(5)]
    for p in params:
        p.grad = torch.full_like(p, 0.5)

    MuonForeach(params, batch_size=2).step()

    assert calls == [
        (2, [(2, 2), (2, 2)]),
        (2, [(2, 2), (2, 2)]),
        (1, [(2, 2)]),
    ]


def test_foreach_step_with_all_none_grads_is_noop(monkeypatch) -> None:
    def fail_foreach_muon(*args, **kwargs):
        raise AssertionError("_foreach_muon should not be called")

    monkeypatch.setattr(optim_foreach_module, "_foreach_muon", fail_foreach_muon)
    params = [nn.Parameter(torch.ones(2, 2)) for _ in range(2)]
    before = [p.detach().clone() for p in params]
    optimizer = MuonForeach(params)

    optimizer.step()

    for p, expected in zip(params, before, strict=True):
        torch.testing.assert_close(p, expected)
        assert p not in optimizer.state


def test_foreach_mixed_dtype_params_are_grouped_separately(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        optim_foreach_module,
        "move_tensors_to_device",
        lambda tensors, _src, _dst: tensors,
    )

    def fake_foreach_muon(p, g, buf, lr_ratio, *args, **kwargs):
        calls.append((p[0].dtype, g[0].dtype, buf[0].dtype, len(p)))

    monkeypatch.setattr(optim_foreach_module, "_foreach_muon", fake_foreach_muon)
    fp32 = nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
    bf16 = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    fp32.grad = torch.full_like(fp32, 0.5)
    bf16.grad = torch.full_like(bf16, 0.5)

    MuonForeach([fp32, bf16]).step()

    assert sorted(calls, key=lambda call: str(call[0])) == [
        (torch.bfloat16, torch.bfloat16, torch.float32, 1),
        (torch.float32, torch.float32, torch.float32, 1),
    ]


def test_register_dtensor_foreach_ops_is_idempotent() -> None:
    optim_foreach_module._register_dtensor_foreach_ops()
    optim_foreach_module._register_dtensor_foreach_ops()

    assert getattr(torch.ops.aten._foreach_sign_.default, "_dtensor_registered") is True


def _foreach_sign_dtensor_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    fulls = [
        torch.linspace(-4, 3, steps=8).view(4, 2),
        torch.linspace(3, -4, steps=8).view(4, 2),
    ]
    ds = [distribute_tensor(f.clone(), mesh, [Shard(0)]) for f in fulls]

    torch._foreach_sign_(cast(list[torch.Tensor], ds))

    for d, full in zip(ds, fulls, strict=True):
        torch.testing.assert_close(d.full_tensor(), full.sign())


def test_registered_dtensor_foreach_sign_smoke() -> None:
    run_distributed(_foreach_sign_dtensor_worker, world_size=2)


@requires_cuda
@pytest.mark.parametrize("strategy", ["newton_schulz", "polar_express"])
@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("wd", [0.0, 0.3])
def test_foreach_matches_base_muon(
    wd: float,
    nesterov: bool,
    cautious: bool,
    strategy: OrthogonalizationStrategy,
) -> None:
    device = "cuda"

    ref_params = _make_params(SHAPES, device, seed=0)
    fe_params = _make_params(SHAPES, device, seed=0)
    for r, f in zip(ref_params, fe_params, strict=True):
        assert torch.equal(r, f), "optimizers must start from identical parameters"

    # lr and wd are deliberately distinct so the buggy decoupled shrink (which used
    # ``1 - lr*lr``) cannot coincide with the correct ``lr_ratio*lr*wd`` term.
    kwargs: dict[str, Any] = dict(
        lr=0.2,
        wd=wd,
        nesterov=nesterov,
        use_cautious_wd=cautious,
        orthogonalization_strategy=strategy,
    )
    ref = Muon(ref_params, **kwargs)
    fe = MuonForeach(fe_params, **kwargs)

    grad_gen = torch.Generator(device=device).manual_seed(123)
    for _ in range(3):
        # Identical gradients for both optimizers each step. Both paths mutate the
        # grad tensor in place, so each optimizer gets its own clone.
        grads = [torch.randn(s, device=device, generator=grad_gen) for s in SHAPES]
        for p, g in zip(ref_params, grads, strict=True):
            p.grad = g.clone()
        for p, g in zip(fe_params, grads, strict=True):
            p.grad = g.clone()

        ref.step()
        fe.step()

    for i, (r, f) in enumerate(zip(ref_params, fe_params, strict=True)):
        assert_close(
            f,
            r,
            rtol=RTOL,
            atol=ATOL,
            max_mismatch_pct=MAX_MISMATCH_PCT,
            msg=f"MuonForeach diverged from Muon for param {i} (shape {tuple(r.shape)})",
        )


# --- distributed: MuonForeach on DTensor parameters --------------------------------
#
# Spawns an nccl world (one rank per GPU) and runs a MuonForeach step on parameters
# sharded across the mesh, checking the result matches a single-process MuonForeach
# step on the equivalent full tensors. This exercises the batched foreach DTensor
# path (``foreach_zeropower`` with its ``redistribute``/``from_local`` round-trips).
# GPU-only: the foreach path moves tensors to CUDA and orthogonalizes via Triton.
#
# NOTE: ``full_tensor()`` is a collective — every rank must call it (never guard the
# comparison behind ``if rank == 0``, or the ranks desync and deadlock).

DIST_SHAPES = [(8, 16), (8, 16), (16, 8)]  # per-rank-dim-0 size is scaled by world_size


def _muon_foreach_dtensor_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cuda", (world_size,))
    torch.manual_seed(0)
    fulls = [torch.randn(n * world_size, m, device="cuda") for n, m in DIST_SHAPES]
    grads = [torch.randn_like(f) for f in fulls]

    # Distributed: params + grads sharded on dim 0 across the mesh.
    dparams = [nn.Parameter(distribute_tensor(f.clone(), mesh, [Shard(0)])) for f in fulls]
    for p, g in zip(dparams, grads, strict=True):
        p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])
    MuonForeach(dparams, lr=0.1, wd=0.0).step()

    # Reference: identical step on full tensors (runs the same on every rank).
    rparams = [nn.Parameter(f.clone()) for f in fulls]
    for p, g in zip(rparams, grads, strict=True):
        p.grad = g.clone()
    MuonForeach(rparams, lr=0.1, wd=0.0).step()

    for dp, rp in zip(dparams, rparams, strict=True):
        dparam = dp.data
        assert isinstance(dparam, DTensor)
        assert_close(
            dparam.full_tensor(),
            rp,
            rtol=RTOL,
            atol=ATOL,
            max_mismatch_pct=MAX_MISMATCH_PCT,
        )


def _muon_foreach_3d_fsdp_uses_fast_path_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cuda", (world_size,))
    torch.manual_seed(0)
    called_fast_path = False
    original_fast_path = optim_foreach_module.foreach_zeropower_3d_fsdp

    def recording_fast_path(*args, **kwargs):
        nonlocal called_fast_path
        called_fast_path = True
        return original_fast_path(*args, **kwargs)

    cast(Any, optim_foreach_module).foreach_zeropower_3d_fsdp = recording_fast_path
    fulls = [torch.randn(4 * world_size, 16, 8, device="cuda") for _ in range(2)]
    grads = [torch.randn_like(f) for f in fulls]
    dparams = [nn.Parameter(distribute_tensor(f.clone(), mesh, [Shard(0)])) for f in fulls]
    for p, g in zip(dparams, grads, strict=True):
        p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])

    MuonForeach(
        [{"params": dparams, "flatten": False}],
        lr=0.01,
        wd=0.0,
        orthogonalization_strategy="newton_schulz",
    ).step()

    assert called_fast_path


@requires_2_gpus
def test_foreach_dtensor_matches_single_process() -> None:
    run_distributed(
        _muon_foreach_dtensor_worker, world_size=2, backend="nccl", device_type="cuda"
    )


@requires_2_gpus
def test_foreach_dtensor_3d_fsdp_uses_fast_path() -> None:
    run_distributed(
        _muon_foreach_3d_fsdp_uses_fast_path_worker,
        world_size=2,
        backend="nccl",
        device_type="cuda",
    )


@requires_cuda
def test_foreach_cpu_offload_matches_cuda_step() -> None:
    shape = (32, 16)
    torch.manual_seed(0)
    cpu_param = nn.Parameter(torch.randn(shape))
    cuda_param = nn.Parameter(cpu_param.detach().clone().cuda())
    grad = torch.randn(shape)
    cpu_param.grad = grad.clone()
    cuda_param.grad = grad.clone().cuda()

    kwargs: dict[str, Any] = dict(lr=0.1, wd=0.0, orthogonalization_strategy="newton_schulz")
    MuonForeach([cpu_param], **kwargs).step()
    MuonForeach([cuda_param], **kwargs).step()

    assert cpu_param.device.type == "cpu"
    assert cpu_param.grad.device.type == "cpu"
    assert_close(
        cpu_param,
        cuda_param.cpu(),
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=MAX_MISMATCH_PCT,
    )


@requires_cuda
def test_foreach_maximize_reused_grad_keeps_maximize_direction(monkeypatch) -> None:
    monkeypatch.setattr(optim_foreach_module, "foreach_zeropower", lambda g, **_: g)
    p = nn.Parameter(torch.ones(2, 2, device="cuda"))
    grad = torch.full_like(p, 0.5)
    p.grad = grad
    optimizer = MuonForeach([p], lr=0.1, wd=0.0, momentum=0.0, nesterov=False, maximize=True)

    optimizer.step()
    after_first = p.detach().clone()
    optimizer.step()

    torch.testing.assert_close(grad, torch.full_like(grad, 0.5))
    torch.testing.assert_close(after_first, torch.full_like(after_first, 1.05))
    torch.testing.assert_close(p, torch.full_like(p, 1.10))
