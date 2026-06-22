"""Distributed test for ``MuonLP`` on DTensor parameters.

``MuonLP`` overrides ``_init_muon_group`` to allocate optimizer state through
``_new_buffer``, which is the DTensor-aware piece: it builds the (optionally
quantized) buffer from the parameter's *local* shard and re-wraps it with
``DTensor.from_local`` using the parameter's mesh + placements. This spawns an nccl
world (one rank per GPU) via :func:`run_distributed` and checks a sharded ``MuonLP``
step matches a single-process step on the equivalent full tensors, and that the
momentum buffer comes back as a DTensor that mirrors the parameter's sharding.

GPU-only (orthogonalization uses the Triton kernels) and requires ``torchao``
(``MuonLP`` lives behind that optional dependency).

NOTE: ``full_tensor()`` is a collective — every rank must call it (never guard the
comparison behind ``if rank == 0``, or the ranks desync and deadlock).

Scope: this covers the base ``MuonLP`` (plain fp32 buffers). The quantized
subclasses (``Muon8bit`` / ``Muon4bit`` / ``MuonFp8``) are *not* exercised here:
the per-parameter ``Muon.muon`` update runs in-place ``mul_``/``add_`` directly on
the optimizer-state tensor, which the torchao ``OptimStateNbit`` subclasses don't
implement — so those fail independently of distribution.
"""

from typing import Any

import pytest
import torch
import torch.nn as nn
from testkit import assert_close, run_distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

try:
    from torchao.optim.subclass_4bit import OptimState4bit  # ty: ignore[unresolved-import]
    from torchao.optim.subclass_8bit import OptimState8bit  # ty: ignore[unresolved-import]
    from torchao.optim.subclass_fp8 import OptimStateFp8  # ty: ignore[unresolved-import]

    from dtensor_muon.optim import Muon, MuonLP
    from dtensor_muon.optim.optim_lp import Muon4bit, Muon8bit, MuonFp8
except ImportError:
    TORCHAO_AVAILABLE = False
else:
    TORCHAO_AVAILABLE = True

requires_torchao = pytest.mark.skipif(
    not TORCHAO_AVAILABLE, reason="MuonLP requires the optional torchao dependency"
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 CUDA devices"
)

# Distributed and single-process run the identical kernel on identical data, so the
# match is effectively exact; keep a small budget for redistribution ordering.
RTOL = 1e-2
ATOL = 2e-2
MAX_MISMATCH_PCT = 1.0

DIST_SHAPES = [(8, 16), (16, 8)]  # dim-0 size is scaled by world_size when sharding

QUANTIZED_OPTIMIZERS = (
    [
        (Muon8bit, OptimState8bit),
        (Muon4bit, OptimState4bit),
        (MuonFp8, OptimStateFp8),
    ]
    if TORCHAO_AVAILABLE
    else []
)


def _dense_buffer(buffer: torch.Tensor) -> torch.Tensor:
    local = buffer.to_local() if isinstance(buffer, DTensor) else buffer
    return local.dequantize() if hasattr(local, "dequantize") else local


def _muon_lp_dtensor_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cuda", (world_size,))
    torch.manual_seed(0)
    fulls = [torch.randn(n * world_size, m, device="cuda") for n, m in DIST_SHAPES]
    grads = [torch.randn_like(f) for f in fulls]

    # Distributed: params + grads sharded on dim 0 across the mesh.
    dparams = [nn.Parameter(distribute_tensor(f.clone(), mesh, [Shard(0)])) for f in fulls]
    for p, g in zip(dparams, grads, strict=True):
        p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])
    dopt = MuonLP(dparams, lr=0.1, wd=0.0)
    dopt.step()

    # Reference: identical step on full tensors (runs the same on every rank).
    rparams = [nn.Parameter(f.clone()) for f in fulls]
    for p, g in zip(rparams, grads, strict=True):
        p.grad = g.clone()
    MuonLP(rparams, lr=0.1, wd=0.0).step()

    for dp, rp in zip(dparams, rparams, strict=True):
        # The momentum buffer is built per-shard then re-wrapped as a DTensor that
        # mirrors the parameter's sharding (this is what _new_buffer guarantees).
        buf = dopt.state[dp]["momentum_buffer"]
        dparam = dp.data
        assert isinstance(buf, DTensor)
        assert isinstance(dparam, DTensor)
        assert buf.placements == dparam.placements
        assert buf.device_mesh == dparam.device_mesh

        assert_close(
            dparam.full_tensor(),
            rp,
            rtol=RTOL,
            atol=ATOL,
            max_mismatch_pct=MAX_MISMATCH_PCT,
        )


def _muon_8bit_dtensor_local_numel_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,))

    # Global numel reaches the quantization threshold, but local shard does not:
    # local shape is (32, 64), numel 2048, so the local buffer stays plain.
    small_local = nn.Parameter(distribute_tensor(torch.randn(64, 64), mesh, [Shard(0)]))
    small_buf = Muon8bit([small_local], block_size=2048)._new_buffer(small_local, signed=True)
    small_dtensor = small_local.data
    assert isinstance(small_buf, DTensor)
    assert isinstance(small_dtensor, DTensor)
    assert small_buf.placements == small_dtensor.placements
    assert small_buf.device_mesh == small_dtensor.device_mesh
    assert small_buf.shape == small_local.shape
    assert type(small_buf.to_local()) is torch.Tensor
    torch.testing.assert_close(
        _dense_buffer(small_buf), torch.zeros_like(_dense_buffer(small_buf))
    )

    # Local shape is (64, 64), numel 4096, and divisible by block_size: quantized.
    quantized_local = nn.Parameter(distribute_tensor(torch.randn(128, 64), mesh, [Shard(0)]))
    quantized_buf = Muon8bit([quantized_local], block_size=2048)._new_buffer(
        quantized_local, signed=True
    )
    quantized_dtensor = quantized_local.data
    assert isinstance(quantized_buf, DTensor)
    assert isinstance(quantized_dtensor, DTensor)
    assert quantized_buf.placements == quantized_dtensor.placements
    assert quantized_buf.device_mesh == quantized_dtensor.device_mesh
    assert quantized_buf.shape == quantized_local.shape
    assert isinstance(quantized_buf.to_local(), OptimState8bit)
    torch.testing.assert_close(
        _dense_buffer(quantized_buf), torch.zeros_like(_dense_buffer(quantized_buf))
    )


@requires_2_gpus
@requires_torchao
def test_muon_lp_dtensor_matches_single_process() -> None:
    run_distributed(_muon_lp_dtensor_worker, world_size=2, backend="nccl", device_type="cuda")


@requires_cuda
@requires_torchao
def test_base_muon_lp_matches_base_muon_single_process() -> None:
    shape = (32, 16)
    torch.manual_seed(0)
    lp_param = nn.Parameter(torch.randn(shape, device="cuda"))
    ref_param = nn.Parameter(lp_param.detach().clone())
    kwargs: dict[str, Any] = dict(lr=0.1, wd=0.0, orthogonalization_strategy="newton_schulz")
    lp = MuonLP([lp_param], **kwargs)
    ref = Muon([ref_param], **kwargs)

    grad_gen = torch.Generator(device="cuda").manual_seed(123)
    for _ in range(2):
        grad = torch.randn(shape, device="cuda", generator=grad_gen)
        lp_param.grad = grad.clone()
        ref_param.grad = grad.clone()
        lp.step()
        ref.step()

    assert_close(lp_param, ref_param, rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT)


@pytest.mark.parametrize(("optimizer_cls", "state_cls"), QUANTIZED_OPTIMIZERS)
@pytest.mark.parametrize(
    ("shape", "block_size", "should_quantize"),
    [
        ((64, 64), 2048, True),
        ((32, 32), 2048, False),
        ((64, 64), 3000, False),
        ((64, 64), 4096, True),
        ((64, 64), 8192, False),
    ],
)
def test_new_buffer_quantization_threshold_and_block_size_cpu(
    optimizer_cls, state_cls, shape: tuple[int, int], block_size: int, should_quantize: bool
) -> None:
    p = nn.Parameter(torch.randn(*shape))
    opt = optimizer_cls([p], block_size=block_size)

    buf = opt._new_buffer(p, signed=True)

    if should_quantize:
        assert isinstance(buf, state_cls)
    else:
        assert type(buf) is torch.Tensor
    assert buf.shape == p.shape
    assert buf.device == p.device
    torch.testing.assert_close(_dense_buffer(buf), torch.zeros_like(_dense_buffer(buf)))


@requires_torchao
def test_new_buffer_dtensor_uses_local_numel_threshold_cpu_gloo() -> None:
    run_distributed(_muon_8bit_dtensor_local_numel_worker, world_size=2)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="Quantized torchao optimizer-state subclasses do not yet implement in-place momentum updates.",
)
@pytest.mark.parametrize(("optimizer_cls", "_state_cls"), QUANTIZED_OPTIMIZERS)
def test_quantized_subclasses_step_once_cpu(optimizer_cls, _state_cls) -> None:
    p = nn.Parameter(torch.randn(64, 64))
    p.grad = torch.randn_like(p)
    opt = optimizer_cls([p], block_size=2048)

    opt.step()


@requires_torchao
def test_bf16_stochastic_round_is_constructor_state_only() -> None:
    p = nn.Parameter(torch.randn(32, 32))
    opt = Muon8bit([p], bf16_stochastic_round=True)

    assert opt.bf16_stochastic_round is True
