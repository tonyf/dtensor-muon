"""Distributed correctness tests for the DTensor code paths.

These exercise the DTensor branches of ``to_local`` / ``zeropower`` /
``foreach_zeropower`` which need a real process group and device mesh. We spawn a
small gloo world on CPU via :func:`run_distributed`, so the tests run anywhere
(no GPUs required) and verify the *distribution plumbing* — sharding,
``full_tensor``, ``redistribute``, ``from_local`` round-trips — matches the plain
single-process result. The Triton kernels are CUDA-only, so every worker uses the
pure-PyTorch iteration (``use_triton=False``).

Each worker seeds RNG identically per rank, so every rank materializes the same
full tensor before sharding; the orthogonalized output (after ``full_tensor``) is
compared against the non-distributed ``zeropower`` of the same input.
"""

import pytest
import torch
from testkit import assert_close, run_distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from muonium.orthogonalize import OrthogonalizationStrategy
from muonium.orthogonalize.orthogonalize import (
    foreach_zeropower,
    foreach_zeropower_3d_fsdp,
    get_dtensor_metadata,
    is_fsdp_3d_sharded,
    zeropower,
)
from muonium.utils.dtensor import to_local

STEPS = 5
# bfloat16 orthogonalization; distributed and reference run the identical kernel on
# identical data, so differences come only from redistribution ordering.
RTOL = 1e-2
ATOL = 2e-2
MAX_MISMATCH_PCT = 1.0


# --- workers (must be module-level so torch.multiprocessing.spawn can pickle them) ---


def _to_local_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    full = torch.randn(8 * world_size, 4)
    d = distribute_tensor(full, mesh, [Shard(0)])

    # to_local() returns just this rank's shard...
    local = to_local(d)
    assert not isinstance(local, DTensor)
    assert local.shape[0] == full.shape[0] // world_size
    assert_close(local, full.chunk(world_size, dim=0)[rank], rtol=0, atol=0)
    # ...full_tensor=True gathers the whole thing on every rank.
    gathered = to_local(d, full_tensor=True)
    assert_close(gathered, full, rtol=0, atol=0)
    # Plain tensors pass straight through unchanged.
    plain = torch.randn(3, 3)
    assert to_local(plain) is plain


def _zeropower_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    full = torch.randn(64, 32)
    d = distribute_tensor(full, mesh, [Shard(0)])

    out = zeropower(d, steps=STEPS, use_triton=False, strategy="newton_schulz")
    assert isinstance(out, DTensor)
    # Distributed result, gathered, must match the single-process computation.
    ref = zeropower(full, steps=STEPS, use_triton=False, strategy="newton_schulz")
    assert_close(
        out.full_tensor(), ref, rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT
    )
    assert out.placements == d.placements


def _foreach_zeropower_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    fulls = [torch.randn(32, 16) for _ in range(4)]
    ds = [distribute_tensor(f, mesh, [Shard(0)]) for f in fulls]

    outs = foreach_zeropower(ds, steps=STEPS, use_triton=False, strategy="newton_schulz")
    refs = foreach_zeropower(fulls, steps=STEPS, use_triton=False, strategy="newton_schulz")
    assert len(outs) == len(refs)
    for out, ref in zip(outs, refs, strict=True):
        assert isinstance(out, DTensor)
        assert out.placements == ds[0].placements
        assert_close(
            out.full_tensor(), ref, rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT
        )


def _foreach_zeropower_3d_fsdp_worker(
    rank: int,
    world_size: int,
    strategy: OrthogonalizationStrategy = "newton_schulz",
) -> None:
    # 3D tensors sharded only on dim 0 (FSDP layout): the fast path operates purely
    # on local shards, then rewraps via DTensor.from_local.
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    fulls = [torch.randn(4 * world_size, 16, 8) for _ in range(3)]
    ds = [distribute_tensor(f, mesh, [Shard(0)]) for f in fulls]
    assert is_fsdp_3d_sharded(ds)

    outs = foreach_zeropower_3d_fsdp(ds, steps=STEPS, use_triton=False, strategy=strategy)
    refs = foreach_zeropower(fulls, steps=STEPS, use_triton=False, strategy=strategy)
    assert len(outs) == len(refs)
    for out, ref in zip(outs, refs, strict=True):
        assert isinstance(out, DTensor)
        assert out.placements == ds[0].placements
        assert_close(
            out.full_tensor(), ref, rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT
        )


def _two_dim_mesh_worker(rank: int, world_size: int) -> None:
    assert world_size == 4
    mesh = init_device_mesh("cpu", (2, 2))
    torch.manual_seed(0)

    full_3d = torch.randn(8, 16, 8)
    fsdp = distribute_tensor(full_3d, mesh, [Shard(0), Replicate()])
    assert is_fsdp_3d_sharded([fsdp])

    replicated_3d = distribute_tensor(full_3d, mesh, [Replicate(), Replicate()])
    assert not is_fsdp_3d_sharded([replicated_3d])

    full = torch.randn(64, 32)
    d = distribute_tensor(full, mesh, [Shard(0), Replicate()])
    out = zeropower(d, steps=STEPS, use_triton=False, strategy="newton_schulz")
    ref = zeropower(full, steps=STEPS, use_triton=False, strategy="newton_schulz")

    assert isinstance(out, DTensor)
    assert out.placements == d.placements
    assert_close(
        out.full_tensor(), ref, rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT
    )


def _is_fsdp_3d_sharded_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    # Sharded on dim 0 -> FSDP layout.
    sharded = distribute_tensor(torch.randn(4 * world_size, 8, 8), mesh, [Shard(0)])
    assert is_fsdp_3d_sharded([sharded])
    # Replicated -> not FSDP-sharded.
    replicated = distribute_tensor(torch.randn(8, 8, 8), mesh, [Replicate()])
    assert not is_fsdp_3d_sharded([replicated])
    # 2D shard -> wrong rank, not the 3D FSDP layout.
    two_d = distribute_tensor(torch.randn(4 * world_size, 8), mesh, [Shard(0)])
    assert not is_fsdp_3d_sharded([two_d])
    # 3D tensor sharded along a non-FSDP dimension is not the fast-path layout.
    wrong_dim = distribute_tensor(torch.randn(8, 8 * world_size, 8), mesh, [Shard(1)])
    assert not is_fsdp_3d_sharded([wrong_dim])
    # Plain tensors are never DTensor-sharded.
    plain = torch.randn(8, 8, 8)
    assert not is_fsdp_3d_sharded([plain])
    assert not is_fsdp_3d_sharded([sharded, plain])


def _zeropower_layout_worker(
    rank: int,
    world_size: int,
    placement_kind: str,
    strategy: OrthogonalizationStrategy,
) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    if placement_kind == "replicate":
        placement = Replicate()
        full = torch.randn(64, 32)
    elif placement_kind == "shard0_uneven":
        placement = Shard(0)
        full = torch.randn(65, 32)
    elif placement_kind == "shard1":
        placement = Shard(1)
        full = torch.randn(64, 32)
    else:
        raise AssertionError(f"unknown placement kind {placement_kind}")

    d = distribute_tensor(full, mesh, [placement])
    out = zeropower(d, steps=STEPS, use_triton=False, strategy=strategy)
    ref = zeropower(full, steps=STEPS, use_triton=False, strategy=strategy)

    assert isinstance(out, DTensor)
    assert out.placements == d.placements
    assert_close(
        out.full_tensor(), ref, rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT
    )


def _foreach_layout_worker(
    rank: int,
    world_size: int,
    placement_kind: str,
    strategy: OrthogonalizationStrategy,
) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    placement = Replicate() if placement_kind == "replicate" else Shard(1)
    fulls = [torch.randn(64, 32) + i for i in range(3)]
    ds = [distribute_tensor(f, mesh, [placement]) for f in fulls]

    outs = foreach_zeropower(ds, steps=STEPS, use_triton=False, strategy=strategy)
    refs = foreach_zeropower(fulls, steps=STEPS, use_triton=False, strategy=strategy)

    assert len(outs) == len(refs)
    for out, ref, d in zip(outs, refs, ds, strict=True):
        assert isinstance(out, DTensor)
        assert out.placements == d.placements
        assert_close(
            out.full_tensor(), ref, rtol=RTOL, atol=ATOL, max_mismatch_pct=MAX_MISMATCH_PCT
        )


def _metadata_and_rejection_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    a = distribute_tensor(torch.randn(8, 8), mesh, [Shard(0)])
    b = distribute_tensor(torch.randn(8, 8), mesh, [Replicate()])
    with pytest.raises(AssertionError):
        get_dtensor_metadata([a, b])

    c = distribute_tensor(torch.randn(16, 8), mesh, [Shard(0)])
    with pytest.raises(AssertionError):
        get_dtensor_metadata([a, c])

    base = torch.randn(8, 8)
    d = distribute_tensor(base.t(), mesh, [Shard(0)])
    with pytest.raises(AssertionError):
        get_dtensor_metadata([a, d])

    two_d = distribute_tensor(torch.randn(8, 8), mesh, [Shard(0)])
    with pytest.raises(AssertionError, match="only works for DTensors"):
        foreach_zeropower_3d_fsdp([two_d], use_triton=False)

    wrong_dim_3d = distribute_tensor(torch.randn(8, 8 * world_size, 8), mesh, [Shard(1)])
    with pytest.raises(AssertionError, match="only works for DTensors"):
        foreach_zeropower_3d_fsdp([wrong_dim_3d], use_triton=False)


# --- tests ---


def test_to_local_distributed() -> None:
    run_distributed(_to_local_worker, world_size=2)


def test_zeropower_dtensor_matches_single_process() -> None:
    run_distributed(_zeropower_worker, world_size=2)


def test_zeropower_dtensor_matches_single_process_world_size_one() -> None:
    run_distributed(_zeropower_worker, world_size=1)


def test_foreach_zeropower_dtensor_matches_single_process() -> None:
    run_distributed(_foreach_zeropower_worker, world_size=2)


@pytest.mark.parametrize("strategy", ["newton_schulz", "polar_express"])
def test_foreach_zeropower_3d_fsdp_matches_single_process(
    strategy: OrthogonalizationStrategy,
) -> None:
    run_distributed(_foreach_zeropower_3d_fsdp_worker, world_size=2, args=(strategy,))


def test_dtensor_two_dim_mesh_layout_round_trip() -> None:
    run_distributed(_two_dim_mesh_worker, world_size=4)


def test_is_fsdp_3d_sharded_classification() -> None:
    run_distributed(_is_fsdp_3d_sharded_worker, world_size=2)


@pytest.mark.parametrize("strategy", ["newton_schulz", "polar_express"])
@pytest.mark.parametrize("placement_kind", ["replicate", "shard0_uneven", "shard1"])
def test_zeropower_dtensor_layouts_match_single_process(
    placement_kind: str, strategy: OrthogonalizationStrategy
) -> None:
    run_distributed(
        _zeropower_layout_worker,
        world_size=2,
        args=(placement_kind, strategy),
    )


@pytest.mark.parametrize("strategy", ["newton_schulz", "polar_express"])
@pytest.mark.parametrize("placement_kind", ["replicate", "shard1"])
def test_foreach_zeropower_dtensor_layouts_match_single_process(
    placement_kind: str, strategy: OrthogonalizationStrategy
) -> None:
    run_distributed(
        _foreach_layout_worker,
        world_size=2,
        args=(placement_kind, strategy),
    )


def test_metadata_consistency_and_fsdp_fast_path_rejection() -> None:
    run_distributed(_metadata_and_rejection_worker, world_size=2)
