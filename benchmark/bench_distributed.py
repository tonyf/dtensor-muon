"""Distributed orthogonalization benchmark over a DTensor/FSDP2 mesh.

This is the piece that justifies the DTensor machinery: orthogonalizing the momentum
update when parameters are sharded across ranks. We time three ways of doing it on the
*same* set of 3D tensors (a stack of per-param momentum buffers sharded on dim 0, the
FSDP layout):

1. **FSDP fast path** — ``foreach_zeropower_3d_fsdp``: works directly on local shards,
   no redistribute. This is what ``MuonForeach.muon`` prefers when ``is_fsdp_3d_sharded``.
2. **General path** — ``foreach_zeropower`` on the same sharded DTensors: stacks,
   redistributes to shard dim 0, orthogonalizes, redistributes back (collective-heavy).
3. **Single-device baseline** — ``foreach_zeropower`` on the replicated full tensors
   (no collectives), the "no sharding" reference each rank computes locally.

Backend is chosen automatically: NCCL/CUDA when ≥2 GPUs are visible (real numbers),
otherwise gloo/CPU (portable, but timings are dominated by spawn/collective overhead and
are *indicative only*).

The spawned worker has no return channel, so rank 0 writes the timings as JSON to a path
the parent passes in (after a ``barrier`` so the per-rank loops stay in step). Timing uses
the fixed-rep :func:`bench_fixed` so every rank issues an identical number of collectives.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

# Ensure the repo root is importable so the spawned children can import ``benchmark.*``
# (multiprocessing's spawn start method propagates the parent's sys.path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.harness import cuda_available, markdown_table  # noqa: E402

# A list of COUNT identical (G, N, M) tensors sharded on dim 0 — mimics COUNT same-shaped
# weight matrices' momentum buffers under FSDP. The FSDP fast path batches uniform-shape
# buffers, so every tensor in the list must share shape/stride; G must divide world_size.
_SHAPE = (16, 2048, 1024)
_COUNT = 8
_QUICK_SHAPE = (8, 256, 256)
_QUICK_COUNT = 4
_STEPS = 5


def _dist_bench_worker(
    rank: int,
    world_size: int,
    *,
    device_type: str,
    shape: tuple[int, int, int],
    count: int,
    steps: int,
    out_path: str,
) -> None:
    import statistics
    import time

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from dtensor_muon.orthogonalize import (
        foreach_zeropower,
        foreach_zeropower_3d_fsdp,
        is_fsdp_3d_sharded,
    )

    cuda = device_type == "cuda"

    def _timed(fn, warmup=10, reps=60) -> float:
        """Barrier-aligned, fixed-rep *median* ms.

        A deterministic rep count keeps the collective-bearing paths in lockstep across
        ranks; the median (not mean) is robust to this power-capped card's clock jitter;
        the leading barrier lines the ranks up before the measured window.
        """
        for _ in range(warmup):
            fn()
        if cuda:
            torch.cuda.synchronize()
        dist.barrier()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(samples)

    mesh = init_device_mesh(device_type, (world_size,))
    torch.manual_seed(0)
    use_triton = cuda

    g, n, m = shape
    assert g % world_size == 0, "dim 0 must divide world_size"
    fulls = [torch.randn(g, n, m, device=device_type) for _ in range(count)]
    ds = [distribute_tensor(f, mesh, [Shard(0)]) for f in fulls]
    assert is_fsdp_3d_sharded(ds), "expected 3D DTensors sharded only on dim 0"

    # Every rank runs every timed call (the general path issues collectives — ranks must
    # stay in lockstep). _timed uses a deterministic rep count for the same reason.
    t_fast = _timed(lambda: foreach_zeropower_3d_fsdp(ds, steps=steps, use_triton=use_triton))
    t_general = _timed(lambda: foreach_zeropower(ds, steps=steps, use_triton=use_triton))
    t_single = _timed(lambda: foreach_zeropower(fulls, steps=steps, use_triton=use_triton))

    dist.barrier()
    if rank == 0:
        Path(out_path).write_text(
            json.dumps(
                {
                    "world_size": world_size,
                    "device": device_type,
                    "fsdp_fast_ms": t_fast,
                    "general_ms": t_general,
                    "single_device_ms": t_single,
                }
            )
        )


def run(quick: bool = False) -> str:
    from test_support.distributed import run_distributed

    if cuda_available() and torch.cuda.device_count() >= 2:
        backend, device_type, world_size, indicative = "nccl", "cuda", 2, False
    else:
        backend, device_type, world_size, indicative = "gloo", "cpu", 2, True

    shape = _QUICK_SHAPE if quick else _SHAPE
    count = _QUICK_COUNT if quick else _COUNT

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    try:
        run_distributed(
            _dist_bench_worker,
            world_size=world_size,
            kwargs=dict(
                device_type=device_type,
                shape=shape,
                count=count,
                steps=_STEPS,
                out_path=out_path,
            ),
            backend=backend,
            device_type=device_type,
        )
        data = json.loads(Path(out_path).read_text())
    finally:
        Path(out_path).unlink(missing_ok=True)

    base = data["single_device_ms"]
    fast, general = data["fsdp_fast_ms"], data["general_ms"]
    rows = [
        ["single-device (replicated, no collectives)", f"{base:.4f}", "1.00x (ref)"],
        ["FSDP fast path (foreach_zeropower_3d_fsdp)", f"{fast:.4f}", f"{base / fast:.2f}x"],
        [
            "general path (foreach_zeropower + redistribute)",
            f"{general:.4f}",
            f"{base / general:.2f}x",
        ],
    ]
    shape = _QUICK_SHAPE if quick else _SHAPE
    count = _QUICK_COUNT if quick else _COUNT
    note = (
        f"Backend **{backend}** on **{device_type}**, world_size **{world_size}**, "
        f"{_STEPS} steps. Workload: {count} × {tuple(shape)} buffers sharded on dim 0. "
        f"Same tensors for all paths; speedup vs the single-device replicated baseline. "
        f"The FSDP fast path orthogonalizes each rank's local shard with no collectives "
        f"(~{base / fast:.1f}× the single-device baseline by splitting the batch across "
        f"ranks) and is **{general / fast:.1f}× faster than the general path**, which "
        f"pays for the redistribute all-gather/scatter. The general path's collective "
        f"time is noisy run-to-run; the fast path is stable."
    )
    if indicative:
        note += (
            "\n\n> ⚠️ **Indicative only** — gloo/CPU timings are dominated by spawn and "
            "collective overhead. Run on ≥2 GPUs for meaningful numbers."
        )
    return note + "\n\n" + markdown_table(["path", "per-call (ms)", "speedup"], rows)


if __name__ == "__main__":
    print(run())
