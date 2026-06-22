# Benchmark results

Regenerate with `uv run python benchmark/run.py`.

- **Device:** 2x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **CUDA:** 13.0
- **PyTorch:** 2.12.1+cu130
- **Python:** 3.12.9
- **Platform:** Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- **Commit:** f7b0cb4

## Triton kernels

### Gram kernel (`gram` vs `x @ x.mT`)

| shape | dtype | torch (ms) | triton (ms) | speedup |
| --- | --- | --- | --- | --- |
| (32, 2048, 1024) | bfloat16 | 1.1234 | 0.6077 | 1.85x |

### Orthogonalization loops (5 steps, bf16; eager vs compiled torch vs triton)

| strategy | shape | eager-torch (ms) | compiled-torch (ms) | triton (ms) | triton vs compiled |
| --- | --- | --- | --- | --- | --- |
| newton_schulz | (32, 2048, 1024) | 12.7727 | 10.1110 | 9.3620 | 1.08x |
| polar_express | (32, 2048, 1024) | 12.7811 | 10.1018 | 9.3657 | 1.08x |

## Single-device optimizers

Device: **cuda**. Workload: 16 weight matrices (shapes [(512, 512), (512, 1536), (512, 2048), (2048, 512)]). Timing a full `optimizer.step()`; speedup vs the naive reference. Newton-Schulz variants verified against naive: Muon (ns), MuonForeach (ns).

| optimizer | step (ms) | speedup |
| --- | --- | --- |
| naive (ns) | 8.6122 | 1.00x (ref) |
| Muon (ns) | 9.3614 | 0.92x |
| Muon (pe) | 9.5561 | 0.90x |
| MuonForeach (ns) | 2.4483 | 3.52x |
| MuonForeach (pe) | 2.6481 | 3.25x |

_`MuonLP` (4/8-bit/fp8 states) is omitted: it optimizes optimizer-state memory, not step time, and its quantized buffers don't support the in-place update path (see `tests/optim/test_optim_lp.py`)._

## Distributed orthogonalization

Backend **nccl** on **cuda**, world_size **2**, 5 steps. Workload: 4 × (8, 256, 256) buffers sharded on dim 0. Same tensors for all paths; speedup vs the single-device replicated baseline. The FSDP fast path orthogonalizes each rank's local shard with no collectives (~0.7× the single-device baseline by splitting the batch across ranks) and is **1.6× faster than the general path**, which pays for the redistribute all-gather/scatter. The general path's collective time is noisy run-to-run; the fast path is stable.

| path | per-call (ms) | speedup |
| --- | --- | --- |
| single-device (replicated, no collectives) | 0.4984 | 1.00x (ref) |
| FSDP fast path (foreach_zeropower_3d_fsdp) | 0.7364 | 0.68x |
| general path (foreach_zeropower + redistribute) | 1.1420 | 0.44x |
