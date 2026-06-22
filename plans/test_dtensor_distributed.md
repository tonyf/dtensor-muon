# Test Plan — DTensor distributed paths + `utils/dtensor.py`

**Test file:** `tests/utils/test_dtensor.py` (covers the CPU/gloo DTensor helper
and orthogonalization paths, including non-default placements and 2D meshes).

Covers the distributed branches of `orthogonalize/orthogonalize.py` and the
`to_local` helper in `utils/dtensor.py`. These run **anywhere** (gloo on CPU,
`use_triton=False`) via `run_distributed`.

> ✅ Prerequisite resolved: `run_distributed` lives in
> `test_support/distributed.py` and is tested directly in
> [test_testing_harness.md](test_testing_harness.md).

## What's under test

- `to_local(tensor, full_tensor=False)` — `.to_local()` shard vs `.full_tensor()`
  gather vs passthrough for plain tensors.
- `zeropower(DTensor)` — `full_tensor()` → orthogonalize → `from_local` replicate
  → `redistribute` back to original placements.
- `foreach_zeropower(list[DTensor])` — stack → redistribute to `Shard(0)` →
  orthogonalize local shards → redistribute back → unbind to per-param DTensors.
- `foreach_zeropower_3d_fsdp(list[DTensor])` — fast path for 3D tensors sharded
  only on dim 0: skip the redistribute, work on local shards directly.
- `is_fsdp_3d_sharded` — the guard selecting the fast path.
- `get_dtensor_metadata` — metadata extraction + cross-tensor consistency check.

## Behavioral contract to cover

| # | Behavior | Test case | Status |
| --- | --- | --- | --- |
| 1 | `to_local` shard / full_tensor / passthrough | `_to_local_worker` | ✅ exists |
| 2 | `zeropower(DTensor)` == single-process `zeropower(full)` | `_zeropower_worker` | ✅ exists |
| 3 | `foreach_zeropower(list[DTensor])` == single-process | `_foreach_zeropower_worker` | ✅ exists |
| 4 | `foreach_zeropower_3d_fsdp` == single-process | `_foreach_zeropower_3d_fsdp_worker` | ✅ exists |
| 5 | `is_fsdp_3d_sharded` classification (Shard0-3D yes; Replicate/2D/plain no) | `_is_fsdp_3d_sharded_worker` | ✅ exists |
| 6 | **Placements round-trip** — output `.placements` equals input's | assert `out.placements == d.placements` (not just `full_tensor` match) | ✅ exists |
| 7 | **`Replicate()` input** to `zeropower`/`foreach_zeropower` | a fully-replicated DTensor orthogonalizes and stays replicated | ✅ exists |
| 8 | **Shard on dim 1** (tensor-parallel, not FSDP) | `zeropower` of a `Shard(1)` tensor matches single-proc; exercises the non-dim-0 redistribute | ✅ exists |
| 9 | **`world_size` other than 2** | run workers at `world_size=1` and a `(2,2)` mesh with `world_size=4`; uneven shapes included | ✅ exists |
| 10 | **Uneven sharding** — dim 0 not divisible by world_size | `(65,32)` over 2 ranks; verify gather + orthogonalize still matches | ✅ exists |
| 11 | `get_dtensor_metadata` consistency assert fires | mismatched placements, shape, and stride | ✅ exists |
| 12 | `foreach_zeropower_3d_fsdp` **rejects** non-FSDP layout | 2D and 3D `Shard(1)` DTensors | ✅ exists |
| 13 | **2D mesh** (FSDP × TP, e.g. `(2,2)`) | `is_fsdp_3d_sharded` with `[Shard(0), Replicate()]`; `zeropower` round-trip on a 2D-mesh DTensor | ✅ exists |
| 14 | Both strategies under DTensor | parametrized over `newton_schulz` / `polar_express` | ✅ exists |
| 15 | `foreach_zeropower` preserves **order** across ranks | distinct per-element inputs; each output matches its own reference, not a permutation | ✅ exists |

## Equivalence reference (the core assertion)

Every worker's success criterion: seed RNG identically per rank → build the same
full tensor → shard it → distributed result `.full_tensor()` must match the
single-process `zeropower(full)` within `(RTOL=1e-2, ATOL=2e-2,
MAX_MISMATCH_PCT=1.0)`. This is sound and should be the template for all new
workers (#6–#15). Keep the per-rank identical seeding.

## Sharp edges / notes

- Workers **must be module-level** (picklable for `mp.spawn`). Build the
  `DeviceMesh` and DTensors *inside* the worker.
- `_foreach_sign_` DTensor op-strategy is registered at `MuonForeach` import
  time; the foreach DTensor path here does not exercise it directly, but a
  `MuonForeach`+DTensor optimizer test does — cross-reference
  [test_optim_foreach.md](test_optim_foreach.md).
- gloo backend has no `async_op` collective overlap guarantees; the code uses
  `redistribute(async_op=True)` — confirm results are still correct (the
  `.to_local()`/`.full_tensor()` after forces completion).
- Add a `requires_world_size(n)` skip helper for cases needing ≥N CPUs/ranks; 2
  is the safe default for CI.
