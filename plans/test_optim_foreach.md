# Test Plan — `optim/optim_foreach.py` (`MuonForeach`)

**Test file:** `tests/optim/test_optim_foreach.py` (covers foreach/base
equivalence, batching, CPU offload, maximize grad reuse, DTensor registration,
and DTensor optimizer paths).

## What the module does

`MuonForeach(Muon)` overrides `muon()` to batch the update with `torch._foreach_*`
ops + batched `foreach_zeropower`. It does **not** fork the rest of `Muon` —
construction, group building, Adam path, checkpointing all come from the base.

Key mechanics:
- `_register_dtensor_foreach_ops()` registers a DTensor op-strategy for
  `_foreach_sign_` at import time (idempotent via `_dtensor_registered` flag).
- `muon()` groups tensors by `(device, dtype)` then by **shape**, optionally
  chunks each shape-group into `batch_size` batches, moves each batch to CUDA
  (CPU-offload support), calls `_foreach_muon`, moves results back.
- `_foreach_muon()` (module-level, low-level): foreach momentum update, Nesterov
  mix, picks `foreach_zeropower_3d_fsdp` when `is_fsdp_3d_sharded(g)` else
  `foreach_zeropower`, scales by `lr_ratio*lr`, applies weight decay, writes `p`.

## ✅ FIXED — weight-decay bug (headline test now lands)

`_foreach_muon` previously ended with:

```python
adjusted_lr = torch._foreach_mul(lr_ratio, lr)
torch._foreach_mul_(u, adjusted_lr)
torch._foreach_mul_(p, 1 - lr * lr)      # <-- BUG: lr², ignores wd / cautious
torch._foreach_add_(p, u, alpha=-1)
```

It did a decoupled-style `p *= (1 - lr*lr)` shrink and **never used
`weight_decay` or `cautious_wd` at all** — the decay coefficient was `lr²` (not
`lr_ratio*lr*weight_decay`), and cautious masking was dropped. This diverged from
the base `Muon.muon()` reference for any `wd != 0`.

**Fix applied:** weight decay is now folded into the update direction `u` before
scaling, mirroring the reference (cautious mask `u*p > 0` when `cautious_wd`,
plain `u += wd*p` otherwise):

```python
if weight_decay != 0:
    if cautious_wd:
        mask = [(u_ * p_ > 0).to(p_.dtype) for u_, p_ in zip(u, p, strict=True)]
        masked_p = torch._foreach_mul(p, mask)
        torch._foreach_add_(u, masked_p, alpha=weight_decay)
    else:
        torch._foreach_add_(u, p, alpha=weight_decay)
adjusted_lr = torch._foreach_mul(lr_ratio, lr)
torch._foreach_mul_(u, adjusted_lr)
torch._foreach_add_(p, u, alpha=-1)
```

**Pinned by:** `test_foreach_matches_base_muon` (below, #1) — `MuonForeach` vs
`Muon` over `wd ∈ {0, 0.3}`, `nesterov`, `cautious`, both strategies, with `lr`
and `wd` chosen so the old `1 - lr*lr` shrink can't coincide with the correct
`lr_ratio*lr*wd` term. Verified: passes with the fix; against the buggy line all
8 `wd≠0` cases fail (~50% element mismatch vs the 5% bf16/cautious-boundary
budget). The 5% budget absorbs the discrete cautious-mask bit-flips that come
from the batched kernel differing from the per-param one at bf16.

## Behavioral contract to cover

| # | Behavior | Test |
| --- | --- | --- |
| 1 | **`MuonForeach` ≈ `Muon` (base) for identical config** | same params/grads/seed, run N steps with both; compare params. Parametrize `wd ∈ {0, 0.1}`, `nesterov ∈ {T,F}`, `cautious ∈ {T,F}`, both strategies. **`wd=0` should match; `wd!=0` exposes the bug above.** `@requires_cuda` (foreach_zeropower triton + CUDA move). |
| 2 | `wd=0` equivalence holds exactly (within bf16 tol) | the clean baseline that *must* pass |
| 3 | Shape grouping: params of **different shapes** in one group all update correctly | mix `(8,16)`,`(16,8)`,`(8,16)`; each matches base `Muon` |
| 4 | Same-shape params get batched together (single `foreach_zeropower` call) | spy/count calls, or just correctness over many same-shape params |
| 5 | `batch_size` chunking: `batch_size < len(group)` produces chunked calls | 5 params, `batch_size=2` records chunks 2/2/1 | ✅ exists |
| 6 | `batch_size` chunking with a count **not divisible** by batch_size | 5 params, `batch_size=2` (last chunk size 1) | ✅ exists |
| 7 | Mixed dtype in one group (fp32 + bf16 params) grouped separately | monkeypatched foreach call records separate dtype buckets | ✅ exists |
| 8 | Empty muon group / all-None grads → no-op (early `len==0` return) | no `_foreach_muon` call, params unchanged | ✅ exists |
| 9 | `maximize=True` foreach path negates grads correctly | vs base `Muon(maximize=True)` |
| 10 | ndim>2 grads: `view_as(p)` writes back the correct shape after orthogonalization | 3D/4D params with `flatten=True` |
| 11 | Adam group still works through `MuonForeach` (inherited path) | mixed muon+adam model steps |
| 12 | `MuonForeach` has no `compile` param but inherits `_init_step_impls`; `_muon_impl == self.muon` | construction smoke |

## CPU-offload path

| # | Behavior | Test |
| --- | --- | --- |
| 13 | Params on **CPU** are moved to CUDA, updated, moved back; final state on CPU matches an all-CUDA run | `@requires_cuda`; allocate params on cpu, grads on cpu |
| 14 | `move_tensors_to_device` no-ops when device types match | cross-ref [test_foreach.md](test_foreach.md) |
| 15 | After offload step, params remain on their original device & dtype | assert `.device`/`.dtype` unchanged |

## DTensor / FSDP fast path

| # | Behavior | Test |
| --- | --- | --- |
| 16 | `_register_dtensor_foreach_ops` is **idempotent** (second import / second call no-ops; `_dtensor_registered` set) | call twice, assert registration flag | ✅ exists |
| 17 | `_foreach_sign_` works on a `TupleStrategy` of DTensors after registration | construct DTensor list, call `torch._foreach_sign_` | ✅ exists |
| 18 | FSDP fast path chosen for 3D Shard(0) DTensors (`is_fsdp_3d_sharded` true → `foreach_zeropower_3d_fsdp`) | NCCL world; step `MuonForeach` on 3D-sharded DTensor params and assert fast path called | ✅ exists |
| 19 | General DTensor path for non-FSDP layouts (`Shard(1)`, 2D) uses `foreach_zeropower` | helper-level coverage exists in `tests/utils/test_dtensor.py`; optimizer-level CPU/gloo coverage is invalid because `_foreach_muon` always moves to CUDA | ⚠️ invalid CPU/gloo assumption |
| 20 | End-to-end `MuonForeach.step()` on a sharded model matches replicated single-proc training | CUDA/NCCL `Shard(0)` optimizer path is covered; CPU/gloo optimizer coverage is invalid because `_foreach_muon` always moves to CUDA | ✅ CUDA/NCCL covered |

## Notes / gating

- Most end-to-end correctness tests are `@requires_cuda` because `_foreach_muon`
  unconditionally moves batches to `torch.device("cuda")` and `foreach_zeropower`
  defaults to `use_triton=True`. Structural tests for grouping/chunking use
  monkeypatching and run on CPU; optimizer DTensor end-to-end tests are
  `requires_2_gpus`/NCCL because `_foreach_muon` still moves to CUDA.
- The `# @torch.compile(dynamic=True)` decorator on `_foreach_muon` is commented
  out — note that the foreach path currently runs eager.
