# Test Plan — `orthogonalize/orthogonalize.py` (single-process dispatch)

**Test file:** `tests/orthogonalize/test_orthogonalize.py` (covers
`zeropower`/`foreach_zeropower` Triton-vs-eager, CPU dispatch contracts,
preconditions, dtype, and plain tensor FSDP guard behavior).

> The DTensor branches of this module (the distributed plumbing) are covered by a
> separate plan: [test_dtensor_distributed.md](test_dtensor_distributed.md). This
> plan is the **single-process / plain-tensor** dispatch surface.

## What the module does

The seam between optimizers and the iteration kernels. Public surface:

- `OrthogonalizationStrategy = Literal["newton_schulz", "polar_express"]`
- `_get_orthogonalization_fn(strategy, use_triton)` — 2×2 dispatch table.
- `zeropower(G, steps, eps, use_triton, strategy)` — single tensor.
- `foreach_zeropower(Gs, steps, eps, use_triton, strategy)` — batched (stacks,
  orthogonalizes, unbinds).
- `get_dtensor_metadata`, `is_fsdp_3d_sharded`, `foreach_zeropower_3d_fsdp` —
  DTensor helpers (see the distributed plan; `is_fsdp_3d_sharded` on plain
  tensors is in-scope here — must return `False`).

## Behavioral contract to cover (plain tensors)

| # | Behavior | Test case | Status |
| --- | --- | --- | --- |
| 1 | `zeropower` Triton matches eager iteration | `(256,128),(128,256),(512,512)` × both strategies | ✅ exists |
| 2 | `foreach_zeropower` Triton matches stacked eager | 4×`(256,128)` × both strategies | ✅ exists |
| 3 | **Dispatch table** maps correctly | `_get_orthogonalization_fn` returns `ns_loop*`/`pe_loop*` for each `(strategy, use_triton)` cell | ✅ exists |
| 4 | **Unknown strategy raises** | `_get_orthogonalization_fn("bogus", …)`, `zeropower`, and `foreach_zeropower` raise `ValueError` | ✅ exists |
| 5 | `foreach_zeropower` returns a **list** of the same length/order as input | len + per-element identity of mapping | ✅ exists |
| 6 | **`zeropower` == `foreach_zeropower` single-element** | `foreach_zeropower([G])[0] ≈ zeropower(G)` for plain tensors | ✅ exists |
| 7 | `use_triton=False` path runs (and matches `use_triton=True` within tol) | both strategies | ✅ exists |
| 8 | Output dtype | inputs are fp32; output is bf16 (iteration runs in bf16, no upcast on return) | ✅ exists |
| 9 | `is_fsdp_3d_sharded` on plain tensors → `False` | plain tensors here; mixed DTensor/plain list covered in distributed plan | ✅ exists |
| 10 | `foreach_zeropower` mixed shapes in one call | list of differing `(N,M)` raises `RuntimeError` from `torch.stack` | ✅ exists |
| 11 | 3D inputs to `foreach_zeropower` (the `"G N M"` annotated case) | list of `(G,N,M)` plain tensors orthogonalize per last-2-dims | ✅ exists |

## Notes / sharp edges

- **CUDA gating:** `use_triton=True` paths need CUDA. The `use_triton=False`
  dispatch table and `ValueError` tests are pure-Python/CPU and should run
  everywhere — split these out from `@requires_cuda` so they run on CPU CI.
- **Reference choice:** continue using the eager `__wrapped__` iteration as
  ground truth (compiled 2D miscompile). The `eager()` helper is already in the
  file.
- `foreach_zeropower` empty list (`Gs=[]`): `Gs[0]` indexes — **will IndexError**.
  Callers guard with `len(params)==0` before calling, but pin this as a
  documented precondition (or add a guard + test).
- The Polar Express `use_triton=True` cell currently maps to `pe_loop_triton`
  (the comment "doesn't have a triton kernel yet" is stale — it does). Test #3
  should assert the *current* mapping so the comment/code drift is visible.
