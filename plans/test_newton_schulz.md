# Test Plan — `orthogonalize/newton_schulz.py`

**Test file:** `tests/orthogonalize/test_newton_schulz.py` (covers Triton-vs-eager
parity, empty-batch passthrough, matrix-sign properties, transpose/zero-step
behavior, coefficient overrides, and a 2D-compile xfail sentinel).

## What the module does

Two implementations of the Newton-Schulz "matrix sign" iteration over the last
two dims of a `(..., N, M)` tensor (operates in bf16):

- `ns_loop(X, steps, *, a, b, c, eps)` — pure PyTorch, `@torch.compile(fullgraph=True)`.
  Reference math. **Known to miscompile on 2D inputs** — tests use the eager
  original `ns_loop.__wrapped__`.
- `ns_loop_triton(X, steps, ...)` — `@torch.compile(fullgraph=True)`,
  uses `gram_` for the `X X^T` and `A A^T` products. CUDA-only.

Both: transpose when `N > M` so the smaller dim is last, normalize by spectral
norm proxy (`X / (‖X‖_F + eps)`), iterate `steps` times, transpose back.

## Behavioral contract to cover

| # | Behavior | Test case | Status |
| --- | --- | --- | --- |
| 1 | Triton matches eager `ns_loop` across shapes | batched wide/tall/square + 2D | ✅ exists |
| 2 | Steps sweep `1,3,5` | parametrized | ✅ exists |
| 3 | Empty batch `(0,N,M)` passthrough | returns input unchanged | ✅ exists |
| 4 | 2D compile regression documented | `xfail(strict=True)` sentinel | ✅ exists |
| 5 | **Output is approximately orthogonal** | for full-rank `X`, `U @ U.T ≈ I` (on the smaller dim) within tolerance — the actual property Muon needs, not just self-consistency | ✅ exists |
| 6 | **Singular-value flattening** | SVD of output has singular values near 1 | ✅ exists |
| 7 | **Preserves left/right singular vectors** | `out ≈ U V^T` (matrix sign) | ✅ exists |
| 8 | Transpose path parity (`N>M`) vs non-transpose (`N<M`) | result for `X` and `X.T` are transposes of each other | ✅ exists |
| 9 | `steps=0` | returns the normalized input | ✅ exists |
| 10 | Coefficient overrides `a,b,c` are honored | non-default coeffs match a hand-rolled one-step iteration | ✅ exists |
| 11 | Convergence: more steps → closer to orthogonal | original monotonic claim is not stable for the aggressive coefficients; replaced by orthogonality, singular-value, and matrix-sign contracts above | ⚠️ invalid assumption |

## The key correctness property

The tests include a "ground truth" check independent of Triton/eager agreement:

```
X = randn(N, M)               # full rank, N != M
out = ns_loop.__wrapped__(X.bf16(), steps=5).float()
# smaller dimension d = min(N, M); the d x d Gram should be ≈ identity
G = out @ out.T  if N <= M else out.T @ out
assert_close(G, eye(d), atol=…)   # orthonormal rows/cols
```

This is the property Muon depends on and is tested for the eager reference across
tall/wide/square shapes, with Triton parity covered by the CUDA tests.

## Edge / robustness cases

- **Rank-deficient input** (e.g. a rank-1 matrix, or duplicated rows): NS sign
  is ill-defined; document and pin the actual behavior (no NaN/Inf; output
  finite). Important because real gradients can be near-low-rank. ✅ covered
- **Already-orthogonal input** is a near-fixed point: `out ≈ X`. ✅ covered
- **Scale invariance.** `ns_loop(c·X) ≈ ns_loop(X)` for `c > 0` (the
  normalization removes scale). Verify across a few magnitudes incl. very small
  and large where `eps` matters. ✅ covered
- **NaN/Inf safety.** No NaNs for the normalization on an all-zero input (the
  `+ eps` guards the divide) — assert output finite. ✅ covered
- **`B=1` batch** vs unbatched 2D produce the same per-matrix result. ✅ covered

## Sharp edges to keep documented

- Keep `test_ns_loop_2d_compile_regression` as a **strict xfail**: it asserts the
  compiled `ns_loop` is wrong on 2D. If it ever starts passing, the upstream
  miscompile was fixed and the eager-via-`__wrapped__` workaround can be removed.
- bf16 tolerances: `RTOL=1e-2, ATOL=2e-2, MAX_MISMATCH_PCT=0.5`. The
  orthogonality tests (#5–#7) need a looser `atol` (bf16 over 5 iters) — tune
  empirically but keep it tight enough to fail on a broken iteration.
