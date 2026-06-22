# Test Plan — `orthogonalize/polar_express.py`

**Test file:** `tests/orthogonalize/test_polar_express.py` (covers Triton-vs-eager
parity, empty-batch, matrix-sign properties, transpose/zero-step behavior,
coefficient table integrity, max-step rejection, and a 2D-compile xfail).

## What the module does

Polar Express orthogonalization (arxiv 2505.16932), adapted from nanochat. Same
shape contract as Newton-Schulz but uses **per-step precomputed coefficients**
`POLAR_EXPRESS_COEFFS` (a list of 5 `(a,b,c)` triples) instead of fixed `a,b,c`,
and normalizes by `‖X‖_F * 1.02 + eps`.

- `pe_loop(X, steps, *, eps)` — pure PyTorch, `@torch.compile(fullgraph=True)`.
  Reference. **Miscompiles on 2D** → tests use `pe_loop.__wrapped__`.
- `pe_loop_triton(X, steps, ...)` — Triton (`gram_`-backed), CUDA-only.

Asserts `steps <= 5`, then iterates `POLAR_EXPRESS_COEFFS[:steps]`.

## Behavioral contract to cover

| # | Behavior | Test case | Status |
| --- | --- | --- | --- |
| 1 | Triton matches eager `pe_loop` across shapes | wide/tall/square + 2D | ✅ exists |
| 2 | Steps sweep `1, 3, len(COEFFS)` | parametrized | ✅ exists |
| 3 | Empty batch passthrough | `(0,256,128)` | ✅ exists |
| 4 | 2D compile regression | `xfail(strict=True)` sentinel | ✅ exists |
| 5 | **Orthogonality property** | `U U^T ≈ I` on smaller dim for full-rank input (ground-truth, independent of both impls) | ✅ exists |
| 6 | **Singular values → 1** | SVD of output has σ ≈ 1 | ✅ exists |
| 7 | **Faster convergence than NS at equal steps** | original claim is not stable for the current aggressive coefficients; not a contract | ⚠️ invalid assumption |
| 8 | `steps > 5` is rejected | `steps=6` raises `AssertionError`; public `zeropower(..., strategy="polar_express", steps=6)` is pinned too | ✅ exists |
| 9 | `steps=0` returns normalized input | loop never runs | ✅ exists |
| 10 | Transpose-path symmetry | result for `X` and `X.T` are transposes | ✅ exists |
| 11 | `COEFFS` table integrity | length is 5; first triple matches the published value (guards accidental edits) | ✅ exists |

## The key correctness property

The tests include the same ground-truth orthogonality and matrix-sign checks as
Newton-Schulz, independent of Triton/eager agreement.

## Edge / robustness cases

- **`steps` rejection** is the PE-specific hazard: `Muon`'s default
  `ns_steps=5` matches the table length, but the option is user-settable.
  `steps > 5` is an `AssertionError`, not a clamp.
- Rank-deficient / already-orthogonal / scale-invariance / all-zero / NaN-Inf:
  same battery as Newton-Schulz (#edge cases there) — PE's `1.02` fudge factor in
  the normalizer slightly under-normalizes by design. ✅ covered
- **`B=1` vs unbatched 2D** agree. ✅ covered

## Sharp edges to keep documented

- Keep `test_pe_loop_2d_compile_regression` as **strict xfail**.
- The `1.02` normalization constant is intentional (keeps σ < 1 before
  iterating). Don't "fix" it; if a test needs tighter tolerance, account for it
  rather than removing it.
- Tolerances: `RTOL=1e-2, ATOL=2e-2, MAX_MISMATCH_PCT=0.5`.
