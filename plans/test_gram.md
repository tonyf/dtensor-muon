# Test Plan — `kernels/gram.py`

**Test file:** `tests/kernels/test_gram.py` (covers forward correctness,
symmetry, shape/masking edge cases, and assertion/contract surface).

## What the module does

A fused, autotuned Triton kernel computing the batched Gram matrix
`y[b] = x[b] @ x[b].T` over the last two dims of a `(..., M, K)` tensor, writing a
`(..., M, M)` output. It computes only the upper-triangle blocks and mirrors them
into the lower triangle. Public surface:

- `gram_(d_in, d_out)` — in-place; writes into a caller-provided **contiguous**
  output. Asserts CUDA, same device, same dtype, `ndim >= 2`, matching batch
  dims, `d_out` square in last two dims, and `d_out.is_contiguous()`.
- `gram(d_in)` — allocating wrapper that makes an empty output and calls `gram_`.

CUDA-only. Gate every test with `@requires_cuda`.

## Behavioral contract to cover

| # | Behavior | Test case | Status |
| --- | --- | --- | --- |
| 1 | Matches `x @ x.mT` for 2D input | `shape=(128,64)` | ✅ exists |
| 2 | Matches reference for 3D / 4D batched | `(32,128,64)`, `(2,32,128,64)` | ✅ exists |
| 3 | Wide case `K > M` | `(32,128,256)` | ✅ exists |
| 4 | fp32 (TF32 path) and bf16 both within tolerance | parametrized dtype | ✅ exists |
| 5 | `gram_` (in-place) matches allocating `gram` and reference | `gram_inplace_wrapper` | ✅ exists |
| 6 | Output is **exactly symmetric** (mirror logic correct) | assert `y == y.mT` (bit-exact, since lower triangle is a copy of the upper) | ✅ exists |
| 7 | Tall case `M > K` (more rows than cols) | `(32,256,64)` | ✅ exists |
| 8 | Diagonal block (`pid_m == pid_n`) vs off-diagonal both correct | shape forcing multiple M-blocks, e.g. `M=300` | ✅ exists |
| 9 | Non-multiple-of-block `M` and `K` (masking / `%M` wrap correctness) | `M=130, K=70` | ✅ exists |
| 10 | Single-element batch and `M=1` / `K=1` degenerate shapes | `(1,1,8)`, `(4,1,1)` | ✅ exists |

## Contract / assertion tests

These documented preconditions in `gram_` are pinned by
`tests/kernels/test_gram.py`:

- **Non-contiguous output rejected.** Pass a transposed (non-contiguous) `d_out`
  → expect `AssertionError` ("d_out must be contiguous").
- **Non-contiguous input is tolerated.** `gram_` calls `.contiguous()` on
  `d_in`; pass a transposed input and confirm the result still matches
  `x @ x.mT` (input copy path).
- **dtype mismatch** between `d_in` and `d_out` → `AssertionError`.
- **device mismatch** (two different CUDA devices, or CPU vs CUDA) →
  `AssertionError`. (Multi-GPU sub-case skip if `device_count() < 2`.)
- **CPU input** → `AssertionError` (the `is_cuda` assert), not a silent wrong
  answer.
- **wrong output shape** — `d_out` last two dims not `(M, M)`, or batch dims not
  matching `d_in` → `AssertionError`.
- **`ndim < 2`** input → `AssertionError`.

## Numerical / robustness cases

- **Determinism / autotune stability.** Run the same input twice; outputs must
  be identical (autotuner picks a config but result must not vary run-to-run).
- **Large `K` accumulation.** `K` spanning many `BLOCK_SIZE_K` iterations (e.g.
  `K=1024`) to exercise the K-loop masking on the final partial block.
- **`x` with extreme magnitudes / zeros.** All-zero input → all-zero output; a
  row of zeros yields zero row/col in the Gram matrix.

## Tolerances

Reuse the module's `_TOL` table: fp32 `(2e-2, 2e-2, 0.1)` (TF32 in `tl.dot`),
bf16 `(1e-2, 1e-2, 0.1)`. The symmetry test (#6) should be **exact** because the
lower triangle is a literal store of the transposed upper block, not a recompute.

## Notes

- The `__main__` Typer CLI in the test file is a manual benchmark harness, not
  pytest-collected — leave it, but the plan's coverage is about the
  `@requires_cuda` pytest functions.
- No backward pass: the kernel is not differentiable and is only used inside
  `@torch.no_grad()` optimizer steps, so `bwd` testing is out of scope.
