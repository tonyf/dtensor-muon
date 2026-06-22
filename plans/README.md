# Test Plans — `dtensor-muon`

This directory holds a test plan per source module / test file. Each plan
catalogs the behavior that must be covered to call the module "correct", maps
each behavior to concrete test cases, and calls out known sharp edges / suspected
bugs that the tests should pin down.

The test files have been expanded against these plans. Treat the per-plan status
tables as the current audit record: most rows are covered, while a small number
are marked as implementation blockers or intentionally invalid assumptions.

## Module → plan → test file map

| Source module | Plan | Test file |
| --- | --- | --- |
| `kernels/gram.py` | [test_gram.md](test_gram.md) | `tests/kernels/test_gram.py` |
| `orthogonalize/newton_schulz.py` | [test_newton_schulz.md](test_newton_schulz.md) | `tests/orthogonalize/test_newton_schulz.py` |
| `orthogonalize/polar_express.py` | [test_polar_express.md](test_polar_express.md) | `tests/orthogonalize/test_polar_express.py` |
| `orthogonalize/orthogonalize.py` (single-proc dispatch) | [test_orthogonalize.md](test_orthogonalize.md) | `tests/orthogonalize/test_orthogonalize.py` |
| `orthogonalize/orthogonalize.py` (DTensor paths) + `utils/dtensor.py` | [test_dtensor_distributed.md](test_dtensor_distributed.md) | `tests/utils/test_dtensor.py` |
| `optim/optim.py` (`Muon` base) | [test_optim.md](test_optim.md) | `tests/optim/test_optim.py` |
| `optim/optim_foreach.py` (`MuonForeach`) | [test_optim_foreach.md](test_optim_foreach.md) | `tests/optim/test_optim_foreach.py` |
| `optim/optim_lp.py` (`MuonLP` + 8/4/fp8) | [test_optim_lp.md](test_optim_lp.md) | `tests/optim/test_optim_lp.py` |
| `utils/foreach.py` | [test_foreach.md](test_foreach.md) | `tests/utils/test_foreach.py` |
| `tests/testkit.py` (+ `test_support/distributed.py`) | [test_testing_harness.md](test_testing_harness.md) | `tests/utils/test_testing.py` (new) |

## Cross-cutting conventions (apply to every plan)

- **CUDA gating.** The Triton kernel (`gram_`) and everything that calls it
  (`ns_loop_triton`, `pe_loop_triton`, `zeropower(..., use_triton=True)`) are
  CUDA-only. Gate with `@requires_cuda` (`pytest.mark.skipif(not
  torch.cuda.is_available())`). On a CPU host these silently skip — a green CPU
  run does **not** mean these paths ran. The DTensor tests are the exception:
  they spawn a **gloo** world on CPU and run anywhere with `use_triton=False`.
- **Compiled-reference trap.** `ns_loop` / `pe_loop` are `@torch.compile`'d and
  *miscompile on 2D inputs*. Compare against the eager original reached via
  `fn.__wrapped__` (the `eager()` helper), never the compiled wrapper. Keep the
  `xfail` sentinel tests that document the miscompile.
- **Precision.** Muon math runs the buffer/param update in fp32 but
  orthogonalizes in bf16. Correctness comparisons use the loose
  `assert_close(rtol≈1e-2, atol≈2e-2, max_mismatch_pct)` from
  `tests/testkit.py`, not bit-exactness.
- **Recompile limit.** `tests/conftest.py` raises Dynamo's cache limits to 256.
  Any new wide parametrization over compiled kernels relies on this; don't lower it.

## Suspected bugs the tests should pin (see individual plans for detail)

1. ✅ **FIXED** — `optim_foreach.py` decoupled weight decay used `1 - lr * lr`
   and ignored `weight_decay` / `cautious_wd`. Now folds (cautious) weight decay
   into the update direction like the base `muon()` reference, pinned by the
   `MuonForeach`-vs-`Muon` equivalence test. (See
   [test_optim_foreach.md](test_optim_foreach.md).)
2. ✅ **FIXED** — distributed test helpers now live outside the shipped package in
   `test_support/distributed.py` and are pinned by `tests/utils/test_testing.py`.
3. ✅ **FIXED** — `maximize` no longer mutates caller-held grads in either base
   or foreach Muon paths; reuse/double-step regressions are pinned in optimizer
   tests.
4. ⚠️ **IMPLEMENTATION BLOCKER** — quantized `Muon8bit` / `Muon4bit` /
   `MuonFp8` step tests are strict-xfailed because the current torchao state
   subclasses do not support the in-place momentum update path. Construction and
   DTensor buffer behavior are covered.
