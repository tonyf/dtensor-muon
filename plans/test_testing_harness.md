# Test Plan — `tests/testkit.py` (+ `test_support/distributed.py`)

**Test file:** `tests/utils/test_testing.py` (covers clone/assert/run-example
behavior, distributed helper process handling, and benchmark/table helpers).

This module is shared by the test suite *and* `benchmark/`. Bugs here produce
false greens (a broken comparison that never fails) or false reds, so it deserves
its own coverage even though it's "just test infra".

## ✅ Fixed blocker

`run_distributed`, `_dist_worker_entry`, and `_find_free_port` live in
`test_support/distributed.py`, outside the shipped `dtensor_muon` package. The
following regressions are pinned:

- `_find_free_port()` returns a bindable port.
- `run_distributed(noop_worker, world_size=1/2)` runs.
- worker assertions propagate to the parent.
- rank environment variables and args/kwargs are forwarded.
- a second distributed run works after a failing one.

## `clone_args`

| # | Behavior | Test |
| --- | --- | --- |
| 1 | Tensors detached + cloned (mutating a clone doesn't touch original) | in-place add on clone, original unchanged |
| 2 | `requires_grad` preserved | grad-requiring tensor stays grad-requiring |
| 3 | Non-tensor args passed through unchanged (identity) | ints/strings/None |
| 4 | Returns a new tuple each call (no shared buffers across calls) | two clones are distinct objects |

## `assert_close`

| # | Behavior | Test |
| --- | --- | --- |
| 5 | Close tensors pass; far tensors raise | basic |
| 6 | Compares in **float32** regardless of input dtype | bf16 inputs upcast |
| 7 | `max_mismatch_pct=None` → strict elementwise (any mismatch raises) | one bad element fails |
| 8 | `max_mismatch_pct=p` tolerates ≤p% mismatched, raises above | construct exactly-p% mismatch boundary |
| 9 | Mismatch percentage computed correctly | known fraction of bad elements |
| 10 | Custom `msg` surfaces in the error | substring check |
| 11 | Empty tensors don't divide-by-zero (`max(numel,1)`) | `()` / size-0 |

## `_as_tensors`

| # | Behavior | Test |
| --- | --- | --- |
| 12 | Single tensor → `[tensor]` | |
| 13 | Tuple/list of tensors → flat list | |
| 14 | Non-tensor element → `AssertionError` | `("a",)` |

## `run_example` (correctness path; benchmark off on CPU)

| # | Behavior | Test |
| --- | --- | --- |
| 15 | Matching kernel vs baseline passes | identity fns |
| 16 | Diverging kernel raises with labeled message | kernel that adds 1 |
| 17 | Output-count mismatch raises | kernel returns 2 tensors, baseline 1 |
| 18 | First baseline is the reference; extra baselines also compared to it | dict of baselines |
| 19 | Multiple kernels each checked against ref | dict of kernels |
| 20 | `benchmark` defaults to `False` on CPU (returns `{}`) | no CUDA → empty dict |
| 21 | In-place kernel can't corrupt inputs for other fns (clone-per-call) | kernel that writes its input; baseline still sees clean input |
| 22 | `bwd=True` compares gradients; raises if no arg `requires_grad` | grad parity + the no-grad-arg assertion |
| 23 | `bwd=True` grad presence/value mismatch raises | one fn produces None grad; one fn has same forward and different backward |

## `run_distributed` / `_dist_worker_entry` / `_find_free_port`

(Run on **gloo/CPU**, `world_size` small. These are the highest-value once the
NameError is fixed.)

| # | Behavior | Test |
| --- | --- | --- |
| 24 | Smoke: trivial worker runs to completion on `world_size=1` and `2` | module-level no-op worker |
| 25 | An `assert` failure inside the worker propagates as an exception to the parent | worker that asserts False → `pytest.raises` |
| 26 | `args`/`kwargs` forwarded to the worker | worker echoes/asserts on them |
| 27 | Each rank gets distinct `RANK`/`LOCAL_RANK`, shared `WORLD_SIZE`/`MASTER_*` | worker asserts env |
| 28 | `_find_free_port` returns a usable, bindable port | bind to it succeeds |
| 29 | Process group torn down even when the worker raises (`finally: destroy_process_group`) | a second `run_distributed` after a failing one still works (no leaked group) |
| 30 | Final `dist.barrier()` keeps ranks in lockstep before teardown | worker with uneven per-rank work still joins cleanly |

## `_print_table` / `do_bench`

| # | Behavior | Test |
| --- | --- | --- |
| 31 | `do_bench` returns a float (CUDA) | `@requires_cuda` smoke |
| 32 | `_print_table` formats without error and marks the ref baseline | capture stdout, assert "(ref)" line present | 

## Notes

- Most of this module is pure Python and runs on CPU; direct tests now exist for
  the harness surfaces instead of relying only on indirect kernel tests.
- `clone_args`/`assert_close`/`run_example` are exercised indirectly by every
  kernel test, but a broken `max_mismatch_pct` boundary or a silently-passing
  comparison would hide real kernel regressions — test them directly.
