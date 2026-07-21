# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`dtensor-muon` is a distributed-ready implementation of the Muon optimizer built on PyTorch
`DTensor`. It runs the orthogonalization (matrix-sign / "zero-power") step across sharded
parameters (FSDP / tensor-parallel meshes) and falls back to fused Adam/AdamW for parameters
Muon doesn't apply to — all from a single optimizer instance. See `README.md` for the user-facing
API, the full option table, and the math behind Muon.

## Commands

The project uses [uv](https://docs.astral.sh/uv/); always prefix commands with `uv run`.

```bash
uv sync                          # core dev environment
uv sync --extra lp               # also install torchao (required for MuonLP)

uv run pytest                    # full test suite
uv run pytest tests/optim/test_optim.py            # a single file
uv run pytest tests/optim/test_optim.py::test_name # a single test
uv run pytest -k "foreach"       # tests matching a keyword

uv run ruff check                # lint (line length 96)
uv run ty                        # type check
```

Most kernel/orthogonalization tests are gated on CUDA (`@requires_cuda` →
`pytest.mark.skipif(not torch.cuda.is_available(), ...)`) and silently skip on a CPU-only host;
don't assume a green run means those paths were exercised. The DTensor tests are the exception —
they spawn a small **gloo** world on CPU and run anywhere.

## Architecture

### Single optimizer, pluggable algorithms dispatched by param group

`Muon` (`optim/optim.py`) is one `torch.optim.Optimizer` that drives the whole Muon family plus
Adam. Each param-group dict carries an `"algorithm"` key: `"adam"`/`"adamw"` take the special-cased
Adam path (`_step_adam_group` → `torch.optim.adam.adam`, the upstream fused kernel); every other
name is resolved through the **algorithm registry** in `optim/algorithms/` (`"muon"` is the
default; `"normuon"` is built in; third parties add variants with `@register_muon_algorithm`).
The constructor normalizes every group through `_build_muon_group` / `_build_adam_group`, which
stamp a `use_muon` flag and resolve each option (lr, wd, momentum, …) with a per-group override
falling back to the constructor default; variant-specific options (declared in
`MuonAlgorithm.options`, e.g. NorMuon's `muon_beta2`) live in the group dict only. `step()`
partitions by `use_muon` and `_step_muon_group` looks up the group's algorithm, gathers state
into `dict[str, list[Tensor]]` per the algorithm's `state_spec`, and calls a lazily-compiled
per-algorithm kernel (`_muon_impls`, `torch.compile(partial(self.muon, algo), dynamic=True)`).
`__setstate__` re-applies defaults (including `algorithm: "muon"` and each algorithm's option
defaults) so checkpoints from older versions load; the algorithm *name string* is what persists
in checkpoints.

Muon only handles 2D+ real tensors; `MuonAlgorithm.validate_param` raises on 1D or complex
params at group build. The intended usage is to route weight matrices to Muon and
norms/biases/embeddings/LM-head to the Adam group. Muon groups may also set `split_sizes`
(2D only, validated in `_validate_split_sizes`) to orthogonalize row blocks of a fused weight
(e.g. QKV) independently, with per-block LR corrections (`split_lr_scales`).

`flatten` (group option, default **False**, matching microsoft/dion) controls how 3D+ params
are shaped before the update, in `_init_muon_group`: False treats them as batches of 2D
matrices — leading dims fold into one batch dim (`grad.flatten(end_dim=-3)`), each trailing
matrix orthogonalized independently, and `lr_ratio` comes from the matrix dims; True (opt-in,
for conv-style weights) collapses to a single `(dim0, -1)` matrix. The False default is what
lets 3D FSDP-stacked DTensors reach the `foreach_zeropower_3d_fsdp` fast path without extra
configuration. State buffers are shaped like the *transformed* grad, so changing `flatten`
for a param group is a state-breaking change.

### Two orthogonal axes: algorithms own the math, optimizer classes own execution

`optim/algorithms/` holds `MuonAlgorithm` strategy objects: `name`, `options` (variant
hyperparameter defaults), `state_spec` (declarative per-param buffers via `BufferSpec`,
`like="grad"|"grad_rows"`), a per-tensor fp32 `update` reference, and an optional batched
`foreach_update` (defaults to looping `update`). Algorithms never touch DTensor logic directly —
they call `orthogonalize_single` / `orthogonalize_batch` from `algorithms/base.py`, which wrap
`zeropower`/`foreach_zeropower`, pick the FSDP fast path, and implement the `split_sizes`
row-block handling. `algorithms/base.py` is also the monkeypatch seam tests use to stub
`zeropower`/`foreach_zeropower`.

Execution strategy is per-group data too, not a class: `Muon.muon()` has two drivers selected
by the group's `foreach` flag — the per-parameter reference loop (`algorithm.update` per
tensor), or the batched shell (group by (device, dtype, shape), chunk by `batch_size`,
move CPU-offloaded batches to CUDA and back, delegate to `algorithm.foreach_update`).
`foreach=None` (default) resolves at group build to True iff every param is on CUDA;
explicit `True` opts CPU(-offloaded) groups into the CUDA round trip and hence requires CUDA.
The same group key drives the base Optimizer's foreach `zero_grad`, as in torch.optim.
`optim.py` registers a DTensor op-strategy for `_foreach_sign_` at import time.

- **`MuonLP`** (`optim/optim_lp.py`, needs the `lp` extra) overrides `_new_state_buffer` so any
  algorithm's `like="grad"` buffers go through `_new_buffer` into a quantized torchao subclass
  (`grad_rows` buffers are tiny and stay fp32). The concrete classes `Muon8bit`, `Muon4bit`,
  `MuonFp8` only override `_subclass_zeros`. Buffers are quantized only when `numel() >= 4096`
  and divisible by `block_size`; DTensor params get the quantized local tensor re-wrapped via
  `DTensor.from_local`. Defaults `foreach=False`: the quantized subclasses don't implement the
  `torch._foreach_*` ops.

### Orthogonalization is a dispatch layer over iteration schemes

`orthogonalize/orthogonalize.py` is the seam between the optimizers and the matrix-sign kernels.
`zeropower` (single tensor) and `foreach_zeropower` (batched) select an implementation via
`_get_orthogonalization_fn(strategy, use_triton)`:

|                  | `use_triton=True` | `use_triton=False` |
| ---------------- | ----------------- | ------------------ |
| `newton_schulz`  | `ns_loop_triton`  | `ns_loop`          |
| `polar_express`  | `pe_loop_triton`  | `pe_loop`          |

All iteration loops live in `orthogonalize/newton_schulz.py` and `orthogonalize/polar_express.py`,
operate on the last two dims of a `(..., N, M)` tensor in bfloat16, and are wrapped in
`@torch.compile`. The accumulation is written **out-of-place** (`X = a * X + B @ X`, not
`X.mul(a).add_(...)`): the in-place form miscompiled under Inductor on 2D inputs. Tests still use
the eager original (via `fn.__wrapped__`) as a stable, compile-independent reference, and
`test_{ns,pe}_loop_2d_compiles_correctly` guard against the regression returning.

DTensor handling also lives here and is the core distributed logic — there are three paths:
- **Single tensor** (`zeropower`): `full_tensor()` to replicate, orthogonalize, redistribute back
  to the original placements.
- **General batched** (`foreach_zeropower`): stack, `redistribute` to shard dim 0 across the mesh,
  orthogonalize local shards, redistribute and unbind back to per-param DTensors.
- **FSDP fast path** (`foreach_zeropower_3d_fsdp`, guarded by `is_fsdp_3d_sharded`): for 3D
  DTensors sharded *only* on dim 0, it skips the redistribute and works directly on local shards —
  the cheap path that makes sharded orthogonalization efficient. `orthogonalize_batch`
  (`optim/algorithms/base.py`) checks `is_fsdp_3d_sharded` and prefers this.

### Triton kernel

`kernels/gram.py` is a fused, autotuned Gram-matrix kernel (`y = x @ x.T` per batch, computing
only the upper triangle and mirroring). `ns_loop_triton` uses it for the `X X^T` and `A A^T`
products inside Newton-Schulz. `gram_` requires a contiguous output and asserts CUDA.

### `utils/`

`tests/testkit.py` holds shared test helpers such as `run_example`, which checks a kernel against
a baseline (cloning args per call so in-place ops can't corrupt shared inputs) and optionally
benchmarks. `test_support/distributed.py` holds `run_distributed`, which spawns `world_size`
subprocesses forming a torch.distributed world and runs a module-level
`worker(rank, world_size, ...)` — assertions inside the worker are the pass/fail signal. Workers
must be module-level (picklable for spawn); build the `DeviceMesh` and DTensors inside the worker.

## Conventions

- `tests/conftest.py` raises Dynamo's `cache_size_limit`/`accumulated_cache_size_limit` to 256
  because parametrized sweeps over many (shape, steps) combos otherwise hit
  `FailOnRecompileLimitHit` (the orthogonalization loops are `fullgraph=True`). If you add wide
  parametrizations over compiled kernels, keep this in mind.
- Muon math runs in fp32 for the buffer/param updates but orthogonalizes in bfloat16.
- Per-param state: `step`, `momentum_buffer`, and a cached `lr_ratio` (`sqrt(max(1, N/M))`) for
  shape-based LR scaling.
