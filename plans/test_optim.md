# Test Plan — `optim/optim.py` (`Muon` base optimizer)

**Test file:** `tests/optim/test_optim.py` (covers constructor validation,
Muon update math, Adam parity, mixed routing, compile paths, and checkpoint
compatibility).

## What the module does

`Muon` is one `torch.optim.Optimizer` driving **two algorithms** dispatched per
param group via an `"algorithm"` key (`"muon"` default | `"adam"` | `"adamw"`).

- Constructor validates lr/eps/betas, normalizes each group through
  `_build_muon_group` / `_build_adam_group` (stamps `use_muon`, resolves
  per-group overrides against constructor defaults).
- `muon(...)` — per-parameter reference update: momentum buffer, Nesterov mix,
  `zeropower`, (cautious) weight decay, shape-scaled lr — all in fp32.
- `_init_muon_group` / `_init_adam_group` — lazy state init + grad collection.
- `_step_muon_group` → `self._muon_impl`; `_step_adam_group` →
  `torch.optim.adam.adam` (fused upstream kernel).
- `step()` partitions groups by `use_muon` and routes.
- `__setstate__` re-applies defaults (checkpoint back-compat).

## Construction / validation (mostly CPU, no CUDA needed)

| # | Behavior | Test |
| --- | --- | --- |
| 1 | Negative lr → `ValueError` | `Muon([p], lr=-1)` |
| 2 | Negative `adam_eps` → `ValueError` | |
| 3 | `adam_betas` out of `[0,1)` (each index) → `ValueError` | beta0=1.0, beta1=-0.1 |
| 4 | Passing a single dict (not list of dicts) → `TypeError` | `Muon({"params": [p]})` |
| 5 | Empty params iterable → `ValueError` | `Muon([])` |
| 6 | Group dict missing `"params"` key → `ValueError` | |
| 7 | Empty group (`params=[]`) is **skipped**, not added | group with no params dropped |
| 8 | Unknown `"algorithm"` → `ValueError` | `algorithm="rmsprop"` |
| 9 | Muon group with a **1D param** → `ValueError` | `Parameter(randn(4))` |
| 10 | Muon group with a **complex param** → `NotImplementedError` | `randn(2,2,dtype=cfloat)` |
| 11 | Bare param list → wrapped as one muon group | `Muon([p2d])` builds a muon group |
| 12 | Per-group override beats constructor default | group `{"lr":0.5}` vs default lr; assert resolved value on `param_groups` |
| 13 | `lr` stored as a **tensor** for both muon and adam groups | `torch.is_tensor(group["lr"])` |
| 14 | Adam group `wd` wrapped as tensor; muon group `wd` left as float | matches `_build_*_group` |
| 15 | 3D+ param allowed in a muon group | `randn(4,8,8)` accepted (ndim≥2) |

## `muon()` update math — per-parameter reference (the heart)

These need the orthogonalization to run; on CPU `zeropower(use_triton=...)`
hits the Triton kernel only with `use_triton=True`. **`muon()` calls
`zeropower(...)` with its default `use_triton=True`** → CUDA-only. So end-to-end
`step()` correctness tests for muon groups are effectively `@requires_cuda`
unless a CPU path is plumbed. Note this gating explicitly.

| # | Behavior | Test |
| --- | --- | --- |
| 16 | One `step()` decreases a quadratic loss / moves params in the update direction | small linear regression, loss after N steps < before |
| 17 | Momentum buffer accumulates: `buf = momentum*buf + grad` | inspect `state["momentum_buffer"]` after a step with known grad |
| 18 | Nesterov vs non-Nesterov differ and match hand-computed `grad += momentum*buf` | two optimizers, same input |
| 19 | `lr_ratio = sqrt(max(1, N/M))` cached correctly | check `state["lr_ratio"]` for tall/wide/square params |
| 20 | Shape-scaled lr actually applied | update magnitude scales with `lr_ratio` |
| 21 | **Cautious WD**: decay applied only where `u*p > 0` | construct `u`,`p` with mixed signs; compare to non-cautious |
| 22 | Non-cautious WD: `u += wd*p` everywhere | `use_cautious_wd=False` |
| 23 | `wd=0` → no weight-decay term | params unaffected by wd path |
| 24 | **`maximize=True`** negates the gradient | sign of update flips vs `maximize=False` |
| 25 | **`maximize` mutates grad in place** (`grad.neg_()`) — verify it does not corrupt a caller-held `.grad` or double-negate on a second `step()` without fresh grads | call `step()` twice; check grad state |
| 26 | `flatten=True` reshapes ndim>2 grads to 2D for orthogonalization, writes back via `view_as` | 4D param updates correctly |
| 27 | `flatten=False` with ndim==3 allowed; ndim>3 asserts | assertion message check |
| 28 | fp32 internal math regardless of param dtype | bf16 / fp16 param still updates correctly (buffer kept fp32) |
| 29 | State `step` increments each `step()` | `state["step"]` counts |
| 30 | Param with `grad is None` is skipped (not in `params_with_grad`) | mixed grads |

## Adam-group path (delegates to upstream fused `adam`)

| # | Behavior | Test |
| --- | --- | --- |
| 31 | Adam group matches `torch.optim.AdamW` on the same params/grads | parity vs reference optimizer over a few steps |
| 32 | `is_adamw` toggles decoupled vs coupled weight decay | `decoupled_weight_decay` flag plumbed |
| 33 | `amsgrad=True` allocates `max_exp_avg_sq` and is used | state key present; parity vs `AdamW(amsgrad=True)` |
| 34 | `fused_adam=False` uses the foreach (non-fused) path | runs on CPU (fused is CUDA-only); assert it still steps |
| 35 | Complex params **allowed** in adam group (`has_complex`) | `randn(2,2,dtype=cfloat)` in adam group steps without error |
| 36 | Adam state `step` device/dtype honors fused vs non-fused | matches upstream `_get_scalar_dtype` logic |

## Mixed-group routing & `step()`

| # | Behavior | Test |
| --- | --- | --- |
| 37 | A model with a muon group (weights) + adam group (biases/norms) steps both | each param moves; muon params orthogonalized, adam params Adam-updated |
| 38 | `step(closure)` returns the closure loss | closure returning a tensor |
| 39 | `step()` runs under `@torch.no_grad()` and `disable_cache_limit()` | no grad tracking on params after step |
| 40 | Param ordering within a group preserved through state mapping | |

## `compile` flag (already partly covered)

| # | Behavior | Test | Status |
| --- | --- | --- | --- |
| 41 | `compile=False` → `_adam_impl is adam`, `_muon_impl == self.muon`; no `adam`/`muon` keys shadow methods | ✅ exists |
| 42 | `compile=True` → both wrapped via `torch.compile(dynamic=True)`, public methods unshadowed | ✅ exists (monkeypatched compile) |
| 43 | `compile=True` end-to-end actually steps (real compile) | `@requires_cuda` smoke tests for Muon and Adam groups | ✅ exists |

## Checkpoint / `__setstate__`

| # | Behavior | Test |
| --- | --- | --- |
| 44 | `state_dict()` → `load_state_dict()` round-trip preserves momentum buffer, step, lr_ratio | save/load, resume, compare to uninterrupted run |
| 45 | Loading a state where `lr` is a **float** (old checkpoint) re-wraps to tensor | construct legacy-shaped state |
| 46 | Loading a state where `step` is a Python float re-wraps to tensor (muon + adam) | both branches of `__setstate__` |
| 47 | Missing-default keys filled (`ns_steps`, `nesterov`, `flatten`, `use_cautious_wd`, `orthogonalization_strategy`) | strip keys, load, assert defaults restored |
| 48 | `compile` attr restored (`getattr(self,'compile',False)`) and `_init_step_impls` re-run after load | `_muon_impl`/`_adam_impl` valid post-load |
| 49 | **`__setstate__` default `orthogonalization_strategy` is `"newton_schulz"`** but constructor default is `"polar_express"` — pin this intentional/legacy mismatch so a checkpoint round-trip doesn't silently change strategy | assert and document |

## Suspected bugs / sharp edges to pin

- **#49 strategy-default mismatch**: constructor defaults `orthogonalization_strategy="polar_express"`,
  but `__setstate__` `setdefault`s `"newton_schulz"`. For a fresh checkpoint the
  key is present so it's preserved; only legacy checkpoints missing the key flip
  to NS. Test both so the behavior is intentional, not accidental.
- ✅ **FIXED** — #25 in-place grad negation under `maximize` is pinned by
  reused-grad tests for both base and foreach optimizers.
- `muon()` orthogonalizes via `zeropower` with `use_triton=True` default →
  end-to-end muon `step()` is CUDA-gated. If a CPU-runnable muon step is desired,
  that's a feature gap worth a test (call path forcing `use_triton=False`).
