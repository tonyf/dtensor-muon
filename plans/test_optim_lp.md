# Test Plan — `optim/optim_lp.py` (`MuonLP`, `Muon8bit`, `Muon4bit`, `MuonFp8`)

**Test file:** `tests/optim/test_optim_lp.py` (covers base `MuonLP`, buffer
construction/quantization thresholds, DTensor local-threshold behavior, and
strict-xfailed quantized stepping characterization).

Requires the `lp` extra (`uv sync --extra lp` → `torchao`). Module raises
`ImportError` at import if torchao is absent, and the public export in
`optim/__init__.py` is guarded by a torchao try/except. **Gate the whole test
module on torchao availability** (`pytest.importorskip("torchao")`).

## What the module does

`MuonLP(Muon)` overrides only `_init_muon_group` to store the **momentum buffer**
in a quantized torchao subclass; everything else (update math via base `muon()`)
is inherited. The concrete classes override only `_subclass_zeros`:

- `Muon8bit` → `OptimState8bit`
- `Muon4bit` → `OptimState4bit`
- `MuonFp8` → `OptimStateFp8` (note: signed arg dropped)

`_new_buffer(p, signed)`:
- `to_local(p)` first (unwrap DTensor).
- Quantize **only when** `local_p.numel() >= 4096` **and** `numel() %
  block_size == 0`; otherwise plain `torch.zeros_like`.
- If `p` is a DTensor, re-wrap the (quantized) local tensor via
  `DTensor.from_local`.
- `.to(p.device)`.

## Behavioral contract to cover

| # | Behavior | Test |
| --- | --- | --- |
| 1 | Import raises a clear error without torchao | hard to test in-process; rely on `importorskip` + a note. Optionally subprocess with torchao hidden. |
| 2 | `MuonLP` (base, default `_subclass_zeros` = plain zeros) ≈ `Muon` | identical config/seed; params match | ✅ exists |
| 3 | Buffer **is quantized** when `numel >= 4096` and divisible by `block_size` | param shape `(64,64)` with `block_size=2048`; assert `OptimState` subclass | ✅ exists |
| 4 | Buffer is **plain** `zeros_like` when `numel < 4096` | `(32,32)`=1024 → plain tensor, not quantized | ✅ exists |
| 5 | Buffer is **plain** when `numel` not divisible by `block_size` | `numel=4096`, `block_size=3000` → plain | ✅ exists |
| 6 | Quantization threshold is on **local** numel (DTensor) | global numel ≥4096 but local shard <4096 stays plain | ✅ exists |
| 7 | Quantized-buffer training **converges** / tracks full-precision within a tolerance | blocked until quantized state subclasses support the in-place momentum update path | ⚠️ blocked |
| 8 | `Muon8bit`, `Muon4bit`, `MuonFp8` each step without error and reduce loss | strict-xfailed one-step smoke documents current implementation failure | ⚠️ strict xfail |
| 9 | Lower precision = larger deviation from fp32 baseline (4bit worse than 8bit) | blocked by #7/#8 | ⚠️ blocked |
| 10 | `MuonFp8._subclass_zeros` ignores `signed` (different signature) | covered by quantized buffer-construction parametrization | ✅ exists |
| 11 | `block_size` constructor arg is plumbed into `_new_buffer` | non-default `block_size`, assert quantization decision changes accordingly | ✅ exists |
| 12 | `bf16_stochastic_round` flag is stored | constructor-state-only behavior pinned | ✅ exists |
| 13 | Momentum buffer **round-trips** through quantization across multiple steps | blocked by #7/#8 | ⚠️ blocked |
| 14 | `compile=True` works with quantized buffers | blocked by #7/#8 | ⚠️ blocked |

## DTensor + quantization (gloo/CPU where possible)

| # | Behavior | Test |
| --- | --- | --- |
| 15 | DTensor param → buffer re-wrapped as DTensor with **matching** placements/mesh/shape/stride | inspect `state["momentum_buffer"]` is a DTensor with `p.placements` | ✅ exists |
| 16 | Quantized + DTensor: local quantized tensor wrapped via `from_local(run_check=False)` | covered for the local-threshold/plain case; positive quantized stepping blocked by #7/#8 | ⚠️ partial |
| 17 | Sharded LP training matches replicated single-proc (within quant tol) | base `MuonLP` DTensor path covered; quantized sharded training blocked by #7/#8 | ⚠️ partial |

## Checkpointing quantized state

| # | Behavior | Test |
| --- | --- | --- |
| 18 | `state_dict`/`load_state_dict` round-trips the quantized buffer | blocked by #7/#8 | ⚠️ blocked |
| 19 | `__setstate__` defaults still apply for LP subclasses | inherited behavior covered in base `Muon` tests | ✅ covered upstream |

## Notes / gating

- The base `muon()` does `momentum_buffer.to(torch.float32)` then
  `momentum_buffer.copy_(momentum_buffer_fp32)` — for a quantized subclass this
  exercises the torchao dequant/quant path on every step. Tolerances for #7/#13
  must be **looser** than fp32 Muon tests; tune per dtype (4bit loosest).
- Quantization tests #3–#6 are pure construction (`_init_muon_group` /
  `_new_buffer`) and may run on **CPU** without orthogonalization — split these
  from the `@requires_cuda` convergence tests so coverage exists on CPU CI.
- Confirm whether `bf16_stochastic_round` is actually consumed; if not, mark it
  as a known dead option rather than testing behavior that doesn't exist.
