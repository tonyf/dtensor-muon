# Test Plan — `utils/foreach.py`

**Test file:** `tests/utils/test_foreach.py` (covers grouping, compile sentinel,
identity/no-copy behavior, device moves, `None`, round-trip, and empty input).

Pure-Python helpers, **no CUDA, no distributed** — these tests run everywhere and
are cheap. High value because both functions feed `MuonForeach`'s correctness.

## What the module does

- `group_tensors_by_shape(tensorlist)` → `dict[shape_tuple, (tensors, indices)]`.
  Groups by `tuple(t.shape)`. **When compiling** (`torch.compiler.is_compiling()`)
  returns a single sentinel bucket `{(0,0): (tensorlist, range(len))}`.
- `move_tensors_to_device(tensors, device_in, device_out)` — moves each tensor to
  `device_out` (non-blocking), passing `None`s through. **No-op when
  `device_in.type == device_out.type`** (returns the input list unchanged).

## `group_tensors_by_shape`

| # | Behavior | Test |
| --- | --- | --- |
| 1 | Distinct shapes → one bucket each | `[(2,3),(4,5),(2,3)]` → `(2,3)` bucket has 2 tensors, indices `[0,2]` |
| 2 | Indices map back to original positions | verify `indices` reconstruct order |
| 3 | Identical shapes all coalesce | N same-shape tensors → single bucket, all N tensors |
| 4 | Empty list → empty dict | `[]` → `{}` |
| 5 | Scalar tensors use `()` key | `torch.tensor(1.0)` → key `()` |
| 6 | Insertion order of first-seen shape preserved (dict order) | first occurrence defines bucket order |
| 7 | Tensors compared by **shape only**, not dtype/device | same shape, different dtype still grouped together |
| 8 | **Compiling sentinel path** | under `torch.compiler.is_compiling()` (monkeypatch to return True) → returns `{(0,0): (list, [0..n-1])}` regardless of shapes |
| 9 | Returned tensor lists are the **same objects** (no clone) | identity check |

## `move_tensors_to_device`

| # | Behavior | Test |
| --- | --- | --- |
| 10 | Same device type → returns input unchanged (identity, no copy) | cpu→cpu returns same list object |
| 11 | Different type → each tensor `.to(device_out)` | `@requires_cuda`: cpu→cuda moves; result on cuda |
| 12 | `None` entries pass through as `None` | mixed `[tensor, None, tensor]` |
| 13 | Round-trip cpu→cuda→cpu preserves values | `@requires_cuda` |
| 14 | Same **type** but different **index** still no-ops (cuda:0 → cuda:1) | per the `.type` check, this is a no-op — pin this (potentially surprising) behavior | 
| 15 | Empty list → empty list | |

## Notes / sharp edges

- #14 is a latent footgun: `move_tensors_to_device` only compares `device.type`,
  so a `cuda:0 → cuda:1` request is silently a **no-op**. `MuonForeach` only ever
  moves between cpu and cuda so it's fine today, but pin the behavior so a future
  multi-GPU caller isn't surprised. Mark `@requires_cuda` + `device_count()>=2`.
- The compiling sentinel (#8) is the contract that lets `MuonForeach.muon`'s
  shape-grouping degrade to "group all" under `torch.compile` (inductor handles
  the real grouping). Worth an explicit test because it's easy to break silently.
- No tolerance concerns — these are exact/structural assertions.
