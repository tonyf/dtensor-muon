# muonium

A performant, distributed-ready implementation of the **Muon** optimizer — and a base for
Muon research — built on PyTorch
[`DTensor`](https://docs.pytorch.org/docs/stable/distributed.tensor.html).
Formerly published as `dtensor-muon`.

- **Distributed by construction.** Orthogonalization runs across sharded parameters
  (FSDP / tensor-parallel meshes), with a collective-free fast path when parameters are
  sharded along dim 0 (the FSDP layout). For expert-parallel MoEs, orthogonalization runs
  directly on the local sharded parameters with no collectives at all.
- **A base for Muon variants.** The update rule is pluggable per param group: baseline
  Muon and [NorMuon](https://arxiv.org/abs/2510.05491) are built in, and third parties
  register their own variants with a few dozen lines — the optimizer supplies state
  allocation, batching, CPU offload, DTensor handling, and `torch.compile` around your
  math. Variant names persist in checkpoints.
- **One optimizer for the whole model.** Weight matrices go to a Muon-family algorithm;
  norms, biases, embeddings, and the LM head fall back to PyTorch's fused Adam/AdamW —
  all selected per param group on a single optimizer instance.
- **Performant.** Compiled update kernels, a batched `foreach` driver, a fused Triton
  Gram-matrix kernel inside Newton-Schulz, and optional 4-bit/8-bit/fp8 optimizer state
  (`MuonLP`).

This is an open-sourcing of work done at Dream3D. This was implemented without AI tools originally. When copying the source over, Claude Fable 5 was used to audit the codebase, write tests and documentation, and fix a bug. Claude is tagged on all the commits he contributed to :)

## What is Muon?

Muon updates 2D+ parameters by replacing the momentum-smoothed gradient with its nearest
semi-orthogonal matrix (the matrix-sign, or "zero power", of the gradient) before applying
it. The orthogonalization is computed iteratively on the GPU, with two schemes available:
classic Newton-Schulz iteration (backed by a fused Triton Gram-matrix kernel) and
[Polar Express](https://arxiv.org/pdf/2505.16932), which uses precomputed coefficients for
faster convergence. Learning rates are scaled by each matrix's shape
(`sqrt(max(1, N/M))`) automatically.

Muon only applies to weight matrices. A single `Muon` instance handles the remaining
parameters (norms, biases, embeddings, the LM head) with PyTorch's fused Adam/AdamW,
selected per param group.

## Installation

```bash
uv pip install muonium
```

Requires Python ≥ 3.12 and PyTorch ≥ 2.12.1. The `lp` extra (`muonium[lp]`) pulls in
[`torchao`](https://github.com/pytorch/ao), needed only for `MuonLP`.

## Usage

Route weight matrices to Muon and everything else to Adam with the `"algorithm"` key on
param groups:

```python
import torch
from muonium import Muon

model = ...

muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
adam_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]

optimizer = Muon(
    [
        {"params": muon_params},                      # "muon" is the default
        {"params": adam_params, "algorithm": "adamw"},
    ],
    lr=1e-3,
    wd=0.1,
)

for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The update runs through one of two drivers, selected by the `foreach` flag (constructor
default or per param group): the per-parameter reference loop, or a batched driver that
groups params by (device, dtype, shape) and uses `torch._foreach_*` ops for higher
throughput. The default (`None`) picks the batched driver automatically for groups whose
params all live on CUDA; passing `foreach=True` explicitly also opts CPU-offloaded params
into a per-batch CUDA round trip (`batch_size` caps how many tensors move at once).

`MuonLP` (experimental, requires the `lp` extra) stores momentum in 4-bit, 8-bit, or fp8 to
cut optimizer-state memory; it defaults to `foreach=False` because the quantized state
subclasses don't support the batched ops.

Weight decay is *cautious* by default — applied only where the update and the parameter
agree in sign (`u * p > 0`), following the Cautious Optimizers technique. The original Muon
has no cautious variant; set `use_cautious_wd=False` for plain decoupled weight decay.

The update kernels are compiled internally, so `optimizer.step()` needs no compile flag. It
can also sit inside a surrounding `torch.compile(fullgraph=False)` region like any PyTorch
optimizer; `fullgraph=True` through the step and differentiating through the update are not
supported.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `lr` | `1e-3` | Learning rate. |
| `wd` | `0.1` | Weight decay. |
| `use_cautious_wd` | `True` | Apply decay only where update and param share a sign; `False` for plain decay. |
| `momentum` | `0.95` | Muon momentum. |
| `nesterov` | `True` | Use Nesterov momentum. |
| `ns_steps` | `5` | Orthogonalization iteration steps. |
| `orthogonalization_strategy` | `"polar_express"` | `"newton_schulz"` or `"polar_express"`. |
| `adam_betas` | `(0.9, 0.95)` | Betas for the Adam path. |
| `is_adamw` | `True` | Decoupled (AdamW) vs. coupled weight decay for the Adam path. |
| `fused_adam` | `None` | Select the fused Adam kernel explicitly; `None` uses PyTorch's default dispatch. |
| `foreach_adam` | `None` | Select the foreach Adam kernel explicitly; `None` uses PyTorch's default dispatch. |
| `foreach` | `None` | Batched Muon driver. `None` = auto (on for all-CUDA groups); `True` also opts CPU-offloaded groups into the CUDA round trip. |
| `batch_size` | `None` | Max tensors per foreach batch (`None` = unbounded). |

Most options can also be overridden per param group.

Muon groups additionally accept `flatten` (default `False`): 3D+ tensors are treated as
batches of 2D matrices, each orthogonalized independently (leading dims fold into the
batch). This is what you want for stacked/FSDP-sharded weights — it is also what lets them
take the sharded fast path (see Benchmarks). Set `flatten=True` — e.g. for convolutional
weights — to collapse them to a single `(dim0, -1)` matrix instead.

### Algorithm variants

The `"algorithm"` key selects the update rule per param group. `"muon"` (default) and the
`"adam"`/`"adamw"` fallback are built in, plus:

| Algorithm | Extra group options | Description |
| --- | --- | --- |
| `"normuon"` | `muon_beta2` (`0.95`) | [NorMuon](https://arxiv.org/abs/2510.05491): normalizes the orthogonalized update by a per-neuron second-moment EMA, then rescales to preserve its Frobenius norm. Adds a `variance_neuron` state buffer (one value per row). |

Variant-specific hyperparameters travel in the group dict, next to the `algorithm` key that
selects them:

```python
optimizer = Muon(
    [
        {"params": hidden_weights},                                    # baseline muon
        {"params": qkv_weights, "algorithm": "normuon", "muon_beta2": 0.95},
        {"params": adam_params, "algorithm": "adamw"},
    ]
)
```

Muon-family groups also accept `split_sizes` (2D params only): row blocks of a fused weight
(e.g. a QKV projection, `split_sizes=(q_rows, k_rows, v_rows)`) are orthogonalized — and,
for NorMuon, normalized — independently, matching the update separate per-block parameters
would receive, while the model keeps the single wide GEMM.

#### Registering your own variant

Third-party packages can add algorithms without forking: subclass `MuonAlgorithm`,
implement the per-tensor `update` (and optionally a batched `foreach_update` using
`torch._foreach_*` ops — the default loops the per-tensor reference), and register it. The
optimizer supplies everything around the math: state allocation from `state_spec`
(including `MuonLP` quantization), batching, CPU offload, DTensor orthogonalization via the
`orthogonalize_single` / `orthogonalize_batch` helpers, and `torch.compile`.

```python
from muonium import BufferSpec, MuonAlgorithm, register_muon_algorithm
from muonium.optim.algorithms import orthogonalize_single


@register_muon_algorithm
class MyVariant(MuonAlgorithm):
    name = "my_variant"
    options = {"alpha": 0.5}  # per-group hyperparams and their defaults
    state_spec = {
        "momentum_buffer": BufferSpec(like="grad", signed=True),
        # add extra per-param buffers here ("grad" or "grad_rows" shaped)
    }

    def update(self, param, grad, state, lr_ratio, *, lr, alpha, ns_steps,
               orthogonalization_strategy, split_sizes, **kwargs):
        buf = state["momentum_buffer"]
        ...  # your math; call orthogonalize_single(g, ns_steps=..., strategy=..., split_sizes=...)


optimizer = Muon([{"params": params, "algorithm": "my_variant", "alpha": 0.7}])
```

The `algorithm` name is stored in checkpoints, so `load_state_dict` restores the right
variant (register it before loading).

## Benchmarks

The suite in [`benchmark/`](benchmark/) covers the kernels, single-device optimizers, and
the distributed orthogonalization paths; `uv run python benchmark/run.py` writes results and
methodology to [`benchmark/RESULTS.md`](benchmark/RESULTS.md) (`--quick` for a fast smoke
run).

Distributed orthogonalization, measured on 2× NVIDIA RTX PRO 6000 Blackwell
(PyTorch 2.12.1+cu130, NCCL, params sharded on dim 0):

| path | per-call (ms) | vs single-device |
| --- | --- | --- |
| single-device (replicated, no collectives) | 42.6 | 1.00× (ref) |
| FSDP fast path (`foreach_zeropower_3d_fsdp`) | 20.4 | 2.09× |
| general path (`foreach_zeropower` + redistribute) | 33.2 | 1.28× |

The fast path applies when parameters are sharded only along dim 0 (the FSDP layout): each
rank orthogonalizes its local shard with no collectives. The general path first
redistributes the stacked batch across the mesh, which costs communication but still beats
replicating.

## Development

```bash
git clone https://github.com/tonyf/muonium.git
cd muonium
uv sync --extra lp   # --extra lp is only needed for MuonLP/torchao

uv run pytest        # test suite
uv run ruff check    # lint
uv run ty            # type check
```

```
src/muonium/
├── optim/            # Muon (+ MuonLP) optimizers and the algorithm registry
├── orthogonalize/    # zeropower dispatch, Newton-Schulz & Polar Express, DTensor handling
├── kernels/          # Triton Gram-matrix kernel
└── utils/            # DTensor and foreach helpers
```
