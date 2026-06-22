# dtensor-muon

A distributed-ready implementation of the **Muon** optimizer built on PyTorch
[`DTensor`](https://docs.pytorch.org/docs/stable/distributed.tensor.html). It runs the
orthogonalization step efficiently across sharded parameters (FSDP / tensor-parallel meshes)
and falls back to Adam/AdamW for the parameters Muon doesn't apply to — all from a single
optimizer instance.

## What is Muon?

Muon updates 2D+ parameters by taking the momentum-smoothed gradient and replacing it with
its nearest semi-orthogonal matrix (the "zero-power" or matrix-sign of the gradient) before
applying it. The orthogonalization is computed iteratively so it stays cheap on the GPU. This
implementation provides two iteration schemes:

- **`newton_schulz`** — the classic Newton-Schulz iteration, with a fused Triton Gram-matrix
  kernel.
- **`polar_express`** — the Polar Express scheme ([arXiv:2505.16932](https://arxiv.org/pdf/2505.16932)),
  which uses precomputed coefficients for faster convergence.

## Features

- **DTensor / distributed first** — orthogonalization is run across the device mesh, with a
  dedicated fast path for parameters sharded only along dim 0 (FSDP).
- **Unified Muon + Adam** — one optimizer handles both. Parameters Muon can't update (1D
  tensors, embeddings, the LM head, etc.) are routed to a fused Adam/AdamW path via per-group
  configuration.
- **Three variants:**
  - `Muon` — the reference per-parameter implementation.
  - `MuonForeach` — batched `foreach` operations for higher throughput.
  - `MuonLP` — quantized (4-bit / 8-bit / fp8) optimizer states via
    [`torchao`](https://github.com/pytorch/ao) for reduced memory.
- **Cautious weight decay** — weight decay applied only where update and parameter share a
  sign.
- **Nesterov momentum**, automatic shape-based learning-rate scaling, optional
  `torch.compile`, and Triton kernels.

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.12.1
- Triton (for the fused Newton-Schulz kernel)
- `torchao` (optional, only for `MuonLP`)

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Install it directly from the repository:

```bash
uv pip install git+https://github.com/tonyf/dtensor-muon.git

# include torchao for the low-precision optimizer (MuonLP)
uv pip install "dtensor-muon[lp] @ git+https://github.com/tonyf/dtensor-muon.git"
```

### Development

Clone the repository and sync the environment:

```bash
git clone https://github.com/tonyf/dtensor-muon.git
cd dtensor-muon
uv sync                  # core install
uv sync --extra lp       # include torchao for the low-precision optimizer
```

## Usage

Muon is applied to 2D+ weight matrices, while norms, biases, and embeddings are typically
left to Adam. Configure this with param groups using the `"algorithm"` key:

```python
import torch
from dtensor_muon import Muon

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

# standard training loop
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Choosing a variant

```python
from dtensor_muon import Muon, MuonForeach, MuonLP   # MuonLP requires the `lp` extra
```

`MuonForeach` and `MuonLP` are drop-in replacements with the same constructor. Use
`MuonForeach` for throughput and `MuonLP` to shrink optimizer-state memory.

### Selecting an orthogonalization strategy

```python
optimizer = Muon(params, orthogonalization_strategy="polar_express")  # or "newton_schulz"
```

### Key options

| Option | Default | Description |
| --- | --- | --- |
| `lr` | `1e-3` | Learning rate. |
| `wd` | `0.1` | Weight decay. |
| `use_cautious_wd` | `True` | Apply weight decay only where update and param share a sign. |
| `momentum` | `0.95` | Muon momentum. |
| `nesterov` | `True` | Use Nesterov momentum. |
| `ns_steps` | `5` | Orthogonalization iteration steps. |
| `orthogonalization_strategy` | `"polar_express"` | `"newton_schulz"` or `"polar_express"`. |
| `adam_betas` | `(0.9, 0.95)` | Betas for the Adam path. |
| `is_adamw` | `True` | Decoupled (AdamW) vs. coupled weight decay for the Adam path. |
| `fused_adam` | `True` | Use the fused Adam kernel. |
| `compile` | `False` | `torch.compile` the per-parameter step. |

Most options can also be overridden per param group.

## Project layout

```
src/dtensor_muon/
├── optim/            # Muon, MuonForeach, MuonLP optimizers
├── orthogonalize/    # zeropower dispatch, Newton-Schulz & Polar Express, DTensor handling
├── kernels/          # Triton Gram-matrix kernel
└── utils/            # DTensor and foreach helpers
```

### Testing and linting

```bash
uv run pytest        # run the test suite
uv run ruff check    # lint
uv run ty            # type check
```
