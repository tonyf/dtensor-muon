"""Single-device optimizer benchmark: naive vs Muon (per-param and foreach drivers) vs MuonLP.

Workload is a synthetic stack of transformer weight matrices (all Muon-eligible 2D+
params). We time a full ``optimizer.step()`` — the whole update, including momentum,
orthogonalization, weight decay, and the param write — which is what actually costs time
in training.

All variants share hyperparameters (``ns_steps=5`` and ``use_cautious_wd=False`` to match
the naive baseline). Muon kernels compile automatically. The foreach driver and ``MuonLP``
move work to CUDA internally, so they only appear on a CUDA host. Before timing, the
Newton-Schulz variants are sanity-checked against the naive reference from identical
seeded params/grads.
"""

import torch

from benchmark.harness import assert_close, bench, cuda_available, markdown_table
from benchmark.naive_muon import NaiveMuon

# (name, count, (rows, cols)); d_model=2048, 12 layers. Every numel is divisible by 2048
# so MuonLP actually quantizes the momentum buffer (its >=4096 & %block_size threshold).
_FULL = [
    ("attn_qkv", 12, (2048, 6144)),
    ("attn_out", 12, (2048, 2048)),
    ("mlp_in", 12, (2048, 8192)),
    ("mlp_out", 12, (8192, 2048)),
]
_QUICK = [
    ("attn_qkv", 4, (512, 1536)),
    ("attn_out", 4, (512, 512)),
    ("mlp_in", 4, (512, 2048)),
    ("mlp_out", 4, (2048, 512)),
]

# Shared hyperparameters (explicit constants so the optimizer constructors get precise
# types rather than the int|float union a dict literal would unpack to).
_LR: float = 1e-3
_WD: float = 0.1
_MOMENTUM: float = 0.95
_NESTEROV: bool = True
_NS_STEPS: int = 5


def _make_grads(shapes, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return [torch.randn(*s, device=device, generator=g) for s in shapes]


def _make_params(shapes, device, seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    return [torch.randn(*s, device=device, generator=g) for s in shapes]


def _shapes(profile):
    return [shape for _, count, shape in profile for _ in range(count)]


def _step_fn(opt, params, grads):
    """A single timed step: refill grads (the muon path consumes them) then step."""

    def _run():
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()

    return _run


def _build_optimizer(kind, strategy, params):
    from muonium.optim import Muon

    if kind == "naive":
        return NaiveMuon(
            params, lr=_LR, wd=_WD, momentum=_MOMENTUM, nesterov=_NESTEROV, ns_steps=_NS_STEPS
        )
    if kind not in ("Muon", "Muon-foreach"):
        raise ValueError(kind)
    return Muon(
        params,
        foreach=kind == "Muon-foreach",
        lr=_LR,
        wd=_WD,
        momentum=_MOMENTUM,
        nesterov=_NESTEROV,
        ns_steps=_NS_STEPS,
        use_cautious_wd=False,
        orthogonalization_strategy=strategy,
    )


def _variants(device):
    """(label, kind, strategy) triples available on this device."""
    v = [
        ("naive (ns)", "naive", "newton_schulz"),
        ("Muon (ns)", "Muon", "newton_schulz"),
        ("Muon (pe)", "Muon", "polar_express"),
    ]
    if device == "cuda":
        v += [
            ("Muon-foreach (ns)", "Muon-foreach", "newton_schulz"),
            ("Muon-foreach (pe)", "Muon-foreach", "polar_express"),
        ]
    # MuonLP's quantized subclasses (Muon8bit/Muon4bit/MuonFp8) are intentionally absent:
    # their torchao OptimStateNbit buffers don't implement the in-place mul_/add_ the
    # update path runs on the optimizer state, so a step() raises (see
    # tests/optim/test_optim_lp.py). MuonLP targets optimizer-state *memory*, not step
    # throughput, so it isn't a meaningful entry in a step-time table.
    return v


def _sanity_check(shapes, device, variants):
    """One step of each Newton-Schulz variant must track the naive reference."""
    grads = _make_grads(shapes, device)
    base_params = _make_params(shapes, device)

    ref = [p.clone() for p in base_params]
    naive = _build_optimizer("naive", "newton_schulz", ref)
    for p, g in zip(ref, grads):
        p.grad = g.clone()
    naive.step()

    notes = []
    for label, kind, strategy in variants:
        if kind == "naive" or strategy != "newton_schulz":
            continue
        params = [p.clone() for p in base_params]
        opt = _build_optimizer(kind, strategy, params)
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()
        for i, (got, want) in enumerate(zip(params, ref)):
            assert_close(got, want, msg=f"{label} param {i} vs naive")
        notes.append(label)
    return notes


def run(quick: bool = False) -> str:
    device = "cuda" if cuda_available() else "cpu"
    profile = _QUICK if quick else _FULL
    shapes = _shapes(profile)
    variants = _variants(device)

    checked = _sanity_check(shapes, device, variants)

    times: dict[str, float] = {}
    for label, kind, strategy in variants:
        params = _make_params(shapes, device)
        grads = _make_grads(shapes, device)
        opt = _build_optimizer(kind, strategy, params)
        times[label] = bench(_step_fn(opt, params, grads))
        del opt, params, grads
        if device == "cuda":
            torch.cuda.empty_cache()

    ref = "naive (ns)"
    base = times[ref]
    rows = [
        [
            label,
            f"{times[label]:.4f}",
            "1.00x (ref)" if label == ref else f"{base / times[label]:.2f}x",
        ]
        for label, _, _ in variants
    ]
    n = len(shapes)
    desc = (
        f"Device: **{device}**. Workload: {n} weight matrices "
        f"(shapes {sorted({s for _, _, s in profile})}). "
        f"Timing a full `optimizer.step()`; speedup vs the naive reference. "
        f"Newton-Schulz variants verified against naive: {', '.join(checked) or 'none'}."
    )
    footnote = (
        "\n\n_`MuonLP` (4/8-bit/fp8 states) is omitted: it optimizes optimizer-state "
        "memory, not step time, and its quantized buffers don't support the in-place "
        "update path (see `tests/optim/test_optim_lp.py`)._"
    )
    return (
        desc + "\n\n" + markdown_table(["optimizer", "step (ms)", "speedup"], rows) + footnote
    )


if __name__ == "__main__":
    print(run())
