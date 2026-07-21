"""Triton kernels vs. their PyTorch references.

- Gram: ``gram(x)`` vs ``x @ x.mT``.
- Newton-Schulz loop: ``ns_loop_triton`` vs the eager ``ns_loop`` (``__wrapped__``).
- Polar Express loop: ``pe_loop_triton`` vs the eager ``pe_loop`` (``__wrapped__``).

The Triton loops and the Gram kernel both require CUDA, so the whole section is skipped
(with a note) on a CPU-only host. We use 3D batched inputs for the loops both because
that's what the optimizer actually orthogonalizes (stacked ``(B, N, M)`` tensors) and
because the compiled loops are known to miscompile on 2D inputs.
"""

from typing import Any, cast

import torch

from benchmark.harness import assert_close, bench, cuda_available, markdown_table

# (batch, rows, cols); a 2D entry exercises the unbatched kernel path.
_GRAM_SHAPES = [(32, 2048, 1024), (32, 1024, 1024), (2048, 2048), (32, 2048, 4096)]
_GRAM_DTYPES = [torch.bfloat16, torch.float32]

# 3D only (see module docstring).
_LOOP_SHAPES = [(32, 2048, 1024), (32, 1024, 1024), (16, 4096, 1024)]
_STEPS = 5

# bf16 gram accumulates in fp32 inside tl.dot; fp32 inputs use TF32 -> looser budget.
_GRAM_TOL = {torch.bfloat16: (1e-2, 1e-2), torch.float32: (2e-2, 2e-2)}


def _quick(shapes):
    return shapes[:1]


def _bench_gram(shapes, dtypes) -> str:
    from muonium.kernels.gram import gram

    rows = []
    for dtype in dtypes:
        for shape in shapes:
            torch.manual_seed(0)
            x = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
            rtol, atol = _GRAM_TOL[dtype]
            # Sanity check, not a precision test: both kernel and reference accumulate in
            # TF32 under matmul precision "high", and large-K shapes drift a little more.
            assert_close(
                gram(x),
                x @ x.mT,
                rtol=rtol,
                atol=atol,
                max_mismatch_pct=2.0,
                msg=f"gram {shape} {dtype}",
            )
            t_triton = bench(lambda x=x: gram(x))
            t_torch = bench(lambda x=x: x @ x.mT)
            rows.append(
                [
                    str(tuple(shape)),
                    str(dtype).replace("torch.", ""),
                    f"{t_torch:.4f}",
                    f"{t_triton:.4f}",
                    f"{t_torch / t_triton:.2f}x",
                ]
            )
    return markdown_table(["shape", "dtype", "torch (ms)", "triton (ms)", "speedup"], rows)


def _bench_loops(shapes) -> str:
    from muonium.orthogonalize.newton_schulz import ns_loop, ns_loop_triton
    from muonium.orthogonalize.polar_express import pe_loop, pe_loop_triton

    # Three variants per strategy: uncompiled eager torch (the correctness reference,
    # reached via __wrapped__), the @torch.compile'd torch loop, and the @torch.compile'd
    # Triton-fused loop. The "as deployed" fight is compiled-torch vs Triton, so the
    # speedup column is Triton vs compiled-torch. (Compiled loops are valid on the 3D
    # inputs used here; they miscompile only on 2D.)
    variants = [
        ("newton_schulz", ns_loop_triton, ns_loop, getattr(ns_loop, "__wrapped__", ns_loop)),
        ("polar_express", pe_loop_triton, pe_loop, getattr(pe_loop, "__wrapped__", pe_loop)),
    ]
    rows = []
    for name, triton_fn, compiled_fn, eager_fn in variants:
        triton_fn = cast(Any, triton_fn)
        compiled_fn = cast(Any, compiled_fn)
        eager_fn = cast(Any, eager_fn)
        for shape in shapes:
            torch.manual_seed(0)
            x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
            ref = eager_fn(x, _STEPS)
            for label, fn in (("triton", triton_fn), ("compiled-torch", compiled_fn)):
                assert_close(
                    fn(x, _STEPS),
                    ref,
                    rtol=1e-2,
                    atol=2e-2,
                    max_mismatch_pct=1.0,
                    msg=f"{name} {label} loop {shape}",
                )
            t_eager = bench(lambda x=x, f=eager_fn: f(x, _STEPS))
            t_torch = bench(lambda x=x, f=compiled_fn: f(x, _STEPS))
            t_triton = bench(lambda x=x, f=triton_fn: f(x, _STEPS))
            rows.append(
                [
                    name,
                    str(tuple(shape)),
                    f"{t_eager:.4f}",
                    f"{t_torch:.4f}",
                    f"{t_triton:.4f}",
                    f"{t_torch / t_triton:.2f}x",
                ]
            )
    return markdown_table(
        [
            "strategy",
            "shape",
            "eager-torch (ms)",
            "compiled-torch (ms)",
            "triton (ms)",
            "triton vs compiled",
        ],
        rows,
    )


def run(quick: bool = False) -> str:
    if not cuda_available():
        return "_Skipped — the Triton Gram kernel and the compiled loops require CUDA._"

    gram_shapes = _quick(_GRAM_SHAPES) if quick else _GRAM_SHAPES
    gram_dtypes = [torch.bfloat16] if quick else _GRAM_DTYPES
    loop_shapes = _quick(_LOOP_SHAPES) if quick else _LOOP_SHAPES

    gram_md = _bench_gram(gram_shapes, gram_dtypes)
    loop_md = _bench_loops(loop_shapes)
    return (
        "### Gram kernel (`gram` vs `x @ x.mT`)\n\n"
        f"{gram_md}\n\n"
        f"### Orthogonalization loops ({_STEPS} steps, bf16; eager vs compiled torch vs triton)\n\n"
        f"{loop_md}"
    )


if __name__ == "__main__":
    print(run())
