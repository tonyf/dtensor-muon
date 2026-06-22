"""Correctness + benchmark helper for tests and benchmarks.

Modeled on ``helion._testing.run_example`` but standalone: it depends only on
``torch`` and ``triton.testing.do_bench`` (both already project dependencies),
so it keeps working after helion is uninstalled.

Typical use in a pytest test::

    from dtensor_muon.utils.testing import run_example

    def test_gram():
        x = torch.randn(8, 256, 128, device="cuda", dtype=torch.float32)
        run_example(gram, lambda x: x @ x.mT, (x,), kernel_name="gram_triton")

Or to benchmark several implementations against a baseline::

    run_example(
        {"triton": gram, "compiled": torch.compile(ref)},
        {"torch": ref},
        (x,),
    )
"""

from __future__ import annotations

import functools
import sys
from typing import Callable, Sequence

import torch

__all__ = ["run_example", "clone_args", "do_bench", "assert_close"]

# A function under test / baseline: takes the (cloned) args, returns a tensor or
# a tuple of tensors.
ExampleFn = Callable[..., "torch.Tensor | tuple[torch.Tensor, ...]"]


def clone_args(args: Sequence[object]) -> tuple[object, ...]:
    """Clone an args tuple so a function can't mutate a shared buffer.

    Tensors are detached and cloned, preserving ``requires_grad`` (needed for
    backward tests). Non-tensor args are passed through unchanged.
    """
    cloned: list[object] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cloned.append(arg.detach().clone().requires_grad_(arg.requires_grad))
        else:
            cloned.append(arg)
    return tuple(cloned)


def _as_tensors(result: object) -> list[torch.Tensor]:
    """Normalize a tensor / tuple-of-tensors return value to a flat list."""
    if isinstance(result, (tuple, list)):
        out: list[torch.Tensor] = []
        for t in result:
            assert isinstance(t, torch.Tensor), f"expected Tensor, got {type(t)}"
            out.append(t)
        return out
    assert isinstance(result, torch.Tensor), f"expected Tensor, got {type(result)}"
    return [result]


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-1,
    max_mismatch_pct: float | None = None,
    msg: str | None = None,
) -> None:
    """``torch.testing.assert_close`` in float32, optionally tolerating a small
    fraction of mismatched elements (useful for low-precision kernels)."""
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    if max_mismatch_pct is None:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=msg)
        return

    mismatched = ~torch.isclose(actual, expected, rtol=rtol, atol=atol)
    pct = 100.0 * mismatched.sum().item() / max(mismatched.numel(), 1)
    if pct > max_mismatch_pct:
        raise AssertionError(
            (msg or "")
            + f"\n{pct:.4f}% of elements mismatched "
            f"(allowed {max_mismatch_pct:.4f}%) at rtol={rtol}, atol={atol}"
        )


def do_bench(fn: Callable[[], object], *, grad_to_none: Sequence[torch.Tensor] | None = None) -> float:
    """Median wall-clock time of ``fn`` in milliseconds via ``triton.testing.do_bench``."""
    from triton.testing import do_bench as _triton_do_bench

    return float(
        _triton_do_bench(
            fn,
            grad_to_none=list(grad_to_none) if grad_to_none else None,
            return_mode="median",
        )
    )


def run_example(
    kernel_fn: ExampleFn | dict[str, ExampleFn],
    baseline_fn: ExampleFn | dict[str, ExampleFn],
    args: tuple[object, ...],
    *,
    kernel_name: str = "kernel",
    baseline_name: str = "torch",
    rtol: float = 1e-2,
    atol: float = 1e-1,
    max_mismatch_pct: float | None = None,
    bwd: bool = False,
    benchmark: bool | None = None,
) -> dict[str, float]:
    """Check correctness against a baseline, then (optionally) benchmark.

    Args:
        kernel_fn: A single callable, or ``{name: callable}`` for several variants.
        baseline_fn: A single callable, or ``{name: callable}``. Correctness is
            checked against the first baseline; every other function (kernels and
            extra baselines) is compared to it.
        args: Positional args passed to every function. Cloned per-call so a
            function that writes in-place can't corrupt the inputs for the others.
        kernel_name / baseline_name: Labels used when a single callable is passed.
        rtol / atol: Tolerances for the float32 comparison.
        max_mismatch_pct: If set, allow this percentage of mismatched elements
            instead of requiring every element to be close.
        bwd: Also check gradients. Args that ``requires_grad`` get a backward pass
            with a shared random grad-output and their ``.grad`` compared.
        benchmark: Run the timing pass and print a table. Defaults to True when
            CUDA is available, False otherwise (correctness-only on CPU).

    Returns:
        ``{name: time_ms}`` for every function benchmarked (empty if benchmarking
        was skipped).
    """
    kernels = kernel_fn if isinstance(kernel_fn, dict) else {kernel_name: kernel_fn}
    baselines = baseline_fn if isinstance(baseline_fn, dict) else {baseline_name: baseline_fn}
    all_fns = {**kernels, **baselines}

    ref_name, ref_fn = next(iter(baselines.items()))

    # --- forward correctness ---
    expected = _as_tensors(ref_fn(*clone_args(args)))
    for name, fn in all_fns.items():
        if name == ref_name:
            continue
        print(f"Checking {name} forward...", file=sys.stderr)
        result = _as_tensors(fn(*clone_args(args)))
        assert len(result) == len(expected), (
            f"{name} returned {len(result)} tensors, baseline returned {len(expected)}"
        )
        for i, (r, e) in enumerate(zip(result, expected, strict=True)):
            assert_close(
                r, e, rtol=rtol, atol=atol, max_mismatch_pct=max_mismatch_pct,
                msg=f"{name}: forward output {i} (shape {tuple(r.shape)}) mismatch",
            )

    # --- backward correctness ---
    if bwd:
        _check_backward(all_fns, ref_name, ref_fn, args, rtol, atol, max_mismatch_pct)

    # --- benchmark ---
    if benchmark is None:
        benchmark = torch.cuda.is_available()
    if not benchmark:
        return {}

    times: dict[str, float] = {}
    for name, fn in all_fns.items():
        cloned = clone_args(args)
        times[name] = do_bench(functools.partial(fn, *cloned))

    _print_table(times, baselines)
    return times


def _check_backward(
    all_fns: dict[str, ExampleFn],
    ref_name: str,
    ref_fn: ExampleFn,
    args: tuple[object, ...],
    rtol: float,
    atol: float,
    max_mismatch_pct: float | None,
) -> None:
    grad_idx = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.requires_grad]
    assert grad_idx, "bwd=True but no arg has requires_grad=True"

    def run_backward(fn: ExampleFn) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        cloned = clone_args(args)
        out = _as_tensors(fn(*cloned))
        grad_outputs = [torch.randn_like(o) for o in out]
        torch.autograd.backward(out, grad_outputs)
        grads = [
            cloned[i].grad.clone() if cloned[i].grad is not None else None  # type: ignore[union-attr]
            for i in grad_idx
        ]
        return grad_outputs, grads

    # Use the same grad-output for baseline and every impl so grads are comparable.
    grad_outputs, baseline_grads = run_backward(ref_fn)
    assert any(g is not None for g in baseline_grads), "baseline produced no gradients"

    for name, fn in all_fns.items():
        if name == ref_name:
            continue
        print(f"Checking {name} backward...", file=sys.stderr)
        cloned = clone_args(args)
        out = _as_tensors(fn(*cloned))
        assert len(out) == len(grad_outputs)
        torch.autograd.backward(out, grad_outputs)
        for k, i in enumerate(grad_idx):
            g = cloned[i].grad  # type: ignore[union-attr]
            e = baseline_grads[k]
            assert (g is None) == (e is None), f"{name}: grad presence mismatch for arg {i}"
            if e is not None:
                assert_close(
                    g, e, rtol=rtol, atol=atol, max_mismatch_pct=max_mismatch_pct,
                    msg=f"{name}: gradient for arg {i} (shape {tuple(g.shape)}) mismatch",
                )


def _print_table(times: dict[str, float], baselines: dict[str, object]) -> None:
    best_baseline = min(times[name] for name in baselines)
    line = "=" * 65
    print(f"\n{line}\nBenchmark Results\n{line}", file=sys.stderr)
    print(f"{'Implementation':<24} {'Time (ms)':<12} {'Speedup':<12}", file=sys.stderr)
    print("-" * 65, file=sys.stderr)
    for name, t in times.items():
        is_ref = name in baselines and t == best_baseline
        speedup = "1.00x (ref)" if is_ref else f"{best_baseline / t:.2f}x"
        print(f"{name:<24} {t:<12.4f} {speedup:<12}", file=sys.stderr)
    print(f"{line}\n", file=sys.stderr)
