"""Shared benchmark plumbing: timing, environment capture, and markdown reporting.

This is deliberately standalone — it depends only on ``torch`` (and ``triton.testing``
on CUDA), not on the test-only ``tests/testkit.py``. Importing it sets the same Dynamo
cache limits and matmul precision the test suite uses so the compiled orthogonalization
loops don't trip the recompile limit during a multi-shape sweep.
"""

import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import torch

# The compiled loops are ``fullgraph=True``; a sweep over many (shape, dtype, steps)
# combos otherwise hits Dynamo's recompile limit (mirrors ``tests/conftest.py``).
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 256
torch.set_float32_matmul_precision("high")


def cuda_available() -> bool:
    return torch.cuda.is_available()


def bench(
    fn: Callable[[], object],
    *,
    grad_to_none: Sequence[torch.Tensor] | None = None,
    warmup: int = 10,
    reps: int = 50,
) -> float:
    """Median wall-clock time of ``fn`` in milliseconds.

    On CUDA this defers to ``triton.testing.do_bench`` (CUDA-event timing, auto-tuned
    rep count). On CPU it falls back to a ``perf_counter`` median, since the triton
    timer assumes a CUDA device.
    """
    if cuda_available():
        from triton.testing import do_bench as _do_bench

        return float(
            _do_bench(
                fn,
                grad_to_none=list(grad_to_none) if grad_to_none else None,
                return_mode="median",
            )
        )

    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 3e-2,
    max_mismatch_pct: float = 5.0,
    msg: str = "",
) -> None:
    """float32 closeness tolerating a small fraction of mismatched elements.

    bf16 orthogonalization produces a few near-zero outliers; the discrete cautious-WD
    mask flips a handful of bits between implementations. Same budget the optimizer
    equivalence tests use (``tests/optim/test_optim_foreach.py``).
    """
    actual = actual.detach().to(torch.float32)
    expected = expected.detach().to(torch.float32)
    mismatched = ~torch.isclose(actual, expected, rtol=rtol, atol=atol)
    pct = 100.0 * mismatched.sum().item() / max(mismatched.numel(), 1)
    if pct > max_mismatch_pct:
        raise AssertionError(
            f"{msg}\n{pct:.4f}% of elements mismatched (allowed {max_mismatch_pct:.4f}%) "
            f"at rtol={rtol}, atol={atol}"
        )


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def capture_env() -> dict[str, str]:
    """Snapshot the hardware/software the numbers were produced on."""
    if torch.cuda.is_available():
        names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
        device = f"{torch.cuda.device_count()}x {', '.join(sorted(names))}"
        cuda = torch.version.cuda or "unknown"
    else:
        device = "CPU only"
        cuda = "n/a"
    return {
        "Device": device,
        "CUDA": cuda,
        "PyTorch": torch.__version__,
        "Python": platform.python_version(),
        "Platform": platform.platform(),
        "Commit": _git_commit(),
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavored markdown table."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body]) if rows else head + "\n" + sep


def speedup_rows(times: dict[str, float], ref: str) -> list[list[str]]:
    """Rows of ``[name, time_ms, speedup-vs-ref]`` for a timing dict."""
    base = times[ref]
    rows = []
    for name, t in times.items():
        speedup = "1.00x (ref)" if name == ref else f"{base / t:.2f}x"
        rows.append([name, f"{t:.4f}", speedup])
    return rows


def env_header_md(env: dict[str, str]) -> str:
    lines = ["# Benchmark results", "", "Regenerate with `uv run python benchmark/run.py`.", ""]
    lines += [f"- **{k}:** {v}" for k, v in env.items()]
    return "\n".join(lines)


def write_results(path: Path, sections: list[tuple[str, str]], env: dict[str, str]) -> None:
    """Write the env header followed by each ``(title, markdown)`` section."""
    parts = [env_header_md(env)]
    for title, md in sections:
        parts.append(f"\n## {title}\n\n{md}")
    path.write_text("\n".join(parts) + "\n")
    print(f"wrote {path}", file=sys.stderr)
