"""Correctness tests for Newton-Schulz orthogonalization.

Checks the Triton-backed iteration (``ns_loop_triton``, which uses the ``gram_``
kernel) against the repository's own PyTorch reference (``ns_loop``). The
reference is run *uncompiled* (via its ``__wrapped__`` original) as a stable
ground truth. ``ns_loop`` previously miscompiled on 2D inputs under
``torch.compile`` (in-place ``add_`` accumulation); that has been fixed by
rewriting the accumulation out-of-place, and ``test_ns_loop_2d_compiles_correctly``
guards against a regression. Both require CUDA (the kernel is CUDA-only), so the
module is skipped when no GPU is available.
"""

from typing import Callable

import pytest
import torch
from testkit import run_example

from dtensor_muon.orthogonalize.newton_schulz import ns_loop, ns_loop_triton

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# bf16 iteration: the triton and eager paths agree to <0.01% of elements at these
# tolerances across every shape/step tested, so the budget below is generous.
RTOL = 1e-2
ATOL = 2e-2
MAX_MISMATCH_PCT = 0.5


def eager(fn: Callable) -> Callable:
    """The original, uncompiled function behind a ``@torch.compile`` wrapper."""
    return getattr(fn, "__wrapped__", fn)


def _orthogonal_gram(x: torch.Tensor) -> torch.Tensor:
    return x @ x.T if x.size(-2) <= x.size(-1) else x.T @ x


def _normalized_ns_input(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return x / (x.norm(dim=(-2, -1), keepdim=True) + eps)


@requires_cuda
@pytest.mark.parametrize("steps", [1, 3, 5])
@pytest.mark.parametrize(
    "shape",
    [
        (8, 256, 128),  # batched, wide (N > M -> transpose path)
        (8, 128, 256),  # batched, tall (N < M)
        (4, 512, 512),  # batched, square
        (256, 128),  # unbatched 2D
        (128, 256),  # unbatched 2D, tall
    ],
)
def test_ns_triton_matches_torch(shape: tuple[int, ...], steps: int) -> None:
    torch.manual_seed(0)
    X = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    run_example(
        ns_loop_triton,
        eager(ns_loop),
        (X, steps),
        kernel_name="triton",
        baseline_name="torch",
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=MAX_MISMATCH_PCT,
        benchmark=False,
    )


@requires_cuda
def test_ns_empty_batch_is_passthrough() -> None:
    X = torch.randn(0, 256, 128, device="cuda", dtype=torch.bfloat16)
    run_example(ns_loop_triton, eager(ns_loop), (X, 5), benchmark=False)


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_ns_eager_output_is_approximately_orthogonal(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    X = torch.randn(*shape, dtype=torch.bfloat16)

    out = eager(ns_loop)(X, 5).float()
    gram = _orthogonal_gram(out)
    singular_values = torch.linalg.svdvals(out)

    torch.testing.assert_close(gram, torch.eye(gram.size(0)), rtol=0.0, atol=0.45)
    assert (singular_values - 1).abs().max() < 0.35


def test_ns_steps_zero_returns_normalized_input() -> None:
    torch.manual_seed(0)
    X = torch.randn(64, 32, dtype=torch.bfloat16)

    out = eager(ns_loop)(X, 0)

    torch.testing.assert_close(out, _normalized_ns_input(X), rtol=0, atol=0)


def test_ns_eager_approximates_matrix_sign() -> None:
    torch.manual_seed(1)
    X = torch.randn(64, 32, dtype=torch.bfloat16)

    out = eager(ns_loop)(X, 5).float()
    u, _, vh = torch.linalg.svd(X.float(), full_matrices=False)

    torch.testing.assert_close(out, u @ vh, rtol=0, atol=0.12)


def test_ns_transpose_path_is_symmetric() -> None:
    torch.manual_seed(2)
    X = torch.randn(64, 32, dtype=torch.bfloat16)

    out = eager(ns_loop)(X, 5)
    transposed_out = eager(ns_loop)(X.T.contiguous(), 5).T

    torch.testing.assert_close(out, transposed_out, rtol=0, atol=0)


def test_ns_coefficient_overrides_match_manual_iteration() -> None:
    torch.manual_seed(3)
    X = torch.randn(16, 8, dtype=torch.float32)
    a, b, c = 2.0, -1.5, 0.25

    normalized = _normalized_ns_input(X)
    gram = normalized @ normalized.T
    update = gram.mul(b).add(gram @ gram, alpha=c)
    expected = normalized.mul(a).add(update @ normalized)

    out = eager(ns_loop)(X, 1, a=a, b=b, c=c)

    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-7)


def test_ns_rank_deficient_and_zero_inputs_stay_finite() -> None:
    torch.manual_seed(4)
    rank_one = torch.outer(torch.randn(32), torch.randn(16)).to(torch.bfloat16)
    zeros = torch.zeros(32, 16, dtype=torch.bfloat16)

    rank_out = eager(ns_loop)(rank_one, 5)
    zero_out = eager(ns_loop)(zeros, 5)

    assert torch.isfinite(rank_out).all()
    assert torch.isfinite(zero_out).all()
    torch.testing.assert_close(zero_out, torch.zeros_like(zero_out), rtol=0, atol=0)


def test_ns_already_orthogonal_input_is_near_fixed_point() -> None:
    torch.manual_seed(5)
    q, _ = torch.linalg.qr(torch.randn(64, 32))
    X = q.to(torch.bfloat16)

    out = eager(ns_loop)(X, 5).float()

    torch.testing.assert_close(out, q, rtol=0, atol=0.03)


def test_ns_scale_invariance() -> None:
    torch.manual_seed(6)
    X = torch.randn(32, 16, dtype=torch.bfloat16)
    expected = eager(ns_loop)(X, 5).float()

    for scale in (1e-2, 1e2, 1e4):
        out = eager(ns_loop)((X.float() * scale).to(torch.bfloat16), 5).float()
        torch.testing.assert_close(out, expected, rtol=0, atol=0.05)


def test_ns_single_batch_matches_unbatched() -> None:
    torch.manual_seed(7)
    X = torch.randn(32, 16, dtype=torch.bfloat16)

    batched = eager(ns_loop)(X.unsqueeze(0), 5)[0]
    unbatched = eager(ns_loop)(X, 5)

    torch.testing.assert_close(batched, unbatched, rtol=0, atol=0)


@requires_cuda
def test_ns_loop_2d_compiles_correctly() -> None:
    """Regression: the compiled ``ns_loop`` must match its eager original on 2D.

    The in-place ``add_`` accumulation used to miscompile here (orthogonality
    error ~40x); it is now written out-of-place. This pins that the compiled
    path stays correct on a bare 2D matrix.
    """
    torch.manual_seed(0)
    X = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    run_example(
        ns_loop,
        eager(ns_loop),
        (X, 5),
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=MAX_MISMATCH_PCT,
        benchmark=False,
    )
