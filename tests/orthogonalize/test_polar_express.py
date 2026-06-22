"""Correctness tests for Polar Express orthogonalization.

Checks the Triton-backed iteration (``pe_loop_triton``, which uses the ``gram_``
kernel) against the repository's own PyTorch reference (``pe_loop``), run
*uncompiled* via its ``__wrapped__`` original as a stable ground truth. As with
Newton-Schulz, the compiled ``pe_loop`` previously miscompiled its in-place
accumulation on 2D inputs; that is fixed (out-of-place accumulation) and
``test_pe_loop_2d_compiles_correctly`` guards against regression. Both require
CUDA (the kernel is CUDA-only).
"""

from typing import Callable

import pytest
import torch
from testkit import run_example

from dtensor_muon.orthogonalize.polar_express import (
    POLAR_EXPRESS_COEFFS,
    pe_loop,
    pe_loop_triton,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

RTOL = 1e-2
ATOL = 2e-2
MAX_MISMATCH_PCT = 0.5


def eager(fn: Callable) -> Callable:
    """The original, uncompiled function behind a ``@torch.compile`` wrapper."""
    return getattr(fn, "__wrapped__", fn)


def _orthogonal_gram(x: torch.Tensor) -> torch.Tensor:
    return x @ x.T if x.size(-2) <= x.size(-1) else x.T @ x


def _normalized_pe_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + eps)


@requires_cuda
@pytest.mark.parametrize("steps", [1, 3, len(POLAR_EXPRESS_COEFFS)])
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
def test_pe_triton_matches_torch(shape: tuple[int, ...], steps: int) -> None:
    torch.manual_seed(0)
    X = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    run_example(
        pe_loop_triton,
        eager(pe_loop),
        (X, steps),
        kernel_name="triton",
        baseline_name="torch",
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=MAX_MISMATCH_PCT,
        benchmark=False,
    )


@requires_cuda
def test_pe_empty_batch_is_passthrough() -> None:
    X = torch.randn(0, 256, 128, device="cuda", dtype=torch.bfloat16)
    run_example(pe_loop_triton, eager(pe_loop), (X, 5), benchmark=False)


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_pe_eager_output_is_approximately_orthogonal(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    X = torch.randn(*shape, dtype=torch.bfloat16)

    out = eager(pe_loop)(X, len(POLAR_EXPRESS_COEFFS)).float()
    gram = _orthogonal_gram(out)
    singular_values = torch.linalg.svdvals(out)

    torch.testing.assert_close(gram, torch.eye(gram.size(0)), rtol=0.0, atol=0.13)
    assert (singular_values - 1).abs().max() < 0.16


def test_pe_steps_zero_returns_normalized_input() -> None:
    torch.manual_seed(0)
    X = torch.randn(64, 32, dtype=torch.bfloat16)

    out = eager(pe_loop)(X, 0)

    torch.testing.assert_close(out, _normalized_pe_input(X), rtol=0, atol=0)


def test_pe_eager_approximates_matrix_sign() -> None:
    torch.manual_seed(1)
    X = torch.randn(64, 32, dtype=torch.bfloat16)

    out = eager(pe_loop)(X, len(POLAR_EXPRESS_COEFFS)).float()
    u, _, vh = torch.linalg.svd(X.float(), full_matrices=False)

    torch.testing.assert_close(out, u @ vh, rtol=0, atol=0.05)


def test_pe_transpose_path_is_symmetric() -> None:
    torch.manual_seed(2)
    X = torch.randn(64, 32, dtype=torch.bfloat16)

    out = eager(pe_loop)(X, len(POLAR_EXPRESS_COEFFS))
    transposed_out = eager(pe_loop)(X.T.contiguous(), len(POLAR_EXPRESS_COEFFS)).T

    torch.testing.assert_close(out, transposed_out, rtol=0, atol=0)


def test_pe_rejects_more_steps_than_coefficients() -> None:
    X = torch.randn(16, 8, dtype=torch.bfloat16)

    with pytest.raises(
        AssertionError,
        match="only supports up to 5 optimization steps",
    ):
        eager(pe_loop)(X, len(POLAR_EXPRESS_COEFFS) + 1)


def test_pe_coefficients_table_integrity() -> None:
    assert len(POLAR_EXPRESS_COEFFS) == 5
    assert POLAR_EXPRESS_COEFFS[0] == (
        8.156554524902461,
        -22.48329292557795,
        15.878769915207462,
    )


def test_pe_rank_deficient_and_zero_inputs_stay_finite() -> None:
    torch.manual_seed(4)
    rank_one = torch.outer(torch.randn(32), torch.randn(16)).to(torch.bfloat16)
    zeros = torch.zeros(32, 16, dtype=torch.bfloat16)

    rank_out = eager(pe_loop)(rank_one, len(POLAR_EXPRESS_COEFFS))
    zero_out = eager(pe_loop)(zeros, len(POLAR_EXPRESS_COEFFS))

    assert torch.isfinite(rank_out).all()
    assert torch.isfinite(zero_out).all()
    torch.testing.assert_close(zero_out, torch.zeros_like(zero_out), rtol=0, atol=0)


def test_pe_already_orthogonal_input_is_near_fixed_point() -> None:
    torch.manual_seed(5)
    q, _ = torch.linalg.qr(torch.randn(64, 32))
    X = q.to(torch.bfloat16)

    out = eager(pe_loop)(X, len(POLAR_EXPRESS_COEFFS)).float()

    torch.testing.assert_close(out, q, rtol=0, atol=0.06)


def test_pe_scale_invariance() -> None:
    torch.manual_seed(6)
    X = torch.randn(32, 16, dtype=torch.bfloat16)
    expected = eager(pe_loop)(X, len(POLAR_EXPRESS_COEFFS)).float()

    for scale in (1e-2, 1e2, 1e4):
        out = eager(pe_loop)(
            (X.float() * scale).to(torch.bfloat16), len(POLAR_EXPRESS_COEFFS)
        ).float()
        torch.testing.assert_close(out, expected, rtol=0, atol=0.06)


def test_pe_single_batch_matches_unbatched() -> None:
    torch.manual_seed(7)
    X = torch.randn(32, 16, dtype=torch.bfloat16)

    batched = eager(pe_loop)(X.unsqueeze(0), len(POLAR_EXPRESS_COEFFS))[0]
    unbatched = eager(pe_loop)(X, len(POLAR_EXPRESS_COEFFS))

    torch.testing.assert_close(batched, unbatched, rtol=0, atol=0)


@requires_cuda
def test_pe_loop_2d_compiles_correctly() -> None:
    """Regression: the compiled ``pe_loop`` must match its eager original on 2D.

    The in-place ``add_`` accumulation used to miscompile here; it is now written
    out-of-place. This pins that the compiled path stays correct on a bare 2D
    matrix.
    """
    torch.manual_seed(0)
    X = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    run_example(
        pe_loop,
        eager(pe_loop),
        (X, 5),
        rtol=RTOL,
        atol=ATOL,
        max_mismatch_pct=MAX_MISMATCH_PCT,
        benchmark=False,
    )
