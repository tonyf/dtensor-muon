"""
Newton-Schulz iteration for matrix orthogonalization.

This module contains the PyTorch implementation of Newton-Schulz iteration.
The triton kernel is in the kernel submodule.
"""

from typing import Annotated

import torch
from torch import Tensor

from dtensor_muon.kernels.gram import gram_


@torch.compile(fullgraph=True)
def ns_loop(
    X: Annotated[Tensor, "B N M"],
    steps: int,
    *,
    a: float = 3.4445,
    b: float = -4.7750,
    c: float = 2.0315,
    eps: float = 1e-7,
) -> Annotated[Tensor, "B N M"]:
    """
    Newton-Schulz iteration for computing matrix sign function.

    Args:
        X: Input tensor of shape (B, N, M) where B is batch size
        steps: Number of iteration steps
        a, b, c: Newton-Schulz coefficients
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized tensor of same shape as input
    """
    if X.size(0) == 0:
        return X

    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.transpose(-2, -1)

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        A = torch.matmul(X, X.transpose(-1, -2))
        B = A.mul(b).add_(torch.matmul(A, A), alpha=c)
        X = X.mul(a).add_(torch.matmul(B, X))

    if transpose:
        X = X.transpose(-2, -1)

    return X


@torch.compile(fullgraph=True, dynamic=True)
def ns_loop_triton(
    X: Tensor,
    steps: int = 5,
    *,
    a: float = 3.4445,
    b: float = -4.7750,
    c: float = 2.0315,
    eps: float = 1e-7,
) -> Tensor:
    """
    Batched/ND Newton-Schulz orthogonalization over the last 2 dims.
    """
    assert X.ndim >= 2
    if X.size(0) == 0:
        return X

    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT  # transpose last two dims

    # Ensure spectral norm <= 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    M = X.size(-2)
    buf_shape = (*X.shape[:-2], M, M)
    A = torch.empty(buf_shape, dtype=X.dtype, device=X.device)
    B = torch.empty_like(A)

    for _ in range(steps):
        # A = X X^T
        gram_(X, A)

        # B = A A^T == A^2 since A is symmetric
        gram_(A, B)

        # B := B = b*A + c*A^2   (reuse B storage)
        B.mul_(c).add_(A, alpha=b)

        # X := a*X + B X
        X = X.mul(a).add_(B @ X)

    if transpose:
        X = X.mT
    return X


if __name__ == "__main__":
    import torch
    from helion._testing import run_example

    X = torch.randn(32, 2048, 1024, device="cuda", dtype=torch.bfloat16)
    steps = 5

    run_example(
        ns_loop_triton,
        ns_loop,
        (X, steps),
        kernel_name="triton",
        baseline_name="torch",
    )
