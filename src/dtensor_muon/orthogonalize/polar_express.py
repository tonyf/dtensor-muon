"""
Polar Express orthogonalization algorithm for Muon optimizer.

Reference: "Polar Express Sign Method for orthogonalization"
https://arxiv.org/pdf/2505.16932 by Noah Amsel et al.

Adapted from nanochat implementation by Karpathy:
https://github.com/karpathy/nanochat/blob/542beb0c8c175af2d52ec7065345dcd8f0162368/nanochat/optim.py#L78
"""

from typing import Annotated

import torch
from torch import Tensor

from dtensor_muon.kernels.gram import gram_

# Polar Express coefficients from https://arxiv.org/pdf/2505.16932
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(fullgraph=True)
def pe_loop(
    X: Annotated[Tensor, "B N M"],
    steps: int,
    *,
    eps: float = 1e-6,
) -> Annotated[Tensor, "B N M"]:
    """
    Polar Express orthogonalization algorithm.

    Computes the matrix sign function (zero-power) using a specialized
    iteration scheme with precomputed coefficients for fast convergence.

    Args:
        X: Input tensor of shape (B, N, M) where B is batch size
        steps: Number of iteration steps (max 5, limited by POLAR_EXPRESS_COEFFS)
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized tensor of same shape as input
    """
    assert steps <= 5, (
        "polar express orthogonalization only supports up to 5 optimization steps."
    )
    if X.size(0) == 0:
        return X

    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.transpose(-2, -1)

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + eps)

    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        A = torch.matmul(X, X.transpose(-1, -2))
        B = A.mul(b).add_(torch.matmul(A, A), alpha=c)
        X = X.mul(a).add_(B @ X)

    if transpose:
        X = X.transpose(-2, -1)

    return X


@torch.compile(fullgraph=True, dynamic=True)
def pe_loop_triton(
    X: Tensor,
    steps: int = 5,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """
    Polar Express orthogonalization algorithm.
    """
    assert steps <= 5, (
        "polar express orthogonalization only supports up to 5 optimization steps."
    )
    assert X.ndim >= 2
    if X.size(0) == 0:
        return X

    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + eps)

    M = X.size(-2)
    buf_shape = (*X.shape[:-2], M, M)
    A = torch.empty(buf_shape, dtype=X.dtype, device=X.device)
    B = torch.empty_like(A)

    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
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
        pe_loop_triton,
        pe_loop,
        (X, steps),
        kernel_name="triton",
        baseline_name="torch",
    )
