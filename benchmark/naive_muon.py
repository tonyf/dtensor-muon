"""A naive, single-file Muon — the honest "straight from the blog post" baseline.

This is what you'd write following Keller Jordan's original Muon: a plain
``torch.optim.Optimizer`` with a per-parameter Python loop, an eager 5-step
Newton-Schulz orthogonalization in bf16, and no DTensor / Triton / ``torch.compile`` /
fused-Adam machinery. The update math mirrors :meth:`muonium.optim.optim.Muon.muon`
(fp32 momentum buffer, Nesterov mixing, ``sqrt(max(1, rows/cols))`` shape-scaled LR) so
the production variants can be benchmarked against it apples-to-apples.

Weight decay here is the simple non-cautious form (``u += wd * p``); set the production
optimizers' ``use_cautious_wd=False`` to match.
"""

import math

import torch
from torch import Tensor

# Classic quintic Newton-Schulz coefficients (Jordan / Bernstein-Newhouse).
_A, _B, _C = 3.4445, -4.7750, 2.0315


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int, eps: float = 1e-7) -> Tensor:
    """Eager matrix-sign of ``G`` via ``steps`` Newton-Schulz iterations, in bf16."""
    assert G.ndim >= 2
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = _B * A + _C * (A @ A)
        X = _A * X + B @ X
    if transpose:
        X = X.mT
    return X


class NaiveMuon(torch.optim.Optimizer):
    """Reference per-parameter Muon. Handles 2D+ params only (no Adam fallback)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        wd: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, wd=wd, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim >= 2, "NaiveMuon only handles 2D+ params"

                p_fp32 = p.to(torch.float32)
                g = p.grad.to(torch.float32)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p_fp32)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                u = zeropower_via_newtonschulz5(update, ns_steps).to(torch.float32)
                u = u.view_as(p_fp32)

                if wd != 0:
                    u.add_(p_fp32, alpha=wd)

                rows, cols = p.shape[-2], p.shape[-1]
                lr_ratio = math.sqrt(max(1.0, rows / cols))
                p_fp32.add_(u, alpha=-lr * lr_ratio)
                p.copy_(p_fp32)
        return loss
