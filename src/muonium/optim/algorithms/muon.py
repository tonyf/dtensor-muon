# Baseline Muon update rule (Keller Jordan's Muon), per-tensor and foreach forms.
import torch
from torch import Tensor

from muonium.orthogonalize import OrthogonalizationStrategy

from .base import (
    MuonAlgorithm,
    orthogonalize_batch,
    orthogonalize_single,
    register_muon_algorithm,
)


@register_muon_algorithm
class MuonBaseline(MuonAlgorithm):
    """The original Muon update: momentum, orthogonalize, shape-scaled apply."""

    name = "muon"

    def update(
        self,
        param: Tensor,
        grad: Tensor,
        state: dict[str, Tensor],
        lr_ratio: Tensor,
        *,
        lr: Tensor,
        weight_decay: float,
        cautious_wd: bool,
        momentum: float,
        nesterov: bool,
        maximize: bool,
        ns_steps: int,
        orthogonalization_strategy: OrthogonalizationStrategy,
        split_sizes: tuple[int, ...] | None,
        **opts,
    ) -> None:
        momentum_buffer = state["momentum_buffer"]
        if maximize:
            grad = -grad

        param_fp32 = param.to(torch.float32)
        grad_fp32 = grad.to(torch.float32)
        momentum_buffer_fp32 = momentum_buffer.to(torch.float32)

        # Keep the compiled math out-of-place until copying state back. This
        # avoids functionalization aliasing the fp32 views with fp32 inputs.
        momentum_buffer_fp32 = momentum_buffer_fp32 * momentum + grad_fp32
        momentum_buffer.copy_(momentum_buffer_fp32)

        # Update gradient
        if nesterov:
            grad_fp32 = grad_fp32 + momentum_buffer_fp32 * momentum
        else:
            grad_fp32 = momentum_buffer_fp32

        # Zero power via orthogonalization (Newton-Schulz or Polar Express)
        u = orthogonalize_single(
            grad_fp32,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
        )
        u = u.view_as(param_fp32)

        # Apply weight decay. Cautious WD (only where u * p > 0) is an addition
        # to the original Muon; cautious_wd=False gives plain decoupled WD.
        if weight_decay != 0:
            if cautious_wd:
                u = u + weight_decay * param_fp32 * (u * param_fp32 > 0)
            else:
                u = u + weight_decay * param_fp32

        # Scale update
        adjusted_lr = lr_ratio * lr
        param.copy_(param_fp32 - adjusted_lr * u)

    def foreach_update(
        self,
        params: list[Tensor],
        grads: list[Tensor],
        state: dict[str, list[Tensor]],
        lr_ratios: list[Tensor],
        *,
        lr: Tensor,
        weight_decay: float,
        cautious_wd: bool,
        momentum: float,
        nesterov: bool,
        maximize: bool,
        ns_steps: int,
        orthogonalization_strategy: OrthogonalizationStrategy,
        split_sizes: tuple[int, ...] | None,
        **opts,
    ) -> None:
        """Batched Muon update. Assumes all tensors are same shape and device."""
        p = params
        g = grads
        buf = state["momentum_buffer"]
        if maximize:
            g = list(torch._foreach_neg(g))

        torch._foreach_mul_(buf, momentum)
        torch._foreach_add_(buf, g)

        if nesterov:
            torch._foreach_add_(g, buf, alpha=momentum)
        else:
            g = buf

        # Independent of u: enqueued before orthogonalization so on the DTensor
        # path it overlaps the async redistribute instead of queuing behind its
        # stream wait (forced at the first pointwise op on u below).
        adjusted_lr = torch._foreach_mul(lr_ratios, lr)

        # Zero power via orthogonalization
        u = orthogonalize_batch(
            g,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
        )
        u = [u_.view_as(p_) for u_, p_ in zip(u, p, strict=True)]

        # Apply weight decay into the update direction (mirrors the per-tensor
        # reference).
        if weight_decay != 0:
            if cautious_wd:
                # u += weight_decay * p, only where u * p > 0
                mask = [(u_ * p_ > 0).to(p_.dtype) for u_, p_ in zip(u, p, strict=True)]
                masked_p = torch._foreach_mul(p, mask)
                torch._foreach_add_(u, masked_p, alpha=weight_decay)
            else:
                torch._foreach_add_(u, p, alpha=weight_decay)

        # scale update
        torch._foreach_mul_(u, adjusted_lr)

        torch._foreach_add_(p, u, alpha=-1)
