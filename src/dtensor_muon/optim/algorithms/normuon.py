# NorMuon: Muon with neuron-wise update normalization (arxiv.org/abs/2510.05491).
import torch
from torch import Tensor

from dtensor_muon.orthogonalize import OrthogonalizationStrategy

from .base import (
    BufferSpec,
    MuonAlgorithm,
    orthogonalize_batch,
    orthogonalize_single,
    register_muon_algorithm,
)


def _normuon_normalize(
    u: Tensor,
    v: Tensor,
    muon_beta2: float,
    split_sizes: tuple[int, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Neuron-wise normalization of an orthogonalized update.

    Maintains a per-row (neuron) second-moment EMA ``v``, divides the update by
    its square root, then rescales to preserve the update's Frobenius norm.
    Returns ``(normalized_u, new_v)`` out-of-place; the caller copies ``new_v``
    back into optimizer state. With ``split_sizes``, each row block is normalized
    independently, matching the update separate per-block parameters would get.
    """
    if split_sizes is not None:
        results = [
            _normuon_normalize(u_block, v_block, muon_beta2)
            for u_block, v_block in zip(
                u.split(list(split_sizes), dim=-2),
                v.split(list(split_sizes), dim=-2),
                strict=True,
            )
        ]
        return (
            torch.cat([r[0] for r in results], dim=-2),
            torch.cat([r[1] for r in results], dim=-2),
        )

    u = u.to(v.dtype)
    norm_u = u.norm(p=2, dim=(-2, -1), keepdim=True)

    # Neuron-wise second moment: mean of squares along the input dimension.
    neuron_norms = (u * u).mean(dim=-1, keepdim=True)
    v_new = torch.lerp(v, neuron_norms, 1 - muon_beta2)

    u = u / (v_new.sqrt() + 1e-8)

    # Rescale to preserve the Frobenius norm of the unnormalized update.
    norm_u_new = u.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return u * (norm_u / norm_u_new), v_new


@register_muon_algorithm
class NorMuon(MuonAlgorithm):
    """Muon plus a neuron-wise second-moment normalization of the update.

    Identical to baseline Muon except the orthogonalized update is normalized
    row-wise by an EMA of per-neuron squared magnitudes (decay ``muon_beta2``)
    and rescaled to keep its Frobenius norm.
    """

    name = "normuon"
    options = {"muon_beta2": 0.95}
    state_spec = {
        "momentum_buffer": BufferSpec(like="grad", signed=True),
        "variance_neuron": BufferSpec(like="grad_rows", signed=False),
    }

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
        muon_beta2: float = 0.95,
        **opts,
    ) -> None:
        momentum_buffer = state["momentum_buffer"]
        variance_neuron = state["variance_neuron"]
        if maximize:
            grad = -grad

        param_fp32 = param.to(torch.float32)
        grad_fp32 = grad.to(torch.float32)
        momentum_buffer_fp32 = momentum_buffer.to(torch.float32)

        # Out-of-place until state copy-back (see MuonBaseline.update).
        momentum_buffer_fp32 = momentum_buffer_fp32 * momentum + grad_fp32
        momentum_buffer.copy_(momentum_buffer_fp32)

        if nesterov:
            grad_fp32 = grad_fp32 + momentum_buffer_fp32 * momentum
        else:
            grad_fp32 = momentum_buffer_fp32

        u = orthogonalize_single(
            grad_fp32,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
        )

        # Neuron-wise normalization on the grad-shaped update, so rows align with
        # the variance buffer under flatten.
        u, v_new = _normuon_normalize(
            u, variance_neuron.to(torch.float32), muon_beta2, split_sizes
        )
        variance_neuron.copy_(v_new)

        u = u.view_as(param_fp32)

        if weight_decay != 0:
            if cautious_wd:
                u = u + weight_decay * param_fp32 * (u * param_fp32 > 0)
            else:
                u = u + weight_decay * param_fp32

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
        muon_beta2: float = 0.95,
        **opts,
    ) -> None:
        """Batched NorMuon update. Assumes all tensors are same shape and device."""
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
        # stream wait (see MuonBaseline.foreach_update).
        adjusted_lr = torch._foreach_mul(lr_ratios, lr)

        u = orthogonalize_batch(
            g,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
        )

        # Normalization runs per tensor: it is a handful of reductions that
        # torch.compile fuses, and staying per-tensor keeps DTensor inputs (both
        # the redistribute path and the FSDP local-shard fast path) correct
        # without placement assumptions.
        normalized = []
        for u_, v in zip(u, state["variance_neuron"], strict=True):
            u_, v_new = _normuon_normalize(u_, v.to(torch.float32), muon_beta2, split_sizes)
            v.copy_(v_new)
            normalized.append(u_)
        u = [u_.view_as(p_) for u_, p_ in zip(normalized, p, strict=True)]

        if weight_decay != 0:
            if cautious_wd:
                mask = [(u_ * p_ > 0).to(p_.dtype) for u_, p_ in zip(u, p, strict=True)]
                masked_p = torch._foreach_mul(p, mask)
                torch._foreach_add_(u, masked_p, alpha=weight_decay)
            else:
                torch._foreach_add_(u, p, alpha=weight_decay)

        torch._foreach_mul_(u, adjusted_lr)

        torch._foreach_add_(p, u, alpha=-1)
