# Foreach-optimized Muon optimizer with batched operations
from typing import cast

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer

from dtensor_muon.orthogonalize import (
    OrthogonalizationStrategy,
    foreach_zeropower,
    foreach_zeropower_3d_fsdp,
    is_fsdp_3d_sharded,
)
from dtensor_muon.utils import group_tensors_by_shape, move_tensors_to_device

from .optim import Muon


def _register_dtensor_foreach_ops():
    """Register _foreach_sign_ with DTensor using official register_op_strategy API."""
    from torch.distributed.tensor._op_schema import (
        OpSpec,
        OpStrategy,
        RuntimeSchemaInfo,
        TupleStrategy,
    )
    from torch.distributed.tensor._ops.utils import register_op_strategy

    op = torch.ops.aten._foreach_sign_.default
    if hasattr(op, "_dtensor_registered"):
        return

    @register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))
    def _(op_schema):
        arg = op_schema.args_schema[0]
        return TupleStrategy(
            [
                OpStrategy([OpSpec(s.output_specs, (s.output_specs,)) for s in c.strategies])
                for c in arg.children
            ]
        )

    setattr(op, "_dtensor_registered", True)


_register_dtensor_foreach_ops()


class MuonForeach(Muon):
    """Muon optimizer with batched foreach operations for better performance."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        wd: float = 0.1,
        use_cautious_wd: bool = True,
        maximize: bool = False,
        # Muon defaults
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        orthogonalization_strategy: OrthogonalizationStrategy = "polar_express",
        # Adam defaults
        amsgrad: bool = False,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        is_adamw: bool = True,
        fused_adam: bool | None = None,
        foreach_adam: bool | None = None,
        # Parallelism
        batch_size: int | None = None,
    ):
        super().__init__(
            params=params,
            lr=lr,
            wd=wd,
            use_cautious_wd=use_cautious_wd,
            maximize=maximize,
            #
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            orthogonalization_strategy=orthogonalization_strategy,
            amsgrad=amsgrad,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            is_adamw=is_adamw,
            fused_adam=fused_adam,
            foreach_adam=foreach_adam,
        )
        self.batch_size = batch_size

    def muon(
        self,
        params: list[Tensor],
        grads: list[Tensor],
        momentum_buffers: list[Tensor],
        lr_ratios: list[Tensor],
        *,
        nesterov: bool,
        lr: Tensor,
        weight_decay: float,
        cautious_wd: bool,
        momentum: float,
        ns_steps: int,
        orthogonalization_strategy: OrthogonalizationStrategy,
        maximize: bool,
    ) -> None:
        """Batched Muon update using foreach operations."""
        if len(params) == 0:
            return

        # Group by device and dtype
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
            cast(
                list[list[Tensor | None]],
                [params, grads, momentum_buffers, lr_ratios],
            ),
        )

        for (device, _), (
            (device_p, device_g, device_buf, device_lr),
            _,
        ) in grouped_tensors.items():
            assert device is not None

            # Group by shape for efficient foreach operations
            device_g = cast(list[Tensor], device_g)
            for _, (_, indices) in group_tensors_by_shape(device_g).items():
                # Chunk if batch_size is set
                batches = (
                    [
                        indices[i : i + self.batch_size]
                        for i in range(0, len(indices), self.batch_size)
                    ]
                    if self.batch_size and len(indices) > self.batch_size
                    else [indices]
                )

                for batch_idx in batches:
                    batch_p_orig = [device_p[i] for i in batch_idx]
                    batch_g_orig = [device_g[i] for i in batch_idx]
                    batch_buf_orig = [device_buf[i] for i in batch_idx]
                    batch_lr_orig = [device_lr[i] for i in batch_idx]

                    # Move to CUDA for processing (handles CPU offload)
                    cuda = torch.device("cuda")
                    batch_p = move_tensors_to_device(batch_p_orig, device, cuda)
                    batch_g = move_tensors_to_device(batch_g_orig, device, cuda)
                    batch_buf = move_tensors_to_device(batch_buf_orig, device, cuda)
                    batch_lr = move_tensors_to_device(batch_lr_orig, device, cuda)

                    _foreach_muon(
                        cast(list[Tensor], batch_p),
                        cast(list[Tensor], batch_g),
                        cast(list[Tensor], batch_buf),
                        cast(list[Tensor], batch_lr),
                        nesterov,
                        lr,
                        weight_decay,
                        cautious_wd,
                        momentum,
                        ns_steps,
                        orthogonalization_strategy,
                        maximize,
                    )

                    # CPU offload mutates CUDA copies; copy those values back to the
                    # original tensors. Same-device batches alias and need no copy.
                    for originals, moved in (
                        (batch_p_orig, batch_p),
                        (batch_g_orig, batch_g),
                        (batch_buf_orig, batch_buf),
                        (batch_lr_orig, batch_lr),
                    ):
                        if originals is not moved:
                            for original, value in zip(originals, moved, strict=True):
                                if original is not None and value is not None:
                                    original.copy_(value.to(original.device))


def _foreach_muon(
    p: list[Tensor],
    g: list[Tensor],
    buf: list[Tensor],
    lr_ratio: list[Tensor],
    nesterov: bool,
    lr: Tensor,
    weight_decay: float,
    cautious_wd: bool,
    momentum: float,
    ns_steps: int,
    orthogonalization_strategy: OrthogonalizationStrategy,
    maximize: bool,
):
    """Low-level foreach Muon update. Assumes all tensors are same shape and device."""
    if maximize:
        g = list(torch._foreach_neg(g))

    torch._foreach_mul_(buf, momentum)
    torch._foreach_add_(buf, g)

    if nesterov:
        torch._foreach_add_(g, buf, alpha=momentum)
    else:
        g = buf

    # Zero power via orthogonalization
    if is_fsdp_3d_sharded(g):
        u = foreach_zeropower_3d_fsdp(
            cast(list[DTensor], g), steps=ns_steps, strategy=orthogonalization_strategy
        )
    else:
        u = foreach_zeropower(g, steps=ns_steps, strategy=orthogonalization_strategy)
    u = [u_.view_as(p_) for u_, p_ in zip(u, p, strict=True)]

    # Apply weight decay into the update direction (mirrors Muon.muon reference).
    if weight_decay != 0:
        if cautious_wd:
            # u += weight_decay * p, only where u * p > 0
            mask = [(u_ * p_ > 0).to(p_.dtype) for u_, p_ in zip(u, p, strict=True)]
            masked_p = torch._foreach_mul(p, mask)
            torch._foreach_add_(u, masked_p, alpha=weight_decay)
        else:
            torch._foreach_add_(u, p, alpha=weight_decay)

    # scale update
    adjusted_lr = torch._foreach_mul(lr_ratio, lr)
    torch._foreach_mul_(u, adjusted_lr)

    torch._foreach_add_(p, u, alpha=-1)
