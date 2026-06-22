# Low-precision Muon optimizer with quantized optimizer states
import math

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import _get_scalar_dtype
from torchao.optim.subclass_4bit import OptimState4bit
from torchao.optim.subclass_8bit import OptimState8bit
from torchao.optim.subclass_fp8 import OptimStateFp8

from dtensor_muon.orthogonalize import OrthogonalizationStrategy
from dtensor_muon.utils import to_local

from .optim import Muon

try:
    import torchao  # noqa: F401
except ImportError:
    raise ImportError("Please install `torchao` package to use low-precision optimizers")


class MuonLP(Muon):
    """Low-precision Muon with quantized optimizer states."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        wd: float = 0.1,
        use_cautious_wd: bool = True,
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
        fused_adam: bool = True,
        maximize: bool = False,
        # Low-precision
        block_size: int = 2048,
        bf16_stochastic_round: bool = False,
        compile: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            wd=wd,
            use_cautious_wd=use_cautious_wd,
            #
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            orthogonalization_strategy=orthogonalization_strategy,
            #
            amsgrad=amsgrad,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            is_adamw=is_adamw,
            fused_adam=fused_adam,
            maximize=maximize,
            compile=compile,
        )
        self.block_size = block_size
        self.bf16_stochastic_round = bf16_stochastic_round

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int) -> Tensor:
        """Override in subclasses to use quantized storage."""
        return torch.zeros_like(p)

    def _new_buffer(self, p: Tensor, signed: bool) -> Tensor:
        """Create a buffer, potentially quantized and DTensor-wrapped."""
        local_p = to_local(p)

        # Follow bitsandbytes: only quantize tensors >= 4096 values
        if local_p.numel() >= 4096 and local_p.numel() % self.block_size == 0:
            out = self._subclass_zeros(local_p, signed, self.block_size)
        else:
            out = torch.zeros_like(local_p)

        # Wrap in DTensor if needed
        if isinstance(p, DTensor):
            out = DTensor.from_local(
                local_tensor=out,
                device_mesh=p.device_mesh,
                placements=p.placements,
                run_check=False,
                shape=p.shape,
                stride=p.stride(),
            )

        return out.to(p.device)

    def _init_muon_group(
        self,
        group,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_buffers: list[Tensor],
        lr_ratios: list[Tensor],
        state_steps: list[Tensor],
    ) -> None:
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue

            params_with_grad.append(p)
            assert not grad.is_sparse, "Muon does not support sparse gradients"
            if grad.ndim > 2:
                assert grad.ndim == 3 or group["flatten"], (
                    f"Got ndim={grad.ndim}. Please set flatten=True"
                )
                grad = grad.view(grad.size(0), -1) if group["flatten"] else grad

            grads.append(grad)
            state = self.state[p]

            # Lazy state initialization with quantized buffers
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                state["momentum_buffer"] = self._new_buffer(p, signed=True)
                state["lr_ratio"] = torch.tensor(
                    math.sqrt(max(1.0, grad.shape[-2] / grad.shape[-1]))
                )

            state["step"] += 1
            momentum_buffers.append(state["momentum_buffer"])
            lr_ratios.append(state["lr_ratio"])
            state_steps.append(state["step"])


class Muon8bit(MuonLP):
    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int) -> Tensor:
        return OptimState8bit.zeros(p.shape, signed, block_size, p.device)


class Muon4bit(MuonLP):
    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int) -> Tensor:
        return OptimState4bit.zeros(p.shape, signed, block_size, p.device)


class MuonFp8(MuonLP):
    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int) -> Tensor:
        return OptimStateFp8.zeros(p.shape, block_size, p.device)
