# Low-precision Muon optimizer with quantized optimizer states
try:
    import torchao  # noqa: F401  # ty: ignore[unresolved-import]
except ImportError:
    raise ImportError("Please install `torchao` package to use low-precision optimizers")

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torchao.optim.subclass_4bit import OptimState4bit  # ty: ignore[unresolved-import]
from torchao.optim.subclass_8bit import OptimState8bit  # ty: ignore[unresolved-import]
from torchao.optim.subclass_fp8 import OptimStateFp8  # ty: ignore[unresolved-import]

from dtensor_muon.orthogonalize import OrthogonalizationStrategy
from dtensor_muon.utils import to_local

from .algorithms import BufferSpec
from .optim import Muon


class MuonLP(Muon):
    """Low-precision Muon with quantized optimizer states."""

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
        foreach_adam: bool | None = None,
        fused_adam: bool | None = None,
        # Execution strategy. Defaults to the per-param driver: the quantized
        # torchao state subclasses do not implement the torch._foreach_* ops the
        # batched driver uses.
        foreach: bool | None = False,
        batch_size: int | None = None,
        # Low-precision
        block_size: int = 2048,
        bf16_stochastic_round: bool = False,
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
            #
            amsgrad=amsgrad,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            is_adamw=is_adamw,
            fused_adam=fused_adam,
            foreach_adam=foreach_adam,
            foreach=foreach,
            batch_size=batch_size,
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

    def _new_state_buffer(self, p: Tensor, grad: Tensor, spec: BufferSpec) -> Tensor:
        """Allocate algorithm state through the quantizing ``_new_buffer`` path.

        Grad-shaped buffers (e.g. the momentum buffer) are built from the
        parameter via ``_new_buffer`` — quantized when eligible and re-wrapped as
        DTensors. Row-shaped buffers (``like="grad_rows"``) are tiny (one value
        per neuron), far below the quantization threshold, so they use the plain
        fp32 allocation.
        """
        if spec.like == "grad":
            return self._new_buffer(p, signed=spec.signed)
        return super()._new_state_buffer(p, grad, spec)


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
