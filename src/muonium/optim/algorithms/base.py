# Base class, registry, and shared helpers for pluggable Muon algorithm variants.
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate

from muonium.orthogonalize import (
    OrthogonalizationStrategy,
    foreach_zeropower,
    foreach_zeropower_3d_fsdp,
    is_fsdp_3d_sharded,
    zeropower,
)

# Group keys owned by the optimizer itself; algorithm options may not shadow them.
RESERVED_GROUP_KEYS = frozenset(
    {
        "params",
        "algorithm",
        "use_muon",
        "lr",
        "wd",
        "use_cautious_wd",
        "momentum",
        "nesterov",
        "ns_steps",
        "orthogonalization_strategy",
        "maximize",
        "flatten",
        "foreach",
        "split_sizes",
    }
)

# Algorithm names routed to the Adam path in Muon.__init__, never to this registry.
ADAM_ALGORITHMS = ("adam", "adamw")


@dataclass(frozen=True)
class BufferSpec:
    """Shape/signedness declaration for one per-parameter state buffer.

    ``like="grad"`` allocates an fp32 buffer shaped like the (possibly flattened)
    gradient; ``like="grad_rows"`` shapes it like ``grad[..., :1]`` (one value per
    row/neuron). ``signed`` is consumed by quantized-state optimizers (``MuonLP``).
    """

    like: Literal["grad", "grad_rows"]
    signed: bool = True


def split_lr_scales(
    shape: torch.Size | tuple[int, ...], split_sizes: tuple[int, ...]
) -> tuple[float, ...]:
    """Per-row-block LR corrections for a fused 2D weight split via ``split_sizes``.

    The optimizer scales the whole update by the full matrix's cached
    ``lr_ratio = sqrt(max(1, N/M))``. Each independently orthogonalized row block
    should instead see the adjustment its own shape would receive as a separate
    parameter, so each block is pre-scaled by ``ratio(block) / ratio(full)``.
    """
    rows, cols = shape[-2], shape[-1]
    full_ratio = math.sqrt(max(1.0, rows / cols))
    return tuple(
        math.sqrt(max(1.0, block_rows / cols)) / full_ratio for block_rows in split_sizes
    )


def _split_orthogonalize_full(
    g: Tensor,
    split_sizes: tuple[int, ...],
    scales: tuple[float, ...],
    *,
    ns_steps: int,
    strategy: OrthogonalizationStrategy,
) -> Tensor:
    """Orthogonalize row blocks of a regular (non-DTensor) 2D tensor independently.

    Equal-height blocks (e.g. K and V of a fused QKV weight under GQA) share one
    batched ``foreach_zeropower`` call, mirroring microsoft/dion.
    """
    blocks = list(g.split(list(split_sizes), dim=-2))

    by_height: dict[int, list[int]] = {}
    for i, block in enumerate(blocks):
        by_height.setdefault(block.size(-2), []).append(i)

    out: list[Tensor | None] = [None] * len(blocks)
    for indices in by_height.values():
        ortho = foreach_zeropower(
            [blocks[i] for i in indices], steps=ns_steps, strategy=strategy
        )
        for i, u in zip(indices, ortho, strict=True):
            out[i] = u * scales[i]
    return torch.cat([u for u in out if u is not None], dim=-2)


def orthogonalize_single(
    g: Tensor,
    *,
    ns_steps: int,
    strategy: OrthogonalizationStrategy,
    split_sizes: tuple[int, ...] | None = None,
):
    """Orthogonalize one tensor. DTensor handling lives inside ``zeropower``.

    With ``split_sizes``, row blocks of a fused 2D weight are orthogonalized
    independently and rescaled so the whole-matrix ``lr_ratio`` applied later
    matches the per-block adjustment separate parameters would receive. DTensor
    inputs take a full-tensor round trip (gather, per-block orthogonalize,
    redistribute back to the original placements) so any sharding is supported.
    """
    if split_sizes is None:
        return zeropower(g, steps=ns_steps, strategy=strategy)

    scales = split_lr_scales(g.shape, split_sizes)
    if isinstance(g, DTensor):
        u = _split_orthogonalize_full(
            g.full_tensor(), split_sizes, scales, ns_steps=ns_steps, strategy=strategy
        )
        u_replicated = DTensor.from_local(
            u,
            device_mesh=g.device_mesh,
            placements=(Replicate(),) * g.device_mesh.ndim,
        )
        return u_replicated.redistribute(placements=g.placements, async_op=True)
    return _split_orthogonalize_full(g, split_sizes, scales, ns_steps=ns_steps, strategy=strategy)


def orthogonalize_batch(
    gs: list[Tensor],
    *,
    ns_steps: int,
    strategy: OrthogonalizationStrategy,
    split_sizes: tuple[int, ...] | None = None,
) -> list[Tensor]:
    """Orthogonalize a same-shape batch, preferring the FSDP local-shard fast path.

    This is the one place algorithms touch distributed logic; variants call this
    helper and stay placement-agnostic. ``split_sizes`` (2D fused weights only)
    orthogonalizes row blocks independently; see :func:`orthogonalize_single`.
    """
    if split_sizes is not None:
        if isinstance(gs[0], DTensor):
            # Correctness path: per-tensor full-tensor round trips.
            return [
                orthogonalize_single(
                    g, ns_steps=ns_steps, strategy=strategy, split_sizes=split_sizes
                )
                for g in gs
            ]
        scales = split_lr_scales(gs[0].shape, split_sizes)
        return [
            _split_orthogonalize_full(
                g, split_sizes, scales, ns_steps=ns_steps, strategy=strategy
            )
            for g in gs
        ]
    if is_fsdp_3d_sharded(gs):
        return cast(
            list[Tensor],
            foreach_zeropower_3d_fsdp(gs, steps=ns_steps, strategy=strategy),  # ty: ignore
        )
    return cast(list[Tensor], foreach_zeropower(gs, steps=ns_steps, strategy=strategy))


class MuonAlgorithm(ABC):
    """A Muon-family update rule, selected per param group via ``"algorithm"``.

    Subclasses own only the math. The optimizer classes own everything around it:
    group building, state allocation (``state_spec``), batching, CPU offload, state
    quantization, and torch.compile. Register subclasses with
    :func:`register_muon_algorithm` and select them by name in a param-group dict.
    """

    name: ClassVar[str]
    # Variant-specific hyperparameters and their defaults. Overridable per param
    # group; resolved by ``Muon._build_muon_group`` and passed to ``update`` /
    # ``foreach_update`` as keyword arguments.
    options: ClassVar[dict[str, Any]] = {}
    # Per-parameter state buffers (beyond the shared ``step`` / ``lr_ratio``).
    state_spec: ClassVar[dict[str, BufferSpec]] = {
        "momentum_buffer": BufferSpec(like="grad", signed=True),
    }

    def validate_param(self, p: Tensor) -> None:
        """Reject parameters this algorithm cannot handle (called at group build)."""
        if p.ndim < 2:
            raise ValueError(
                "Muon only supports 2D+ parameters; found a 1D tensor in a Muon group"
            )
        if torch.is_complex(p):
            raise NotImplementedError(
                "Complex parameters are not supported in Muon. Add these parameters "
                "to the Adam group or use a different optimizer."
            )

    @abstractmethod
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
        """Per-tensor fp32 reference update; mutates ``param`` and ``state`` in place.

        Keep compiled math out-of-place until copying back into state (see the
        aliasing note in ``MuonBaseline.update``).
        """

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
        """Batched update over same-(device, dtype, shape) tensors.

        Default falls back to the per-tensor reference; override with
        ``torch._foreach_*`` ops for performance.
        """
        for i, (param, grad, lr_ratio) in enumerate(
            zip(params, grads, lr_ratios, strict=True)
        ):
            self.update(
                param,
                grad,
                {key: buffers[i] for key, buffers in state.items()},
                lr_ratio,
                lr=lr,
                weight_decay=weight_decay,
                cautious_wd=cautious_wd,
                momentum=momentum,
                nesterov=nesterov,
                maximize=maximize,
                ns_steps=ns_steps,
                orthogonalization_strategy=orthogonalization_strategy,
                split_sizes=split_sizes,
                **opts,
            )


_REGISTRY: dict[str, MuonAlgorithm] = {}


def register_muon_algorithm(cls: type[MuonAlgorithm]) -> type[MuonAlgorithm]:
    """Register a :class:`MuonAlgorithm` subclass under ``cls.name``.

    Usable as a class decorator. Third-party packages can register their own
    variants and select them via ``{"params": ..., "algorithm": "<name>"}``.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls.__qualname__} must define a non-empty string `name`")
    name = name.lower()
    if name in ADAM_ALGORITHMS:
        raise ValueError(f"'{name}' is reserved for the Adam path")
    if name in _REGISTRY and type(_REGISTRY[name]).__qualname__ != cls.__qualname__:
        raise ValueError(f"Muon algorithm '{name}' is already registered")
    shadowed = RESERVED_GROUP_KEYS.intersection(cls.options)
    if shadowed:
        raise ValueError(
            f"{cls.__qualname__}.options shadow reserved group keys: {sorted(shadowed)}"
        )
    _REGISTRY[name] = cls()
    return cls


def get_algorithm(name: str) -> MuonAlgorithm:
    """Look up a registered algorithm by name (case-insensitive)."""
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{name}'. Registered Muon algorithms: "
            f"{sorted(_REGISTRY)}; 'adam' and 'adamw' select the Adam path."
        ) from None


def registered_algorithms() -> tuple[str, ...]:
    """Names of all registered Muon algorithms (excludes 'adam'/'adamw')."""
    return tuple(sorted(_REGISTRY))
