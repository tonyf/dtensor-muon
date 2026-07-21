"""
Shared orthogonalization utilities for Muon optimizer.

This module provides the dispatch logic and DTensor handling for
different orthogonalization strategies (Newton-Schulz, Polar Express).
"""

from typing import Annotated, Literal, cast

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard

from .newton_schulz import ns_loop, ns_loop_triton
from .polar_express import pe_loop, pe_loop_triton

OrthogonalizationStrategy = Literal["newton_schulz", "polar_express"]


def get_dtensor_metadata(dtensors: DTensor | list[DTensor], run_check: bool = True):
    """Extract metadata from DTensor(s) for reconstruction after operations."""
    tensor = dtensors if isinstance(dtensors, DTensor) else dtensors[0]
    metadata = {
        "device_mesh": tensor.device_mesh,
        "placements": tensor.placements,
        "shape": tensor.shape,
        "stride": tensor.stride(),
    }

    # The consistency assert builds dicts of stride SymInts, which Dynamo cannot
    # represent when this runs inside a compiled region under dynamic shapes —
    # skip the purely defensive check there.
    if run_check and isinstance(dtensors, list) and not torch.compiler.is_compiling():
        assert all(
            metadata
            == {
                "device_mesh": dtensor.device_mesh,
                "placements": dtensor.placements,
                "shape": dtensor.shape,
                "stride": dtensor.stride(),
            }
            for dtensor in dtensors
        )

    return metadata


def _get_orthogonalization_fn(strategy: OrthogonalizationStrategy, use_triton: bool):
    """Get the orthogonalization function based on strategy."""
    if strategy == "newton_schulz":
        return ns_loop_triton if use_triton else ns_loop
    elif strategy == "polar_express":
        return pe_loop_triton if use_triton else pe_loop
    else:
        raise ValueError(f"Unknown orthogonalization strategy: {strategy}")


def _validate_orthogonalization_args(
    strategy: OrthogonalizationStrategy,
    steps: int,
) -> None:
    if strategy == "polar_express":
        assert steps <= 5, (
            "polar express orthogonalization only supports up to 5 optimization steps."
        )


def zeropower(
    G: Annotated[Tensor, "N M"] | Annotated[DTensor, "N M"],
    steps: int = 5,
    eps: float = 1e-7,
    use_triton: bool = True,
    strategy: OrthogonalizationStrategy = "newton_schulz",
) -> Annotated[Tensor, "N M"] | Annotated[DTensor, "N M"]:
    """
    Compute zero-power (sign) of a single gradient tensor using orthogonalization.

    Args:
        G: Gradient tensor (2D, regular or DTensor)
        steps: Number of iteration steps for the algorithm
        eps: Small constant for numerical stability
        strategy: Orthogonalization algorithm to use:
            - "newton_schulz": Classic Newton-Schulz iteration (default)
            - "polar_express": Polar Express algorithm (arxiv.org/pdf/2505.16932)

    Returns:
        Orthogonalized tensor with same structure as input
    """
    _validate_orthogonalization_args(strategy, steps)

    # Select the orthogonalization function
    orthogonalize_fn = _get_orthogonalization_fn(strategy, use_triton)

    if isinstance(G, DTensor):
        X = G.full_tensor()
        U = orthogonalize_fn(X.bfloat16(), steps=steps, eps=eps)
        U = DTensor.from_local(
            U, device_mesh=G.device_mesh, placements=(Replicate(),) * G.device_mesh.ndim
        )
        return U.redistribute(placements=G.placements, async_op=True)
    else:
        return orthogonalize_fn(G.bfloat16(), steps=steps, eps=eps)


def foreach_zeropower(
    Gs: list[Annotated[Tensor, "N M"]]
    | list[Annotated[DTensor, "N M"]]
    | list[Annotated[Tensor, "G N M"]]
    | list[Annotated[DTensor, "G N M"]],
    steps: int = 5,
    eps: float = 1e-7,
    use_triton: bool = True,
    strategy: OrthogonalizationStrategy = "newton_schulz",
) -> (
    list[Annotated[Tensor, "N M"]]
    | list[Annotated[DTensor, "N M"]]
    | list[Annotated[Tensor, "G N M"]]
    | list[Annotated[DTensor, "G N M"]]
):
    """
    Compute zero-power (sign) of gradients using orthogonalization.

    Args:
        Gs: List of gradient tensors (2D or 3D, regular or DTensor)
        steps: Number of iteration steps for the algorithm
        eps: Small constant for numerical stability
        use_triton: Whether to use triton kernel (only for newton_schulz)
        strategy: Orthogonalization algorithm to use:
            - "newton_schulz": Classic Newton-Schulz iteration (default)
            - "polar_express": Polar Express algorithm (arxiv.org/pdf/2505.16932)

    Returns:
        List of orthogonalized tensors with same structure as input
    """
    _validate_orthogonalization_args(strategy, steps)

    orthogonalize_fn = _get_orthogonalization_fn(strategy, use_triton)

    if isinstance(Gs[0], DTensor):
        Gs = cast(list[DTensor], Gs)
        metadata = get_dtensor_metadata(Gs)
        shard_0_pl = (Shard(0), *[Replicate()] * (metadata["device_mesh"].ndim - 1))

        X = cast(DTensor, torch.stack(list(Gs), dim=0).bfloat16())

        # Distribute X chunks across devices for orthogonalization
        X_dist = X.redistribute(placements=shard_0_pl, async_op=True)
        U_dist = orthogonalize_fn(X_dist.to_local(), steps=steps, eps=eps)
        U_dist = DTensor.from_local(U_dist, **get_dtensor_metadata(X_dist))
        U = U_dist.redistribute(placements=X.placements, async_op=True)

        # Unbind back into list (on local tensors)
        U_local = U.to_local()
        Us_local = list(U_local.unbind(0))
        Us = [DTensor.from_local(u, **metadata) for u in Us_local]

        return Us

    else:
        X = torch.stack([g for g in Gs])
        Us = orthogonalize_fn(X.bfloat16(), steps=steps, eps=eps)
        return list(Us.unbind(0))


def is_fsdp_3d_sharded(xs: list[DTensor] | list[Tensor]) -> bool:
    return all(
        (
            isinstance(x, DTensor)
            and x.ndim == 3
            and sum(p == Shard(0) for p in x.placements) == 1
            and sum(p == Replicate() for p in x.placements) == len(x.placements) - 1
        )
        for x in xs
    )


def foreach_zeropower_3d_fsdp(
    Gs: list[Annotated[DTensor, "G N M"]],
    steps: int = 5,
    eps: float = 1e-7,
    use_triton: bool = True,
    strategy: OrthogonalizationStrategy = "newton_schulz",
) -> (
    list[Annotated[Tensor, "N M"]]
    | list[Annotated[DTensor, "N M"]]
    | list[Annotated[Tensor, "G N M"]]
    | list[Annotated[DTensor, "G N M"]]
):
    assert is_fsdp_3d_sharded(Gs), (
        "foreach_zeropower_3d_fsdp only works for DTensors that are *only* sharded along the first dimension"
    )

    orthogonalize_fn = _get_orthogonalization_fn(strategy, use_triton)
    metadata = get_dtensor_metadata(Gs)

    # List[B, G_shard, N, M], distributed across G shards
    X_dist = torch.stack([g.to_local().bfloat16() for g in Gs])
    U_dist = orthogonalize_fn(X_dist, steps=steps, eps=eps)

    # Unbind back into list (on local tensors)
    Us_local = list(U_dist.unbind(0))
    Us = [DTensor.from_local(u, **metadata) for u in Us_local]

    return Us
