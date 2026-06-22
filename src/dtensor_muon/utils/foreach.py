from typing import Sequence, cast

import torch
from torch import Tensor
from torch.utils._foreach_utils import Indices


def group_tensors_by_shape(
    tensorlist: list[Tensor],
) -> dict[tuple[int, ...], tuple[list[Tensor], Indices]]:
    """Group a list of tensors by their shape.

    Returns a dict mapping each shape (as a tuple of ints) to a pair:
    (list_of_tensors_with_that_shape, list_of_original_indices).

    Skips grouping when compiling since inductor will handle this during lowering.
    """
    if torch.compiler.is_compiling():
        # Sentinel key; downstream code should treat this as a single "group all" bucket.
        return {(0, 0): (tensorlist, list(range(len(tensorlist))))}

    groups: dict[tuple[int, ...], tuple[list[Tensor], Indices]] = {}

    for idx, t in enumerate(tensorlist):
        shape_key = tuple(t.shape)  # Scalars will use an empty tuple ()
        if shape_key in groups:
            tensors, indices = groups[shape_key]
            tensors.append(t)
            indices.append(idx)
        else:
            groups[shape_key] = ([t], [idx])

    return groups


def move_tensors_to_device(
    tensors: Sequence[Tensor | None],
    device_in: torch.device,
    device_out: torch.device,
) -> list[Tensor | None]:
    if device_in.type == device_out.type:
        return cast(list[Tensor | None], tensors)
    return [t.to(device_out, non_blocking=True) if t is not None else None for t in tensors]
