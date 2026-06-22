from torch import Tensor
from torch.distributed.tensor import DTensor


def to_local(tensor: Tensor | DTensor, full_tensor: bool = False) -> Tensor:
    if full_tensor:
        return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
    else:
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
