"""dtensor-muon: a distributed-ready Muon optimizer built on PyTorch DTensor."""

from .optim import Muon, MuonForeach
from .orthogonalize import OrthogonalizationStrategy

__all__ = [
    "Muon",
    "MuonForeach",
    "OrthogonalizationStrategy",
]

try:
    from .optim import MuonLP

    __all__ += ["MuonLP"]
except ImportError:
    pass
