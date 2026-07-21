"""muonium: a distributed-ready Muon optimizer built on PyTorch DTensor."""

from .optim import (
    BufferSpec,
    Muon,
    MuonAlgorithm,
    get_algorithm,
    register_muon_algorithm,
    registered_algorithms,
)
from .orthogonalize import OrthogonalizationStrategy

__all__ = [
    "BufferSpec",
    "Muon",
    "MuonAlgorithm",
    "OrthogonalizationStrategy",
    "get_algorithm",
    "register_muon_algorithm",
    "registered_algorithms",
]

try:
    from .optim import MuonLP

    __all__ += ["MuonLP"]
except ImportError:
    pass
