from .algorithms import (
    BufferSpec,
    MuonAlgorithm,
    get_algorithm,
    register_muon_algorithm,
    registered_algorithms,
)
from .optim import Muon

__all__ = [
    "BufferSpec",
    "Muon",
    "MuonAlgorithm",
    "get_algorithm",
    "register_muon_algorithm",
    "registered_algorithms",
]

try:
    import torchao  # noqa: F401  # ty: ignore[unresolved-import]

    from .optim_lp import MuonLP

    __all__ += ["MuonLP"]
except ImportError:
    pass
