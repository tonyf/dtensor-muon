from .optim import Muon
from .optim_foreach import MuonForeach

__all__ = ["Muon", "MuonForeach"]

try:
    import torchao  # noqa: F401  # ty: ignore[unresolved-import]

    from .optim_lp import MuonLP

    __all__ += ["MuonLP"]
except ImportError:
    pass
