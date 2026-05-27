"""cuPF: GPU-accelerated Newton-Raphson power flow solver."""

from ._cupf import (
    BackendKind,
    ComputePolicy,
    AdjointCacheMode,
    CuDSSAlgorithm,
    CuDSSOptions,
    AdjointOptions,
    AdjointResult,
    NRConfig,
    SolveOptions,
    NewtonOptions,
    NRResult,
    NewtonSolver,
)

__all__ = [
    "BackendKind",
    "ComputePolicy",
    "AdjointCacheMode",
    "CuDSSAlgorithm",
    "CuDSSOptions",
    "AdjointOptions",
    "AdjointResult",
    "NRConfig",
    "SolveOptions",
    "NewtonOptions",
    "NRResult",
    "NewtonSolver",
]

try:
    from .torch_autograd import CuPFFunction
except Exception:
    CuPFFunction = None
else:
    __all__.append("CuPFFunction")
