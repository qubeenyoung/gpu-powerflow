"""cuPF: GPU-accelerated Newton-Raphson power flow solver."""

from ._cupf import (
    BackendKind,
    ComputePolicy,
    JacobianBuilderType,
    NewtonAlgorithm,
    NRConfig,
    NewtonOptions,
    NRResult,
    NewtonSolver,
)

PrecisionMode = ComputePolicy
NRResultF64 = NRResult

__all__ = [
    "BackendKind",
    "ComputePolicy",
    "PrecisionMode",
    "JacobianBuilderType",
    "NewtonAlgorithm",
    "NRConfig",
    "NewtonOptions",
    "NRResult",
    "NRResultF64",
    "NewtonSolver",
]
