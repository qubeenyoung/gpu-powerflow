"""cuPF: GPU-accelerated Newton-Raphson power flow solver."""


def _preload_python_cuda_dependencies() -> None:
    try:
        import ctypes
        import pathlib
        import sys
    except Exception:
        return

    for base in map(pathlib.Path, sys.path):
        cublas_lt = base / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if not cublas_lt.exists():
            continue
        try:
            ctypes.CDLL(str(cublas_lt), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return


_preload_python_cuda_dependencies()

def _load_extension():
    import importlib
    import pathlib

    package_dir = pathlib.Path(__file__).resolve().parent
    local_extension = any(package_dir.glob("_cupf*.so")) or any(
        package_dir.glob("_cupf*.pyd")
    )
    if local_extension:
        return importlib.import_module("._cupf", __name__)
    try:
        return importlib.import_module("_cupf")
    except ImportError:
        return importlib.import_module("._cupf", __name__)


_cupf = _load_extension()

BackendKind = _cupf.BackendKind
ComputePolicy = _cupf.ComputePolicy
AdjointCacheMode = _cupf.AdjointCacheMode
CuDSSAlgorithm = _cupf.CuDSSAlgorithm
CuDSSOptions = _cupf.CuDSSOptions
AdjointOptions = _cupf.AdjointOptions
AdjointResult = _cupf.AdjointResult
NRConfig = _cupf.NRConfig
SolveOptions = _cupf.SolveOptions
NewtonOptions = _cupf.NewtonOptions
NRResult = _cupf.NRResult
NewtonSolver = _cupf.NewtonSolver

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
    from .torch_autograd import CuPFFunction, solve
except Exception:
    CuPFFunction = None
    solve = None
else:
    if hasattr(NewtonSolver, "solve_with_adjoint_cache_torch") and hasattr(
        NewtonSolver, "solve_adjoint_torch"
    ):
        __all__.append("CuPFFunction")
        __all__.append("solve")
    else:
        CuPFFunction = None
        solve = None
