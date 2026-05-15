from __future__ import annotations

import csv
import importlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import scipy.io
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as exc:  # pragma: no cover - exercised only on missing scipy envs
    sp = None
    spla = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"


@dataclass
class CaseData:
    name: str
    path: Path
    metadata: dict[str, Any]
    ybus: Any
    indptr: np.ndarray
    indices: np.ndarray
    ybus_data: np.ndarray
    sbus: np.ndarray
    v0: np.ndarray
    pv: np.ndarray
    pq: np.ndarray

    @property
    def n_bus(self) -> int:
        return int(self.ybus.shape[0])

    @property
    def n_branch_like(self) -> int:
        for key in ("n_branch", "n_branches", "nbranch", "nl"):
            if key in self.metadata:
                return int(self.metadata[key])
        # The dump format often does not store branch rows. nnz(Ybus) is the
        # most stable topology-size proxy available in this dataset layout.
        return int(self.ybus.nnz)

    @property
    def pvpq(self) -> np.ndarray:
        return np.r_[self.pv, self.pq].astype(np.int32, copy=False)


@dataclass
class PFResult:
    v: np.ndarray
    converged: bool
    iterations: int
    final_mismatch: float


@dataclass
class NativeAdjointResult:
    grad_load_p: np.ndarray
    grad_load_q: np.ndarray
    lambda_vec: np.ndarray
    success: bool
    backend: str
    transpose_solve_backend: str
    used_adjoint_cache: bool
    adjoint_cache_matches_final_state: bool
    reused_forward_factorization: bool
    reused_final_state_factorization: bool
    refactorized_for_backward: bool
    used_explicit_transpose: bool
    jt_residual_norm: float
    sign_convention: str
    used_python_scipy: bool
    includes_host_device_transfer: bool
    zero_copy: bool
    torch_extension_zero_copy: bool
    raw_pointer_api_used: bool
    current_stream_integrated: bool
    jt_symbolic_analyzed_at_initialize: bool
    jt_values_transposed_on_device: bool
    jt_factorized_during_forward_cache: bool
    jt_refactorized_during_backward: bool
    host_roundtrip_for_jt_transpose: bool
    solve_time_ms: float
    transpose_solve_time_ms: float
    factorization_time_ms: float
    total_time_ms: float


def require_scipy() -> None:
    if sp is None or spla is None:
        raise RuntimeError(f"scipy is required for this benchmark: {_SCIPY_IMPORT_ERROR}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def dtype_to_numpy(dtype: str) -> np.dtype:
    if dtype == "float32":
        return np.dtype(np.float32)
    if dtype == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported dtype: {dtype}")


def complex_dtype(dtype: str) -> np.dtype:
    return np.dtype(np.complex64 if dtype == "float32" else np.complex128)


def list_case_dirs(
    dataset_dir: Path,
    *,
    num_datasets: int | None = None,
    case_names: Sequence[str] | None = None,
) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    if case_names:
        dirs = [dataset_dir / name for name in case_names]
    else:
        dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
        if num_datasets is not None:
            dirs = dirs[: int(num_datasets)]

    missing = [str(p) for p in dirs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing case directories: " + ", ".join(missing))
    return dirs


def load_complex_pairs(path: Path, dtype: str = "float64") -> np.ndarray:
    real_dtype = dtype_to_numpy(dtype)
    values: list[complex] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue
            real, imag = stripped.split()[:2]
            values.append(complex(float(real), float(imag)))
    return np.asarray(values, dtype=complex_dtype(dtype)).astype(
        np.result_type(real_dtype, 1j), copy=False
    )


def load_int_values(path: Path) -> np.ndarray:
    values: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue
            values.append(int(stripped.split()[0]))
    return np.asarray(values, dtype=np.int32)


def load_case(case_dir: Path, dtype: str = "float64") -> CaseData:
    require_scipy()
    case_dir = Path(case_dir)
    ybus = scipy.io.mmread(case_dir / "dump_Ybus.mtx").tocsr()
    ybus = ybus.astype(complex_dtype(dtype), copy=False)
    ybus.sort_indices()

    sbus = load_complex_pairs(case_dir / "dump_Sbus.txt", dtype=dtype)
    v0 = load_complex_pairs(case_dir / "dump_V.txt", dtype=dtype)
    pv = load_int_values(case_dir / "dump_pv.txt")
    pq = load_int_values(case_dir / "dump_pq.txt")
    metadata_path = case_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if ybus.shape[0] != ybus.shape[1]:
        raise ValueError(f"{case_dir.name}: Ybus must be square")
    if sbus.shape[0] != ybus.shape[0] or v0.shape[0] != ybus.shape[0]:
        raise ValueError(f"{case_dir.name}: Sbus/V0 length does not match Ybus")

    return CaseData(
        name=case_dir.name,
        path=case_dir,
        metadata=metadata,
        ybus=ybus,
        indptr=np.asarray(ybus.indptr, dtype=np.int32),
        indices=np.asarray(ybus.indices, dtype=np.int32),
        ybus_data=np.asarray(ybus.data, dtype=np.complex128),
        sbus=np.asarray(sbus, dtype=np.complex128),
        v0=np.asarray(v0, dtype=np.complex128),
        pv=pv,
        pq=pq,
    )


def build_jacobian(case: CaseData, v: np.ndarray) -> Any:
    require_scipy()
    ybus = case.ybus.astype(np.complex128, copy=False)
    v = np.asarray(v, dtype=np.complex128)
    i_bus = ybus @ v
    vm = np.maximum(np.abs(v), 1e-8)
    v_norm = v / vm

    diag_v = sp.diags(v, 0, format="csr")
    diag_i_bus = sp.diags(i_bus, 0, format="csr")
    diag_v_norm = sp.diags(v_norm, 0, format="csr")

    d_s_d_vm = diag_v @ np.conj(ybus @ diag_v_norm) + np.conj(diag_i_bus) @ diag_v_norm
    d_s_d_va = 1j * diag_v @ np.conj(diag_i_bus - ybus @ diag_v)

    pvpq = case.pvpq
    pq = case.pq
    j11 = d_s_d_va[pvpq, :][:, pvpq].real
    j12 = d_s_d_vm[pvpq, :][:, pq].real
    j21 = d_s_d_va[pq, :][:, pvpq].imag
    j22 = d_s_d_vm[pq, :][:, pq].imag
    return sp.vstack([sp.hstack([j11, j12]), sp.hstack([j21, j22])], format="csr")


def mismatch_vector(case: CaseData, v: np.ndarray, sbus: np.ndarray | None = None) -> np.ndarray:
    if sbus is None:
        sbus = case.sbus
    mis = v * np.conj(case.ybus @ v) - sbus
    return np.r_[mis[case.pv].real, mis[case.pq].real, mis[case.pq].imag].astype(np.float64)


def cpu_newton_pf(
    case: CaseData,
    *,
    sbus: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    tolerance: float = 1e-8,
    max_iter: int = 50,
) -> PFResult:
    require_scipy()
    sbus = case.sbus if sbus is None else np.asarray(sbus, dtype=np.complex128)
    v = np.array(case.v0 if v0 is None else v0, dtype=np.complex128, copy=True)
    va = np.angle(v)
    vm = np.abs(v)
    pvpq = case.pvpq
    n_pvpq = int(pvpq.size)

    converged = False
    final_mismatch = math.inf
    completed = 0
    for iteration in range(int(max_iter)):
        f = mismatch_vector(case, v, sbus)
        final_mismatch = float(np.max(np.abs(f))) if f.size else 0.0
        completed = iteration
        if np.isfinite(final_mismatch) and final_mismatch <= tolerance:
            converged = True
            break

        jac = build_jacobian(case, v)
        dx = spla.spsolve(jac, f)
        if not np.all(np.isfinite(dx)):
            break

        va[pvpq] -= dx[:n_pvpq]
        vm[case.pq] -= dx[n_pvpq:]
        v = vm * np.exp(1j * va)
        completed = iteration + 1

    if not converged:
        f = mismatch_vector(case, v, sbus)
        final_mismatch = float(np.max(np.abs(f))) if f.size else 0.0
        converged = bool(np.isfinite(final_mismatch) and final_mismatch <= tolerance)
    return PFResult(v=v, converged=converged, iterations=completed, final_mismatch=final_mismatch)


def deterministic_weights(n_bus: int, dtype: str = "float64") -> tuple[np.ndarray, np.ndarray]:
    real_dtype = dtype_to_numpy(dtype)
    # Smooth non-constant weights avoid cancellation while staying reproducible.
    idx = np.arange(n_bus, dtype=real_dtype)
    w_vm = (1.0 + ((idx % 17.0) / 17.0)).astype(real_dtype)
    w_va = (0.25 + ((idx % 23.0) / 31.0)).astype(real_dtype)
    return w_vm, w_va


def scalar_voltage_loss(v: np.ndarray, w_vm: np.ndarray, w_va: np.ndarray) -> float:
    return float(np.sum(w_vm * np.abs(v)) + np.sum(w_va * np.angle(v)))


def implicit_load_gradients(
    case: CaseData,
    v: np.ndarray,
    w_vm: np.ndarray,
    w_va: np.ndarray,
) -> dict[str, np.ndarray]:
    """CPU/SciPy implicit-adjoint reference.

    This is intentionally kept as a reference/debug helper only. Benchmark rows
    labelled cupf_native_adjoint or cupf_pybind must call the native cuPF
    solve_adjoint pybind API instead.
    """
    jac = build_jacobian(case, v)
    pvpq = case.pvpq
    pq = case.pq
    d_loss_dx = np.r_[w_va[pvpq], w_vm[pq]].astype(np.float64)
    lam = spla.spsolve(jac.T, d_loss_dx)

    grad_sbus_p = np.zeros(case.n_bus, dtype=np.float64)
    grad_sbus_q = np.zeros(case.n_bus, dtype=np.float64)
    grad_sbus_p[pvpq] = lam[: pvpq.size]
    grad_sbus_q[pq] = lam[pvpq.size :]

    return {
        "load_p": -grad_sbus_p,
        "load_q": -grad_sbus_q,
        "sbus_p": grad_sbus_p,
        "sbus_q": grad_sbus_q,
    }


def sample_parameter_indices(
    case: CaseData,
    parameter_name: str,
    *,
    fd_samples: int,
    full: bool,
    seed: int,
) -> np.ndarray:
    if parameter_name == "load_p":
        candidates = case.pvpq
    elif parameter_name == "load_q":
        candidates = case.pq
    else:
        raise ValueError(f"Unsupported finite-difference parameter: {parameter_name}")
    candidates = np.asarray(candidates, dtype=np.int32)
    if full or fd_samples <= 0 or fd_samples >= candidates.size:
        return np.sort(candidates)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=int(fd_samples), replace=False))


def perturb_load_sbus(case: CaseData, parameter_name: str, index: int, delta: float) -> np.ndarray:
    # load_p/load_q are demand variables, while Sbus is net injection. Increasing
    # load therefore decreases the corresponding Sbus component.
    sbus = np.array(case.sbus, dtype=np.complex128, copy=True)
    if parameter_name == "load_p":
        sbus[index] -= float(delta)
    elif parameter_name == "load_q":
        sbus[index] -= 1j * float(delta)
    else:
        raise ValueError(f"Unsupported finite-difference parameter: {parameter_name}")
    return sbus


def relative_error(actual: np.ndarray, expected: np.ndarray, eps: float) -> np.ndarray:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    denom = np.maximum(np.abs(expected), eps)
    return np.abs(actual - expected) / denom


def summarize_array(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return {
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "mean": mean(vals),
        "median": median(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def ensure_output_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=default)
        handle.write("\n")


def import_cupf(extra_python_paths: Sequence[Path] | None = None) -> Any:
    if extra_python_paths:
        for path in reversed([str(Path(p)) for p in extra_python_paths]):
            if path not in sys.path:
                sys.path.insert(0, path)
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    try:
        return importlib.import_module("cupf")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import cupf. Build/install the pybind package first, for example: "
            "CUPF_WITH_CUDA=ON python3 -m pip install -e ./cuPF. "
            "For CUDA wheels that use NVIDIA Python cuDSS/cuBLAS packages, prepend their library "
            "directories to LD_LIBRARY_PATH if libcudss/libcublas symbols are not found."
        ) from exc


def _compute_policy(cupf: Any, compute: str) -> Any:
    normalized = compute.lower()
    if normalized == "fp64":
        return cupf.ComputePolicy.FP64
    if normalized == "fp32":
        return cupf.ComputePolicy.FP32
    if normalized == "mixed":
        return cupf.ComputePolicy.Mixed
    raise ValueError(f"Unsupported cuPF compute policy: {compute}")


def cupf_compute_torch_dtype(compute: str) -> Any:
    import torch

    return torch.float64 if compute.lower() == "fp64" else torch.float32


def native_adjoint_result_from_pybind(
    result: Any,
    *,
    grad_load_p: np.ndarray | None = None,
    grad_load_q: np.ndarray | None = None,
    lambda_vec: np.ndarray | None = None,
) -> NativeAdjointResult:
    if grad_load_p is None:
        grad_load_p = np.asarray(getattr(result, "grad_load_p_numpy", []), dtype=np.float64)
    if grad_load_q is None:
        grad_load_q = np.asarray(getattr(result, "grad_load_q_numpy", []), dtype=np.float64)
    if lambda_vec is None:
        lambda_vec = np.asarray(getattr(result, "lambda_numpy", []), dtype=np.float64)
    return NativeAdjointResult(
        grad_load_p=grad_load_p,
        grad_load_q=grad_load_q,
        lambda_vec=lambda_vec,
        success=bool(result.success),
        backend=str(result.backend),
        transpose_solve_backend=str(result.transpose_solve_backend),
        used_adjoint_cache=bool(result.used_adjoint_cache),
        adjoint_cache_matches_final_state=bool(result.adjoint_cache_matches_final_state),
        reused_forward_factorization=bool(result.reused_forward_factorization),
        reused_final_state_factorization=bool(result.reused_final_state_factorization),
        refactorized_for_backward=bool(result.refactorized_for_backward),
        used_explicit_transpose=bool(result.used_explicit_transpose),
        jt_residual_norm=float(result.jt_residual_norm),
        sign_convention=str(result.sign_convention),
        used_python_scipy=bool(result.used_python_scipy),
        includes_host_device_transfer=bool(result.includes_host_device_transfer),
        zero_copy=bool(result.zero_copy),
        torch_extension_zero_copy=bool(getattr(result, "torch_extension_zero_copy", False)),
        raw_pointer_api_used=bool(getattr(result, "raw_pointer_api_used", False)),
        current_stream_integrated=bool(getattr(result, "current_stream_integrated", False)),
        jt_symbolic_analyzed_at_initialize=bool(getattr(result, "jt_symbolic_analyzed_at_initialize", False)),
        jt_values_transposed_on_device=bool(getattr(result, "jt_values_transposed_on_device", False)),
        jt_factorized_during_forward_cache=bool(getattr(result, "jt_factorized_during_forward_cache", False)),
        jt_refactorized_during_backward=bool(getattr(result, "jt_refactorized_during_backward", False)),
        host_roundtrip_for_jt_transpose=bool(getattr(result, "host_roundtrip_for_jt_transpose", False)),
        solve_time_ms=float(result.solve_time_ms),
        transpose_solve_time_ms=float(result.transpose_solve_time_ms),
        factorization_time_ms=float(result.factorization_time_ms),
        total_time_ms=float(result.total_time_ms),
    )


class CupfSolverSession:
    """cuPF pybind solver wrapper that preserves the forward state for backward."""

    def __init__(
        self,
        case: CaseData,
        *,
        cupf: Any,
        backend: str,
        compute: str,
        tolerance: float = 1e-8,
        max_iter: int = 50,
    ) -> None:
        self.case = case
        self.cupf = cupf
        self.backend = backend
        self.compute = compute
        self.cfg = cupf.NRConfig()
        self.cfg.tolerance = float(tolerance)
        self.cfg.max_iter = int(max_iter)

        opts = cupf.NewtonOptions()
        opts.backend = cupf.BackendKind.CUDA if backend == "cuda" else cupf.BackendKind.CPU
        opts.compute = _compute_policy(cupf, compute)
        self.solver = cupf.NewtonSolver(opts)
        self.solver.initialize(
            case.indptr,
            case.indices,
            case.ybus_data,
            case.n_bus,
            case.n_bus,
            case.pv,
            case.pq,
        )
        self.batch_size = 1
        self.last_pf: PFResult | None = None

    def forward(
        self,
        *,
        sbus: np.ndarray | None = None,
        v0: np.ndarray | None = None,
        batch_size: int = 1,
        prepare_adjoint_cache: bool = False,
        allow_explicit_transpose_fallback: bool | None = None,
    ) -> PFResult:
        case = self.case
        solve_options = self.cupf.SolveOptions()
        solve_options.prepare_adjoint_cache = bool(prepare_adjoint_cache)
        solve_options.allow_explicit_transpose_fallback = (
            self.backend == "cuda"
            if allow_explicit_transpose_fallback is None
            else bool(allow_explicit_transpose_fallback)
        )
        sbus_arr = np.asarray(case.sbus if sbus is None else sbus, dtype=np.complex128)
        v0_arr = np.asarray(case.v0 if v0 is None else v0, dtype=np.complex128)
        use_batch_api = int(batch_size) != 1 or sbus_arr.ndim == 2 or v0_arr.ndim == 2

        if not use_batch_api:
            result = self.solver.solve(
                case.indptr,
                case.indices,
                case.ybus_data,
                case.n_bus,
                case.n_bus,
                sbus_arr,
                v0_arr,
                case.pv,
                case.pq,
                self.cfg,
                solve_options,
            )
            v = np.asarray(getattr(result, "V_numpy", result.V), dtype=np.complex128)
            self.batch_size = 1
            self.last_pf = PFResult(
                v=v,
                converged=bool(result.converged),
                iterations=int(result.iterations),
                final_mismatch=float(result.final_mismatch),
            )
            return self.last_pf

        if sbus_arr.ndim == 2:
            if sbus_arr.shape != (int(batch_size), case.n_bus):
                raise ValueError("batched sbus must have shape [batch_size, n_bus]")
            sbus_batch = np.asarray(sbus_arr, dtype=np.complex128)
        else:
            sbus_batch = np.repeat(sbus_arr[None, :], int(batch_size), axis=0)
        if v0_arr.ndim == 2:
            if v0_arr.shape != (int(batch_size), case.n_bus):
                raise ValueError("batched v0 must have shape [batch_size, n_bus]")
            v0_batch = np.asarray(v0_arr, dtype=np.complex128)
        else:
            v0_batch = np.repeat(v0_arr[None, :], int(batch_size), axis=0)

        result = self.solver.solve_batch(
            case.indptr,
            case.indices,
            case.ybus_data,
            case.n_bus,
            case.n_bus,
            sbus_batch,
            v0_batch,
            case.pv,
            case.pq,
            self.cfg,
            solve_options,
        )
        v = np.asarray(getattr(result, "V_numpy", result.V), dtype=np.complex128).reshape(
            int(batch_size), case.n_bus
        )
        converged = np.asarray(getattr(result, "converged_numpy", result.converged), dtype=np.uint8)
        iterations = np.asarray(getattr(result, "iterations_numpy", result.iterations), dtype=np.int32)
        mismatch = np.asarray(getattr(result, "final_mismatch_numpy", result.final_mismatch), dtype=np.float64)
        self.batch_size = int(batch_size)
        self.last_pf = PFResult(
            v=v,
            converged=bool(np.all(converged != 0)),
            iterations=int(np.max(iterations)) if iterations.size else 0,
            final_mismatch=float(np.max(np.abs(mismatch))) if mismatch.size else math.nan,
        )
        return self.last_pf

    def solve_native_adjoint(
        self,
        *,
        grad_vm: np.ndarray,
        grad_va: np.ndarray,
        reuse_forward_factorization: bool = False,
        require_cached_factorization: bool = True,
        allow_refactorize_for_backward: bool = False,
        allow_explicit_transpose_fallback: bool | None = None,
        check_residual: bool = True,
    ) -> NativeAdjointResult:
        if self.last_pf is None:
            raise RuntimeError("CupfSolverSession.solve_native_adjoint requires a prior forward() call")
        if not hasattr(self.cupf, "AdjointOptions"):
            raise RuntimeError("Installed cupf package does not expose native AdjointOptions/solve_adjoint")

        opts = self.cupf.AdjointOptions()
        opts.reuse_forward_factorization = bool(reuse_forward_factorization)
        opts.allow_refactorize = bool(allow_refactorize_for_backward)
        opts.require_cached_factorization = bool(require_cached_factorization)
        opts.allow_refactorize_for_backward = bool(allow_refactorize_for_backward)
        opts.allow_explicit_transpose_fallback = (
            self.backend == "cuda"
            if allow_explicit_transpose_fallback is None
            else bool(allow_explicit_transpose_fallback)
        )
        opts.compute_load_gradients = True
        opts.check_residual = bool(check_residual)

        grad_va_arr = np.asarray(grad_va, dtype=np.float64)
        grad_vm_arr = np.asarray(grad_vm, dtype=np.float64)
        result = self.solver.solve_adjoint(grad_va_arr, grad_vm_arr, self.case.pv, self.case.pq, opts)

        grad_load_p = np.asarray(result.grad_load_p_numpy, dtype=np.float64)
        grad_load_q = np.asarray(result.grad_load_q_numpy, dtype=np.float64)
        lambda_vec = np.asarray(result.lambda_numpy, dtype=np.float64)
        return native_adjoint_result_from_pybind(
            result,
            grad_load_p=grad_load_p,
            grad_load_q=grad_load_q,
            lambda_vec=lambda_vec,
        )

    def solve_native_adjoint_raw_cuda(
        self,
        *,
        grad_vm_tensor: Any,
        grad_va_tensor: Any,
        grad_load_p_out: Any,
        grad_load_q_out: Any,
        require_cached_factorization: bool = True,
    ) -> NativeAdjointResult:
        if self.last_pf is None:
            raise RuntimeError("CupfSolverSession.solve_native_adjoint_raw_cuda requires a prior forward() call")
        for name, tensor in (
            ("grad_vm_tensor", grad_vm_tensor),
            ("grad_va_tensor", grad_va_tensor),
            ("grad_load_p_out", grad_load_p_out),
            ("grad_load_q_out", grad_load_q_out),
        ):
            if not getattr(tensor, "is_cuda", False):
                raise ValueError(f"{name} must be a CUDA tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if grad_vm_tensor.shape != grad_va_tensor.shape:
            raise ValueError("grad_vm_tensor and grad_va_tensor shapes must match")
        if grad_load_p_out.shape != grad_va_tensor.shape or grad_load_q_out.shape != grad_va_tensor.shape:
            raise ValueError("gradient output tensors must match grad input shape")
        if len(grad_va_tensor.shape) != 2 or grad_va_tensor.shape[1] != self.case.n_bus:
            raise ValueError("grad tensors must have shape [batch_size, n_bus]")
        if str(grad_va_tensor.dtype).endswith("float64"):
            dtype_name = "float64"
        elif str(grad_va_tensor.dtype).endswith("float32"):
            dtype_name = "float32"
        else:
            raise ValueError("raw CUDA adjoint supports float32 or float64 tensors")

        opts = self.cupf.AdjointOptions()
        opts.require_cached_factorization = bool(require_cached_factorization)
        opts.allow_refactorize = False
        opts.allow_refactorize_for_backward = False
        opts.allow_explicit_transpose_fallback = True
        raw_method = getattr(self.solver, "solve_adjoint_cuda_raw_unsafe", self.solver.solve_adjoint_cuda_raw)
        result = raw_method(
            int(grad_va_tensor.data_ptr()),
            int(grad_vm_tensor.data_ptr()),
            int(grad_load_p_out.data_ptr()),
            int(grad_load_q_out.data_ptr()),
            int(grad_va_tensor.shape[0]),
            int(grad_va_tensor.shape[1]),
            dtype_name,
            opts,
        )
        return native_adjoint_result_from_pybind(
            result,
            grad_load_p=np.empty((0,), dtype=np.float64),
            grad_load_q=np.empty((0,), dtype=np.float64),
            lambda_vec=np.empty((0,), dtype=np.float64),
        )

    def make_torch_static_inputs(self, *, batch_size: int, device: Any) -> dict[str, Any]:
        import torch

        dtype = cupf_compute_torch_dtype(self.compute)
        case = self.case
        return {
            "sbus_base_re": torch.as_tensor(case.sbus.real, dtype=dtype, device=device).contiguous(),
            "sbus_base_im": torch.as_tensor(case.sbus.imag, dtype=dtype, device=device).contiguous(),
            "v0_va": torch.as_tensor(np.angle(case.v0), dtype=dtype, device=device).contiguous(),
            "v0_vm": torch.as_tensor(np.abs(case.v0), dtype=dtype, device=device).contiguous(),
            "load_p": torch.zeros((int(batch_size), case.n_bus), dtype=dtype, device=device),
            "load_q": torch.zeros((int(batch_size), case.n_bus), dtype=dtype, device=device),
            "va_out": torch.empty((int(batch_size), case.n_bus), dtype=dtype, device=device),
            "vm_out": torch.empty((int(batch_size), case.n_bus), dtype=dtype, device=device),
        }

    def forward_torch_extension(
        self,
        tensors: dict[str, Any],
        *,
        prepare_adjoint_cache: bool = False,
        allow_explicit_transpose_fallback: bool | None = None,
    ) -> Any:
        if not hasattr(self.solver, "solve_with_adjoint_cache_torch"):
            raise RuntimeError("Installed cupf package does not expose torch extension zero-copy API")
        solve_options = self.cupf.SolveOptions()
        solve_options.prepare_adjoint_cache = bool(prepare_adjoint_cache)
        solve_options.allow_explicit_transpose_fallback = (
            self.backend == "cuda"
            if allow_explicit_transpose_fallback is None
            else bool(allow_explicit_transpose_fallback)
        )
        result = self.solver.solve_with_adjoint_cache_torch(
            tensors["sbus_base_re"],
            tensors["sbus_base_im"],
            tensors["load_p"],
            tensors["load_q"],
            tensors["v0_va"],
            tensors["v0_vm"],
            tensors["va_out"],
            tensors["vm_out"],
            self.cfg,
            solve_options,
        )
        self.batch_size = int(tensors["load_p"].shape[0])
        self.last_pf = PFResult(
            v=np.empty((0,), dtype=np.complex128),
            converged=bool(result.success),
            iterations=0,
            final_mismatch=math.nan,
        )
        return result

    def solve_native_adjoint_torch_extension(
        self,
        *,
        grad_vm_tensor: Any,
        grad_va_tensor: Any,
        grad_load_p_out: Any,
        grad_load_q_out: Any,
        require_cached_factorization: bool = True,
        check_residual: bool = False,
        copy_to_numpy: bool = False,
    ) -> NativeAdjointResult:
        if not hasattr(self.solver, "solve_adjoint_torch"):
            raise RuntimeError("Installed cupf package does not expose solve_adjoint_torch")
        opts = self.cupf.AdjointOptions()
        opts.require_cached_factorization = bool(require_cached_factorization)
        opts.allow_refactorize = False
        opts.allow_refactorize_for_backward = False
        opts.allow_explicit_transpose_fallback = True
        opts.compute_load_gradients = True
        opts.check_residual = bool(check_residual)
        result = self.solver.solve_adjoint_torch(
            grad_va_tensor,
            grad_vm_tensor,
            grad_load_p_out,
            grad_load_q_out,
            opts,
        )
        if copy_to_numpy:
            grad_p = grad_load_p_out.detach().cpu().numpy().astype(np.float64, copy=False)
            grad_q = grad_load_q_out.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            grad_p = np.empty((0,), dtype=np.float64)
            grad_q = np.empty((0,), dtype=np.float64)
        return native_adjoint_result_from_pybind(
            result,
            grad_load_p=grad_p,
            grad_load_q=grad_q,
            lambda_vec=np.empty((0,), dtype=np.float64),
        )


def run_cupf_pf(
    case: CaseData,
    *,
    cupf: Any,
    backend: str,
    compute: str,
    sbus: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    tolerance: float = 1e-8,
    max_iter: int = 50,
    batch_size: int = 1,
) -> PFResult:
    opts = cupf.NewtonOptions()
    opts.backend = cupf.BackendKind.CUDA if backend == "cuda" else cupf.BackendKind.CPU
    opts.compute = _compute_policy(cupf, compute)

    cfg = cupf.NRConfig()
    cfg.tolerance = float(tolerance)
    cfg.max_iter = int(max_iter)

    solver = cupf.NewtonSolver(opts)
    solver.initialize(case.indptr, case.indices, case.ybus_data, case.n_bus, case.n_bus, case.pv, case.pq)

    sbus_arr = np.asarray(case.sbus if sbus is None else sbus, dtype=np.complex128)
    v0_arr = np.asarray(case.v0 if v0 is None else v0, dtype=np.complex128)
    use_batch_api = batch_size != 1 or sbus_arr.ndim == 2 or v0_arr.ndim == 2
    if not use_batch_api:
        result = solver.solve(
            case.indptr,
            case.indices,
            case.ybus_data,
            case.n_bus,
            case.n_bus,
            sbus_arr,
            v0_arr,
            case.pv,
            case.pq,
            cfg,
        )
        v = np.asarray(getattr(result, "V_numpy", result.V), dtype=np.complex128)
        return PFResult(
            v=v,
            converged=bool(result.converged),
            iterations=int(result.iterations),
            final_mismatch=float(result.final_mismatch),
        )

    if sbus_arr.ndim == 2:
        if sbus_arr.shape != (int(batch_size), case.n_bus):
            raise ValueError("batched sbus must have shape [batch_size, n_bus]")
        sbus_batch = np.asarray(sbus_arr, dtype=np.complex128)
    else:
        sbus_batch = np.repeat(sbus_arr[None, :], int(batch_size), axis=0)
    if v0_arr.ndim == 2:
        if v0_arr.shape != (int(batch_size), case.n_bus):
            raise ValueError("batched v0 must have shape [batch_size, n_bus]")
        v0_batch = np.asarray(v0_arr, dtype=np.complex128)
    else:
        v0_batch = np.repeat(v0_arr[None, :], int(batch_size), axis=0)
    result = solver.solve_batch(
        case.indptr,
        case.indices,
        case.ybus_data,
        case.n_bus,
        case.n_bus,
        sbus_batch,
        v0_batch,
        case.pv,
        case.pq,
        cfg,
    )
    v = np.asarray(getattr(result, "V_numpy", result.V), dtype=np.complex128).reshape(batch_size, case.n_bus)
    converged = np.asarray(getattr(result, "converged_numpy", result.converged), dtype=np.uint8)
    iterations = np.asarray(getattr(result, "iterations_numpy", result.iterations), dtype=np.int32)
    mismatch = np.asarray(getattr(result, "final_mismatch_numpy", result.final_mismatch), dtype=np.float64)
    return PFResult(
        v=v,
        converged=bool(np.all(converged != 0)),
        iterations=int(np.max(iterations)) if iterations.size else 0,
        final_mismatch=float(np.max(np.abs(mismatch))) if mismatch.size else math.nan,
    )


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)
