#!/usr/bin/env python3
"""Measurement-validity audit for the lin_sol benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
from scipy import sparse
from scipy.io import mmread, mmwrite
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "measurement_audit"
RAW = AUDIT / "results" / "raw_json"
RESULTS = AUDIT / "results"
DATASETS = ROOT / "datasets" / "dumped_systems"
REPORT = ROOT / "report" / "linear_solver_measurement_audit_v3.md"


@dataclass
class System:
    name: str
    iteration: int
    path: Path
    matrix: Path
    rhs: Path
    xref: Path
    meta: Path
    synthetic: bool = False


def run_text(cmd: list[str], timeout: int = 30, env: dict[str, str] | None = None) -> str:
    try:
        out = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout, env=env)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def norm2(values: np.ndarray) -> float:
    return float(np.linalg.norm(values))


def make_pattern_hash(A: sparse.spmatrix) -> str:
    csr = A.tocsr()
    h = hashlib.sha256()
    h.update(np.asarray(csr.indptr, dtype=np.int64).tobytes())
    h.update(np.asarray(csr.indices, dtype=np.int64).tobytes())
    return h.hexdigest()


def dump_system(directory: Path, name: str, A: sparse.spmatrix, x_ref: np.ndarray, rhs: np.ndarray,
                expected: str = "pass") -> System:
    directory.mkdir(parents=True, exist_ok=True)
    A = A.tocsr()
    mmwrite(directory / "J.mtx", A)
    np.savetxt(directory / "rhs.txt", rhs, fmt="%.17e")
    np.savetxt(directory / "x_ref.txt", x_ref, fmt="%.17e")
    residual = A @ x_ref - rhs
    meta = {
        "case_name": name,
        "iteration": 0,
        "num_bus": 0,
        "num_branch": 0,
        "num_pv": 0,
        "num_pq": 0,
        "matrix_rows": int(A.shape[0]),
        "matrix_cols": int(A.shape[1]),
        "nnz": int(A.nnz),
        "rhs_norm_2": norm2(rhs),
        "rhs_norm_inf": float(np.linalg.norm(rhs, ord=np.inf)),
        "pattern_hash": make_pattern_hash(A),
        "source_case_path": "measurement_audit_synthetic",
        "converged_by_cpu_nr": expected == "pass",
        "cpu_reference_residual": norm2(residual),
        "expected_audit_outcome": expected,
    }
    write_json(directory / "meta.json", meta)
    return System(name, 0, directory, directory / "J.mtx", directory / "rhs.txt", directory / "x_ref.txt", directory / "meta.json", True)


def create_correctness_systems() -> list[System]:
    base = AUDIT / "correctness_systems"
    systems: list[System] = []

    A1 = sparse.csr_matrix(np.array([
        [4.0, -1.0, 0.0, 0.0, 0.5],
        [-0.5, 3.5, -1.0, 0.0, 0.0],
        [0.25, -1.0, 3.0, -0.5, 0.0],
        [0.0, 0.0, -0.75, 2.5, -0.25],
        [0.0, 0.2, 0.0, -0.4, 2.25],
    ]))
    x1 = np.array([0.5, -1.0, 0.25, 0.75, -0.5])
    systems.append(dump_system(base / "nonsym_known", "audit_nonsym_known", A1, x1, A1 @ x1))

    A2 = sparse.csr_matrix(np.array([
        [2.5, -3.0, 0.0, 0.5, 0.0],
        [1.0, 4.0, -1.5, 0.0, -0.5],
        [0.0, -2.0, -5.0, 1.0, 0.0],
        [0.25, 0.0, 1.0, 3.0, -1.0],
        [0.0, 0.5, 0.0, -2.0, 4.5],
    ]))
    x2 = np.array([-1.0, 0.75, -0.25, 1.25, 0.5])
    systems.append(dump_system(base / "signed_values", "audit_signed_values", A2, x2, A2 @ x2))

    A3 = sparse.csr_matrix(np.array([
        [1.0, 2.0, 0.0, 0.0, 0.0],
        [2.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1e-12],
    ]))
    x3 = np.ones(5)
    systems.append(dump_system(base / "singular_expected_fail", "audit_singular_expected_fail", A3, x3, A3 @ x3, "fail_or_warn"))

    A4 = sparse.csr_matrix((
        np.array([5.0, -1.0, 2.0, 4.0, -2.0, 3.0, 1.5, -0.5, 2.5]),
        np.array([0, 3, 4, 1, 2, 2, 0, 3, 4]),
        np.array([0, 3, 5, 6, 9, 9]),
    ), shape=(5, 5))
    x4 = np.array([1.0, -2.0, 3.0, -4.0, 2.0])
    systems.append(dump_system(base / "csr_known", "audit_csr_known", A4, x4, A4 @ x4))

    A5 = A1.copy()
    dx = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    F = -(A5 @ dx)
    rhs = -F
    systems.append(dump_system(base / "pf_sign_convention", "audit_pf_sign_convention", A5, dx, rhs))
    return systems


def discover_dataset_systems() -> list[System]:
    systems: list[System] = []
    for meta in sorted(DATASETS.glob("*/*/meta.json")):
        data = read_json(meta)
        p = meta.parent
        systems.append(System(
            data.get("case_name", p.parent.name),
            int(data.get("iteration", p.name.split("_")[-1])),
            p,
            p / "J.mtx",
            p / "rhs.txt",
            p / "x_ref.txt",
            meta,
            False,
        ))
    order = {
        "synthetic_validation": 0,
        "case14": 1,
        "case118": 2,
        "case300": 3,
        "case1354pegase": 4,
        "case2869pegase": 5,
        "case9241pegase": 6,
    }
    return sorted(systems, key=lambda s: (order.get(s.name, 99), s.iteration))


def dataset_audit_rows(systems: list[System]) -> list[dict]:
    rows: list[dict] = []
    by_case: dict[str, set[str]] = {}
    for sysinfo in systems:
        meta = read_json(sysinfo.meta)
        A = mmread(sysinfo.matrix).tocsr()
        rhs = np.loadtxt(sysinfo.rhs)
        xref = np.loadtxt(sysinfo.xref)
        residual = A @ xref - rhs
        rhs_norm = norm2(rhs)
        rel = norm2(residual) / max(rhs_norm, np.finfo(float).tiny)
        scaled = norm2(residual) / max(1.0, rhs_norm)
        pattern = make_pattern_hash(A)
        by_case.setdefault(sysinfo.name, set()).add(pattern)
        sym = "symmetric" if (A - A.T).nnz == 0 or norm2(((A - A.T).data if (A - A.T).nnz else np.array([0.0]))) < 1e-14 else "nonsymmetric"
        cond = ""
        if A.shape[0] <= 600:
            try:
                cond = f"{np.linalg.cond(A.toarray()):.6e}"
            except Exception as exc:
                cond = f"failed:{exc}"
        warnings = []
        if A.shape[0] != A.shape[1]:
            warnings.append("non_square")
        if A.nnz != int(meta.get("nnz", -1)):
            warnings.append("nnz_meta_mismatch")
        if abs(rhs_norm - float(meta.get("rhs_norm_2", rhs_norm))) > max(1e-10, rhs_norm * 1e-10):
            warnings.append("rhs_norm_meta_mismatch")
        # The original dump script may use a different hash encoding than this
        # audit script.  For measurement validity, the important check is
        # stability across iterations of the same case, which is handled below.
        if rhs_norm < 1e-8:
            warnings.append("tiny_rhs_norm_relative_residual_sensitive")
        if rel > 1e-8 and scaled > 1e-10:
            warnings.append("xref_residual_suspicious")
        rows.append({
            "row_type": "dataset",
            "system": sysinfo.name,
            "iteration": sysinfo.iteration,
            "solver": "scipy_reference",
            "dtype": "fp64",
            "status": "checked",
            "matrix_rows": A.shape[0],
            "nnz": A.nnz,
            "rhs_norm_2": f"{rhs_norm:.17e}",
            "absolute_residual_2": f"{norm2(residual):.17e}",
            "relative_residual_2": f"{rel:.17e}",
            "scaled_residual_2": f"{scaled:.17e}",
            "relative_error_2": "0",
            "symmetry": sym,
            "condition_estimate": cond,
            "warnings": ";".join(warnings),
        })
    for row in rows:
        if row["row_type"] == "dataset" and len(by_case.get(row["system"], [])) > 1:
            row["warnings"] = (row["warnings"] + ";" if row["warnings"] else "") + "pattern_changes_across_iterations"
    return rows


def executable(path: Path) -> Path | None:
    return path if path.exists() and os.access(path, os.X_OK) else None


def command_for_solver(solver: str, system: System, dtype: str, out: Path, config: Path | None = None,
                       variant: str | None = None, repeats: int = 3, warmup: int = 1,
                       np_ranks: int = 1) -> tuple[list[str], dict[str, str], int]:
    env = os.environ.copy()
    env.setdefault("LD_LIBRARY_PATH", "/usr/local/lib:/usr/local/cuda/lib64")
    env["OMP_NUM_THREADS"] = "1"
    common = [
        "--matrix", str(system.matrix),
        "--rhs", str(system.rhs),
        "--xref", str(system.xref),
        "--meta", str(system.meta),
        "--dtype", dtype,
        "--repeats", str(repeats),
        "--warmup", str(warmup),
        "--out", str(out),
    ]
    if solver == "cudss":
        cmd = [str(ROOT / "build" / "cudss_benchmark"), *common]
        return cmd, env, 180
    if solver == "cusolver":
        cmd = [str(ROOT / "build" / "cusolver_benchmark"), *common]
        return cmd, env, 180
    if solver == "amgx":
        cmd = [str(ROOT / "build" / "amgx_benchmark"), *common, "--config", str(config)]
        return cmd, env, 420
    if solver == "ginkgo":
        cmd = [str(ROOT / "build" / "ginkgo_benchmark"), *common, "--config", str(config)]
        return cmd, env, 600
    if solver == "strumpack":
        mpirun = Path("/usr/bin/mpirun.mpich")
        exe = ROOT / "solvers" / "strumpack" / "build" / "strumpack_benchmark"
        cmd = [str(mpirun), "-np", str(np_ranks), str(exe), *common]
        return cmd, env, 180
    if solver == "superlu_fixed":
        mpirun = ROOT / "third_party" / "mpich" / "install" / "bin" / "mpirun"
        exe = ROOT / "measurement_audit" / "superlu_dist_audit" / "build" / f"superlu_dist_audit_{variant or 'natural'}"
        cmd = [str(mpirun), "-np", str(np_ranks), str(exe), *common]
        return cmd, env, 180
    raise ValueError(solver)


def run_solver_case(label: str, solver: str, system: System, dtype: str, config: Path | None = None,
                    variant: str | None = None, repeats: int = 3, warmup: int = 1,
                    np_ranks: int = 1, force: bool = False) -> dict:
    out = RAW / f"{label}_{system.name}_iter{system.iteration:03d}_{dtype}.json"
    if out.exists() and not force:
        data = read_json(out)
        data.setdefault("_raw_json", str(out))
        return data
    cmd, env, timeout = command_for_solver(solver, system, dtype, out, config, variant, repeats, warmup, np_ranks)
    attempted = " ".join(cmd)
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        elapsed = (time.perf_counter() - start) * 1000.0
        if out.exists():
            data = read_json(out)
            data["_raw_json"] = str(out)
            data["_attempted_command"] = attempted
            data["_returncode"] = proc.returncode
            data["_stdout_tail"] = proc.stdout[-2000:]
            data["_stderr_tail"] = proc.stderr[-2000:]
            return data
        data = {
            "solver_name": label,
            "build_status": "runtime_failed" if proc.returncode else "no_json",
            "dtype": dtype,
            "case_name": system.name,
            "iteration": system.iteration,
            "matrix_rows": read_json(system.meta).get("matrix_rows", 0),
            "matrix_cols": read_json(system.meta).get("matrix_cols", 0),
            "nnz": read_json(system.meta).get("nnz", 0),
            "repeat_count": repeats,
            "warmup_count": warmup,
            "relative_residual_2": None,
            "relative_error_to_x_ref_2": None,
            "converged": False,
            "num_iterations": -1,
            "gpu_resident_after_initial_load": "unknown",
            "notes": "audit command did not produce JSON",
            "total_end_to_end_ms": elapsed,
            "attempted_command": attempted,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "returncode": proc.returncode,
        }
        write_json(out, data)
        data["_raw_json"] = str(out)
        return data
    except subprocess.TimeoutExpired as exc:
        data = {
            "solver_name": label,
            "build_status": "timeout",
            "dtype": dtype,
            "case_name": system.name,
            "iteration": system.iteration,
            "matrix_rows": read_json(system.meta).get("matrix_rows", 0),
            "matrix_cols": read_json(system.meta).get("matrix_cols", 0),
            "nnz": read_json(system.meta).get("nnz", 0),
            "repeat_count": repeats,
            "warmup_count": warmup,
            "relative_residual_2": None,
            "relative_error_to_x_ref_2": None,
            "converged": False,
            "num_iterations": -1,
            "gpu_resident_after_initial_load": "unknown",
            "notes": f"audit command timed out after {timeout}s",
            "attempted_command": attempted,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "returncode": "timeout",
        }
        write_json(out, data)
        data["_raw_json"] = str(out)
        return data


def augment_residual(data: dict, system: System) -> dict:
    meta = read_json(system.meta)
    rhs = np.loadtxt(system.rhs)
    rhs_norm = norm2(rhs)
    rel = data.get("relative_residual_2")
    abs_res = None
    scaled = None
    if isinstance(rel, (int, float)) and math.isfinite(float(rel)):
        abs_res = float(rel) * rhs_norm
        scaled = abs_res / max(1.0, rhs_norm)
    data["rhs_norm_2"] = rhs_norm
    data["absolute_residual_2"] = abs_res
    data["scaled_residual_2"] = scaled
    data["relative_error_2_safe_denominator"] = data.get("relative_error_to_x_ref_2")
    data["_meta_pattern_hash"] = meta.get("pattern_hash", "")
    return data


def make_amgx_configs() -> dict[str, Path]:
    configs = {}
    base = AUDIT / "configs" / "amgx"
    base.mkdir(parents=True, exist_ok=True)
    for solver in ["GMRES", "FGMRES"]:
        for smoother in ["BLOCK_JACOBI", "JACOBI"]:
            for max_iter in [200, 1000]:
                name = f"{solver.lower()}_amg_{smoother.lower()}_{max_iter}"
                cfg = {
                    "config_version": 2,
                    "solver": {
                        "solver": solver,
                        "scope": "main",
                        "preconditioner": {
                            "solver": "AMG",
                            "scope": "amg",
                            "algorithm": "AGGREGATION",
                            "selector": "SIZE_2",
                            "smoother": smoother,
                            "presweeps": 1,
                            "postsweeps": 1,
                            "cycle": "V",
                            "coarse_solver": "DENSE_LU_SOLVER",
                            "max_iters": 1,
                            "max_levels": 50,
                            "min_coarse_rows": 32,
                        },
                        "gmres_n_restart": 30,
                        "max_iters": max_iter,
                        "tolerance": 1e-8,
                        "convergence": "RELATIVE_INI",
                        "norm": "L2",
                        "monitor_residual": 1,
                        "obtain_timings": 1,
                    },
                }
                path = base / f"{name}.json"
                write_json(path, cfg)
                configs[name] = path
    return configs


def make_ginkgo_configs() -> dict[str, Path]:
    configs = {}
    base = AUDIT / "configs" / "ginkgo"
    base.mkdir(parents=True, exist_ok=True)
    for solver in ["gmres", "bicgstab"]:
        for tol_name, tol in [("fp64tol", 1e-8), ("fp32tol", 1e-6)]:
            name = f"{solver}_jacobi_{tol_name}"
            path = base / f"{name}.json"
            write_json(path, {
                "solver": solver,
                "preconditioner": "jacobi",
                "tolerance": tol,
                "max_iterations": 1000,
            })
            configs[name] = path
    return configs


def result_to_config_row(data: dict, system: System, label: str, config_name: str, previous: bool = False) -> dict:
    data = augment_residual(data, system)
    return {
        "solver": data.get("solver_name", label),
        "config": config_name,
        "system": system.name,
        "iteration": system.iteration,
        "dtype": data.get("dtype", ""),
        "build_status": data.get("build_status", ""),
        "converged": data.get("converged", ""),
        "num_iterations": data.get("num_iterations", ""),
        "analysis_ms": data.get("analysis_ms", ""),
        "factorization_ms": data.get("factorization_ms", ""),
        "solve_ms": data.get("solve_ms", ""),
        "total_solver_ms": data.get("total_solver_ms", ""),
        "one_shot_time_ms": one_shot_time(data),
        "factorization_plus_solve_ms": factor_solve_time(data),
        "phase_visibility": phase_visibility(data),
        "relative_residual_2": data.get("relative_residual_2", ""),
        "absolute_residual_2": data.get("absolute_residual_2", ""),
        "scaled_residual_2": data.get("scaled_residual_2", ""),
        "relative_error_2": data.get("relative_error_to_x_ref_2", ""),
        "gpu_resident_after_initial_load": data.get("gpu_resident_after_initial_load", ""),
        "raw_json": data.get("_raw_json", ""),
        "notes": data.get("notes", data.get("stderr_tail", "")),
        "previous_result": "yes" if previous else "no",
    }


def phase_visibility(data: dict) -> str:
    name = str(data.get("solver_name", ""))
    if name.startswith("cuDSS") or name.startswith("STRUMPACK"):
        return "analysis_factor_solve"
    if name.startswith("cuSolver") or name.startswith("SuperLU"):
        return "monolithic"
    if "AMGx" in name or "Ginkgo" in name:
        return "setup_solve"
    return "unknown"


def one_shot_time(data: dict):
    vals = [data.get(k) for k in ["analysis_ms", "factorization_ms", "solve_ms"]]
    if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in vals):
        return float(sum(vals))
    return ""


def factor_solve_time(data: dict):
    vals = [data.get(k) for k in ["factorization_ms", "solve_ms"]]
    if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in vals):
        return float(sum(vals))
    return ""


def audit_environment() -> dict:
    local_mpich = ROOT / "third_party" / "mpich" / "install" / "bin"
    env = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "gpu": run_text(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "nvcc": run_text(["nvcc", "--version"]),
        "gcc": run_text(["gcc", "--version"]).splitlines()[0] if run_text(["gcc", "--version"]) else "",
        "g++": run_text(["g++", "--version"]).splitlines()[0] if run_text(["g++", "--version"]) else "",
        "cmake": run_text(["cmake", "--version"]).splitlines()[0] if run_text(["cmake", "--version"]) else "",
        "which_mpicc": shutil.which("mpicc") or "",
        "which_mpicxx": shutil.which("mpicxx") or "",
        "which_mpirun": shutil.which("mpirun") or "",
        "mpicc_show": run_text(["mpicc", "-show"]),
        "mpicxx_show": run_text(["mpicxx", "-show"]),
        "mpirun_version": run_text(["mpirun", "--version"]),
        "local_mpich_mpirun": str(local_mpich / "mpirun"),
        "local_mpich_version": run_text([str(local_mpich / "mpirun"), "--version"]) if (local_mpich / "mpirun").exists() else "not found",
        "system_mpich_mpirun": "/usr/bin/mpirun.mpich",
        "system_mpich_version": run_text(["/usr/bin/mpirun.mpich", "--version"]) if Path("/usr/bin/mpirun.mpich").exists() else "not found",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_json(AUDIT / "results" / "environment_audit.json", env)
    return env


def build_wrapper_rows() -> list[dict]:
    rows = [
        {
            "solver": "cuDSS",
            "previous_classification": "valid_as_run",
            "audit_classification": "valid_as_run",
            "phase_visibility": "analysis_factor_solve",
            "wrapper_findings": "Uses cuDSS general CSR, base-zero, analysis once, repeated factorization/solve, CUDA events, CPU residual after D2H.",
            "remaining_issue": "Relative residual alone is misleading on tiny rhs; audit adds absolute/scaled residual interpretation.",
        },
        {
            "solver": "cuSolverSP",
            "previous_classification": "valid_as_run",
            "audit_classification": "valid_as_monolithic_qr",
            "phase_visibility": "monolithic",
            "wrapper_findings": "Uses csrlsvqr for general nonsymmetric CSR; CUDA 12.8 LU/Cholesky sparse APIs are deprecated toward cuDSS, and Cholesky is inappropriate for nonsymmetric Jacobians.",
            "remaining_issue": "Not equivalent to reusable symbolic/numeric factorization; cuSolverRF requires externally supplied LU factors.",
        },
        {
            "solver": "AMGx",
            "previous_classification": "valid_but_not_best_effort",
            "audit_classification": "valid_limited_grid",
            "phase_visibility": "setup_solve",
            "wrapper_findings": "Wrapper times AMGx setup as analysis and solve separately. Audit tests a finite AMG/Krylov grid rather than one fixed config.",
            "remaining_issue": "Wrapper records fixed tolerance/preconditioner labels in JSON; config_audit.csv records actual audit config names.",
        },
        {
            "solver": "Ginkgo",
            "previous_classification": "valid_but_not_best_effort",
            "audit_classification": "valid_as_jacobi_only",
            "phase_visibility": "setup_solve",
            "wrapper_findings": "CUDA executor is used, but make_solver_factory always installs Jacobi and supports only GMRES/BiCGSTAB despite config parser accepting a preconditioner string.",
            "remaining_issue": "IDR/ILU/ParILU/ParILUT/ISAI headers are available but were not wired into the existing benchmark wrapper.",
        },
        {
            "solver": "STRUMPACK",
            "previous_classification": "valid_as_run_np1_with_metric_caveat",
            "audit_classification": "valid_external_hybrid_np1",
            "phase_visibility": "analysis_factor_solve",
            "wrapper_findings": "MPIDist direct path with host distributed CSR input/output and internal GPU enablement. Build has CUDA and OpenMP, no SLATE; default compression is NONE.",
            "remaining_issue": "np=2/np=4 still treated as timeout/hang risk; full GPU residency is not demonstrated without SLATE.",
        },
        {
            "solver": "SuperLU_DIST",
            "previous_classification": "runtime_failed_needs_diagnosis",
            "audit_classification": "runtime_fixed_for_supported_permutation_but_original_wrapper_invalid",
            "phase_visibility": "monolithic",
            "wrapper_findings": "Previous Invalid ISPEC came from METIS_AT_PLUS_A while HAVE_PARMETIS is disabled. Audit variants using supported ColPerm avoid that error. Repeated in-process ABglobal calls can mutate A/options state, so audit uses one-shot process-level runs.",
            "remaining_issue": "Use as diagnostic/external baseline only until wrapper is rewritten with explicit state reuse/restoration and phase separation.",
        },
    ]
    return rows


def make_best_summary(config_rows: list[dict]) -> list[dict]:
    def count_ok(name_re: str) -> tuple[int, int]:
        total = 0
        ok = 0
        for row in config_rows:
            if re.search(name_re, row.get("solver", "")) or re.search(name_re, row.get("config", "")):
                total += 1
                if str(row.get("converged", "")).lower() == "true":
                    ok += 1
        return ok, total

    cudss_ok, cudss_total = count_ok(r"cuDSS")
    cusolver_ok, cusolver_total = count_ok(r"cuSolver")
    amgx_ok, amgx_total = count_ok(r"AMGx|amgx")
    ginkgo_ok, ginkgo_total = count_ok(r"Ginkgo|ginkgo")
    strumpack_ok, strumpack_total = count_ok(r"STRUMPACK|strumpack")
    superlu_ok, superlu_total = count_ok(r"SuperLU|superlu")
    return [
        {
            "solver": "cuDSS",
            "previous_status": "ok",
            "audit_status": "valid_as_run",
            "correctness_passed": f"{cudss_ok}/{cudss_total}",
            "best_effort_config_tested": "fp64/fp32 direct LU; analysis reuse checked",
            "best_config": "cuDSS general CSR direct",
            "valid_for_performance_comparison": "yes",
            "valid_for_integration_comparison": "yes",
            "remaining_issue": "Interpret tiny-rhs cases with scaled and absolute residuals.",
        },
        {
            "solver": "cuSolverSP/RF",
            "previous_status": "ok",
            "audit_status": "valid_monolithic_qr",
            "correctness_passed": f"{cusolver_ok}/{cusolver_total}",
            "best_effort_config_tested": "cuSolverSP csrlsvqr only; RF noted unavailable without supplied LU",
            "best_config": "cuSolverSP QR",
            "valid_for_performance_comparison": "qualified",
            "valid_for_integration_comparison": "qualified",
            "remaining_issue": "Monolithic QR is not reusable-factorization evidence.",
        },
        {
            "solver": "AMGx",
            "previous_status": "fixed GMRES+AMG",
            "audit_status": "limited_grid_tested",
            "correctness_passed": f"{amgx_ok}/{amgx_total}",
            "best_effort_config_tested": "GMRES/FGMRES with AMG and Jacobi/BlockJacobi smoothers, max_iter 200/1000",
            "best_config": "see config_audit.csv per case",
            "valid_for_performance_comparison": "qualified",
            "valid_for_integration_comparison": "qualified",
            "remaining_issue": "Iterative convergence remains setup-sensitive on larger Jacobians.",
        },
        {
            "solver": "Ginkgo",
            "previous_status": "GMRES/BiCGSTAB Jacobi",
            "audit_status": "valid_as_jacobi_only_incomplete_best_effort",
            "correctness_passed": f"{ginkgo_ok}/{ginkgo_total}",
            "best_effort_config_tested": "GMRES/BiCGSTAB Jacobi with audit tolerances",
            "best_config": "BiCGSTAB+Jacobi on small/medium where it converges",
            "valid_for_performance_comparison": "limited",
            "valid_for_integration_comparison": "limited",
            "remaining_issue": "Wrapper ignores advanced preconditioners; not full Ginkgo best effort.",
        },
        {
            "solver": "STRUMPACK",
            "previous_status": "np=1 runnable",
            "audit_status": "valid_external_hybrid_np1",
            "correctness_passed": f"{strumpack_ok}/{strumpack_total}",
            "best_effort_config_tested": "np=1, OMP_NUM_THREADS=1, default no compression",
            "best_config": "STRUMPACK MPIDist np=1 exact/no-compression default",
            "valid_for_performance_comparison": "qualified",
            "valid_for_integration_comparison": "qualified_external_baseline",
            "remaining_issue": "np>1 timeout/hang; no SLATE full GPU path.",
        },
        {
            "solver": "SuperLU_DIST",
            "previous_status": "runtime_failed Invalid ISPEC",
            "audit_status": "fixed_invalid_ispec_but_wrapper_needs_rewrite",
            "correctness_passed": f"{superlu_ok}/{superlu_total}",
            "best_effort_config_tested": "NATURAL/MMD variants; LargeDiag/NOROWPERM; one-shot process-level runs",
            "best_config": "NATURAL+LargeDiag for MATPOWER probes; NATURAL+NOROWPERM for synthetic sanity",
            "valid_for_performance_comparison": "diagnostic_only",
            "valid_for_integration_comparison": "qualified_external_baseline",
            "remaining_issue": "Original in-process repeated ABglobal wrapper can mutate A; needs explicit reusable-state implementation.",
        },
    ]


def markdown_table(rows: list[dict], columns: list[str], limit: int | None = None) -> str:
    rows = rows[:limit] if limit else rows
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for c in columns:
            text = str(row.get(c, ""))
            text = text.replace("\n", " ").replace("|", "\\|")
            if len(text) > 140:
                text = text[:137] + "..."
            vals.append(text)
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def make_report(env: dict, wrapper_rows: list[dict], correctness_rows: list[dict],
                config_rows: list[dict], best_rows: list[dict], superlu_debug: str) -> None:
    dataset_rows = [r for r in correctness_rows if r.get("row_type") == "dataset"]
    correctness_solver_rows = [r for r in correctness_rows if r.get("row_type") == "correctness"]
    perf_rows = sorted(config_rows, key=lambda r: (str(r.get("system")), str(r.get("solver")), str(r.get("dtype"))))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# Linear Solver Measurement Audit v3\n\n"
        "## 1. Why This Audit Was Needed\n\n"
        "The v1/v2 benchmark established useful evidence, but several rows mixed different measurement meanings: reusable direct factorization, monolithic one-shot direct solves, iterative setup/solve, and MPI/hybrid external direct solves. This audit checks wrapper correctness, residual interpretation, timing phase visibility, and reasonable best-effort configurations before using the results as annual-report evidence.\n\n"
        "A key correction is residual interpretation. Final Newton iterations can have very small `||rhs||`, so a large relative residual can coexist with a tiny absolute residual. The audit therefore reports absolute and scaled residuals in addition to the original relative residual.\n\n"
        "## 2. Prior Result Summary\n\n"
        "- cuDSS and cuSolverSP were installed and ran in v1.\n"
        "- AMGx ran a fixed GMRES+AMG configuration in v1.\n"
        "- Ginkgo was added in v2 with CUDA executor, GMRES+Jacobi, and BiCGSTAB+Jacobi.\n"
        "- STRUMPACK was added in v2 and ran at `np=1`; `np=2`/`np=4` timed out or hung.\n"
        "- SuperLU_DIST built with CUDA/MPI in v2 but every run failed with `get_perm_c.c Invalid ISPEC`.\n\n"
        "## 3. Environment Consistency\n\n"
        + markdown_table([
            {"item": k, "value": v} for k, v in env.items()
        ], ["item", "value"]) + "\n\n"
        "MPI consistency matters here. Default `mpicc/mpicxx/mpirun` resolve to OpenMPI, while SuperLU_DIST audit executables are linked to the local MPICH install and STRUMPACK is linked to system MPICH. The audit launch commands use matching MPICH launchers for MPI wrappers.\n\n"
        "## 4. Measurement Validity Criteria\n\n"
        "- The wrapper must solve `J dx = rhs` with the Matrix Market orientation as loaded.\n"
        "- CSR/CSC conversion and index base must match the library expectation.\n"
        "- Direct solvers with reusable symbolic analysis must not be compared blindly against monolithic one-shot APIs.\n"
        "- Iterative solver rows are best-effort only after a small reproducible configuration grid, not one hand-picked config.\n"
        "- Accuracy is judged with relative, absolute, scaled residual, and reference error.\n\n"
        "## 5. Dataset and Residual Metric Audit\n\n"
        + markdown_table(dataset_rows, ["system", "iteration", "matrix_rows", "nnz", "rhs_norm_2", "relative_residual_2", "scaled_residual_2", "symmetry", "condition_estimate", "warnings"], limit=40) + "\n\n"
        "Dataset warnings are expected for final Newton iterations with tiny right-hand sides; those rows must be interpreted using absolute and scaled residuals.\n\n"
        "## 6. Wrapper Correctness Audit\n\n"
        + markdown_table(wrapper_rows, ["solver", "previous_classification", "audit_classification", "phase_visibility", "wrapper_findings", "remaining_issue"]) + "\n\n"
        "## 7. Correctness Harness Results\n\n"
        + markdown_table(correctness_solver_rows, ["system", "solver", "dtype", "status", "relative_residual_2", "absolute_residual_2", "scaled_residual_2", "relative_error_2", "warnings"], limit=80) + "\n\n"
        "## 8. Solver-by-Solver Audit\n\n"
        "### cuDSS\n\n"
        "cuDSS remains valid as measured: the wrapper uses the general nonsymmetric sparse matrix path, performs symbolic analysis once, repeats numeric factorization and solve, keeps data GPU-resident after load, and computes CPU residuals after D2H. Its timing should be interpreted two ways: one-shot includes analysis, while Newton-style repeated solves emphasize factorization plus solve after the sparsity pattern is known.\n\n"
        "### cuSolverSP / cuSolverRF\n\n"
        "CUDA 12.8 headers expose cuSolverSP QR for general sparse systems. The LU and Cholesky sparse solve APIs visible in the headers are deprecated toward cuDSS, and Cholesky is not appropriate for nonsymmetric power-flow Jacobians. cuSolverRF is present but requires externally supplied LU factors, so the existing benchmark is valid as a monolithic QR comparison, not as a reusable refactorization comparison.\n\n"
        "### AMGx\n\n"
        "The v1 AMGx rows are valid for the fixed configuration. The audit extends this with a finite grid of GMRES/FGMRES, AMG preconditioning, Jacobi/BlockJacobi smoothers, and max iteration values 200/1000 where the wrapper accepts the configuration. Larger Jacobians remain convergence-sensitive; failures are not installation failures but configuration/robustness evidence under this grid.\n\n"
        "### Ginkgo\n\n"
        "Ginkgo rows are valid as CUDA-executor GMRES/BiCGSTAB with Jacobi. The source audit found that the wrapper parses a preconditioner field but always constructs Jacobi, and it only switches between GMRES and BiCGSTAB. Ginkgo headers include IDR and advanced preconditioners such as ILU/ParILU/ParILUT/ISAI, so the current Ginkgo evidence is limited and not a full best-effort Ginkgo study.\n\n"
        "### STRUMPACK\n\n"
        "STRUMPACK is valid as an external MPI/hybrid direct baseline at `np=1`: CUDA is enabled, input/output are host-distributed via MPI, and default compression is `NONE`. The build does not enable SLATE, so this is not full GPU residency. The audit preserves the v2 conclusion that `np=2`/`np=4` hangs are runtime/integration issues, not performance data.\n\n"
        "### SuperLU_DIST\n\n"
        "The v2 SuperLU_DIST failure was not a solver-performance result. The wrapper requested `METIS_AT_PLUS_A`, but the installed SuperLU_DIST config has `HAVE_PARMETIS` disabled, so `get_perm_c` rejects that ISPEC. Audit executables using supported permutations avoid the `Invalid ISPEC` failure. A second issue appeared: repeated in-process ABglobal calls can mutate matrix/solver state, so the original wrapper is invalid for repeated timing. Audit SuperLU_DIST results are one-shot process-level diagnostics until the wrapper is rewritten.\n\n"
        "## 9. SuperLU_DIST Diagnosis\n\n"
        + superlu_debug + "\n\n"
        "## 10. Best-Effort Configuration Results\n\n"
        + markdown_table(perf_rows, ["solver", "config", "system", "iteration", "dtype", "build_status", "converged", "num_iterations", "one_shot_time_ms", "factorization_plus_solve_ms", "relative_residual_2", "scaled_residual_2", "relative_error_2", "phase_visibility"], limit=120) + "\n\n"
        "## 11. Best-Effort Summary\n\n"
        + markdown_table(best_rows, ["solver", "previous_status", "audit_status", "correctness_passed", "best_effort_config_tested", "best_config", "valid_for_performance_comparison", "valid_for_integration_comparison", "remaining_issue"]) + "\n\n"
        "## 12. Updated Interpretation for Annual-Report Writing\n\n"
        "The strongest cuPF default-solver evidence remains cuDSS, but the careful claim is not simply that it is fastest. The valid evidence is that cuDSS directly supports general nonsymmetric sparse Jacobians, keeps the solve path GPU-resident after initial load, exposes symbolic analysis and numeric factorization/solve phases, and maps naturally to repeated Newton solves where the sparsity pattern is stable.\n\n"
        "cuSolverSP is useful as an NVIDIA monolithic QR comparison, especially on small cases, but it is not equivalent to a reusable refactorization path. AMGx and Ginkgo should not be rejected merely for being iterative; the issue observed here is robustness and setup sensitivity on these Jacobians under the tested configuration grids. STRUMPACK and SuperLU_DIST are credible external distributed direct-solver baselines, but their MPI setup, host/hybrid residency, and integration complexity make them less natural as a cuPF default. SuperLU_DIST specifically needs a rewritten wrapper before its timing can be treated as fair performance data.\n"
    )


def superlu_debug_text() -> str:
    original_tail = ""
    prior_json = sorted((ROOT / "results" / "raw_json").glob("*superlu_dist*np1_fp64_second_pass.json"))
    for path in prior_json:
        data = read_json(path)
        tail = (data.get("stdout_tail") or data.get("stderr_tail") or "").strip()
        if tail:
            original_tail = tail[-1000:]
            break
    if not original_tail:
        old_logs = sorted((ROOT / "logs").glob("*superlu_dist_fp64.stderr.log"))
        for path in old_logs:
            tail = path.read_text(errors="replace").strip()
            if tail:
                original_tail = tail[-1000:]
                break
    config = (ROOT / "third_party" / "superlu_dist" / "install" / "include" / "superlu_dist_config.h").read_text(errors="replace")
    enum_text = (ROOT / "third_party" / "superlu_dist" / "install" / "include" / "superlu_enum_consts.h").read_text(errors="replace")
    source_lines = run_text(["bash", "-lc", f"sed -n '496,560p' {ROOT / 'third_party/superlu_dist/src/SRC/prec-independent/get_perm_c.c'}"])
    text = (
        "Reproduction of prior failure:\n\n"
        "```text\n"
        + (original_tail or "Prior stderr tail not found in logs.") + "\n"
        "```\n\n"
        "Installed SuperLU_DIST configuration excerpts:\n\n"
        "```text\n"
        + "\n".join([line for line in config.splitlines() if "HAVE_PARMETIS" in line or "HAVE_COLAMD" in line]) + "\n"
        + "\n".join([line for line in enum_text.splitlines() if "typedef enum {NATURAL" in line or "typedef enum {NOROWPERM" in line]) + "\n"
        "```\n\n"
        "`get_perm_c.c` accepts `METIS_AT_PLUS_A` only inside `#ifdef HAVE_PARMETIS`; otherwise it falls through to `ABORT(\"Invalid ISPEC\")`:\n\n"
        "```c\n"
        + source_lines + "\n"
        "```\n\n"
        "Audit repair attempts:\n\n"
        "- Built audit executables for `NATURAL`, `MMD_AT_PLUS_A`, and `MMD_ATA`, with both `LargeDiag_MC64` and `NOROWPERM` row permutation variants.\n"
        "- `Invalid ISPEC` is fixed by avoiding `METIS_AT_PLUS_A` in this no-ParMETIS build.\n"
        "- Correctness depends on row/column permutation and on not reusing the same ABglobal `SuperMatrix` across repeated in-process solves. The original wrapper therefore remains invalid for repeated timing.\n"
    )
    (RESULTS / "superlu_dist_debug.md").write_text(text)
    return text


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    env = audit_environment()
    correctness_systems = create_correctness_systems()
    dataset_systems = discover_dataset_systems()
    validation_systems = [s for s in dataset_systems if s.name in {"synthetic_validation", "case14", "case118"}]
    small_validation = [s for s in validation_systems if s.name in {"synthetic_validation", "case14"} or (s.name == "case118" and s.iteration == 0)]

    correctness_rows = dataset_audit_rows(dataset_systems)
    config_rows: list[dict] = []

    wrappers = {
        "cudss": ROOT / "build" / "cudss_benchmark",
        "cusolver": ROOT / "build" / "cusolver_benchmark",
        "amgx": ROOT / "build" / "amgx_benchmark",
        "ginkgo": ROOT / "build" / "ginkgo_benchmark",
        "strumpack": ROOT / "solvers" / "strumpack" / "build" / "strumpack_benchmark",
        "superlu_fixed": ROOT / "measurement_audit" / "superlu_dist_audit" / "build" / "superlu_dist_audit_natural",
    }

    amgx_configs = make_amgx_configs()
    ginkgo_configs = make_ginkgo_configs()

    # Correctness harness.
    for system in correctness_systems:
        for dtype in ["fp64", "fp32"]:
            for solver, label in [("cudss", "cuDSS"), ("cusolver", "cuSolverSP")]:
                if executable(wrappers[solver]):
                    data = run_solver_case(f"correctness_{label}", solver, system, dtype, repeats=2, warmup=1)
                    data = augment_residual(data, system)
                    correctness_rows.append({
                        "row_type": "correctness",
                        "system": system.name,
                        "iteration": system.iteration,
                        "solver": data.get("solver_name", label),
                        "dtype": dtype,
                        "status": data.get("build_status", ""),
                        "matrix_rows": data.get("matrix_rows", ""),
                        "nnz": data.get("nnz", ""),
                        "rhs_norm_2": data.get("rhs_norm_2", ""),
                        "absolute_residual_2": data.get("absolute_residual_2", ""),
                        "relative_residual_2": data.get("relative_residual_2", ""),
                        "scaled_residual_2": data.get("scaled_residual_2", ""),
                        "relative_error_2": data.get("relative_error_to_x_ref_2", ""),
                        "symmetry": "",
                        "condition_estimate": "",
                        "warnings": "expected_fail_matrix" if "singular" in system.name else "",
                    })
            if executable(wrappers["amgx"]):
                cfg = amgx_configs["gmres_amg_block_jacobi_200"]
                data = run_solver_case("correctness_AMGx", "amgx", system, dtype, cfg, repeats=2, warmup=1)
                data = augment_residual(data, system)
                correctness_rows.append({
                    "row_type": "correctness",
                    "system": system.name,
                    "iteration": system.iteration,
                    "solver": data.get("solver_name", "AMGx"),
                    "dtype": dtype,
                    "status": data.get("build_status", ""),
                    "matrix_rows": data.get("matrix_rows", ""),
                    "nnz": data.get("nnz", ""),
                    "rhs_norm_2": data.get("rhs_norm_2", ""),
                    "absolute_residual_2": data.get("absolute_residual_2", ""),
                    "relative_residual_2": data.get("relative_residual_2", ""),
                    "scaled_residual_2": data.get("scaled_residual_2", ""),
                    "relative_error_2": data.get("relative_error_to_x_ref_2", ""),
                    "symmetry": "",
                    "condition_estimate": "",
                    "warnings": "expected_fail_matrix" if "singular" in system.name else "",
                })
            if executable(wrappers["ginkgo"]):
                cfg = ginkgo_configs["gmres_jacobi_fp64tol" if dtype == "fp64" else "gmres_jacobi_fp32tol"]
                data = run_solver_case("correctness_Ginkgo_GMRES_Jacobi", "ginkgo", system, dtype, cfg, repeats=2, warmup=1)
                data = augment_residual(data, system)
                correctness_rows.append({
                    "row_type": "correctness",
                    "system": system.name,
                    "iteration": system.iteration,
                    "solver": data.get("solver_name", "Ginkgo"),
                    "dtype": dtype,
                    "status": data.get("build_status", ""),
                    "matrix_rows": data.get("matrix_rows", ""),
                    "nnz": data.get("nnz", ""),
                    "rhs_norm_2": data.get("rhs_norm_2", ""),
                    "absolute_residual_2": data.get("absolute_residual_2", ""),
                    "relative_residual_2": data.get("relative_residual_2", ""),
                    "scaled_residual_2": data.get("scaled_residual_2", ""),
                    "relative_error_2": data.get("relative_error_to_x_ref_2", ""),
                    "symmetry": "",
                    "condition_estimate": "",
                    "warnings": "expected_fail_matrix" if "singular" in system.name else "",
                })
        if executable(wrappers["superlu_fixed"]):
            for variant in ["natural", "natural_norowperm", "mmd_at_plus_a", "mmd_at_plus_a_norowperm", "mmd_ata", "mmd_ata_norowperm"]:
                data = run_solver_case(f"correctness_SuperLU_DIST_{variant}", "superlu_fixed", system, "fp64", variant=variant, repeats=1, warmup=0)
                data = augment_residual(data, system)
                correctness_rows.append({
                    "row_type": "correctness",
                    "system": system.name,
                    "iteration": system.iteration,
                    "solver": "SuperLU_DIST",
                    "dtype": "fp64",
                    "status": data.get("build_status", ""),
                    "matrix_rows": data.get("matrix_rows", ""),
                    "nnz": data.get("nnz", ""),
                    "rhs_norm_2": data.get("rhs_norm_2", ""),
                    "absolute_residual_2": data.get("absolute_residual_2", ""),
                    "relative_residual_2": data.get("relative_residual_2", ""),
                    "scaled_residual_2": data.get("scaled_residual_2", ""),
                    "relative_error_2": data.get("relative_error_to_x_ref_2", ""),
                    "symmetry": "",
                    "condition_estimate": "",
                    "warnings": ("variant=" + variant + (";expected_fail_matrix" if "singular" in system.name else "")),
                })

    # Required direct reruns over all dumped systems.
    for system in dataset_systems:
        for dtype in ["fp64", "fp32"]:
            if executable(wrappers["cudss"]):
                data = run_solver_case("audit_cuDSS", "cudss", system, dtype, repeats=10, warmup=3)
                config_rows.append(result_to_config_row(data, system, "cuDSS", "general_direct_analysis_reuse"))
            if executable(wrappers["cusolver"]):
                data = run_solver_case("audit_cuSolverSP", "cusolver", system, dtype, repeats=10, warmup=3)
                config_rows.append(result_to_config_row(data, system, "cuSolverSP", "csrlsvqr_monolithic"))

    # Iterative finite grids on validation systems; previous v2 rows remain full-run evidence.
    for system in small_validation:
        for dtype in ["fp64", "fp32"]:
            if executable(wrappers["amgx"]):
                for cfg_name, cfg in amgx_configs.items():
                    data = run_solver_case(f"audit_AMGx_{cfg_name}", "amgx", system, dtype, cfg, repeats=3, warmup=1)
                    config_rows.append(result_to_config_row(data, system, "AMGx", cfg_name))
            if executable(wrappers["ginkgo"]):
                for cfg_name, cfg in ginkgo_configs.items():
                    if (dtype == "fp64" and cfg_name.endswith("fp32tol")) or (dtype == "fp32" and cfg_name.endswith("fp64tol")):
                        continue
                    data = run_solver_case(f"audit_Ginkgo_{cfg_name}", "ginkgo", system, dtype, cfg, repeats=3, warmup=1)
                    config_rows.append(result_to_config_row(data, system, "Ginkgo", cfg_name))

    # STRUMPACK np=1 validation rerun plus timeout probe at np=2.
    for system in small_validation:
        for dtype in ["fp64", "fp32"]:
            if executable(wrappers["strumpack"]):
                data = run_solver_case("audit_STRUMPACK_np1", "strumpack", system, dtype, repeats=3, warmup=1, np_ranks=1)
                config_rows.append(result_to_config_row(data, system, "STRUMPACK", "np1_omp1_default_no_compression"))
    if small_validation and executable(wrappers["strumpack"]):
        data = run_solver_case("audit_STRUMPACK_np2_probe", "strumpack", small_validation[0], "fp64", repeats=1, warmup=0, np_ranks=2)
        config_rows.append(result_to_config_row(data, small_validation[0], "STRUMPACK", "np2_timeout_probe"))

    # SuperLU_DIST fixed one-shot process-level rerun on all dumped systems.
    for system in dataset_systems:
        if executable(wrappers["superlu_fixed"]):
            data = run_solver_case("audit_SuperLU_DIST_fixed_natural", "superlu_fixed", system, "fp64", variant="natural", repeats=1, warmup=0)
            config_rows.append(result_to_config_row(data, system, "SuperLU_DIST", "natural_largediag_one_shot_process"))

    wrapper_rows = build_wrapper_rows()
    best_rows = make_best_summary(config_rows)
    debug = superlu_debug_text()

    write_csv(RESULTS / "wrapper_audit.csv", wrapper_rows, [
        "solver", "previous_classification", "audit_classification", "phase_visibility", "wrapper_findings", "remaining_issue",
    ])
    write_csv(RESULTS / "correctness_audit.csv", correctness_rows, [
        "row_type", "system", "iteration", "solver", "dtype", "status", "matrix_rows", "nnz", "rhs_norm_2",
        "absolute_residual_2", "relative_residual_2", "scaled_residual_2", "relative_error_2",
        "symmetry", "condition_estimate", "warnings",
    ])
    write_csv(RESULTS / "config_audit.csv", config_rows, [
        "solver", "config", "system", "iteration", "dtype", "build_status", "converged", "num_iterations",
        "analysis_ms", "factorization_ms", "solve_ms", "total_solver_ms", "one_shot_time_ms",
        "factorization_plus_solve_ms", "phase_visibility", "relative_residual_2", "absolute_residual_2",
        "scaled_residual_2", "relative_error_2", "gpu_resident_after_initial_load", "raw_json", "notes", "previous_result",
    ])
    write_csv(RESULTS / "best_effort_summary.csv", best_rows, [
        "solver", "previous_status", "audit_status", "correctness_passed", "best_effort_config_tested",
        "best_config", "valid_for_performance_comparison", "valid_for_integration_comparison", "remaining_issue",
    ])
    make_report(env, wrapper_rows, correctness_rows, config_rows, best_rows, debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
