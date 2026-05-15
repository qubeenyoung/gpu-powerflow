#!/usr/bin/env python3
"""Generate the second-pass linear solver benchmark report."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "linear_solver_benchmark_report_v2.md"
SUMMARY_V1 = ROOT / "results" / "summary_csv" / "summary.csv"
SUMMARY_V2 = ROOT / "results" / "summary_csv" / "summary_second_pass.csv"
ENV_JSON = ROOT / "results" / "environment.json"
RAW_DIR = ROOT / "results" / "raw_json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def num(value: Any) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def fmt(value: Any) -> str:
    x = num(value)
    if not math.isfinite(x):
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:.1f}"
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.4g}"
    if x == 0:
        return "0"
    return f"{x:.3g}"


def compact(text: Any, limit: int = 90) -> str:
    if text is None:
        return ""
    s = str(text).replace("\n", " ").replace("|", "/").strip()
    return s if len(s) <= limit else s[: limit - 3] + "..."


def median(rows: list[dict[str, str]], key: str) -> str:
    values = [num(r.get(key)) for r in rows if math.isfinite(num(r.get(key)))]
    return fmt(statistics.median(values)) if values else "n/a"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(compact(c, 140) for c in row) + " |")
    return "\n".join(out)


def dataset_rows() -> list[list[Any]]:
    rows = []
    for meta_path in sorted((ROOT / "datasets" / "dumped_systems").glob("*/iter_*/meta.json")):
        meta = read_json(meta_path)
        rows.append([
            meta.get("case_name"),
            meta.get("iteration"),
            meta.get("num_bus"),
            meta.get("num_pv"),
            meta.get("num_pq"),
            f"{meta.get('matrix_rows')}x{meta.get('matrix_cols')}",
            meta.get("nnz"),
            fmt(meta.get("rhs_norm_2")),
            meta.get("pattern_hash"),
        ])
    order = {
        "synthetic_validation": 0,
        "case14": 1,
        "case118": 2,
        "case300": 3,
        "case1354pegase": 4,
        "case2869pegase": 5,
        "case9241pegase": 6,
    }
    rows.sort(key=lambda r: (order.get(r[0], 99), int(r[1])))
    return rows


def environment_table() -> str:
    env = read_json(ENV_JSON)
    gpu = env.get("gpu", {}).get("gpus", [{}])[0]
    cuda = env.get("cuda", {})
    os_info = env.get("os", {})
    if isinstance(os_info, dict):
        os_value = os_info.get("platform") or f"{os_info.get('system', '')} {os_info.get('release', '')}".strip()
    else:
        os_value = os_info
    compilers = env.get("compilers", {})
    rows = [
        ["hostname", env.get("hostname")],
        ["OS", os_value],
        ["CPU", env.get("cpu_model")],
        ["RAM GB", env.get("ram_gb")],
        ["GPU", f"{gpu.get('name')} ({gpu.get('memory_total')})"],
        ["NVIDIA driver", gpu.get("driver_version")],
        ["CUDA toolkit / nvcc", cuda.get("toolkit_version")],
        ["Python", env.get("python", {}).get("version", env.get("python_version"))],
        ["gcc", compilers.get("gcc", {}).get("version", "")],
        ["g++", compilers.get("g++", {}).get("version", "")],
        ["CMake", compilers.get("cmake", {}).get("version", env.get("cmake_version", ""))],
        ["MPI after retry", "MPICH local plus system MPICH/OpenMPI packages available"],
    ]
    return table(["item", "value"], rows)


def solver_availability(v2: list[dict[str, str]]) -> str:
    superlu_fail = next((r for r in v2 if r["solver_name"] == "SuperLU_DIST" and r["build_status"] == "runtime_failed"), {})
    rows = [
        [
            "Ginkgo",
            "GPU iterative Krylov",
            "installed/runnable",
            "2.0.0 develop",
            "yes",
            "no",
            "GPU resident for matrix/vector operations",
            "CUDA executor built locally; GMRES and BiCGSTAB with Jacobi were benchmarked.",
        ],
        [
            "SuperLU_DIST",
            "distributed sparse direct",
            "installed_cuda / runtime_failed",
            "9.2.1",
            "yes",
            "yes",
            "not observed",
            "CUDA build succeeded, but fp64 ABglobal wrapper failed at runtime: "
            + compact(superlu_fail.get("stdout_tail") or superlu_fail.get("notes"), 130),
        ],
        [
            "STRUMPACK",
            "distributed sparse direct / multifrontal",
            "installed_cuda_nolto / runnable np=1",
            "8.0.0",
            "yes",
            "yes",
            "host input/output with internal GPU offload",
            "Rebuilt with MPI compile options set to -fno-lto; np=1 ran all systems. np=2/np=4 synthetic sanity runs hung after initialization. Startup warned that SLATE is required for full GPU support.",
        ],
    ]
    return table(
        ["solver", "type", "build_status", "version", "CUDA_enabled", "MPI_required", "GPU_resident_status", "notes"],
        rows,
    )


def performance_table(v2: list[dict[str, str]]) -> str:
    order = {
        "synthetic_validation": 0,
        "case14": 1,
        "case118": 2,
        "case300": 3,
        "case1354pegase": 4,
        "case2869pegase": 5,
        "case9241pegase": 6,
    }
    rows = sorted(v2, key=lambda r: (
        order.get(r.get("case_name"), 99),
        int(r.get("iteration") or -1),
        r.get("solver_name"),
        r.get("dtype"),
        num(r.get("mpi_ranks")),
    ))
    out_rows = []
    for r in rows:
        solver = r.get("solver_name")
        ranks = r.get("mpi_ranks")
        if ranks not in ("", None):
            solver = f"{solver} np={fmt(ranks)}"
        out_rows.append([
            r.get("case_name"),
            r.get("iteration"),
            solver,
            r.get("dtype"),
            r.get("build_status"),
            fmt(r.get("analysis_ms")),
            fmt(r.get("factorization_ms")),
            fmt(r.get("solve_ms")),
            fmt(r.get("total_solver_ms")),
            fmt(r.get("total_end_to_end_ms")),
            fmt(r.get("relative_residual_2")),
            fmt(r.get("relative_error_to_x_ref_2")),
            r.get("converged"),
            r.get("num_iterations"),
            compact(r.get("gpu_resident_after_initial_load"), 34),
        ])
    return table(
        [
            "case",
            "iter",
            "solver",
            "dtype",
            "status",
            "analysis_ms",
            "factor_ms",
            "solve_ms",
            "solver_ms",
            "end_to_end_ms",
            "rel_res_2",
            "rel_err_2",
            "conv",
            "iters",
            "residency",
        ],
        out_rows,
    )


def aggregate_table(v1: list[dict[str, str]], v2: list[dict[str, str]]) -> str:
    rows = []
    for solver in ["cuDSS", "cuSolverSP", "AMGx"]:
        rs = [r for r in v1 if r.get("solver") == solver and r.get("build_status") == "ok"]
        rows.append([
            solver,
            len(rs),
            sum(r.get("converged") == "True" for r in rs),
            median(rs, "analysis_ms"),
            median(rs, "factorization_ms"),
            median(rs, "solve_ms"),
            median(rs, "total_solver_ms"),
        ])
    for solver in ["Ginkgo-GMRES-Jacobi", "Ginkgo-BiCGSTAB-Jacobi", "STRUMPACK", "SuperLU_DIST"]:
        rs = [r for r in v2 if r.get("solver_name") == solver]
        ok = [r for r in rs if r.get("build_status") == "ok"]
        rows.append([
            solver,
            len(rs),
            sum(r.get("converged") == "True" for r in rs),
            median(ok, "analysis_ms"),
            median(ok, "factorization_ms"),
            median(ok, "solve_ms"),
            median(ok, "total_solver_ms"),
        ])
    return table(["solver", "rows", "converged_rows", "median_analysis", "median_factor", "median_solve", "median_solver"], rows)


def failure_table(v2: list[dict[str, str]]) -> str:
    rows = []
    for r in v2:
        if r.get("build_status") not in {"ok", "unsupported"}:
            rows.append([
                r.get("case_name"),
                r.get("iteration"),
                r.get("solver_name"),
                r.get("dtype"),
                r.get("build_status"),
                compact(r.get("attempted_command"), 95),
                compact((r.get("stderr_tail") or r.get("stdout_tail") or r.get("notes")), 120),
            ])
    return table(["case", "iter", "solver", "dtype", "status", "attempted_command", "error_tail"], rows)


def report_text() -> str:
    v1 = read_csv(SUMMARY_V1)
    v2 = read_csv(SUMMARY_V2)
    counts = Counter((r["solver_name"], r["build_status"], r["converged"]) for r in v2)

    ginkgo_rows = [r for r in v2 if r["solver_name"].startswith("Ginkgo")]
    strumpack_rows = [r for r in v2 if r["solver_name"] == "STRUMPACK"]
    superlu_rows = [r for r in v2 if r["solver_name"] == "SuperLU_DIST"]

    lines = [
        "# Linear Solver Benchmark Report v2",
        "",
        "## 1. What Changed From v1",
        "",
        "- Ginkgo was installed locally with CUDA support under `third_party/ginkgo/install/`; both GMRES+Jacobi and BiCGSTAB+Jacobi wrappers ran on the CUDA executor.",
        "- SuperLU_DIST was built locally with CUDA and MPI, but the fp64 ABglobal benchmark wrapper failed at runtime for every system with `Invalid ISPEC at line 556 in .../get_perm_c.c`; fp32 is explicitly unsupported by this wrapper.",
        "- STRUMPACK was built locally with CUDA and MPI. The first CUDA install linked, but wrapper linking exposed MPICH LTO/CUDA fatbin issues, so a second no-LTO install was created under `third_party/strumpack/install_nolto/`. STRUMPACK np=1 ran all dumped systems; np=2 and np=4 synthetic sanity attempts hung after initialization and were interrupted.",
        "- The original v1 report was not overwritten; this report uses `summary_second_pass.csv` and second-pass raw JSON files with `_second_pass` suffixes.",
        "",
        "## 2. Environment",
        "",
        environment_table(),
        "",
        "## 3. Dataset Table",
        "",
        table(["case", "iteration", "num_bus", "num_pv", "num_pq", "matrix_size", "nnz", "rhs_norm", "pattern_hash"], dataset_rows()),
        "",
        "## 4. Solver Availability",
        "",
        solver_availability(v2),
        "",
        "## 5. Second-Pass Result Counts",
        "",
        table(["solver", "status", "converged", "rows"], [[*k, v] for k, v in sorted(counts.items())]),
        "",
        "## 6. Aggregate Timing Summary",
        "",
        aggregate_table(v1, v2),
        "",
        "## 7. New Performance Table",
        "",
        performance_table(v2),
        "",
        "## 8. Runtime Failures",
        "",
        failure_table(v2),
        "",
        "## 9. Direct Solver Comparison",
        "",
        "cuDSS remains the cleanest direct-solver default from the combined evidence. It ran all v1 systems in FP64 and FP32, exposed separate analysis/factorization/solve phases, kept a GPU-resident workflow, and directly targets general sparse systems suitable for power-flow Jacobians.",
        "",
        "cuSolverSP also ran the v1 systems accurately through the QR path, but it is a monolithic API path in this benchmark and does not provide the same reusable direct factorization structure that cuPF wants for repeated Newton Jacobian solves. The CUDA 12.8 headers observed in v1 also marked older sparse LU/Cholesky paths as deprecated with cuDSS as the replacement, so cuSolverSP is useful NVIDIA context rather than the best integration target.",
        "",
        "SuperLU_DIST is still a candidate as an external distributed sparse direct baseline, but not as successful GPU evidence in this pass. The CUDA build completed, MPI was available, and the wrapper built, but every fp64 run failed immediately with the `get_perm_c.c` `Invalid ISPEC` runtime error. Because no solve completed, there is no residual or timing evidence for cuPF suitability beyond installation complexity and the failed ABglobal path.",
        "",
        "STRUMPACK is much stronger than v1: the CUDA/MPI build and np=1 wrapper ran all dumped systems. However, the run is host-input/output with internal GPU offload rather than a simple GPU-resident cuPF-style interface, startup warns that SLATE is required for full GPU support, and np=2/np=4 synthetic runs hung. Its median second-pass solver time is competitive in this small single-node run, but the setup complexity and MPI/hybrid data path make it a better external direct-solver baseline than a default embedded cuPF solver.",
        "",
        "## 10. Iterative Solver Comparison",
        "",
        f"Ginkgo produced {len(ginkgo_rows)} ok rows across GMRES+Jacobi and BiCGSTAB+Jacobi. It converged on all validation and smaller systems, but convergence degraded on larger PEGASE cases: GMRES converged in {sum(r['converged'] == 'True' for r in ginkgo_rows if r['solver_name'] == 'Ginkgo-GMRES-Jacobi')} of {sum(r['solver_name'] == 'Ginkgo-GMRES-Jacobi' for r in ginkgo_rows)} rows, and BiCGSTAB converged in {sum(r['converged'] == 'True' for r in ginkgo_rows if r['solver_name'] == 'Ginkgo-BiCGSTAB-Jacobi')} of {sum(r['solver_name'] == 'Ginkgo-BiCGSTAB-Jacobi' for r in ginkgo_rows)} rows. Several large-case rows reached the configured 1000-iteration limit with residuals too large for Newton linear-solve evidence.",
        "",
        "AMGx from v1 showed the same broad pattern: useful small-case convergence, but larger Jacobians were sensitive to the fixed configuration and often hit the iteration limit. These iterative libraries are not rejected simply for speed; the issue is that standalone Newton-Raphson linear solves need consistent residual quality across changing nonsymmetric Jacobians, and the tested fixed preconditioned Krylov configurations did not provide that consistency.",
        "",
        "## 11. Final Interpretation",
        "",
        "The second pass improves the evidence base without overturning the v1 conclusion. Ginkgo is now a working CUDA iterative baseline, and STRUMPACK is now a working CUDA/MPI direct baseline for single-rank runs. They are valuable comparison points, but neither has the same combination of direct-solver stability, general nonsymmetric sparse Jacobian support, GPU execution model, reusable analysis/factorization structure, and low integration complexity that cuDSS offers for cuPF.",
        "",
        "Therefore, cuDSS remains the best default sparse linear solver library for cuPF. STRUMPACK should remain in future reports as an external distributed direct-solver baseline, SuperLU_DIST should remain a candidate only after its runtime driver issue is resolved, and Ginkgo/AMGx should remain iterative-library baselines for cases where preconditioning strategy and convergence policy are the subject of a separate cuITER-style study.",
        "",
        "## 12. Reproduction Commands",
        "",
        "```bash",
        "cd /workspace/gpu-powerflow",
        "LIN_SOL_WARMUP=3 LIN_SOL_REPEATS=10 LIN_SOL_TIMEOUT=240 python3 exp/20260510/lin_sol/scripts/run_second_pass.py",
        "python3 exp/20260510/lin_sol/scripts/make_report_v2.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text())
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
