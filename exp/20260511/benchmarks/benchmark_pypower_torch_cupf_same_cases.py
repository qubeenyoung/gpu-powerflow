#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_ROOT = REPO_ROOT.parent / "gpu-powerflow-master"
BENCH_DIR = REPO_ROOT / "exp" / "20260511" / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
if MASTER_ROOT.exists() and str(MASTER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASTER_ROOT))

from benchmark_cupf_vs_torch import make_torch_inputs, torch_newton_pf  # noqa: E402
from torch_csr_implicit import (  # noqa: E402
    make_torch_csr_inputs,
    torch_csr_newton_pf,
    torch_spsolve_available,
)
from utils import CupfSolverSession, import_cupf, list_case_dirs, load_case  # noqa: E402

try:
    from pypower.ppoption import ppoption  # type: ignore  # noqa: E402
    from python.pypower.newtonpf import my_newtonpf  # type: ignore  # noqa: E402
except Exception as exc:  # pragma: no cover
    ppoption = None
    my_newtonpf = None
    _PYPOWER_IMPORT_ERROR = exc
else:
    _PYPOWER_IMPORT_ERROR = None


DEFAULT_CASES = ["case118", "case300", "case1354pegase", "case3375wp", "case6468rte"]
FIELDNAMES = [
    "case_name",
    "n_bus",
    "implementation",
    "backend",
    "dtype",
    "batch_size",
    "warmup",
    "repeats",
    "status",
    "converged",
    "iterations",
    "total_time_ms_mean",
    "total_time_ms_median",
    "total_time_ms_std",
    "total_time_ms_min",
    "total_time_ms_max",
    "per_scenario_time_ms_mean",
    "speedup_vs_pypower",
    "speedup_vs_torch_dense_gpu",
    "speedup_vs_torch_csr_implicit_gpu",
    "notes",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PYPOWER, PyTorch dense GPU, and cuPF reference forward speed on identical cases."
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps")
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "pypower_torch_dense_cupf_reference_same_cases",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--cupf-compute", choices=["fp64", "fp32", "mixed"], default="mixed")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    return parser.parse_args()


def summarize(times_ms: list[float]) -> dict[str, float]:
    if not times_ms:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(mean(times_ms)),
        "median": float(median(times_ms)),
        "std": float(pstdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "min": float(min(times_ms)),
        "max": float(max(times_ms)),
    }


def row(
    *,
    case_name: str,
    n_bus: int,
    implementation: str,
    backend: str,
    dtype: str,
    warmup: int,
    repeats: int,
    status: str,
    times_ms: list[float] | None = None,
    converged: bool | str = "",
    iterations: int | str = "",
    notes: str = "",
    error: str = "",
) -> dict[str, Any]:
    stats = summarize(times_ms or [])
    return {
        "case_name": case_name,
        "n_bus": n_bus,
        "implementation": implementation,
        "backend": backend,
        "dtype": dtype,
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "status": status,
        "converged": converged,
        "iterations": iterations,
        "total_time_ms_mean": stats["mean"],
        "total_time_ms_median": stats["median"],
        "total_time_ms_std": stats["std"],
        "total_time_ms_min": stats["min"],
        "total_time_ms_max": stats["max"],
        "per_scenario_time_ms_mean": stats["mean"],
        "speedup_vs_pypower": math.nan,
        "speedup_vs_torch_dense_gpu": math.nan,
        "speedup_vs_torch_csr_implicit_gpu": math.nan,
        "notes": notes,
        "error": error,
    }


def time_cpu(fn: Callable[[], Any], *, warmup: int, repeats: int) -> tuple[str, list[float], list[Any], str]:
    try:
        for _ in range(warmup):
            fn()
        times_ms: list[float] = []
        results: list[Any] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            results.append(result)
        return "ok", times_ms, results, ""
    except Exception as exc:  # pragma: no cover - benchmark path
        return "error", [], [], "".join(traceback.format_exception_only(type(exc), exc)).strip()


def time_cuda(fn: Callable[[], Any], *, warmup: int, repeats: int) -> tuple[str, list[float], list[Any], str]:
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times_ms: list[float] = []
        results: list[Any] = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
            results.append(result)
        return "ok", times_ms, results, ""
    except torch.OutOfMemoryError as exc:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return "oom", [], [], "".join(traceback.format_exception_only(type(exc), exc)).strip()
    except Exception as exc:  # pragma: no cover - benchmark path
        return "error", [], [], "".join(traceback.format_exception_only(type(exc), exc)).strip()


def run_pypower(case: Any, *, warmup: int, repeats: int, tolerance: float, max_iter: int) -> dict[str, Any]:
    if my_newtonpf is None or ppoption is None:
        return row(
            case_name=case.name,
            n_bus=case.n_bus,
            implementation="pypower_newtonpf",
            backend="cpu",
            dtype="float64",
            warmup=warmup,
            repeats=repeats,
            status="error",
            notes="PYPOWER import failed",
            error=str(_PYPOWER_IMPORT_ERROR),
        )

    ppopt = ppoption(PF_TOL=float(tolerance), PF_MAX_IT=int(max_iter), VERBOSE=0, OUT_ALL=0)

    def fn() -> Any:
        return my_newtonpf(
            case.ybus,
            case.sbus,
            case.v0,
            np.asarray([], dtype=np.int32),
            case.pv,
            case.pq,
            ppopt=ppopt,
            emit_status=False,
        )

    status, times, results, error = time_cpu(fn, warmup=warmup, repeats=repeats)
    converged = bool(all(getattr(result, "converged", False) for result in results)) if results else ""
    iterations = max((int(getattr(result, "iterations", 0)) for result in results), default="")
    return row(
        case_name=case.name,
        n_bus=case.n_bus,
        implementation="pypower_newtonpf",
        backend="cpu",
        dtype="float64",
        warmup=warmup,
        repeats=repeats,
        status=status,
        times_ms=times,
        converged=converged,
        iterations=iterations,
        notes="PYPOWER Newton kernel over dumped Ybus/Sbus/V0/pv/pq; no MATPOWER file load or Ybus build.",
        error=error,
    )


def run_torch_dense_gpu(case: Any, *, device: torch.device, dtype: str, warmup: int, repeats: int, tolerance: float, max_iter: int) -> dict[str, Any]:
    tensors = make_torch_inputs(case, 1, device, dtype)

    def fn() -> Any:
        with torch.no_grad():
            return torch_newton_pf(
                tensors["ybus"],
                tensors["sbus_base"],
                tensors["v0"],
                tensors["pv"],
                tensors["pq"],
                tolerance=tolerance,
                max_iter=max_iter,
            )

    status, times, _results, error = time_cuda(fn, warmup=warmup, repeats=repeats)
    return row(
        case_name=case.name,
        n_bus=case.n_bus,
        implementation="torch_dense_gpu",
        backend="cuda",
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
        status=status,
        times_ms=times,
        converged="",
        iterations="",
        notes="Dense Ybus/Jacobian PyTorch Newton path on CUDA; timed with torch.cuda.Event.",
        error=error,
    )


def run_torch_csr_implicit_gpu(
    case: Any,
    *,
    device: torch.device,
    dtype: str,
    warmup: int,
    repeats: int,
    tolerance: float,
    max_iter: int,
) -> dict[str, Any]:
    available, reason = torch_spsolve_available(device.type, dtype)
    if not available:
        return row(
            case_name=case.name,
            n_bus=case.n_bus,
            implementation="torch_csr_implicit_gpu",
            backend="cuda",
            dtype=dtype,
            warmup=warmup,
            repeats=repeats,
            status="skipped_unavailable",
            notes="Requires torch.sparse.spsolve on CUDA CSR. Current PyTorch build does not provide the needed CUDA sparse solver.",
            error=reason,
        )
    static = make_torch_csr_inputs(case, device=device, dtype=dtype)

    def fn() -> Any:
        with torch.no_grad():
            return torch_csr_newton_pf(static, tolerance=tolerance, max_iter=max_iter)

    status, times, _results, error = time_cuda(fn, warmup=warmup, repeats=repeats)
    return row(
        case_name=case.name,
        n_bus=case.n_bus,
        implementation="torch_csr_implicit_gpu",
        backend="cuda",
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
        status=status,
        times_ms=times,
        converged="",
        iterations="",
        notes="CSR Ybus/Jacobian PyTorch Newton path on CUDA with torch.sparse.spsolve; backward is intended as implicit-adjoint, not unrolled autograd.",
        error=error,
    )


def run_cupf_reference(
    case: Any,
    *,
    cupf: Any,
    device: torch.device,
    compute: str,
    warmup: int,
    repeats: int,
    tolerance: float,
    max_iter: int,
) -> dict[str, Any]:
    session = CupfSolverSession(
        case,
        cupf=cupf,
        backend="cuda",
        compute=compute,
        tolerance=tolerance,
        max_iter=max_iter,
    )
    tensors = session.make_torch_static_inputs(batch_size=1, device=device)

    def fn() -> Any:
        return session.forward_torch_extension(
            tensors,
            prepare_adjoint_cache=False,
            allow_explicit_transpose_fallback=True,
        )

    status, times, results, error = time_cuda(fn, warmup=warmup, repeats=repeats)
    converged = bool(all(getattr(result, "success", False) for result in results)) if results else ""
    return row(
        case_name=case.name,
        n_bus=case.n_bus,
        implementation="cupf_reference",
        backend="cuda",
        dtype=compute,
        warmup=warmup,
        repeats=repeats,
        status=status,
        times_ms=times,
        converged=converged,
        iterations="",
        notes="cuPF CUDA forward via torch extension zero-copy, prepare_adjoint_cache=False.",
        error=error,
    )


def add_speedups(rows: list[dict[str, Any]]) -> None:
    by_case: dict[str, dict[str, float]] = {}
    for item in rows:
        if item["status"] != "ok":
            continue
        value = float(item["total_time_ms_mean"])
        if not math.isfinite(value):
            continue
        by_case.setdefault(str(item["case_name"]), {})[str(item["implementation"])] = value

    for item in rows:
        value = float(item["total_time_ms_mean"])
        if item["status"] != "ok" or not math.isfinite(value) or value <= 0.0:
            continue
        baselines = by_case.get(str(item["case_name"]), {})
        pypower = baselines.get("pypower_newtonpf", math.nan)
        torch_dense = baselines.get("torch_dense_gpu", math.nan)
        torch_csr = baselines.get("torch_csr_implicit_gpu", math.nan)
        if math.isfinite(pypower):
            item["speedup_vs_pypower"] = pypower / value
        if math.isfinite(torch_dense):
            item["speedup_vs_torch_dense_gpu"] = torch_dense / value
        if math.isfinite(torch_csr):
            item["speedup_vs_torch_csr_implicit_gpu"] = torch_csr / value


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_speedups(rows)
    csv_path = output_dir / "same_case_speed_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "config": {
            "cases": args.cases,
            "dataset_dir": str(args.dataset_dir),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "torch_dense_device": "cuda",
            "torch_dense_dtype": args.dtype,
            "torch_csr_implicit_device": "cuda",
            "torch_csr_implicit_dtype": args.dtype,
            "cupf_reference_backend": "cuda",
            "cupf_reference_compute": args.cupf_compute,
            "pypower_scope": "Newton kernel over dumped Ybus/Sbus/V0/pv/pq",
            "timing": "torch.cuda.Event for CUDA paths; time.perf_counter for PYPOWER CPU path",
        },
        "rows": rows,
    }
    (output_dir / "same_case_speed_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Same-Case Forward Speed Comparison",
        "",
        f"- Cases: {', '.join(args.cases)}",
        f"- Warmup/repeats: {args.warmup}/{args.repeats}",
        "- PyTorch dense: CUDA dense Ybus/Jacobian path, torch.cuda.Event timing.",
        "- PyTorch CSR implicit: CUDA CSR Ybus/Jacobian path using torch.sparse.spsolve when available; intended backward is implicit adjoint.",
        "- PYPOWER: CPU PYPOWER Newton kernel over the same dumped Ybus/Sbus/V0/pv/pq inputs; MATPOWER file loading/Ybus construction excluded.",
        f"- cuPF reference: CUDA cuPF forward via torch extension zero-copy, compute={args.cupf_compute}, no adjoint cache.",
        "",
        "| case | buses | implementation | status | mean ms | median ms | speedup vs PYPOWER | speedup vs PyTorch dense | speedup vs PyTorch CSR |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        def fmt(value: Any) -> str:
            try:
                fval = float(value)
            except Exception:
                return ""
            return f"{fval:.3f}" if math.isfinite(fval) else ""

        lines.append(
            f"| {item['case_name']} | {item['n_bus']} | {item['implementation']} | {item['status']} | "
            f"{fmt(item['total_time_ms_mean'])} | {fmt(item['total_time_ms_median'])} | "
            f"{fmt(item['speedup_vs_pypower'])} | {fmt(item['speedup_vs_torch_dense_gpu'])} | "
            f"{fmt(item['speedup_vs_torch_csr_implicit_gpu'])} |"
        )
    (output_dir / "same_case_speed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because this comparison fixes PyTorch dense to the GPU path")
    device = torch.device("cuda")
    cupf = import_cupf([])
    case_dirs = list_case_dirs(args.dataset_dir, case_names=args.cases)

    rows: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        case = load_case(case_dir, dtype=args.dtype)
        print(f"[case] {case.name} ({case.n_bus} buses)", flush=True)
        rows.append(
            run_pypower(case, warmup=args.warmup, repeats=args.repeats, tolerance=args.tolerance, max_iter=args.max_iter)
        )
        rows.append(
            run_torch_dense_gpu(
                case,
                device=device,
                dtype=args.dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
        )
        rows.append(
            run_torch_csr_implicit_gpu(
                case,
                device=device,
                dtype=args.dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
        )
        rows.append(
            run_cupf_reference(
                case,
                cupf=cupf,
                device=device,
                compute=args.cupf_compute,
                warmup=args.warmup,
                repeats=args.repeats,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
        )
        write_outputs(args.output_dir, rows, args)

    write_outputs(args.output_dir, rows, args)
    print(f"wrote {args.output_dir / 'same_case_speed_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
