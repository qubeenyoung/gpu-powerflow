from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .matpower_data import (
    DEFAULT_DATASET_ROOT,
    PreprocessedCase,
    ReferenceResult,
    load_case,
    mismatch_norm,
    resolve_case_paths,
    save_dump_case,
    solve_reference,
    voltage_error,
    write_manifest_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluator" / "results"

RUN_FIELDNAMES = [
    "mode",
    "case_name",
    "case_path",
    "backend",
    "compute",
    "cuda_linear_solver",
    "repeat_idx",
    "warmup",
    "success",
    "error_message",
    "n_bus",
    "ybus_nnz",
    "n_ref",
    "n_pv",
    "n_pq",
    "initialize_ms",
    "solve_ms",
    "device_solve_ms",
    "cupf_reported_total_ms",
    "cupf_converged",
    "cupf_iterations",
    "cupf_final_mismatch",
    "output_mismatch",
    "reference_converged",
    "reference_iterations",
    "reference_final_mismatch",
    "max_abs_v_error",
    "rms_abs_v_error",
    "max_abs_vm_error",
    "max_abs_va_error",
]


def now_run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class CsvSink:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "CsvSink":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cases", nargs="*", help="Case names or .m paths. Defaults to all case*.m files.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases after resolution.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=now_run_name())
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--reference-tolerance", type=float, default=1e-10)
    parser.add_argument("--reference-max-iter", type=int, default=80)
    parser.add_argument("--continue-on-error", action="store_true", default=True)


def add_cupf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--compute", choices=("fp64", "fp32", "mixed"), default="mixed")
    parser.add_argument("--cuda-linear-solver", choices=("cudss", "custom"), default="cudss")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--cupf-python-path",
        action="append",
        type=Path,
        default=[],
        help="Extra path containing cupf package or _cupf extension. Can be repeated.",
    )


def output_dir(args: argparse.Namespace, mode: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return DEFAULT_OUTPUT_ROOT / args.run_name / mode


def selected_case_paths(args: argparse.Namespace) -> list[Path]:
    paths = resolve_case_paths(args.dataset_root, args.cases)
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
    return paths


def manifest(args: argparse.Namespace, mode: str, cases: list[Path]) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dataset_root": str(args.dataset_root),
        "cases": [str(path) for path in cases],
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "reference_tolerance": args.reference_tolerance,
        "reference_max_iter": args.reference_max_iter,
        "warmup": getattr(args, "warmup", None),
        "repeats": getattr(args, "repeats", None),
        "backend": getattr(args, "backend", None),
        "compute": getattr(args, "compute", None),
        "cuda_linear_solver": getattr(args, "cuda_linear_solver", None),
    }


def load_case_and_reference(path: Path, args: argparse.Namespace) -> tuple[PreprocessedCase, ReferenceResult]:
    case = load_case(path)
    reference = solve_reference(case, args.reference_tolerance, args.reference_max_iter)
    return case, reference


def prepare_command(args: argparse.Namespace) -> None:
    cases = selected_case_paths(args)
    out = output_dir(args, "prepare")
    dump_root = out / "dumps"
    rows: list[dict[str, object]] = []
    for path in cases:
        try:
            case, reference = load_case_and_reference(path, args)
            case_dir = save_dump_case(case, dump_root, reference)
            rows.append(
                {
                    "case_name": case.case_name,
                    "case_path": str(path),
                    "dump_dir": str(case_dir),
                    "success": True,
                    "error_message": "",
                    "n_bus": int(case.ybus.shape[0]),
                    "ybus_nnz": int(case.ybus.nnz),
                    "n_ref": int(case.ref.size),
                    "n_pv": int(case.pv.size),
                    "n_pq": int(case.pq.size),
                    "reference_converged": reference.converged,
                    "reference_iterations": reference.iterations,
                    "reference_final_mismatch": reference.final_mismatch,
                }
            )
            print(f"[prepare][OK] {case.case_name} -> {case_dir}", flush=True)
        except Exception as exc:
            rows.append({"case_name": path.stem, "case_path": str(path), "success": False, "error_message": repr(exc)})
            print(f"[prepare][FAIL] {path}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
    write_manifest_csv(out / "manifest.csv", rows)
    (out / "run.json").write_text(json.dumps(manifest(args, "prepare", cases), indent=2, sort_keys=True), encoding="utf-8")
    print(f"[prepare] wrote {out}")


def add_cupf_import_paths(extra_paths: list[Path]) -> None:
    candidates = [REPO_ROOT / "cuPF" / "python", *extra_paths]
    for build_dir in (REPO_ROOT / "cuPF").glob("build*"):
        candidates.append(build_dir)
        candidates.extend(path.parent for path in build_dir.rglob("_cupf*.so"))
    for path in candidates:
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def import_cupf(extra_paths: list[Path]):
    add_cupf_import_paths(extra_paths)
    try:
        return importlib.import_module("cupf")
    except Exception:
        return importlib.import_module("_cupf")


def cupf_options(cupf: Any, args: argparse.Namespace) -> Any:
    opts = cupf.NewtonOptions()
    opts.backend = cupf.BackendKind.CUDA if args.backend == "cuda" else cupf.BackendKind.CPU
    opts.compute = {
        "fp64": cupf.ComputePolicy.FP64,
        "fp32": cupf.ComputePolicy.FP32,
        "mixed": cupf.ComputePolicy.Mixed,
    }[args.compute]
    opts.cuda_linear_solver = {
        "cudss": cupf.CudaLinearSolverKind.CuDSS,
        "custom": cupf.CudaLinearSolverKind.Custom,
    }[args.cuda_linear_solver]
    return opts


def cupf_config(cupf: Any, args: argparse.Namespace) -> Any:
    config = cupf.NRConfig()
    config.tolerance = args.tolerance
    config.max_iter = args.max_iter
    return config


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return
    except Exception:
        pass
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def base_row(mode: str, case: PreprocessedCase, reference: ReferenceResult, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": mode,
        "case_name": case.case_name,
        "case_path": str(case.source_path),
        "backend": args.backend,
        "compute": args.compute,
        "cuda_linear_solver": args.cuda_linear_solver,
        "warmup": args.warmup,
        "success": True,
        "error_message": "",
        "n_bus": int(case.ybus.shape[0]),
        "ybus_nnz": int(case.ybus.nnz),
        "n_ref": int(case.ref.size),
        "n_pv": int(case.pv.size),
        "n_pq": int(case.pq.size),
        "reference_converged": reference.converged,
        "reference_iterations": reference.iterations,
        "reference_final_mismatch": reference.final_mismatch,
    }


def python_solve_once(cupf: Any, case: PreprocessedCase, args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray]:
    opts = cupf_options(cupf, args)
    config = cupf_config(cupf, args)
    solver = cupf.NewtonSolver(opts)
    synchronize_cuda()
    init_start = time.perf_counter()
    solver.initialize(
        case.ybus.indptr,
        case.ybus.indices,
        case.ybus.data,
        case.ybus.shape[0],
        case.ybus.shape[1],
        case.pv,
        case.pq,
    )
    synchronize_cuda()
    initialize_ms = (time.perf_counter() - init_start) * 1000.0

    solve_start = time.perf_counter()
    result = solver.solve(
        case.ybus.indptr,
        case.ybus.indices,
        case.ybus.data,
        case.ybus.shape[0],
        case.ybus.shape[1],
        case.sbus,
        case.v0,
        case.pv,
        case.pq,
        config,
    )
    synchronize_cuda()
    solve_ms = (time.perf_counter() - solve_start) * 1000.0
    voltage = np.asarray(result.V_numpy, dtype=np.complex128)
    row = {
        "initialize_ms": initialize_ms,
        "solve_ms": solve_ms,
        "device_solve_ms": "",
        "cupf_reported_total_ms": "",
        "cupf_converged": bool(result.converged),
        "cupf_iterations": int(result.iterations),
        "cupf_final_mismatch": float(result.final_mismatch),
        "output_mismatch": mismatch_norm(case.ybus, case.sbus, voltage, case.pv, case.pq),
    }
    return row, voltage


def python_command(args: argparse.Namespace) -> None:
    cupf = import_cupf(args.cupf_python_path)
    cases = selected_case_paths(args)
    out = output_dir(args, "python")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(manifest(args, "python", cases), indent=2, sort_keys=True), encoding="utf-8")
    with CsvSink(out / "runs.csv", RUN_FIELDNAMES) as sink:
        for path in cases:
            try:
                case, reference = load_case_and_reference(path, args)
                for _ in range(args.warmup):
                    python_solve_once(cupf, case, args)
                for repeat_idx in range(args.repeats):
                    row, voltage = python_solve_once(cupf, case, args)
                    row = {**base_row("python", case, reference, args), **row, **voltage_error(voltage, reference.voltage)}
                    row["repeat_idx"] = repeat_idx
                    sink.write(row)
                    print(
                        f"[python][OK] {case.case_name} repeat={repeat_idx} "
                        f"init_ms={row['initialize_ms']:.3f} solve_ms={row['solve_ms']:.3f} "
                        f"err={row['max_abs_v_error']:.3e}",
                        flush=True,
                    )
            except Exception as exc:
                fail = {"mode": "python", "case_name": path.stem, "case_path": str(path), "success": False, "error_message": repr(exc)}
                sink.write(fail)
                print(f"[python][FAIL] {path}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
    print(f"[python] wrote {out / 'runs.csv'}")


def torch_dtype(args: argparse.Namespace):
    import torch

    return torch.float64 if args.compute == "fp64" else torch.float32


def torch_solve_once(cupf: Any, case: PreprocessedCase, args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    if args.backend != "cuda":
        raise ValueError("torch evaluator requires --backend cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA is not available")
    if not hasattr(cupf.NewtonSolver, "solve_with_adjoint_cache_torch"):
        raise RuntimeError("cuPF was not built with CUPF_WITH_TORCH=ON")

    opts = cupf_options(cupf, args)
    config = cupf_config(cupf, args)
    solver = cupf.NewtonSolver(opts)
    torch.cuda.synchronize()
    init_start = time.perf_counter()
    solver.initialize(
        case.ybus.indptr,
        case.ybus.indices,
        case.ybus.data,
        case.ybus.shape[0],
        case.ybus.shape[1],
        case.pv,
        case.pq,
    )
    torch.cuda.synchronize()
    initialize_ms = (time.perf_counter() - init_start) * 1000.0

    device = torch.device("cuda")
    dtype = torch_dtype(args)
    sbus_base_re = torch.as_tensor(case.sbus.real, device=device, dtype=dtype)
    sbus_base_im = torch.as_tensor(case.sbus.imag, device=device, dtype=dtype)
    v0_complex = torch.as_tensor(case.v0, device=device)
    v0_va = torch.angle(v0_complex).to(dtype)
    v0_vm = torch.abs(v0_complex).to(dtype)
    load_p = torch.zeros((1, case.ybus.shape[0]), device=device, dtype=dtype)
    load_q = torch.zeros_like(load_p)
    va_out = torch.empty_like(load_p)
    vm_out = torch.empty_like(load_q)
    solve_options = cupf.SolveOptions()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    solve_start = time.perf_counter()
    start_event.record()
    result = solver.solve_with_adjoint_cache_torch(
        sbus_base_re,
        sbus_base_im,
        load_p,
        load_q,
        v0_va,
        v0_vm,
        va_out,
        vm_out,
        config,
        solve_options,
    )
    end_event.record()
    torch.cuda.synchronize()
    solve_ms = (time.perf_counter() - solve_start) * 1000.0
    device_ms = float(start_event.elapsed_time(end_event))

    va = va_out.detach().cpu().numpy().reshape(-1).astype(np.float64)
    vm = vm_out.detach().cpu().numpy().reshape(-1).astype(np.float64)
    voltage = vm * np.exp(1j * va)
    output_mismatch = mismatch_norm(case.ybus, case.sbus, voltage, case.pv, case.pq)
    row = {
        "initialize_ms": initialize_ms,
        "solve_ms": solve_ms,
        "device_solve_ms": device_ms,
        "cupf_reported_total_ms": float(getattr(result, "total_time_ms", 0.0)),
        "cupf_converged": output_mismatch <= args.tolerance,
        "cupf_iterations": "",
        "cupf_final_mismatch": "",
        "output_mismatch": output_mismatch,
    }
    return row, voltage


def torch_command(args: argparse.Namespace) -> None:
    cupf = import_cupf(args.cupf_python_path)
    cases = selected_case_paths(args)
    out = output_dir(args, "torch")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(manifest(args, "torch", cases), indent=2, sort_keys=True), encoding="utf-8")
    with CsvSink(out / "runs.csv", RUN_FIELDNAMES) as sink:
        for path in cases:
            try:
                case, reference = load_case_and_reference(path, args)
                for _ in range(args.warmup):
                    torch_solve_once(cupf, case, args)
                for repeat_idx in range(args.repeats):
                    row, voltage = torch_solve_once(cupf, case, args)
                    row = {**base_row("torch", case, reference, args), **row, **voltage_error(voltage, reference.voltage)}
                    row["repeat_idx"] = repeat_idx
                    sink.write(row)
                    print(
                        f"[torch][OK] {case.case_name} repeat={repeat_idx} "
                        f"init_ms={row['initialize_ms']:.3f} solve_ms={row['solve_ms']:.3f} "
                        f"device_ms={row['device_solve_ms']:.3f} err={row['max_abs_v_error']:.3e}",
                        flush=True,
                    )
            except Exception as exc:
                fail = {"mode": "torch", "case_name": path.stem, "case_path": str(path), "success": False, "error_message": repr(exc)}
                sink.write(fail)
                print(f"[torch][FAIL] {path}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
    print(f"[torch] wrote {out / 'runs.csv'}")


def find_cpp_executable(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = sorted((REPO_ROOT / "cuPF").glob("build*/**/cupf_cpp_evaluate"))
    candidates += sorted(REPO_ROOT.glob("build*/**/cupf_cpp_evaluate"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("cupf_cpp_evaluate not found. Build with -DBUILD_EVALUATORS=ON.")


def cpp_command(args: argparse.Namespace) -> None:
    cases = selected_case_paths(args)
    out = output_dir(args, "cpp")
    dump_root = args.dump_root if args.dump_root is not None else out / "dumps"
    dump_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    if not args.skip_prepare:
        for path in cases:
            try:
                case, reference = load_case_and_reference(path, args)
                case_dir = save_dump_case(case, dump_root, reference)
                manifest_rows.append({"case_name": case.case_name, "case_path": str(path), "dump_dir": str(case_dir), "success": True})
                print(f"[cpp][prepare][OK] {case.case_name} -> {case_dir}", flush=True)
            except Exception as exc:
                manifest_rows.append({"case_name": path.stem, "case_path": str(path), "success": False, "error_message": repr(exc)})
                print(f"[cpp][prepare][FAIL] {path}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
        write_manifest_csv(out / "prepare_manifest.csv", manifest_rows)

    exe = find_cpp_executable(args.executable)
    cmd = [
        str(exe),
        "--case-root",
        str(dump_root),
        "--output-dir",
        str(out),
        "--backend",
        args.backend,
        "--compute",
        args.compute,
        "--cuda-linear-solver",
        args.cuda_linear_solver,
        "--tolerance",
        str(args.tolerance),
        "--max-iter",
        str(args.max_iter),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
    ]
    if args.cases:
        cmd.extend(["--cases", *[Path(case).stem for case in args.cases]])
    print("[cpp] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    (out / "run.json").write_text(json.dumps(manifest(args, "cpp", cases), indent=2, sort_keys=True), encoding="utf-8")
    print(f"[cpp] wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cuPF evaluator for Python, torch, and C++ backends.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert MATPOWER .m cases to cuPF dump directories.")
    add_common_args(prepare)
    prepare.set_defaults(func=prepare_command)

    py = subparsers.add_parser("python", help="Evaluate the pybind cuPF solve() path.")
    add_common_args(py)
    add_cupf_args(py)
    py.set_defaults(func=python_command)

    torch_parser = subparsers.add_parser("torch", help="Evaluate the torch zero-copy cuPF path.")
    add_common_args(torch_parser)
    add_cupf_args(torch_parser)
    torch_parser.set_defaults(func=torch_command)

    cpp = subparsers.add_parser("cpp", help="Prepare dumps and run the C++ evaluator executable.")
    add_common_args(cpp)
    add_cupf_args(cpp)
    cpp.add_argument("--dump-root", type=Path, default=None)
    cpp.add_argument("--skip-prepare", action="store_true")
    cpp.add_argument("--executable", type=Path, default=None)
    cpp.set_defaults(func=cpp_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
