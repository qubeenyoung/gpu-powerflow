"""Run representative cuPF variants through the pybind ``NewtonSolver`` API."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from . import eval_common, matpower_data
from .cupf_variants import BUILD_DIRS, filter_variants, find_cupf_module, gpu_available, variants_for_entrypoint


def _insert_import_paths(extra_paths: list[Path]) -> None:
    candidates = [eval_common.REPO_ROOT, eval_common.REPO_ROOT / "cuPF" / "python", *extra_paths]
    for path in candidates:
        if path.exists():
            text = str(path)
            if text in sys.path:
                sys.path.remove(text)
            sys.path.insert(0, text)


def _import_cupf(module_dir: Path):
    _insert_import_paths([module_dir])
    try:
        return importlib.import_module("cupf")
    except Exception:
        return importlib.import_module("_cupf")


def _cupf_options(cupf: Any, variant: dict[str, Any]) -> Any:
    opts = cupf.NewtonOptions()
    opts.backend = cupf.BackendKind.CUDA if variant["backend"] == "cuda" else cupf.BackendKind.CPU
    opts.compute = {
        "fp64": cupf.ComputePolicy.FP64,
        "fp32": cupf.ComputePolicy.FP32,
        "mixed": cupf.ComputePolicy.Mixed,
    }[variant["compute"]]
    opts.cpu_linear_solver = {
        "klu": cupf.CpuLinearSolverKind.KLU,
        "umfpack": cupf.CpuLinearSolverKind.UMFPACK,
    }[variant.get("cpu_linear_solver", "klu")]
    opts.cuda_jacobian = {
        "edge": cupf.CudaJacobianKind.Edge,
        "edge_atomic": cupf.CudaJacobianKind.EdgeAtomic,
        "vertex_warp": cupf.CudaJacobianKind.VertexWarp,
    }[variant.get("cuda_jacobian", "edge")]
    opts.cuda_linear_solver = {
        "cudss": cupf.CudaLinearSolverKind.CuDSS,
        "custom": cupf.CudaLinearSolverKind.Custom,
    }[variant["cuda_linear_solver"]]
    return opts


def _cupf_config(cupf: Any, args: argparse.Namespace) -> Any:
    config = cupf.NRConfig()
    config.tolerance = args.tolerance
    config.max_iter = args.max_iter
    return config


def _synchronize_cuda() -> None:
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


def _solve_once(cupf: Any, case: matpower_data.PreprocessedCase, args: argparse.Namespace, variant: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    solver = cupf.NewtonSolver(_cupf_options(cupf, variant))
    config = _cupf_config(cupf, args)

    _synchronize_cuda()
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
    _synchronize_cuda()
    initialize_ms = (time.perf_counter() - init_start) * 1000.0

    _synchronize_cuda()
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
    _synchronize_cuda()
    solve_ms = (time.perf_counter() - solve_start) * 1000.0
    voltage = np.asarray(result.V_numpy, dtype=np.complex128).reshape(-1)
    mismatch = matpower_data.mismatch_norm(case.ybus, case.sbus, voltage, case.pv, case.pq)
    return (
        {
            "initialize_ms": initialize_ms,
            "solve_ms": solve_ms,
            "total_ms": initialize_ms + solve_ms,
            "converged": eval_common.bool_for_csv(bool(result.converged)),
            "iterations": int(result.iterations),
            "output_mismatch": mismatch,
        },
        voltage,
    )


def run_variant(args: argparse.Namespace, variant: dict[str, Any]) -> None:
    paths = eval_common.selected_case_paths(args)
    out = eval_common.variant_dir(args, variant["variant"])
    build_dir = BUILD_DIRS[variant["build_key"]]
    manifest = eval_common.manifest(
        args,
        "cupf",
        variant["variant"],
        paths,
        backend=variant["backend"],
        compute=variant["compute"],
        linear_solver=variant["linear_solver"],
        jacobian=variant.get("jacobian", ""),
        entrypoint="pybind",
        build_key=variant["build_key"],
    )

    if variant["requires_gpu"] and not gpu_available():
        eval_common.write_skip(out, "requires a CUDA GPU device (none available on this host)", manifest)
        print(f"[{variant['variant']}][SKIP] no CUDA GPU", flush=True)
        return
    module_dir = find_cupf_module(build_dir)
    if module_dir is None:
        eval_common.write_skip(out, f"_cupf module not found under {build_dir}", manifest)
        print(f"[{variant['variant']}][SKIP] missing _cupf module under {build_dir}", flush=True)
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "SKIPPED.txt").unlink(missing_ok=True)
    eval_common.write_json(out / "run.json", manifest)
    cupf = _import_cupf(module_dir)

    with eval_common.CsvSink(out / "runs.csv") as sink:
        for path in paths:
            try:
                case, reference = eval_common.load_case_and_reference(path, args)
                for repeat_idx in range(args.warmup + args.repeats):
                    warmup = repeat_idx < args.warmup
                    measured, voltage = _solve_once(cupf, case, args, variant)
                    row = {
                        "mode": "cupf",
                        "variant": variant["variant"],
                        "case_name": case.case_name,
                        "case_path": str(path),
                        "backend": variant["backend"],
                        "compute": variant["compute"],
                        "linear_solver": variant["linear_solver"],
                        "jacobian": variant.get("jacobian", ""),
                        "entrypoint": "pybind",
                        "repeat_idx": repeat_idx - args.warmup if not warmup else repeat_idx,
                        "warmup": eval_common.bool_for_csv(warmup),
                        "success": "1",
                        "error_message": "",
                        **eval_common.dimensions(case),
                        **measured,
                        **eval_common.reference_fields(reference),
                        **matpower_data.voltage_error(voltage, reference.voltage),
                    }
                    sink.write(row)
                    if not warmup:
                        print(
                            f"[{variant['variant']}][OK] {case.case_name} repeat={row['repeat_idx']} "
                            f"init_ms={row['initialize_ms']:.3f} solve_ms={row['solve_ms']:.3f} "
                            f"iters={row['iterations']} resid={row['output_mismatch']:.3e}",
                            flush=True,
                        )
            except Exception as exc:
                sink.write(
                    {
                        "mode": "cupf",
                        "variant": variant["variant"],
                        "case_name": path.stem,
                        "case_path": str(path),
                        "backend": variant["backend"],
                        "compute": variant["compute"],
                        "linear_solver": variant["linear_solver"],
                        "jacobian": variant.get("jacobian", ""),
                        "entrypoint": "pybind",
                        "success": "0",
                        "converged": "0",
                        "error_message": repr(exc),
                    }
                )
                print(f"[{variant['variant']}][FAIL] {path}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
    print(f"[{variant['variant']}] wrote {out / 'runs.csv'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cuPF pybind benchmark variants.")
    eval_common.add_common_args(parser)
    parser.add_argument("--variants", nargs="*", help="Variant IDs. Defaults to all representative pybind variants.")
    parser.add_argument("--include-diagnostic-fp32", action="store_true")
    parser.add_argument("--single-process", action="store_true", help=argparse.SUPPRESS)
    return parser


def _spawn_single_variant(args: argparse.Namespace, variant_id: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "python.tests.run_cupf_pybind",
        "--single-process",
        "--variants",
        variant_id,
        "--dataset-root",
        str(args.dataset_root),
        "--run-name",
        args.run_name,
        "--output-root",
        str(args.output_root),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--tolerance",
        str(args.tolerance),
        "--max-iter",
        str(args.max_iter),
        "--reference-tolerance",
        str(args.reference_tolerance),
        "--reference-max-iter",
        str(args.reference_max_iter),
        "--no-aggregate",
    ]
    if args.cases:
        cmd += ["--cases", *args.cases]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    variants = filter_variants(variants_for_entrypoint(args.include_diagnostic_fp32, "pybind"), args.variants)
    eval_common.run_root(args).mkdir(parents=True, exist_ok=True)
    if len(variants) > 1 and not args.single_process:
        for variant in variants:
            _spawn_single_variant(args, variant["variant"])
    else:
        for variant in variants:
            run_variant(args, variant)
    if not args.no_aggregate:
        from .aggregate_results import aggregate

        aggregate(eval_common.run_root(args))


if __name__ == "__main__":
    main()
