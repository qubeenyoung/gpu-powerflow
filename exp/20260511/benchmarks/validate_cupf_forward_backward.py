from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.utils import (  # type: ignore
        DEFAULT_DATASET_DIR,
        CupfSolverSession,
        cpu_newton_pf,
        deterministic_weights,
        ensure_output_dir,
        import_cupf,
        list_case_dirs,
        load_case,
        markdown_table,
        perturb_load_sbus,
        relative_error,
        run_cupf_pf,
        sample_parameter_indices,
        scalar_voltage_loss,
        set_seed,
        summarize_array,
        write_csv,
        write_json,
    )
else:
    from .utils import (
        DEFAULT_DATASET_DIR,
        CupfSolverSession,
        cpu_newton_pf,
        deterministic_weights,
        ensure_output_dir,
        import_cupf,
        list_case_dirs,
        load_case,
        markdown_table,
        perturb_load_sbus,
        relative_error,
        run_cupf_pf,
        sample_parameter_indices,
        scalar_voltage_loss,
        set_seed,
        summarize_array,
        write_csv,
        write_json,
    )


FORWARD_FIELDS = [
    "dataset_id",
    "dataset_name",
    "converged_cpu",
    "converged_cupf",
    "mean_relative_error",
    "max_relative_error",
    "mean_relative_error_vm",
    "max_relative_error_vm",
    "mean_relative_error_va",
    "max_relative_error_va",
    "num_buses",
    "num_branches",
    "dtype",
    "device",
]

BACKWARD_FIELDS = [
    "dataset_id",
    "dataset_name",
    "parameter_name",
    "num_checked_elements",
    "finite_difference_h",
    "mean_relative_error",
    "max_relative_error",
    "mean_abs_error",
    "max_abs_error",
    "backend",
    "used_adjoint_cache",
    "adjoint_cache_matches_final_state",
    "transpose_solve_backend",
    "reused_forward_factorization",
    "reused_final_state_factorization",
    "refactorized_for_backward",
    "used_explicit_transpose",
    "used_python_scipy",
    "includes_host_device_transfer",
    "zero_copy",
    "torch_extension_zero_copy",
    "raw_pointer_api_used",
    "current_stream_integrated",
    "jt_symbolic_analyzed_at_initialize",
    "jt_values_transposed_on_device",
    "jt_factorized_during_forward_cache",
    "jt_refactorized_during_backward",
    "host_roundtrip_for_jt_transpose",
    "factorization_time_ms",
    "transpose_solve_time_ms",
    "timing_scope",
    "jt_residual_norm",
    "sign_convention",
    "dtype",
    "device",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cuPF pybind forward results and native cuPF adjoint/backward gradients."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--num-datasets", type=int, default=78)
    parser.add_argument("--cases", nargs="*", default=None, help="Optional case directory names under dataset-dir.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--cupf-compute", choices=["fp64", "fp32", "mixed"], default="fp64")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--fd-params", nargs="+", default=["load_p", "load_q"])
    parser.add_argument("--fd-h", type=float, default=1e-4)
    parser.add_argument("--fd-samples", type=int, default=128)
    parser.add_argument("--fd-full", action="store_true")
    parser.add_argument("--reuse-forward-factorization", action="store_true")
    parser.add_argument("--no-prepare-adjoint-cache", action="store_true")
    parser.add_argument("--allow-explicit-transpose-fallback", action="store_true", default=None)
    parser.add_argument(
        "--allow-backward-refactorize",
        action="store_true",
        help="Allow solve_adjoint() to assemble/factorize on cache miss. Default keeps benchmark on cached backward path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--cupf-python-path", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/cupf_validation"))
    parser.add_argument(
        "--skip-backward",
        action="store_true",
        help="Only run forward validation. Useful for quick pybind smoke tests.",
    )
    return parser.parse_args()


def finite_difference_gradient(
    case: Any,
    *,
    cupf: Any,
    backend: str,
    compute: str,
    parameter_name: str,
    indices: np.ndarray,
    h: float,
    w_vm: np.ndarray,
    w_va: np.ndarray,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, list[str]]:
    fd = np.full(indices.shape, np.nan, dtype=np.float64)
    notes: list[str] = []
    for pos, idx in enumerate(indices):
        sbus_plus = perturb_load_sbus(case, parameter_name, int(idx), h)
        sbus_minus = perturb_load_sbus(case, parameter_name, int(idx), -h)
        try:
            plus = run_cupf_pf(
                case,
                cupf=cupf,
                backend=backend,
                compute=compute,
                sbus=sbus_plus,
                tolerance=tolerance,
                max_iter=max_iter,
            )
            minus = run_cupf_pf(
                case,
                cupf=cupf,
                backend=backend,
                compute=compute,
                sbus=sbus_minus,
                tolerance=tolerance,
                max_iter=max_iter,
            )
            if not (plus.converged and minus.converged):
                notes.append(f"{parameter_name}[{int(idx)}]: non-converged finite-difference endpoint")
                continue
            loss_plus = scalar_voltage_loss(plus.v, w_vm, w_va)
            loss_minus = scalar_voltage_loss(minus.v, w_vm, w_va)
            fd[pos] = (loss_plus - loss_minus) / (2.0 * h)
        except Exception as exc:
            notes.append(f"{parameter_name}[{int(idx)}]: {type(exc).__name__}: {exc}")
    return fd, notes


def run() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_output_dir(args.output_dir)
    eps = float(args.eps if args.eps is not None else (1e-6 if args.dtype == "float32" else 1e-8))
    backend = "cuda" if args.device == "cuda" else "cpu"
    cupf = import_cupf(args.cupf_python_path)

    case_dirs = list_case_dirs(args.dataset_dir, num_datasets=args.num_datasets, case_names=args.cases)
    forward_rows: list[dict[str, Any]] = []
    backward_rows: list[dict[str, Any]] = []
    global_forward_errors: list[float] = []
    global_vm_errors: list[float] = []
    global_va_errors: list[float] = []
    backward_errors: dict[str, list[float]] = {name: [] for name in args.fd_params}
    backward_abs_errors: dict[str, list[float]] = {name: [] for name in args.fd_params}
    notes: list[str] = []

    cpu_converged_count = 0
    cupf_converged_count = 0
    both_converged_count = 0

    for dataset_id, case_dir in enumerate(case_dirs):
        case = load_case(case_dir, dtype=args.dtype)
        try:
            cpu = cpu_newton_pf(case, tolerance=args.tolerance, max_iter=args.max_iter)
        except Exception as exc:
            cpu = None
            notes.append(f"{case.name}: CPU reference failed: {type(exc).__name__}: {exc}")

        cupf_session = None
        try:
            cupf_session = CupfSolverSession(
                case,
                cupf=cupf,
                backend=backend,
                compute=args.cupf_compute,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
            cupf_result = cupf_session.forward(
                prepare_adjoint_cache=not args.no_prepare_adjoint_cache,
                allow_explicit_transpose_fallback=(
                    backend == "cuda"
                    if args.allow_explicit_transpose_fallback is None
                    else args.allow_explicit_transpose_fallback
                ),
            )
        except Exception as exc:
            cupf_result = None
            notes.append(f"{case.name}: cuPF forward failed: {type(exc).__name__}: {exc}")

        converged_cpu = bool(cpu and cpu.converged)
        converged_cupf = bool(cupf_result and cupf_result.converged)
        cpu_converged_count += int(converged_cpu)
        cupf_converged_count += int(converged_cupf)

        row = {
            "dataset_id": dataset_id,
            "dataset_name": case.name,
            "converged_cpu": converged_cpu,
            "converged_cupf": converged_cupf,
            "mean_relative_error": math.nan,
            "max_relative_error": math.nan,
            "mean_relative_error_vm": math.nan,
            "max_relative_error_vm": math.nan,
            "mean_relative_error_va": math.nan,
            "max_relative_error_va": math.nan,
            "num_buses": case.n_bus,
            "num_branches": case.n_branch_like,
            "dtype": args.dtype,
            "device": args.device,
        }

        if cpu is not None and cupf_result is not None and converged_cpu and converged_cupf:
            both_converged_count += 1
            vm_cpu = np.abs(cpu.v)
            vm_cupf = np.abs(cupf_result.v)
            va_cpu = np.angle(cpu.v)
            va_cupf = np.angle(cupf_result.v)
            err_vm = relative_error(vm_cupf, vm_cpu, eps)
            err_va = relative_error(va_cupf, va_cpu, eps)
            err_all = np.r_[err_vm, err_va]
            row.update(
                {
                    "mean_relative_error": float(np.nanmean(err_all)),
                    "max_relative_error": float(np.nanmax(err_all)),
                    "mean_relative_error_vm": float(np.nanmean(err_vm)),
                    "max_relative_error_vm": float(np.nanmax(err_vm)),
                    "mean_relative_error_va": float(np.nanmean(err_va)),
                    "max_relative_error_va": float(np.nanmax(err_va)),
                }
            )
            global_forward_errors.extend(err_all[np.isfinite(err_all)].tolist())
            global_vm_errors.extend(err_vm[np.isfinite(err_vm)].tolist())
            global_va_errors.extend(err_va[np.isfinite(err_va)].tolist())
        forward_rows.append(row)

        if args.skip_backward or cupf_result is None or cupf_session is None or not converged_cupf:
            continue

        w_vm, w_va = deterministic_weights(case.n_bus, dtype=args.dtype)
        try:
            if backend == "cuda" and hasattr(cupf_session.solver, "solve_adjoint_torch"):
                import torch

                torch_session = CupfSolverSession(
                    case,
                    cupf=cupf,
                    backend=backend,
                    compute=args.cupf_compute,
                    tolerance=args.tolerance,
                    max_iter=args.max_iter,
                )
                tensors = torch_session.make_torch_static_inputs(batch_size=1, device=torch.device("cuda"))
                torch_session.forward_torch_extension(
                    tensors,
                    prepare_adjoint_cache=not args.no_prepare_adjoint_cache,
                    allow_explicit_transpose_fallback=(
                        backend == "cuda"
                        if args.allow_explicit_transpose_fallback is None
                        else args.allow_explicit_transpose_fallback
                    ),
                )
                ext_dtype = tensors["load_p"].dtype
                w_vm_ext, w_va_ext = deterministic_weights(
                    case.n_bus,
                    dtype="float64" if ext_dtype == torch.float64 else "float32",
                )
                grad_vm_t = torch.as_tensor(w_vm_ext[None, :], dtype=ext_dtype, device="cuda").contiguous()
                grad_va_t = torch.as_tensor(w_va_ext[None, :], dtype=ext_dtype, device="cuda").contiguous()
                grad_p_t = torch.empty_like(tensors["load_p"])
                grad_q_t = torch.empty_like(tensors["load_q"])
                native_grads = torch_session.solve_native_adjoint_torch_extension(
                    grad_vm_tensor=grad_vm_t,
                    grad_va_tensor=grad_va_t,
                    grad_load_p_out=grad_p_t,
                    grad_load_q_out=grad_q_t,
                    require_cached_factorization=True,
                    check_residual=False,
                    copy_to_numpy=True,
                )
            else:
                native_grads = cupf_session.solve_native_adjoint(
                    grad_vm=w_vm,
                    grad_va=w_va,
                    reuse_forward_factorization=args.reuse_forward_factorization,
                    require_cached_factorization=True,
                    allow_refactorize_for_backward=args.allow_backward_refactorize,
                    allow_explicit_transpose_fallback=(
                        backend == "cuda"
                        if args.allow_explicit_transpose_fallback is None
                        else args.allow_explicit_transpose_fallback
                    ),
                    check_residual=True,
                )
        except Exception as exc:
            notes.append(f"{case.name}: native cuPF backward failed: {type(exc).__name__}: {exc}")
            continue
        if native_grads.used_python_scipy:
            notes.append(f"{case.name}: native cuPF backward unexpectedly reported Python/SciPy usage")

        for param_name in args.fd_params:
            param_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(param_name))
            indices = sample_parameter_indices(
                case,
                param_name,
                fd_samples=args.fd_samples,
                full=args.fd_full,
                seed=args.seed + dataset_id * 1009 + param_seed,
            )
            fd_grad, fd_notes = finite_difference_gradient(
                case,
                cupf=cupf,
                backend=backend,
                compute=args.cupf_compute,
                parameter_name=param_name,
                indices=indices,
                h=args.fd_h,
                w_vm=w_vm,
                w_va=w_va,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
            notes.extend(f"{case.name}: {note}" for note in fd_notes)
            if param_name == "load_p":
                grad = native_grads.grad_load_p[0, indices]
            elif param_name == "load_q":
                grad = native_grads.grad_load_q[0, indices]
            else:
                notes.append(f"{case.name}: unsupported native backward parameter {param_name}")
                continue
            valid = np.isfinite(fd_grad) & np.isfinite(grad)
            abs_err = np.abs(grad[valid] - fd_grad[valid])
            rel_err = relative_error(grad[valid], fd_grad[valid], eps) if np.any(valid) else np.array([])
            backward_errors.setdefault(param_name, []).extend(rel_err[np.isfinite(rel_err)].tolist())
            backward_abs_errors.setdefault(param_name, []).extend(abs_err[np.isfinite(abs_err)].tolist())

            backward_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": case.name,
                    "parameter_name": param_name,
                    "num_checked_elements": int(np.count_nonzero(valid)),
                    "finite_difference_h": args.fd_h,
                    "mean_relative_error": float(np.nanmean(rel_err)) if rel_err.size else math.nan,
                    "max_relative_error": float(np.nanmax(rel_err)) if rel_err.size else math.nan,
                    "mean_abs_error": float(np.nanmean(abs_err)) if abs_err.size else math.nan,
                    "max_abs_error": float(np.nanmax(abs_err)) if abs_err.size else math.nan,
                    "backend": native_grads.backend,
                    "used_adjoint_cache": native_grads.used_adjoint_cache,
                    "adjoint_cache_matches_final_state": native_grads.adjoint_cache_matches_final_state,
                    "transpose_solve_backend": native_grads.transpose_solve_backend,
                    "reused_forward_factorization": native_grads.reused_forward_factorization,
                    "reused_final_state_factorization": native_grads.reused_final_state_factorization,
                    "refactorized_for_backward": native_grads.refactorized_for_backward,
                    "used_explicit_transpose": native_grads.used_explicit_transpose,
                    "used_python_scipy": native_grads.used_python_scipy,
                    "includes_host_device_transfer": native_grads.includes_host_device_transfer,
                    "zero_copy": native_grads.zero_copy,
                    "torch_extension_zero_copy": native_grads.torch_extension_zero_copy,
                    "raw_pointer_api_used": native_grads.raw_pointer_api_used,
                    "current_stream_integrated": native_grads.current_stream_integrated,
                    "jt_symbolic_analyzed_at_initialize": native_grads.jt_symbolic_analyzed_at_initialize,
                    "jt_values_transposed_on_device": native_grads.jt_values_transposed_on_device,
                    "jt_factorized_during_forward_cache": native_grads.jt_factorized_during_forward_cache,
                    "jt_refactorized_during_backward": native_grads.jt_refactorized_during_backward,
                    "host_roundtrip_for_jt_transpose": native_grads.host_roundtrip_for_jt_transpose,
                    "factorization_time_ms": native_grads.factorization_time_ms,
                    "transpose_solve_time_ms": native_grads.transpose_solve_time_ms,
                    "timing_scope": "backward_only_cached",
                    "jt_residual_norm": native_grads.jt_residual_norm,
                    "sign_convention": native_grads.sign_convention,
                    "dtype": args.dtype,
                    "device": args.device,
                }
            )

    forward_summary = {
        "num_datasets": len(case_dirs),
        "cpu_converged": cpu_converged_count,
        "cupf_converged": cupf_converged_count,
        "both_converged": both_converged_count,
        "aggregate_policy": "relative-error aggregate includes only datasets where CPU and cuPF both converged",
        "overall": summarize_array(global_forward_errors),
        "vm": summarize_array(global_vm_errors),
        "va": summarize_array(global_va_errors),
    }
    backward_summary = {
        "implementation": "cupf_native_adjoint",
        "python_scipy_used_as_cupf_backward": False,
        "factorization_reuse_policy": (
            "reuse requested" if args.reuse_forward_factorization else "reuse not requested"
        ),
        "refactorize_allowed": args.allow_backward_refactorize,
        "parameters": {
            param_name: {
                "relative_error": summarize_array(backward_errors.get(param_name, [])),
                "absolute_error": summarize_array(backward_abs_errors.get(param_name, [])),
            }
            for param_name in args.fd_params
        },
    }
    legacy_backward_summary = {
        param_name: {
            "relative_error": summarize_array(backward_errors.get(param_name, [])),
            "absolute_error": summarize_array(backward_abs_errors.get(param_name, [])),
        }
        for param_name in args.fd_params
    }
    backward_summary.update({"legacy_parameter_view": legacy_backward_summary})

    write_csv(output_dir / "forward_summary.csv", forward_rows, FORWARD_FIELDS)
    write_csv(output_dir / "backward_summary.csv", backward_rows, BACKWARD_FIELDS)
    write_json(
        output_dir / "summary.json",
        {
            "config": vars(args),
            "forward": forward_summary,
            "backward": backward_summary,
            "notes": notes,
        },
    )

    md_lines = [
        "# cuPF Forward/Native Backward Validation",
        "",
        f"- datasets: {len(case_dirs)}",
        f"- CPU converged: {cpu_converged_count}/{len(case_dirs)}",
        f"- cuPF converged: {cupf_converged_count}/{len(case_dirs)}",
        f"- both converged: {both_converged_count}/{len(case_dirs)}",
        f"- forward aggregate policy: {forward_summary['aggregate_policy']}",
        "",
        "## Forward Aggregate",
        markdown_table(
            ["metric", "mean", "max"],
            [
                ["all", forward_summary["overall"]["mean"], forward_summary["overall"]["max"]],
                ["Vm", forward_summary["vm"]["mean"], forward_summary["vm"]["max"]],
                ["Va", forward_summary["va"]["mean"], forward_summary["va"]["max"]],
            ],
        ),
        "",
        "## Backward Aggregate",
        markdown_table(
            ["parameter", "mean rel err", "max rel err", "mean abs err", "max abs err"],
            [
                [
                    name,
                    backward_summary["parameters"][name]["relative_error"]["mean"],
                    backward_summary["parameters"][name]["relative_error"]["max"],
                    backward_summary["parameters"][name]["absolute_error"]["mean"],
                    backward_summary["parameters"][name]["absolute_error"]["max"],
                ]
                for name in args.fd_params
            ],
        ),
    ]
    if notes:
        md_lines.extend(["", "## Notes", *[f"- {note}" for note in notes[:50]]])
        if len(notes) > 50:
            md_lines.append(f"- ... {len(notes) - 50} more notes in summary.json")
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote validation outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
