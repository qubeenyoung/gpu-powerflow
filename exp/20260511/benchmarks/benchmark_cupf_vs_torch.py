from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.utils import (  # type: ignore
        DEFAULT_DATASET_DIR,
        CupfSolverSession,
        deterministic_weights,
        ensure_output_dir,
        import_cupf,
        list_case_dirs,
        load_case,
        markdown_table,
        run_cupf_pf,
        set_seed,
        summarize_array,
        write_csv,
        write_json,
    )
else:
    from .utils import (
        DEFAULT_DATASET_DIR,
        CupfSolverSession,
        deterministic_weights,
        ensure_output_dir,
        import_cupf,
        list_case_dirs,
        load_case,
        markdown_table,
        run_cupf_pf,
        set_seed,
        summarize_array,
        write_csv,
        write_json,
    )


RUNTIME_FIELDS = [
    "implementation",
    "backend",
    "batch_size",
    "phase",
    "timing_scope",
    "warmup",
    "repeats",
    "total_time_ms_mean",
    "total_time_ms_median",
    "total_time_ms_std",
    "total_time_ms_min",
    "total_time_ms_max",
    "per_scenario_time_ms_mean",
    "speedup_vs_torch",
    "includes_host_device_transfer",
    "zero_copy",
    "torch_extension_zero_copy",
    "raw_pointer_api_used",
    "current_stream_integrated",
    "used_adjoint_cache",
    "adjoint_cache_matches_final_state",
    "jt_symbolic_analyzed_at_initialize",
    "jt_values_transposed_on_device",
    "jt_factorized_during_forward_cache",
    "jt_refactorized_during_backward",
    "host_roundtrip_for_jt_transpose",
    "reused_forward_factorization",
    "reused_final_state_factorization",
    "refactorized_for_backward",
    "used_explicit_transpose",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cuPF pybind against a torch tensor Newton PF baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--case-name", type=str, default=None, help="Case directory name. Defaults to the first case.")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64, 256])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--cupf-compute", choices=["fp64", "fp32", "mixed"], default="mixed")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-dense-buses", type=int, default=2000)
    parser.add_argument("--cupf-python-path", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/cupf_benchmark"))
    return parser.parse_args()


def torch_dtypes(dtype: str) -> tuple[torch.dtype, torch.dtype]:
    if dtype == "float32":
        return torch.float32, torch.complex64
    return torch.float64, torch.complex128


def torch_voltage_loss(v: torch.Tensor, w_vm: torch.Tensor, w_va: torch.Tensor) -> torch.Tensor:
    return (torch.abs(v) * w_vm[None, :] + torch.angle(v) * w_va[None, :]).sum()


def torch_newton_pf(
    ybus: torch.Tensor,
    sbus: torch.Tensor,
    v0: torch.Tensor,
    pv: torch.Tensor,
    pq: torch.Tensor,
    *,
    tolerance: float,
    max_iter: int,
) -> torch.Tensor:
    v = v0
    va = torch.angle(v)
    vm = torch.abs(v)
    pvpq = torch.cat([pv, pq])
    n_pvpq = int(pvpq.numel())
    eye_dtype = va.dtype

    for _ in range(int(max_iter)):
        i_bus = v @ ybus.T
        mis = v * torch.conj(i_bus) - sbus
        f = torch.cat([mis[:, pv].real, mis[:, pq].real, mis[:, pq].imag], dim=1)
        if bool((torch.max(torch.abs(f), dim=1).values <= tolerance).all().detach().cpu().item()):
            break

        vm_safe = torch.clamp(torch.abs(v), min=1e-8)
        v_norm = v / vm_safe
        diag_i = torch.diag_embed(i_bus)
        y_diag_v = ybus.unsqueeze(0) * v[:, None, :]
        d_s_d_va = 1j * v[:, :, None] * torch.conj(diag_i - y_diag_v)

        y_diag_vnorm = ybus.unsqueeze(0) * v_norm[:, None, :]
        diag_term = torch.diag_embed(torch.conj(i_bus) * v_norm)
        d_s_d_vm = v[:, :, None] * torch.conj(y_diag_vnorm) + diag_term

        j11 = d_s_d_va.index_select(1, pvpq).index_select(2, pvpq).real
        j12 = d_s_d_vm.index_select(1, pvpq).index_select(2, pq).real
        j21 = d_s_d_va.index_select(1, pq).index_select(2, pvpq).imag
        j22 = d_s_d_vm.index_select(1, pq).index_select(2, pq).imag
        jac = torch.cat([torch.cat([j11, j12], dim=2), torch.cat([j21, j22], dim=2)], dim=1)

        dx = torch.linalg.solve(jac, f.to(dtype=eye_dtype).unsqueeze(-1)).squeeze(-1)
        d_va = torch.zeros_like(va)
        d_vm = torch.zeros_like(vm)
        d_va = d_va.scatter(1, pvpq[None, :].expand(v.shape[0], -1), dx[:, :n_pvpq])
        d_vm = d_vm.scatter(1, pq[None, :].expand(v.shape[0], -1), dx[:, n_pvpq:])
        va = va - d_va
        vm = vm - d_vm
        v = torch.polar(vm, va)
    return v


def make_torch_inputs(case: Any, batch_size: int, device: torch.device, dtype: str) -> dict[str, torch.Tensor]:
    real_dtype, complex_dtype = torch_dtypes(dtype)
    ybus_dense = case.ybus.toarray()
    ybus = torch.as_tensor(ybus_dense, dtype=complex_dtype, device=device)
    sbus_base = torch.as_tensor(case.sbus, dtype=complex_dtype, device=device).repeat(batch_size, 1)
    v0 = torch.as_tensor(case.v0, dtype=complex_dtype, device=device).repeat(batch_size, 1)
    pv = torch.as_tensor(case.pv.astype(np.int64), dtype=torch.long, device=device)
    pq = torch.as_tensor(case.pq.astype(np.int64), dtype=torch.long, device=device)
    w_vm_np, w_va_np = deterministic_weights(case.n_bus, dtype=dtype)
    w_vm = torch.as_tensor(w_vm_np, dtype=real_dtype, device=device)
    w_va = torch.as_tensor(w_va_np, dtype=real_dtype, device=device)
    return {
        "ybus": ybus,
        "sbus_base": sbus_base,
        "v0": v0,
        "pv": pv,
        "pq": pq,
        "w_vm": w_vm,
        "w_va": w_va,
    }


class CupfNativePF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        load_delta_p: torch.Tensor,
        load_delta_q: torch.Tensor,
        case: Any,
        cupf: Any,
        backend: str,
        compute: str,
        tolerance: float,
        max_iter: int,
    ) -> torch.Tensor:
        device = load_delta_p.device
        out_dtype = load_delta_p.dtype
        delta_p = load_delta_p.detach().cpu().numpy()
        delta_q = load_delta_q.detach().cpu().numpy()
        sbus = case.sbus[None, :] - (delta_p + 1j * delta_q)
        session = CupfSolverSession(
            case,
            cupf=cupf,
            backend=backend,
            compute=compute,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        result = session.forward(
            sbus=sbus,
            batch_size=int(load_delta_p.shape[0]),
            prepare_adjoint_cache=True,
            allow_explicit_transpose_fallback=(backend == "cuda"),
        )
        v = np.asarray(result.v, dtype=np.complex128).reshape(load_delta_p.shape[0], case.n_bus)
        state_np = np.stack([np.abs(v), np.angle(v)], axis=-1)
        ctx.case = case
        ctx.session = session
        return torch.as_tensor(state_np, dtype=out_dtype, device=device)

    @staticmethod
    def backward(ctx: Any, grad_state: torch.Tensor) -> tuple[Any, ...]:
        grad_np = grad_state.detach().cpu().numpy()
        native = ctx.session.solve_native_adjoint(
            grad_vm=grad_np[:, :, 0].astype(np.float64, copy=False),
            grad_va=grad_np[:, :, 1].astype(np.float64, copy=False),
            reuse_forward_factorization=False,
            require_cached_factorization=True,
            allow_refactorize_for_backward=False,
            allow_explicit_transpose_fallback=True,
            check_residual=False,
        )
        grad_p = torch.as_tensor(native.grad_load_p, dtype=grad_state.dtype, device=grad_state.device)
        grad_q = torch.as_tensor(native.grad_load_q, dtype=grad_state.dtype, device=grad_state.device)
        return grad_p, grad_q, None, None, None, None, None, None


def time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
    prepare: Callable[[], Any] | None = None,
) -> list[float]:
    for _ in range(int(warmup)):
        if prepare is not None:
            prepare()
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(int(repeats)):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if prepare is not None:
                prepare()
            torch.cuda.synchronize()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
        else:
            if prepare is not None:
                prepare()
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def stats_row(
    *,
    implementation: str,
    backend: str,
    batch_size: int,
    phase: str,
    timing_scope: str,
    warmup: int,
    repeats: int,
    times_ms: list[float],
    includes_host_device_transfer: bool = False,
    zero_copy: bool = False,
    torch_extension_zero_copy: bool = False,
    raw_pointer_api_used: bool = False,
    current_stream_integrated: bool = False,
    used_adjoint_cache: bool = False,
    adjoint_cache_matches_final_state: bool = False,
    jt_symbolic_analyzed_at_initialize: bool = False,
    jt_values_transposed_on_device: bool = False,
    jt_factorized_during_forward_cache: bool = False,
    jt_refactorized_during_backward: bool = False,
    host_roundtrip_for_jt_transpose: bool = False,
    reused_forward_factorization: bool = False,
    reused_final_state_factorization: bool = False,
    refactorized_for_backward: bool = False,
    used_explicit_transpose: bool = False,
) -> dict[str, Any]:
    stats = summarize_array(times_ms)
    return {
        "implementation": implementation,
        "backend": backend,
        "batch_size": batch_size,
        "phase": phase,
        "timing_scope": timing_scope,
        "warmup": warmup,
        "repeats": repeats,
        "total_time_ms_mean": stats["mean"],
        "total_time_ms_median": stats["median"],
        "total_time_ms_std": stats["std"],
        "total_time_ms_min": stats["min"],
        "total_time_ms_max": stats["max"],
        "per_scenario_time_ms_mean": stats["mean"] / float(batch_size) if np.isfinite(stats["mean"]) else math.nan,
        "speedup_vs_torch": math.nan,
        "includes_host_device_transfer": includes_host_device_transfer,
        "zero_copy": zero_copy,
        "torch_extension_zero_copy": torch_extension_zero_copy,
        "raw_pointer_api_used": raw_pointer_api_used,
        "current_stream_integrated": current_stream_integrated,
        "used_adjoint_cache": used_adjoint_cache,
        "adjoint_cache_matches_final_state": adjoint_cache_matches_final_state,
        "jt_symbolic_analyzed_at_initialize": jt_symbolic_analyzed_at_initialize,
        "jt_values_transposed_on_device": jt_values_transposed_on_device,
        "jt_factorized_during_forward_cache": jt_factorized_during_forward_cache,
        "jt_refactorized_during_backward": jt_refactorized_during_backward,
        "host_roundtrip_for_jt_transpose": host_roundtrip_for_jt_transpose,
        "reused_forward_factorization": reused_forward_factorization,
        "reused_final_state_factorization": reused_final_state_factorization,
        "refactorized_for_backward": refactorized_for_backward,
        "used_explicit_transpose": used_explicit_transpose,
    }


def run() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_output_dir(args.output_dir)
    requested_device = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    backend = "cuda" if requested_device == "cuda" else "cpu"
    cupf = import_cupf(args.cupf_python_path)

    case_names = [args.case_name] if args.case_name else None
    case_dir = list_case_dirs(args.dataset_dir, num_datasets=1, case_names=case_names)[0]
    case = load_case(case_dir, dtype=args.dtype)
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    if args.device == "cuda" and requested_device == "cpu":
        notes.append("CUDA requested but torch.cuda.is_available() is false; recorded CPU-only timings.")
    if case.n_bus > args.max_dense_buses:
        notes.append(
            f"torch_python baseline skipped: case has {case.n_bus} buses, "
            f"above --max-dense-buses={args.max_dense_buses}."
        )

    for batch_size in args.batch_sizes:
        if case.n_bus <= args.max_dense_buses:
            tensors = make_torch_inputs(case, batch_size, device, args.dtype)

            def torch_forward() -> torch.Tensor:
                with torch.no_grad():
                    return torch_newton_pf(
                        tensors["ybus"],
                        tensors["sbus_base"],
                        tensors["v0"],
                        tensors["pv"],
                        tensors["pq"],
                        tolerance=args.tolerance,
                        max_iter=args.max_iter,
                    )

            def torch_backward() -> None:
                delta_p = torch.zeros((batch_size, case.n_bus), dtype=tensors["w_vm"].dtype, device=device, requires_grad=True)
                delta_q = torch.zeros((batch_size, case.n_bus), dtype=tensors["w_vm"].dtype, device=device, requires_grad=True)
                sbus = tensors["sbus_base"] - torch.complex(delta_p, delta_q)
                v = torch_newton_pf(
                    tensors["ybus"],
                    sbus,
                    tensors["v0"],
                    tensors["pv"],
                    tensors["pq"],
                    tolerance=args.tolerance,
                    max_iter=args.max_iter,
                )
                loss = torch_voltage_loss(v, tensors["w_vm"], tensors["w_va"])
                loss.backward()

            rows.append(
                stats_row(
                    implementation="torch_python",
                    backend="torch_linalg_solve",
                    batch_size=batch_size,
                    phase="forward",
                    timing_scope="forward_only",
                    warmup=args.warmup,
                    repeats=args.repeats,
                    times_ms=time_callable(torch_forward, warmup=args.warmup, repeats=args.repeats, device=device),
                )
            )
            rows.append(
                stats_row(
                    implementation="torch_python",
                    backend="torch_linalg_solve",
                    batch_size=batch_size,
                    phase="backward",
                    timing_scope="forward_plus_backward",
                    warmup=args.warmup,
                    repeats=args.repeats,
                    times_ms=time_callable(torch_backward, warmup=args.warmup, repeats=args.repeats, device=device),
                )
            )

        w_vm_np, w_va_np = deterministic_weights(case.n_bus, dtype=args.dtype)
        real_dtype, _ = torch_dtypes(args.dtype)
        w_vm_t = torch.as_tensor(w_vm_np, dtype=real_dtype, device=device)
        w_va_t = torch.as_tensor(w_va_np, dtype=real_dtype, device=device)

        torch_ext_meta: dict[str, Any] = {
            "backend": f"cupf_{backend}_{args.cupf_compute}",
            "includes_host_device_transfer": False,
            "zero_copy": True,
            "torch_extension_zero_copy": True,
            "raw_pointer_api_used": False,
            "current_stream_integrated": True,
            "used_adjoint_cache": False,
            "adjoint_cache_matches_final_state": False,
            "jt_symbolic_analyzed_at_initialize": False,
            "jt_values_transposed_on_device": False,
            "jt_factorized_during_forward_cache": False,
            "jt_refactorized_during_backward": False,
            "host_roundtrip_for_jt_transpose": False,
            "reused_forward_factorization": False,
            "reused_final_state_factorization": False,
            "refactorized_for_backward": False,
            "used_explicit_transpose": backend == "cuda",
        }
        torch_ext_session: CupfSolverSession | None = None
        torch_ext_tensors: dict[str, Any] | None = None
        torch_ext_grad_va: torch.Tensor | None = None
        torch_ext_grad_vm: torch.Tensor | None = None
        torch_ext_grad_p: torch.Tensor | None = None
        torch_ext_grad_q: torch.Tensor | None = None

        def make_torch_extension_session() -> None:
            nonlocal torch_ext_session, torch_ext_tensors, torch_ext_grad_va, torch_ext_grad_vm, torch_ext_grad_p, torch_ext_grad_q
            torch_ext_session = CupfSolverSession(
                case,
                cupf=cupf,
                backend=backend,
                compute=args.cupf_compute,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
            torch_ext_tensors = torch_ext_session.make_torch_static_inputs(
                batch_size=batch_size,
                device=device,
            )
            ext_dtype = torch_ext_tensors["load_p"].dtype
            w_vm_ext_np, w_va_ext_np = deterministic_weights(
                case.n_bus,
                dtype="float64" if ext_dtype == torch.float64 else "float32",
            )
            torch_ext_grad_vm = torch.as_tensor(
                np.repeat(w_vm_ext_np[None, :], int(batch_size), axis=0),
                dtype=ext_dtype,
                device=device,
            ).contiguous()
            torch_ext_grad_va = torch.as_tensor(
                np.repeat(w_va_ext_np[None, :], int(batch_size), axis=0),
                dtype=ext_dtype,
                device=device,
            ).contiguous()
            torch_ext_grad_p = torch.empty_like(torch_ext_tensors["load_p"])
            torch_ext_grad_q = torch.empty_like(torch_ext_tensors["load_q"])

        def update_torch_ext_meta(native: Any) -> None:
            nonlocal torch_ext_meta
            torch_ext_meta = {
                "backend": native.backend,
                "includes_host_device_transfer": native.includes_host_device_transfer,
                "zero_copy": native.zero_copy,
                "torch_extension_zero_copy": native.torch_extension_zero_copy,
                "raw_pointer_api_used": native.raw_pointer_api_used,
                "current_stream_integrated": native.current_stream_integrated,
                "used_adjoint_cache": native.used_adjoint_cache,
                "adjoint_cache_matches_final_state": native.adjoint_cache_matches_final_state,
                "jt_symbolic_analyzed_at_initialize": native.jt_symbolic_analyzed_at_initialize,
                "jt_values_transposed_on_device": native.jt_values_transposed_on_device,
                "jt_factorized_during_forward_cache": native.jt_factorized_during_forward_cache,
                "jt_refactorized_during_backward": native.jt_refactorized_during_backward,
                "host_roundtrip_for_jt_transpose": native.host_roundtrip_for_jt_transpose,
                "reused_forward_factorization": native.reused_forward_factorization,
                "reused_final_state_factorization": native.reused_final_state_factorization,
                "refactorized_for_backward": native.refactorized_for_backward,
                "used_explicit_transpose": native.used_explicit_transpose,
            }

        def torch_ext_stats_kwargs() -> dict[str, Any]:
            return {k: v for k, v in torch_ext_meta.items() if k != "backend"}

        def cupf_torch_extension_forward_only() -> None:
            if torch_ext_session is None or torch_ext_tensors is None:
                raise RuntimeError("torch extension session was not prepared")
            meta = torch_ext_session.forward_torch_extension(
                torch_ext_tensors,
                prepare_adjoint_cache=False,
                allow_explicit_transpose_fallback=(backend == "cuda"),
            )
            update_torch_ext_meta(meta)

        def cupf_torch_extension_forward_with_cache() -> None:
            if torch_ext_session is None or torch_ext_tensors is None:
                raise RuntimeError("torch extension session was not prepared")
            meta = torch_ext_session.forward_torch_extension(
                torch_ext_tensors,
                prepare_adjoint_cache=True,
                allow_explicit_transpose_fallback=(backend == "cuda"),
            )
            update_torch_ext_meta(meta)

        def cupf_torch_extension_backward_only() -> None:
            if (
                torch_ext_session is None
                or torch_ext_grad_va is None
                or torch_ext_grad_vm is None
                or torch_ext_grad_p is None
                or torch_ext_grad_q is None
            ):
                raise RuntimeError("torch extension backward tensors were not prepared")
            native = torch_ext_session.solve_native_adjoint_torch_extension(
                grad_vm_tensor=torch_ext_grad_vm,
                grad_va_tensor=torch_ext_grad_va,
                grad_load_p_out=torch_ext_grad_p,
                grad_load_q_out=torch_ext_grad_q,
                require_cached_factorization=True,
                check_residual=False,
                copy_to_numpy=False,
            )
            update_torch_ext_meta(native)

        def cupf_torch_extension_forward_plus_backward() -> None:
            cupf_torch_extension_forward_with_cache()
            cupf_torch_extension_backward_only()

        if backend == "cuda" and hasattr(cupf.NewtonSolver, "solve_with_adjoint_cache_torch"):
            make_torch_extension_session()
            try:
                rows.append(
                    stats_row(
                        implementation="cupf_torch_extension_zero_copy",
                        backend=torch_ext_meta["backend"],
                        batch_size=batch_size,
                        phase="forward",
                        timing_scope="forward_only",
                        warmup=args.warmup,
                        repeats=args.repeats,
                        times_ms=time_callable(cupf_torch_extension_forward_only, warmup=args.warmup, repeats=args.repeats, device=device),
                        **torch_ext_stats_kwargs(),
                    )
                )
            except Exception as exc:
                notes.append(f"cupf_torch_extension_zero_copy forward_only skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

            make_torch_extension_session()
            try:
                rows.append(
                    stats_row(
                        implementation="cupf_torch_extension_zero_copy",
                        backend=torch_ext_meta["backend"],
                        batch_size=batch_size,
                        phase="forward",
                        timing_scope="forward_with_adjoint_cache",
                        warmup=args.warmup,
                        repeats=args.repeats,
                        times_ms=time_callable(cupf_torch_extension_forward_with_cache, warmup=args.warmup, repeats=args.repeats, device=device),
                        **torch_ext_stats_kwargs(),
                    )
                )
            except Exception as exc:
                notes.append(f"cupf_torch_extension_zero_copy forward_with_adjoint_cache skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

            make_torch_extension_session()
            try:
                backward_only_times = time_callable(
                    cupf_torch_extension_backward_only,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    device=device,
                    prepare=cupf_torch_extension_forward_with_cache,
                )
                rows.append(
                    stats_row(
                        implementation="cupf_torch_extension_zero_copy",
                        backend=torch_ext_meta["backend"],
                        batch_size=batch_size,
                        phase="backward",
                        timing_scope="backward_only_cached",
                        warmup=args.warmup,
                        repeats=args.repeats,
                        times_ms=backward_only_times,
                        **torch_ext_stats_kwargs(),
                    )
                )
            except Exception as exc:
                notes.append(f"cupf_torch_extension_zero_copy backward_only_cached skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

            make_torch_extension_session()
            try:
                rows.append(
                    stats_row(
                        implementation="cupf_torch_extension_zero_copy",
                        backend=torch_ext_meta["backend"],
                        batch_size=batch_size,
                        phase="backward",
                        timing_scope="forward_plus_backward_cached",
                        warmup=args.warmup,
                        repeats=args.repeats,
                        times_ms=time_callable(cupf_torch_extension_forward_plus_backward, warmup=args.warmup, repeats=args.repeats, device=device),
                        **torch_ext_stats_kwargs(),
                    )
                )
            except Exception as exc:
                notes.append(f"cupf_torch_extension_zero_copy forward_plus_backward_cached skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

        def cupf_forward() -> Any:
            return run_cupf_pf(
                case,
                cupf=cupf,
                backend=backend,
                compute=args.cupf_compute,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
                batch_size=batch_size,
            )

        def cupf_forward_with_cache() -> Any:
            session = CupfSolverSession(
                case,
                cupf=cupf,
                backend=backend,
                compute=args.cupf_compute,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
            return session.forward(
                batch_size=batch_size,
                prepare_adjoint_cache=True,
                allow_explicit_transpose_fallback=(backend == "cuda"),
            )

        def cupf_backward() -> None:
            delta_p = torch.zeros((batch_size, case.n_bus), dtype=real_dtype, device=device, requires_grad=True)
            delta_q = torch.zeros((batch_size, case.n_bus), dtype=real_dtype, device=device, requires_grad=True)
            state = CupfNativePF.apply(
                delta_p,
                delta_q,
                case,
                cupf,
                backend,
                args.cupf_compute,
                args.tolerance,
                args.max_iter,
            )
            loss = (state[:, :, 0] * w_vm_t[None, :] + state[:, :, 1] * w_va_t[None, :]).sum()
            loss.backward()

        grad_vm_np = np.repeat(w_vm_np[None, :], int(batch_size), axis=0)
        grad_va_np = np.repeat(w_va_np[None, :], int(batch_size), axis=0)
        backward_session: CupfSolverSession | None = None
        cupf_backward_meta: dict[str, Any] = {
            "backend": f"cupf_{backend}_{args.cupf_compute}",
            "includes_host_device_transfer": True,
            "reused_forward_factorization": False,
            "refactorized_for_backward": True,
        }

        def prepare_cupf_backward_only() -> None:
            nonlocal backward_session, cupf_backward_meta
            backward_session = CupfSolverSession(
                case,
                cupf=cupf,
                backend=backend,
                compute=args.cupf_compute,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
            backward_session.forward(
                batch_size=batch_size,
                prepare_adjoint_cache=True,
                allow_explicit_transpose_fallback=(backend == "cuda"),
            )

        def cupf_backward_only() -> None:
            nonlocal cupf_backward_meta
            if backward_session is None:
                raise RuntimeError("cupf backward-only prepare callback did not create a session")
            native = backward_session.solve_native_adjoint(
                grad_vm=grad_vm_np,
                grad_va=grad_va_np,
                reuse_forward_factorization=False,
                require_cached_factorization=True,
                allow_refactorize_for_backward=False,
                allow_explicit_transpose_fallback=True,
                check_residual=False,
            )
            cupf_backward_meta = {
                "backend": native.backend,
                "includes_host_device_transfer": native.includes_host_device_transfer,
                "zero_copy": native.zero_copy,
                "used_adjoint_cache": native.used_adjoint_cache,
                "adjoint_cache_matches_final_state": native.adjoint_cache_matches_final_state,
                "reused_forward_factorization": native.reused_forward_factorization,
                "reused_final_state_factorization": native.reused_final_state_factorization,
                "refactorized_for_backward": native.refactorized_for_backward,
                "used_explicit_transpose": native.used_explicit_transpose,
            }

        zc_session: CupfSolverSession | None = None
        zc_meta: dict[str, Any] = {
            "backend": f"cupf_{backend}_{args.cupf_compute}",
            "includes_host_device_transfer": False,
            "zero_copy": True,
            "torch_extension_zero_copy": False,
            "raw_pointer_api_used": True,
            "current_stream_integrated": False,
            "used_adjoint_cache": True,
            "adjoint_cache_matches_final_state": True,
            "jt_symbolic_analyzed_at_initialize": False,
            "jt_values_transposed_on_device": True,
            "jt_factorized_during_forward_cache": True,
            "jt_refactorized_during_backward": False,
            "host_roundtrip_for_jt_transpose": False,
            "reused_forward_factorization": False,
            "reused_final_state_factorization": True,
            "refactorized_for_backward": False,
            "used_explicit_transpose": backend == "cuda",
        }
        zc_dtype = torch.float64 if args.cupf_compute == "fp64" else torch.float32
        w_vm_zc_np, w_va_zc_np = deterministic_weights(
            case.n_bus, dtype="float64" if zc_dtype == torch.float64 else "float32"
        )
        grad_vm_zc = torch.as_tensor(
            np.repeat(w_vm_zc_np[None, :], int(batch_size), axis=0),
            dtype=zc_dtype,
            device=device,
        ).contiguous()
        grad_va_zc = torch.as_tensor(
            np.repeat(w_va_zc_np[None, :], int(batch_size), axis=0),
            dtype=zc_dtype,
            device=device,
        ).contiguous()
        grad_p_zc = torch.empty((batch_size, case.n_bus), dtype=zc_dtype, device=device)
        grad_q_zc = torch.empty((batch_size, case.n_bus), dtype=zc_dtype, device=device)

        def prepare_cupf_zero_copy_backward() -> None:
            nonlocal zc_session
            zc_session = CupfSolverSession(
                case,
                cupf=cupf,
                backend=backend,
                compute=args.cupf_compute,
                tolerance=args.tolerance,
                max_iter=args.max_iter,
            )
            zc_session.forward(
                batch_size=batch_size,
                prepare_adjoint_cache=True,
                allow_explicit_transpose_fallback=(backend == "cuda"),
            )

        def cupf_zero_copy_backward_only() -> None:
            nonlocal zc_meta
            if zc_session is None:
                raise RuntimeError("zero-copy prepare callback did not create a session")
            native = zc_session.solve_native_adjoint_raw_cuda(
                grad_vm_tensor=grad_vm_zc,
                grad_va_tensor=grad_va_zc,
                grad_load_p_out=grad_p_zc,
                grad_load_q_out=grad_q_zc,
                require_cached_factorization=True,
            )
            zc_meta = {
                "backend": native.backend,
                "includes_host_device_transfer": native.includes_host_device_transfer,
                "zero_copy": native.zero_copy,
                "torch_extension_zero_copy": native.torch_extension_zero_copy,
                "raw_pointer_api_used": native.raw_pointer_api_used,
                "current_stream_integrated": native.current_stream_integrated,
                "used_adjoint_cache": native.used_adjoint_cache,
                "adjoint_cache_matches_final_state": native.adjoint_cache_matches_final_state,
                "jt_symbolic_analyzed_at_initialize": native.jt_symbolic_analyzed_at_initialize,
                "jt_values_transposed_on_device": native.jt_values_transposed_on_device,
                "jt_factorized_during_forward_cache": native.jt_factorized_during_forward_cache,
                "jt_refactorized_during_backward": native.jt_refactorized_during_backward,
                "host_roundtrip_for_jt_transpose": native.host_roundtrip_for_jt_transpose,
                "reused_forward_factorization": native.reused_forward_factorization,
                "reused_final_state_factorization": native.reused_final_state_factorization,
                "refactorized_for_backward": native.refactorized_for_backward,
                "used_explicit_transpose": native.used_explicit_transpose,
            }

        try:
            rows.append(
                stats_row(
                    implementation="cupf_pybind_numpy",
                    backend=f"cupf_{backend}_{args.cupf_compute}",
                    batch_size=batch_size,
                    phase="forward",
                    timing_scope="forward_only",
                    warmup=args.warmup,
                    repeats=args.repeats,
                    times_ms=time_callable(cupf_forward, warmup=args.warmup, repeats=args.repeats, device=device),
                    includes_host_device_transfer=True,
                )
            )
        except Exception as exc:
            notes.append(f"cupf_pybind forward skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

        try:
            rows.append(
                stats_row(
                    implementation="cupf_pybind_numpy",
                    backend=f"cupf_{backend}_{args.cupf_compute}",
                    batch_size=batch_size,
                    phase="forward",
                    timing_scope="forward_with_adjoint_cache",
                    warmup=args.warmup,
                    repeats=args.repeats,
                    times_ms=time_callable(
                        cupf_forward_with_cache,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        device=device,
                    ),
                    includes_host_device_transfer=True,
                    zero_copy=False,
                )
            )
        except Exception as exc:
            notes.append(f"cupf_pybind forward_with_adjoint_cache skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

        try:
            cupf_backward_times = time_callable(
                cupf_backward,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            rows.append(
                stats_row(
                    implementation="cupf_pybind_numpy",
                    backend=cupf_backward_meta["backend"],
                    batch_size=batch_size,
                    phase="backward",
                    timing_scope="forward_plus_backward_cached",
                    warmup=args.warmup,
                    repeats=args.repeats,
                    times_ms=cupf_backward_times,
                    includes_host_device_transfer=True,
                    zero_copy=False,
                    used_adjoint_cache=True,
                    adjoint_cache_matches_final_state=True,
                    reused_forward_factorization=False,
                    reused_final_state_factorization=True,
                    refactorized_for_backward=False,
                    used_explicit_transpose=backend == "cuda",
                )
            )
        except Exception as exc:
            notes.append(f"cupf_pybind backward skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

        try:
            cupf_backward_only_times = time_callable(
                cupf_backward_only,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
                prepare=prepare_cupf_backward_only,
            )
            rows.append(
                stats_row(
                    implementation="cupf_pybind_numpy",
                    backend=cupf_backward_meta["backend"],
                    batch_size=batch_size,
                    phase="backward",
                    timing_scope="backward_only_cached",
                    warmup=args.warmup,
                    repeats=args.repeats,
                    times_ms=cupf_backward_only_times,
                    includes_host_device_transfer=cupf_backward_meta["includes_host_device_transfer"],
                    zero_copy=cupf_backward_meta["zero_copy"],
                    used_adjoint_cache=cupf_backward_meta["used_adjoint_cache"],
                    adjoint_cache_matches_final_state=cupf_backward_meta["adjoint_cache_matches_final_state"],
                    reused_forward_factorization=cupf_backward_meta["reused_forward_factorization"],
                    reused_final_state_factorization=cupf_backward_meta["reused_final_state_factorization"],
                    refactorized_for_backward=cupf_backward_meta["refactorized_for_backward"],
                    used_explicit_transpose=cupf_backward_meta["used_explicit_transpose"],
                )
            )
        except Exception as exc:
            notes.append(f"cupf_pybind backward-only skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

        if backend == "cuda":
            try:
                zc_times = time_callable(
                    cupf_zero_copy_backward_only,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    device=device,
                    prepare=prepare_cupf_zero_copy_backward,
                )
                rows.append(
                    stats_row(
                        implementation="cupf_raw_pointer_unsafe",
                        backend=zc_meta["backend"],
                        batch_size=batch_size,
                        phase="backward",
                        timing_scope="backward_only_cached",
                        warmup=args.warmup,
                        repeats=args.repeats,
                        times_ms=zc_times,
                        includes_host_device_transfer=zc_meta["includes_host_device_transfer"],
                        zero_copy=zc_meta["zero_copy"],
                        torch_extension_zero_copy=zc_meta["torch_extension_zero_copy"],
                        raw_pointer_api_used=zc_meta["raw_pointer_api_used"],
                        current_stream_integrated=zc_meta["current_stream_integrated"],
                        used_adjoint_cache=zc_meta["used_adjoint_cache"],
                        adjoint_cache_matches_final_state=zc_meta["adjoint_cache_matches_final_state"],
                        jt_symbolic_analyzed_at_initialize=zc_meta["jt_symbolic_analyzed_at_initialize"],
                        jt_values_transposed_on_device=zc_meta["jt_values_transposed_on_device"],
                        jt_factorized_during_forward_cache=zc_meta["jt_factorized_during_forward_cache"],
                        jt_refactorized_during_backward=zc_meta["jt_refactorized_during_backward"],
                        host_roundtrip_for_jt_transpose=zc_meta["host_roundtrip_for_jt_transpose"],
                        reused_forward_factorization=zc_meta["reused_forward_factorization"],
                        reused_final_state_factorization=zc_meta["reused_final_state_factorization"],
                        refactorized_for_backward=zc_meta["refactorized_for_backward"],
                        used_explicit_transpose=zc_meta["used_explicit_transpose"],
                    )
                )
            except Exception as exc:
                notes.append(f"cupf_raw_pointer_unsafe backward-only skipped for batch={batch_size}: {type(exc).__name__}: {exc}")

    torch_baselines: dict[tuple[int, str], float] = {}
    for row in rows:
        if row["implementation"] == "torch_python":
            torch_baselines[
                (int(row["batch_size"]), str(row["phase"]), str(row["timing_scope"]))
            ] = float(row["total_time_ms_mean"])
    for row in rows:
        key = (int(row["batch_size"]), str(row["phase"]), str(row["timing_scope"]))
        torch_mean = torch_baselines.get(key, math.nan)
        row_mean = float(row["total_time_ms_mean"])
        if row["implementation"] == "torch_python" and np.isfinite(row_mean):
            row["speedup_vs_torch"] = 1.0
        elif np.isfinite(torch_mean) and np.isfinite(row_mean) and row_mean > 0.0:
            row["speedup_vs_torch"] = torch_mean / row_mean

    write_csv(output_dir / "runtime_summary.csv", rows, RUNTIME_FIELDS)
    write_json(
        output_dir / "runtime_summary.json",
        {
            "config": vars(args),
            "case": {"name": case.name, "num_buses": case.n_bus, "num_branches_proxy": case.n_branch_like},
            "timing_method": "torch.cuda.Event with synchronize on CUDA; time.perf_counter on CPU",
            "rows": rows,
            "notes": notes,
        },
    )

    md_lines = [
        "# cuPF vs Torch Runtime Benchmark",
        "",
        f"- case: {case.name}",
        f"- buses: {case.n_bus}",
        f"- device: {requested_device}",
        f"- dtype: {args.dtype}",
        "",
        markdown_table(
            [
                "implementation",
                "backend",
                "batch",
                "phase",
                "scope",
                "mean ms",
                "per scenario ms",
                "speedup vs torch",
            ],
            [
                [
                    row["implementation"],
                    row["backend"],
                    row["batch_size"],
                    row["phase"],
                    row["timing_scope"],
                    row["total_time_ms_mean"],
                    row["per_scenario_time_ms_mean"],
                    row["speedup_vs_torch"],
                ]
                for row in rows
            ],
        ),
    ]
    if notes:
        md_lines.extend(["", "## Notes", *[f"- {note}" for note in notes]])
    (output_dir / "runtime_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote runtime benchmark outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
