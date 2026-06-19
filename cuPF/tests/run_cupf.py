#!/usr/bin/env python3
"""Unified cuPF power-flow runner — one entry point for all three execution paths.

    python3 cuPF/tests/run_cupf.py --path cpp    --backend cuda-custom --precision tf32 --mode solve
    python3 cuPF/tests/run_cupf.py --path python --backend cpu-klu     --precision fp64 --mode debug
    python3 cuPF/tests/run_cupf.py --path torch  --backend cuda-cudss  --precision mixed --grad

Execution paths:
  cpp     the `cupf_bench` C++ benchmark (CUDA only; reads a dump-case dir)
  python  the `_cupf` pybind module (CPU klu/umfpack + CUDA cudss/custom)
  torch   the torch autograd wrapper (CUDA only; optional --grad does a backward pass)

Measurement modes (the test is about *what* we measure, not batch vs single):
  solve      (default) optimized initialize() + solve() wall time — no per-stage sync
  operators  per-operator breakdown (ibus/mismatch/jacobian/factor/solve/vupd) — cpp only
  debug      per-system convergence + residual

Backends/precision are a single option surface; only what a path supports is applied
(CPU is python-only; tf32 is the custom backend only; per-operator timing is cpp-only).
Cases are resolved + preprocessed with benchmark/common/matpower_data (pandapower).
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.common import matpower_data  # noqa: E402  (reuse case loading / dump writing)

CUPF_BUILD = REPO_ROOT / "cuPF" / "build"
CPP_BENCH = CUPF_BUILD / "tests" / "cupf_bench"

BACKENDS = ("cuda-custom", "cuda-cudss", "cpu-klu", "cpu-umfpack")
PRECISIONS = ("fp64", "fp32", "mixed", "tf32")
JACOBIANS = ("edge", "edge_atomic", "vertex_warp")
PATHS = ("cpp", "python", "torch")
MODES = ("solve", "operators", "debug")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unified cuPF power-flow runner.")
    p.add_argument("--path", choices=PATHS, default="cpp")
    p.add_argument("--backend", choices=BACKENDS, default="cuda-custom")
    p.add_argument("--precision", choices=PRECISIONS, default="tf32")
    p.add_argument("--mode", choices=MODES, default="solve")
    p.add_argument("--jacobian", choices=JACOBIANS, default="edge",
                   help="GPU Jacobian assembly kind (CUDA paths only).")
    p.add_argument("--cases", nargs="*", default=["case118"],
                   help="MATPOWER case names/paths (resolved under --dataset-root).")
    p.add_argument("--dataset-root", type=Path, default=matpower_data.DEFAULT_DATASET_ROOT)
    p.add_argument("--batch", type=int, default=1, help="Batch size (not the focus; default 1).")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--scale-step", type=float, default=0.001,
                   help="Per-scenario load scaling so batched systems differ.")
    # custom-solver knobs (cuda-custom only)
    p.add_argument("--matching", choices=("none", "structural"), default="none")
    p.add_argument("--pivot", choices=("none", "shift"), default="shift")
    p.add_argument("--pivot-eps", type=float, default=1e-8)
    p.add_argument("--grad", action="store_true",
                   help="torch path only: also run a backward (adjoint) pass.")
    p.add_argument("--build", action="store_true",
                   help="(Re)build cuPF with the flags this path needs, then run.")
    return p.parse_args(argv)


def validate(args):
    # tf32 is a CUDA-custom factor precision; CPU ignores precision (always FP64).
    if args.precision == "tf32" and args.backend != "cuda-custom" and not args.backend.startswith("cpu-"):
        sys.exit("tf32 is a custom-solver factor precision; use --backend cuda-custom.")
    if args.mode == "operators" and args.path != "cpp":
        sys.exit("--mode operators is cpp-only (the pybind module exposes no per-operator timers).")
    if args.path == "torch" and args.backend.startswith("cpu-"):
        sys.exit("--path torch is CUDA-only; use --path python or cpp for CPU klu/umfpack.")
    if args.path == "torch" and args.mode != "solve":
        sys.exit("--path torch supports --mode solve only (forward/backward timing).")
    if args.path == "cpp" and args.backend.startswith("cpu-") and args.batch > 1:
        sys.exit("CPU backend supports batch=1 only; use --batch 1.")


def resolve_cases(args):
    return matpower_data.resolve_case_paths(args.dataset_root, args.cases)


def build_cupf(need_bindings: bool):
    """Configure + build cuPF. Bindings/torch are needed for python/torch paths."""
    import torch  # noqa: F401  (only to locate its cmake prefix when needed)
    cmd = ["cmake", "-S", str(REPO_ROOT / "cuPF"), "-B", str(CUPF_BUILD),
           "-DCMAKE_BUILD_TYPE=Release", "-DWITH_CUDA=ON",
           "-DENABLE_TIMING=ON", "-DCUPF_ENABLE_CUSTOM_SOLVER=ON"]
    if need_bindings:
        cmd += ["-DBUILD_PYTHON_BINDINGS=ON", "-DCUPF_WITH_TORCH=ON",
                f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}"]
    subprocess.run(cmd, check=True)
    subprocess.run(["cmake", "--build", str(CUPF_BUILD), "-j"], check=True)


# --- path: cpp (cupf_bench) --------------------------------------------------
def run_cpp(args, cases):
    if not CPP_BENCH.exists():
        sys.exit(f"{CPP_BENCH} not built. Run with --build, or:\n"
                 f"  cmake -S cuPF -B cuPF/build -DWITH_CUDA=ON -DENABLE_TIMING=ON "
                 f"-DCUPF_ENABLE_CUSTOM_SOLVER=ON -DBUILD_EVALUATORS=ON && cmake --build cuPF/build -j")
    if args.backend.startswith("cpu-"):
        spec = args.backend                       # cpu-klu / cpu-umfpack (FP64; precision ignored)
    else:
        solver = "custom" if args.backend == "cuda-custom" else "cudss"
        spec = f"{solver}-{args.precision}"
    with tempfile.TemporaryDirectory(prefix="cupf-dump-") as tmp:
        for case_path in cases:
            case = matpower_data.load_case(case_path)
            dump_dir = matpower_data.save_dump_case(case, tmp)
            cmd = [str(CPP_BENCH), str(dump_dir), spec, str(args.batch), args.mode,
                   str(args.repeats), str(args.max_iter), str(args.scale_step),
                   args.matching, args.pivot, str(args.pivot_eps), args.jacobian]
            print(f"\n== {case.case_name} | cpp spec={spec} mode={args.mode} "
                  f"jac={args.jacobian} B={args.batch} ==", flush=True)
            subprocess.run(cmd, check=True)


# --- path: python (_cupf pybind) --------------------------------------------
def _import_cupf():
    so = next(CUPF_BUILD.rglob("_cupf*.so"), None)
    if so is None:
        sys.exit("_cupf module not built. Run with --build, or build cuPF with "
                 "-DBUILD_PYTHON_BINDINGS=ON.")
    sys.path.insert(0, str(so.parent))
    return importlib.import_module("_cupf")


def _python_options(cupf, args):
    o = cupf.NewtonOptions()
    if args.backend.startswith("cpu-"):
        o.backend = cupf.BackendKind.CPU
        o.cpu_linear_solver = (cupf.CpuLinearSolverKind.KLU if args.backend == "cpu-klu"
                               else cupf.CpuLinearSolverKind.UMFPACK)
        o.compute = cupf.ComputePolicy.FP64  # CPU is FP64 internally
        return o
    o.backend = cupf.BackendKind.CUDA
    o.cuda_linear_solver = (cupf.CudaLinearSolverKind.Custom if args.backend == "cuda-custom"
                            else cupf.CudaLinearSolverKind.CuDSS)
    o.compute = {"fp64": cupf.ComputePolicy.FP64, "fp32": cupf.ComputePolicy.FP32,
                 "mixed": cupf.ComputePolicy.Mixed, "tf32": cupf.ComputePolicy.Mixed}[args.precision]
    o.cuda_jacobian = {"edge": cupf.CudaJacobianKind.Edge,
                       "edge_atomic": cupf.CudaJacobianKind.EdgeAtomic,
                       "vertex_warp": cupf.CudaJacobianKind.VertexWarp}[args.jacobian]
    if args.backend == "cuda-custom":
        none_match = getattr(cupf.CustomMatchingMode, "None")   # "None" is a Python keyword
        none_pivot = getattr(cupf.CustomPivotStrategy, "None")
        o.custom.precision = {"fp64": cupf.CustomPrecision.FP64, "fp32": cupf.CustomPrecision.FP32,
                              "mixed": cupf.CustomPrecision.FP32, "tf32": cupf.CustomPrecision.TF32}[args.precision]
        o.custom.matching = (cupf.CustomMatchingMode.Structural if args.matching == "structural" else none_match)
        o.custom.pivot_strategy = (cupf.CustomPivotStrategy.StaticDiagonalShift if args.pivot == "shift"
                                   else none_pivot)
        o.custom.shift_retry_epsilon = args.pivot_eps
    return o


def _make_batch(c, B, scale_step):
    import numpy as np
    n = c.ybus.shape[0]
    scales = (1.0 + scale_step * np.arange(B)).reshape(B, 1)
    sbus = (c.sbus.reshape(1, n) * scales).astype(np.complex128)
    v0 = np.tile(c.v0.reshape(1, n), (B, 1)).astype(np.complex128)
    return sbus, v0


def run_python(args, cases):
    import numpy as np
    cupf = _import_cupf()
    opts = _python_options(cupf, args)
    B = args.batch

    if args.mode == "solve":
        print("case,buses,batch,backend,precision,init_ms,solve_ms,solve_per_sys_ms,converged,relres", flush=True)
    else:  # debug
        print("case,buses,batch,backend,precision,system,iterations,converged,relres", flush=True)

    for case_path in cases:
        c = matpower_data.load_case(case_path)
        yb = c.ybus
        ip, idx, data = yb.indptr.astype(np.int32), yb.indices.astype(np.int32), yb.data
        n = yb.shape[0]
        pv, pq = c.pv.astype(np.int32), c.pq.astype(np.int32)
        sbus, v0 = _make_batch(c, B, args.scale_step)
        cfg = cupf.NRConfig(); cfg.tolerance = args.tol; cfg.max_iter = args.max_iter
        sopt = cupf.SolveOptions()

        # initialize() — timed (solve_batch returns host arrays, so wall timing is clean)
        t = time.perf_counter()
        solver = cupf.NewtonSolver(opts)
        solver.initialize(ip, idx, data, n, n, pv, pq)
        init_ms = (time.perf_counter() - t) * 1000.0

        # Prefer batch path; fall back to single-solve loop where unsupported (CPU).
        use_batch = B > 1
        if use_batch:
            try:
                _solve(solver, ip, idx, data, n, sbus, v0, pv, pq, cfg, sopt, True)
            except RuntimeError:
                use_batch = False
        for _ in range(2):  # warmup
            _solve(solver, ip, idx, data, n, sbus, v0, pv, pq, cfg, sopt, use_batch)
        runs = [_solve(solver, ip, idx, data, n, sbus, v0, pv, pq, cfg, sopt, use_batch)
                for _ in range(args.repeats)]
        solve_ms, V, iters, conv = min(runs, key=lambda r: r[0])

        if args.mode == "solve":
            relres = matpower_data.mismatch_norm(yb, sbus[0], V[0], c.pv, c.pq)
            print(f"{c.case_name},{n},{B},{args.backend},{args.precision},"
                  f"{init_ms:.3f},{solve_ms:.3f},{solve_ms/B:.4f},{bool(np.all(conv))},{relres:.3e}", flush=True)
        else:  # debug — per system
            for b in range(B):
                relres = matpower_data.mismatch_norm(yb, sbus[b], V[b], c.pv, c.pq)
                it = iters[b] if isinstance(iters, (list, tuple)) else iters
                cv = conv[b] if isinstance(conv, (list, tuple)) else conv
                print(f"{c.case_name},{n},{B},{args.backend},{args.precision},{b},{it},{bool(cv)},{relres:.3e}",
                      flush=True)


def _solve(solver, ip, idx, data, n, sbus, v0, pv, pq, cfg, sopt, use_batch):
    """One measured run over sbus.shape[0] systems. Returns (wall_ms, V[B,n], iters, conv)."""
    import numpy as np
    t = time.perf_counter()
    if use_batch:
        r = solver.solve_batch(ip, idx, data, n, n, sbus, v0, pv, pq, cfg, sopt)
        return ((time.perf_counter() - t) * 1000.0,
                np.asarray(r.V_numpy).reshape(-1, n), r.iterations, r.converged)
    Vs, iters, conv = [], [], []
    for b in range(sbus.shape[0]):
        r = solver.solve(ip, idx, data, n, n, sbus[b], v0[b], pv, pq, cfg, sopt)
        Vs.append(np.asarray(r.V_numpy)); iters.append(r.iterations); conv.append(r.converged)
    return ((time.perf_counter() - t) * 1000.0, np.array(Vs), iters, conv)


# --- path: torch (autograd) --------------------------------------------------
def run_torch(args, cases):
    import numpy as np
    import torch
    cupf_mod = _import_cupf()
    sys.path.insert(0, str(REPO_ROOT / "cuPF" / "python"))
    import cupf as cupf_pkg  # the torch wrapper package (cupf.solve)

    B = args.batch
    dtype = torch.float64 if args.precision == "fp64" else torch.float32
    dev = "cuda"
    opts = _python_options(cupf_mod, args)
    solver = cupf_mod.NewtonSolver(opts)

    print("case,buses,batch,backend,precision,grad,fwd_ms", flush=True)
    for case_path in cases:
        c = matpower_data.load_case(case_path)
        yb = c.ybus
        n = yb.shape[0]
        solver.initialize(yb.indptr.astype(np.int32), yb.indices.astype(np.int32), yb.data,
                          n, n, c.pv.astype(np.int32), c.pq.astype(np.int32))
        sbus_re = torch.tensor(c.sbus.real, dtype=dtype, device=dev)
        sbus_im = torch.tensor(c.sbus.imag, dtype=dtype, device=dev)
        v0_vm = torch.tensor(np.abs(c.v0), dtype=dtype, device=dev)
        v0_va = torch.tensor(np.angle(c.v0), dtype=dtype, device=dev)
        load_p = torch.zeros((B, n), dtype=dtype, device=dev, requires_grad=args.grad)
        load_q = torch.zeros((B, n), dtype=dtype, device=dev, requires_grad=args.grad)
        cfg = cupf_mod.NRConfig(); cfg.tolerance = args.tol; cfg.max_iter = args.max_iter
        sopt = cupf_mod.SolveOptions(); sopt.prepare_adjoint_cache = args.grad

        def once():
            va, vm = cupf_pkg.solve(load_p, load_q, solver, sbus_base_re=sbus_re, sbus_base_im=sbus_im,
                                    v0_va=v0_va, v0_vm=v0_vm, config=cfg, solve_options=sopt)
            if args.grad:
                (va.sum() + vm.sum()).backward()
            torch.cuda.synchronize()

        once()  # warmup
        t = time.perf_counter(); once(); fwd_ms = (time.perf_counter() - t) * 1000.0
        print(f"{c.case_name},{n},{B},{args.backend},{args.precision},{args.grad},{fwd_ms:.3f}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    validate(args)
    if args.build:
        build_cupf(need_bindings=args.path in ("python", "torch"))
    cases = resolve_cases(args)
    {"cpp": run_cpp, "python": run_python, "torch": run_torch}[args.path](args, cases)


if __name__ == "__main__":
    main()
