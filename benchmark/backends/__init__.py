"""Physics backends for the PINN benchmark.

Every backend plays the *physics* role behind the trainable dummy layer:

  * non-differentiable backends (scipy/pypower/pandapower, cupf-pybind, exapf,
    matpower) run a forward Newton solve as a fixed *physics oracle* — they
    provide a reference voltage V_ref (for a validation metric) and a solve time.
    Only the dummy MLP trains (decision 2a).
  * cupf-torch is *differentiable*: it can sit inside the autograd graph, so it
    additionally runs a forward+backward demo (gradient through the NR solve).

A backend returns (V_ref or None, solve_seconds). V_ref is optional — the PINN
residual loss does not need it; it is only used for a reported validation MAE.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.common import matpower_data  # noqa: E402

CUPF_BUILD = REPO_ROOT / "cuPF" / "build"

# name -> differentiable?
BACKENDS = {
    "pypower": False,
    "pandapower": False,
    "cupf-pybind": False,
    "cupf-torch": True,
    "exapf": False,
    "matpower": False,
}


def is_differentiable(name: str) -> bool:
    return BACKENDS.get(name, False)


# --- CPU reference (pypower / pandapower share the pandapower-stack Newton) ----
def _cpu_reference(ct):
    t = time.perf_counter()
    ref = matpower_data.solve_reference(ct["case"], tolerance=1e-10, max_iter=50)
    return np.asarray(ref.voltage), time.perf_counter() - t


# --- cuPF pybind forward solve (CPU or CUDA) ----------------------------------
def _import_cupf():
    so = next(CUPF_BUILD.rglob("_cupf*.so"), None)
    if so is None:
        raise RuntimeError("_cupf not built; build cuPF with -DBUILD_PYTHON_BINDINGS=ON")
    if str(so.parent) not in sys.path:
        sys.path.insert(0, str(so.parent))
    import _cupf
    return _cupf


def _cupf_reference(ct):
    cupf = _import_cupf()
    c = ct["case"]
    yb = c.ybus
    ip, idx, data = yb.indptr.astype(np.int32), yb.indices.astype(np.int32), yb.data
    n = yb.shape[0]
    pv, pq = c.pv.astype(np.int32), c.pq.astype(np.int32)
    opts = cupf.NewtonOptions()
    opts.backend = cupf.BackendKind.CUDA
    opts.compute = cupf.ComputePolicy.FP64
    opts.cuda_linear_solver = cupf.CudaLinearSolverKind.CuDSS
    cfg = cupf.NRConfig(); cfg.tolerance = 1e-8; cfg.max_iter = 30
    solver = cupf.NewtonSolver(opts)
    solver.initialize(ip, idx, data, n, n, pv, pq)
    t = time.perf_counter()
    r = solver.solve(ip, idx, data, n, n, c.sbus.astype(np.complex128),
                     c.v0.astype(np.complex128), pv, pq, cfg, cupf.SolveOptions())
    return np.asarray(r.V_numpy), time.perf_counter() - t


# --- exapf / matpower: forward-solve timing via their runners (best effort) ----
def _subprocess_reference(name, ct):
    """Run the external solver for a wall-clock forward time. Returns (None, secs)
    or (None, None) if the toolchain is unavailable — V_ref is not extracted."""
    import subprocess
    case = ct["name"]
    here = Path(__file__).resolve().parent      # benchmark/backends/
    if name == "exapf":
        cmd = ["julia", str(here / "exapf" / "run_powerflow.jl"), case]
    else:  # matpower
        cmd = ["bash", str(here / "matlab" / "run_matpower_case.bash"), case]
    try:
        t = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return None, time.perf_counter() - t
    except Exception:
        return None, None  # toolchain/license unavailable — skip the oracle


def reference_solve(name: str, ct):
    """Forward physics solve (the oracle). Returns (V_ref|None, solve_seconds|None)."""
    if name in ("pypower", "pandapower"):
        return _cpu_reference(ct)
    if name in ("cupf-pybind", "cupf-torch"):
        return _cupf_reference(ct)
    if name in ("exapf", "matpower"):
        return _subprocess_reference(name, ct)
    raise ValueError(f"unknown backend {name}")


# --- cupf-torch: differentiable forward+backward demo -------------------------
def diff_solve_demo(ct):
    """Gradient through the NR solve: ∂(Σ Vm)/∂load via the torch autograd wrapper.
    Returns (forward_ms, backward_ms) or None if torch/cuDSS is unavailable."""
    import torch
    _import_cupf()
    if str(REPO_ROOT / "cuPF" / "python") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "cuPF" / "python"))
    import cupf as cupf_pkg

    c = ct["case"]
    yb = c.ybus
    n = yb.shape[0]
    opts = cupf_pkg._cupf.NewtonOptions()
    opts.backend = cupf_pkg._cupf.BackendKind.CUDA
    opts.compute = cupf_pkg._cupf.ComputePolicy.Mixed
    # adjoint/transpose solve (backward) is implemented for cuDSS only.
    opts.cuda_linear_solver = cupf_pkg._cupf.CudaLinearSolverKind.CuDSS
    solver = cupf_pkg._cupf.NewtonSolver(opts)
    solver.initialize(yb.indptr.astype(np.int32), yb.indices.astype(np.int32), yb.data,
                      n, n, c.pv.astype(np.int32), c.pq.astype(np.int32))

    dev, dt = "cuda", torch.float32
    load_p = torch.zeros((1, n), device=dev, dtype=dt, requires_grad=True)
    load_q = torch.zeros((1, n), device=dev, dtype=dt, requires_grad=True)
    sbus_re = torch.tensor(c.sbus.real, device=dev, dtype=dt)
    sbus_im = torch.tensor(c.sbus.imag, device=dev, dtype=dt)
    v0_vm = torch.tensor(np.abs(c.v0), device=dev, dtype=dt)
    v0_va = torch.tensor(np.angle(c.v0), device=dev, dtype=dt)
    cfg = cupf_pkg._cupf.NRConfig(); cfg.tolerance = 1e-6; cfg.max_iter = 20
    sopt = cupf_pkg._cupf.SolveOptions(); sopt.prepare_adjoint_cache = True

    def fwd():
        return cupf_pkg.solve(load_p, load_q, solver, sbus_base_re=sbus_re, sbus_base_im=sbus_im,
                              v0_va=v0_va, v0_vm=v0_vm, config=cfg, solve_options=sopt)

    va, vm = fwd(); torch.cuda.synchronize()  # warmup
    t = time.perf_counter(); va, vm = fwd(); torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t) * 1000.0
    t = time.perf_counter(); (va.sum() + vm.sum()).backward(); torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - t) * 1000.0
    return fwd_ms, bwd_ms
