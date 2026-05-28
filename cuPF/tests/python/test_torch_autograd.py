from __future__ import annotations

import importlib

import numpy as np


def import_cupf():
    try:
        return importlib.import_module("cupf")
    except Exception:
        return importlib.import_module("_cupf")


def two_bus_case():
    y = 1.0 - 5.0j
    slack_v = 1.0 + 0.0j
    pq_v = 0.97 - 0.05j
    pq_current = y * (pq_v - slack_v)
    pq_sbus = pq_v * np.conjugate(pq_current)
    return {
        "indptr": np.array([0, 2, 4], dtype=np.int32),
        "indices": np.array([0, 1, 0, 1], dtype=np.int32),
        "data": np.array([y, -y, -y, y], dtype=np.complex128),
        "sbus": np.array([0.0 + 0.0j, pq_sbus], dtype=np.complex128),
        "v0": np.array([slack_v, 1.0 + 0.0j], dtype=np.complex128),
        "pv": np.array([], dtype=np.int32),
        "pq": np.array([1], dtype=np.int32),
    }


def main() -> None:
    try:
        import torch
    except Exception:
        print("torch is not available; skipping cuPF torch autograd smoke")
        return
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping cuPF torch autograd smoke")
        return

    cupf = import_cupf()
    if getattr(cupf, "solve", None) is None:
        print("cupf.solve autograd helper is not available; skipping")
        return
    if not hasattr(cupf.NewtonSolver, "solve_with_adjoint_cache_torch"):
        print("cuPF was not built with torch extension methods; skipping")
        return

    data = two_bus_case()
    options = cupf.NewtonOptions()
    options.backend = cupf.BackendKind.CUDA
    options.compute = cupf.ComputePolicy.Mixed
    solver = cupf.NewtonSolver(options)
    solver.initialize(
        data["indptr"],
        data["indices"],
        data["data"],
        2,
        2,
        data["pv"],
        data["pq"],
    )

    device = torch.device("cuda")
    dtype = torch.float32
    load_p = torch.zeros((1, 2), device=device, dtype=dtype, requires_grad=True)
    load_q = torch.zeros((1, 2), device=device, dtype=dtype, requires_grad=True)
    sbus_base_re = torch.tensor(data["sbus"].real, device=device, dtype=dtype)
    sbus_base_im = torch.tensor(data["sbus"].imag, device=device, dtype=dtype)
    v0_va = torch.angle(torch.tensor(data["v0"], device=device, dtype=torch.complex64)).to(dtype)
    v0_vm = torch.abs(torch.tensor(data["v0"], device=device, dtype=torch.complex64)).to(dtype)

    config = cupf.NRConfig()
    config.tolerance = 1e-6
    config.max_iter = 20
    solve_options = cupf.SolveOptions()

    va, vm = cupf.solve(
        load_p,
        load_q,
        solver,
        sbus_base_re=sbus_base_re,
        sbus_base_im=sbus_base_im,
        v0_va=v0_va,
        v0_vm=v0_vm,
        config=config,
        solve_options=solve_options,
    )
    assert va.shape == load_p.shape
    assert vm.shape == load_q.shape
    assert va.is_cuda and vm.is_cuda
    assert va.dtype == dtype and vm.dtype == dtype

    loss = va[:, 1].sum() + vm[:, 1].sum()
    loss.backward()
    assert load_p.grad is not None
    assert load_q.grad is not None
    assert load_p.grad.shape == load_p.shape
    assert load_q.grad.shape == load_q.shape
    assert torch.isfinite(load_p.grad).all()
    assert torch.isfinite(load_q.grad).all()


if __name__ == "__main__":
    main()
