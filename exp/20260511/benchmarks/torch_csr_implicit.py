from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch


def torch_dtypes(dtype: str) -> tuple[torch.dtype, torch.dtype]:
    if dtype == "float32":
        return torch.float32, torch.complex64
    if dtype == "float64":
        return torch.float64, torch.complex128
    raise ValueError(f"Unsupported dtype: {dtype}")


@lru_cache(maxsize=8)
def torch_spsolve_available(device_type: str, dtype_name: str) -> tuple[bool, str]:
    if device_type != "cuda":
        return False, "torch.sparse.spsolve CPU path is not used for this CUDA baseline"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    real_dtype, _ = torch_dtypes(dtype_name)
    try:
        crow = torch.tensor([0, 2, 4], dtype=torch.int64, device="cuda")
        col = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
        val = torch.tensor([4.0, 1.0, 2.0, 3.0], dtype=real_dtype, device="cuda")
        mat = torch.sparse_csr_tensor(crow, col, val, size=(2, 2), device="cuda")
        rhs = torch.tensor([1.0, 2.0], dtype=real_dtype, device="cuda")
        _ = torch.sparse.spsolve(mat, rhs)
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"


@dataclass
class TorchCSRInputs:
    n_bus: int
    n_state: int
    ybus: torch.Tensor
    y_values: torch.Tensor
    sbus: torch.Tensor
    v0: torch.Tensor
    pv: torch.Tensor
    pq: torch.Tensor
    pvpq: torch.Tensor
    jac_crow: torch.Tensor
    jac_col: torch.Tensor
    jac_row_bus: torch.Tensor
    jac_col_bus: torch.Tensor
    jac_y_pos: torch.Tensor
    jac_kind: torch.Tensor


def _build_reduced_jacobian_pattern(case: Any) -> dict[str, np.ndarray | int]:
    n_bus = int(case.n_bus)
    pv = np.asarray(case.pv, dtype=np.int64)
    pq = np.asarray(case.pq, dtype=np.int64)
    pvpq = np.r_[pv, pq].astype(np.int64, copy=False)
    n_pvpq = int(pvpq.size)
    n_pq = int(pq.size)
    n_state = n_pvpq + n_pq

    angle_col = np.full(n_bus, -1, dtype=np.int64)
    vm_col = np.full(n_bus, -1, dtype=np.int64)
    p_row = np.full(n_bus, -1, dtype=np.int64)
    q_row = np.full(n_bus, -1, dtype=np.int64)
    angle_col[pvpq] = np.arange(n_pvpq, dtype=np.int64)
    vm_col[pq] = n_pvpq + np.arange(n_pq, dtype=np.int64)
    p_row[pvpq] = np.arange(n_pvpq, dtype=np.int64)
    q_row[pq] = n_pvpq + np.arange(n_pq, dtype=np.int64)

    indptr = np.asarray(case.indptr, dtype=np.int64)
    indices = np.asarray(case.indices, dtype=np.int64)
    diag_pos = np.full(n_bus, -1, dtype=np.int64)
    entries: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    for i in range(n_bus):
        for pos in range(int(indptr[i]), int(indptr[i + 1])):
            j = int(indices[pos])
            if i == j:
                diag_pos[i] = pos
            if p_row[i] >= 0:
                if angle_col[j] >= 0:
                    entries[(int(p_row[i]), int(angle_col[j]))] = (i, j, pos, 0)
                if vm_col[j] >= 0:
                    entries[(int(p_row[i]), int(vm_col[j]))] = (i, j, pos, 1)
            if q_row[i] >= 0:
                if angle_col[j] >= 0:
                    entries[(int(q_row[i]), int(angle_col[j]))] = (i, j, pos, 2)
                if vm_col[j] >= 0:
                    entries[(int(q_row[i]), int(vm_col[j]))] = (i, j, pos, 3)

    for i in range(n_bus):
        ypos = int(diag_pos[i])
        if p_row[i] >= 0 and angle_col[i] >= 0:
            entries.setdefault((int(p_row[i]), int(angle_col[i])), (i, i, ypos, 0))
        if p_row[i] >= 0 and vm_col[i] >= 0:
            entries.setdefault((int(p_row[i]), int(vm_col[i])), (i, i, ypos, 1))
        if q_row[i] >= 0 and angle_col[i] >= 0:
            entries.setdefault((int(q_row[i]), int(angle_col[i])), (i, i, ypos, 2))
        if q_row[i] >= 0 and vm_col[i] >= 0:
            entries.setdefault((int(q_row[i]), int(vm_col[i])), (i, i, ypos, 3))

    sorted_items = sorted(entries.items())
    crow = np.zeros(n_state + 1, dtype=np.int64)
    col = np.empty(len(sorted_items), dtype=np.int64)
    row_bus = np.empty(len(sorted_items), dtype=np.int64)
    col_bus = np.empty(len(sorted_items), dtype=np.int64)
    y_pos = np.empty(len(sorted_items), dtype=np.int64)
    kind = np.empty(len(sorted_items), dtype=np.int64)
    for idx, ((row_idx, col_idx), payload) in enumerate(sorted_items):
        crow[row_idx + 1] += 1
        col[idx] = col_idx
        row_bus[idx], col_bus[idx], y_pos[idx], kind[idx] = payload
    np.cumsum(crow, out=crow)
    return {
        "n_state": n_state,
        "crow": crow,
        "col": col,
        "row_bus": row_bus,
        "col_bus": col_bus,
        "y_pos": y_pos,
        "kind": kind,
    }


def make_torch_csr_inputs(case: Any, *, device: torch.device, dtype: str) -> TorchCSRInputs:
    real_dtype, complex_dtype = torch_dtypes(dtype)
    pattern = _build_reduced_jacobian_pattern(case)
    y_crow = torch.as_tensor(np.asarray(case.indptr, dtype=np.int64), dtype=torch.int64, device=device)
    y_col = torch.as_tensor(np.asarray(case.indices, dtype=np.int64), dtype=torch.int64, device=device)
    y_values = torch.as_tensor(np.asarray(case.ybus_data), dtype=complex_dtype, device=device)
    ybus = torch.sparse_csr_tensor(y_crow, y_col, y_values, size=(case.n_bus, case.n_bus), device=device)
    pv = torch.as_tensor(np.asarray(case.pv, dtype=np.int64), dtype=torch.long, device=device)
    pq = torch.as_tensor(np.asarray(case.pq, dtype=np.int64), dtype=torch.long, device=device)
    return TorchCSRInputs(
        n_bus=int(case.n_bus),
        n_state=int(pattern["n_state"]),
        ybus=ybus,
        y_values=y_values,
        sbus=torch.as_tensor(np.asarray(case.sbus), dtype=complex_dtype, device=device),
        v0=torch.as_tensor(np.asarray(case.v0), dtype=complex_dtype, device=device),
        pv=pv,
        pq=pq,
        pvpq=torch.cat([pv, pq]),
        jac_crow=torch.as_tensor(pattern["crow"], dtype=torch.int64, device=device),
        jac_col=torch.as_tensor(pattern["col"], dtype=torch.int64, device=device),
        jac_row_bus=torch.as_tensor(pattern["row_bus"], dtype=torch.long, device=device),
        jac_col_bus=torch.as_tensor(pattern["col_bus"], dtype=torch.long, device=device),
        jac_y_pos=torch.as_tensor(pattern["y_pos"], dtype=torch.long, device=device),
        jac_kind=torch.as_tensor(pattern["kind"], dtype=torch.long, device=device),
    )


def _sparse_mv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(mat, vec.unsqueeze(1)).squeeze(1)


def csr_jacobian_values(static: TorchCSRInputs, v: torch.Tensor, i_bus: torch.Tensor) -> torch.Tensor:
    real_dtype = torch.float32 if v.dtype == torch.complex64 else torch.float64
    vm_safe = torch.clamp(torch.abs(v), min=1e-8)
    v_norm = v / vm_safe
    row_bus = static.jac_row_bus
    col_bus = static.jac_col_bus
    y_pos = static.jac_y_pos
    valid_y = y_pos >= 0
    safe_y_pos = torch.clamp(y_pos, min=0)
    y_ij = torch.where(valid_y, static.y_values[safe_y_pos], torch.zeros((), dtype=v.dtype, device=v.device))
    v_i = v[row_bus]
    v_j = v[col_bus]
    vnorm_i = v_norm[row_bus]
    vnorm_j = v_norm[col_bus]
    is_diag = row_bus == col_bus
    diag_i = torch.where(is_diag, i_bus[row_bus], torch.zeros((), dtype=v.dtype, device=v.device))
    d_va = 1j * v_i * torch.conj(diag_i - y_ij * v_j)
    d_vm = v_i * torch.conj(y_ij * vnorm_j) + torch.where(
        is_diag,
        torch.conj(i_bus[row_bus]) * vnorm_i,
        torch.zeros((), dtype=v.dtype, device=v.device),
    )
    kind = static.jac_kind
    values = torch.empty(kind.shape, dtype=real_dtype, device=v.device)
    values = torch.where(kind == 0, d_va.real, values)
    values = torch.where(kind == 1, d_vm.real, values)
    values = torch.where(kind == 2, d_va.imag, values)
    values = torch.where(kind == 3, d_vm.imag, values)
    return values


def make_csr_jacobian(static: TorchCSRInputs, v: torch.Tensor, i_bus: torch.Tensor) -> torch.Tensor:
    values = csr_jacobian_values(static, v, i_bus)
    return torch.sparse_csr_tensor(
        static.jac_crow,
        static.jac_col,
        values,
        size=(static.n_state, static.n_state),
        device=v.device,
    )


def torch_csr_newton_pf(
    static: TorchCSRInputs,
    *,
    sbus: torch.Tensor | None = None,
    tolerance: float,
    max_iter: int,
    return_cache: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    sbus_vec = static.sbus if sbus is None else sbus
    v = static.v0.clone()
    va = torch.angle(v)
    vm = torch.abs(v)
    n_pvpq = int(static.pvpq.numel())
    n_pq = int(static.pq.numel())
    jac = None
    for _ in range(int(max_iter)):
        i_bus = _sparse_mv(static.ybus, v)
        mis = v * torch.conj(i_bus) - sbus_vec
        f = torch.cat([mis[static.pv].real, mis[static.pq].real, mis[static.pq].imag], dim=0)
        if bool((torch.max(torch.abs(f)) <= tolerance).detach().cpu().item()):
            break
        jac = make_csr_jacobian(static, v, i_bus)
        dx = torch.sparse.spsolve(jac, f)
        va = va.clone()
        vm = vm.clone()
        va[static.pvpq] = va[static.pvpq] - dx[:n_pvpq]
        vm[static.pq] = vm[static.pq] - dx[n_pvpq : n_pvpq + n_pq]
        v = torch.polar(vm, va)
    if return_cache:
        if jac is None:
            i_bus = _sparse_mv(static.ybus, v)
            jac = make_csr_jacobian(static, v, i_bus)
        return v, jac
    return v


class TorchCSRImplicitPF(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, load_p: torch.Tensor, load_q: torch.Tensor, static: TorchCSRInputs, tolerance: float, max_iter: int) -> torch.Tensor:
        with torch.no_grad():
            sbus = static.sbus - torch.complex(load_p, load_q)
            v, jac = torch_csr_newton_pf(
                static,
                sbus=sbus,
                tolerance=float(tolerance),
                max_iter=int(max_iter),
                return_cache=True,
            )
        ctx.static = static
        ctx.save_for_backward(v, jac)
        return torch.stack([torch.abs(v), torch.angle(v)], dim=-1)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        static = ctx.static
        _v, jac = ctx.saved_tensors
        grad_vm = grad_out[:, 0]
        grad_va = grad_out[:, 1]
        grad_state = torch.cat([grad_va[static.pvpq], grad_vm[static.pq]], dim=0).contiguous()
        jac_t = torch.transpose(jac, 0, 1).to_sparse_csr()
        lam = torch.sparse.spsolve(jac_t, grad_state)
        grad_p = torch.zeros(static.n_bus, dtype=grad_state.dtype, device=grad_state.device)
        grad_q = torch.zeros(static.n_bus, dtype=grad_state.dtype, device=grad_state.device)
        n_pvpq = int(static.pvpq.numel())
        n_pq = int(static.pq.numel())
        grad_p[static.pvpq] = -lam[:n_pvpq]
        grad_q[static.pq] = -lam[n_pvpq : n_pvpq + n_pq]
        return grad_p, grad_q, None, None, None
