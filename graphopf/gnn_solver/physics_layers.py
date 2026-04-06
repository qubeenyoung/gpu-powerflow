# models/physics_layer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cupy as cp
import torch
import torch.nn as nn
from torch.autograd import Function

from nvmath.bindings import cudss

from gnn_solver.grid_constants import GridConstants
from newton_method.cupy.newtonpf import NewtonSolver


@dataclass(frozen=True)
class PhysicsLayerConfig:
    batch_size: int = 4
    tol: float = 1e-6
    max_iters: int = 10


def _torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    if not x.is_cuda:
        raise RuntimeError("Expected CUDA tensor for zero-copy conversion to CuPy.")
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x))


def _cupy_to_torch(x: cp.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.utils.dlpack.from_dlpack(x.toDlpack())
    if t.device != device:
        t = t.to(device=device)
    return t


def _build_v0(
    batch_size: int,
    nbus: int,
    spv_bus_idx: cp.ndarray,   # (nspv,)
    vm_spv: cp.ndarray,        # (B, nspv)
    vm_pq_default: float = 1.0,
) -> cp.ndarray:
    vm = cp.full((batch_size, nbus), float(vm_pq_default), dtype=cp.float64)
    vm[:, spv_bus_idx] = vm_spv.astype(cp.float64, copy=False)

    va = cp.zeros((batch_size, nbus), dtype=cp.float64)

    v = cp.empty((batch_size, nbus), dtype=cp.complex128)
    cp.multiply(vm, cp.cos(va), out=v.real)
    cp.multiply(vm, cp.sin(va), out=v.imag)
    return v


def _build_sbus(
    batch_size: int,
    nbus: int,
    pv_bus_idx: cp.ndarray,    # (npv,)
    pq_bus_idx: cp.ndarray,    # (npq,)
    pd: cp.ndarray,            # (B, nbus)
    qd: cp.ndarray,            # (B, nbus)
    pg: cp.ndarray,            # (B, npv)
) -> cp.ndarray:
    sbus = cp.zeros((batch_size, nbus), dtype=cp.complex128)

    p_spec = cp.zeros((batch_size, nbus), dtype=cp.float64)
    q_spec = cp.zeros((batch_size, nbus), dtype=cp.float64)

    # PV buses: P specified = Pg - Pd, Q unspecified (set 0, not used in mismatch for PV)
    p_spec[:, pv_bus_idx] = pg.astype(cp.float64, copy=False) - pd[:, pv_bus_idx].astype(cp.float64, copy=False)

    # PQ buses: P specified = -Pd, Q specified = -Qd
    p_spec[:, pq_bus_idx] = -pd[:, pq_bus_idx].astype(cp.float64, copy=False)
    q_spec[:, pq_bus_idx] = -qd[:, pq_bus_idx].astype(cp.float64, copy=False)

    sbus.real = p_spec
    sbus.imag = q_spec
    return sbus


def _csr_to_dense_batch(
    indptr: cp.ndarray,     # (nJ+1,)
    indices: cp.ndarray,    # (nnz,)
    data: cp.ndarray,       # (B*nnz,) float32/float64
    nJ: int,
    batch_size: int,
    nnz: int,
) -> cp.ndarray:
    d = data.reshape(batch_size, nnz)
    J = cp.zeros((batch_size, nJ, nJ), dtype=cp.float64)
    for r in range(nJ):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        cols = indices[start:end]
        J[:, r, cols] = d[:, start:end].astype(cp.float64, copy=False)
    return J


class PhysicsLayerFn(Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,          # (B, 2*nbus) = [Pd | Qd]
        z_partial: torch.Tensor,       # (B, npv+nspv) = [Pg(PV) | Vm(SPV)]
        grid: GridConstants,
        cfg: PhysicsLayerConfig,
        solver: NewtonSolver,
    ) -> torch.Tensor:
        if inputs.ndim != 2:
            raise RuntimeError(f"inputs must be 2D, got shape={tuple(inputs.shape)}")
        if z_partial.ndim != 2:
            raise RuntimeError(f"z_partial must be 2D, got shape={tuple(z_partial.shape)}")

        device = inputs.device
        batch_size = int(inputs.shape[0])
        nbus = int(grid.nbus)

        if batch_size != int(cfg.batch_size):
            raise RuntimeError(f"batch_size mismatch: inputs B={batch_size}, cfg.batch_size={cfg.batch_size}")
        if inputs.shape[1] != 2 * nbus:
            raise RuntimeError(f"inputs dim mismatch: got {inputs.shape[1]}, expected {2*nbus}")

        pv_in_spv = grid.pv_in_spv_tensor.to(device=device)
        spv = grid.spv_tensor.to(device=device)

        npv = int(pv_in_spv.numel())
        nspv = int(spv.numel())
        if z_partial.shape[1] != (npv + nspv):
            raise RuntimeError(f"z_partial dim mismatch: got {z_partial.shape[1]}, expected {npv + nspv}")

        pv_bus_idx_t = pv_in_spv.to(torch.int64)
        spv_bus_idx_t = spv.to(torch.int64)

        pd_t = inputs[:, :nbus]
        qd_t = inputs[:, nbus:]

        pg_t = z_partial[:, :npv]
        vm_spv_t = z_partial[:, npv:]

        pd = _torch_to_cupy(pd_t.contiguous())
        qd = _torch_to_cupy(qd_t.contiguous())
        pg = _torch_to_cupy(pg_t.contiguous())
        vm_spv = _torch_to_cupy(vm_spv_t.contiguous())

        pv_bus_idx = _torch_to_cupy(pv_bus_idx_t.contiguous()).astype(cp.int32, copy=False)
        pq_bus_idx = solver.pq  # cp.int32
        spv_bus_idx = _torch_to_cupy(spv_bus_idx_t.contiguous()).astype(cp.int32, copy=False)

        sbus = _build_sbus(batch_size, nbus, pv_bus_idx, pq_bus_idx, pd, qd, pg)
        v0 = _build_v0(batch_size, nbus, spv_bus_idx, vm_spv, vm_pq_default=1.0)

        v = solver.solve(sbus=sbus, v0=v0, tolerance=float(cfg.tol), max_iter=int(cfg.max_iters))

        va = cp.angle(v).astype(cp.float32, copy=False)
        vm = cp.abs(v).astype(cp.float32, copy=False)
        state_cp = cp.concatenate([va, vm], axis=1)

        ctx.grid = grid
        ctx.cfg = cfg
        ctx.solver = solver
        ctx.save_for_backward(inputs, z_partial)

        return _cupy_to_torch(state_cp, device=device)

    @staticmethod
    def backward(ctx, dl_dstate: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        grid: GridConstants = ctx.grid
        solver: NewtonSolver = ctx.solver

        inputs, z_partial = ctx.saved_tensors

        device = dl_dstate.device
        batch_size = int(inputs.shape[0])
        nbus = int(grid.nbus)

        pv_in_spv = grid.pv_in_spv_tensor.to(device=device)
        spv = grid.spv_tensor.to(device=device)

        npv = int(pv_in_spv.numel())
        nspv = int(spv.numel())

        if dl_dstate.shape[1] != 2 * nbus:
            raise RuntimeError(f"dl_dstate dim mismatch: got {dl_dstate.shape[1]}, expected {2 * nbus}")

        pv_bus_idx_t = pv_in_spv.to(torch.int64)
        spv_bus_idx_t = spv.to(torch.int64)

        dl_dva_t = dl_dstate[:, :nbus].contiguous()
        dl_dvm_t = dl_dstate[:, nbus:].contiguous()

        pv_bus_idx = _torch_to_cupy(pv_bus_idx_t.contiguous()).astype(cp.int32, copy=False)
        pq_bus_idx = solver.pq
        spv_bus_idx = _torch_to_cupy(spv_bus_idx_t.contiguous()).astype(cp.int32, copy=False)

        dl_dva = _torch_to_cupy(dl_dva_t)
        dl_dvm = _torch_to_cupy(dl_dvm_t)

        g_pv = dl_dva[:, pv_bus_idx]
        g_pq_va = dl_dva[:, pq_bus_idx]
        g_pq_vm = dl_dvm[:, pq_bus_idx]
        g = cp.concatenate([g_pv, g_pq_va, g_pq_vm], axis=1).astype(cp.float32, copy=False)

        nJ = int(solver.jacobian.R)
        if g.shape[1] != nJ:
            raise RuntimeError(f"g dim mismatch: got {g.shape[1]}, expected {nJ}")

        # Solve (J^T) * lambda = g using cuDSS (JT is expected to be up-to-date via solver.solve -> jacobian.update_JT)
        solver.F[:] = g.reshape(-1)
        solver.dx.fill(0.0)

        cudss.execute(
            solver.dss_handle,
            cudss.Phase.REFACTORIZATION,
            solver.dss_config,
            solver.dss_data,
            solver.dss_A_T,
            solver.dss_X,
            solver.dss_B,
        )
        cudss.execute(
            solver.dss_handle,
            cudss.Phase.SOLVE,
            solver.dss_config,
            solver.dss_data,
            solver.dss_A_T,
            solver.dss_X,
            solver.dss_B,
        )

        lam = solver.dx.reshape(batch_size, -1).astype(cp.float32, copy=False)

        lam_pv = lam[:, :npv]
        lam_pq_p = lam[:, npv:npv + int(solver.npq)]
        lam_pq_q = lam[:, npv + int(solver.npq):]

        grad_pd = cp.zeros((batch_size, nbus), dtype=cp.float32)
        grad_qd = cp.zeros((batch_size, nbus), dtype=cp.float32)

        grad_pd[:, pv_bus_idx] += -lam_pv
        grad_pd[:, pq_bus_idx] += -lam_pq_p
        grad_qd[:, pq_bus_idx] += -lam_pq_q

        grad_inputs = cp.empty((batch_size, 2 * nbus), dtype=cp.float32)
        grad_inputs[:, :nbus] = grad_pd
        grad_inputs[:, nbus:] = grad_qd

        grad_z = cp.zeros((batch_size, npv + nspv), dtype=cp.float32)
        grad_z[:, :npv] = lam_pv
        # grad_z[:, npv:] = 0.0  # Vm(SPV) gradient intentionally zero

        grad_inputs_t = _cupy_to_torch(grad_inputs, device=device)
        grad_z_t = _cupy_to_torch(grad_z, device=device)

        return grad_inputs_t, grad_z_t, None, None, None


class PhysicsLayer(nn.Module):
    def __init__(self, grid: GridConstants, config: PhysicsLayerConfig) -> None:
        super().__init__()
        self.grid = grid
        self.config = config

        self.newton_solver = NewtonSolver(
            ybus=grid.Ybus,
            pv=grid.pv,
            pq=grid.pq,
            batch_size=config.batch_size,
        )

    def forward(self, inputs: torch.Tensor, guess: torch.Tensor) -> torch.Tensor:
        pv_in_spv_map = self.grid.pv_in_spv_tensor
        spv_bus_ids = self.grid.spv_tensor

        pv_count = int(pv_in_spv_map.numel())
        spv_count = int(spv_bus_ids.numel())
        if guess.shape[1] != (pv_count + spv_count):
            raise RuntimeError(
                f"guess dim mismatch: got {guess.shape[1]}, expected {pv_count + spv_count}"
            )

        z_unit = torch.sigmoid(guess)

        p_lower = self.grid.pmin_tensor.to(device=guess.device, dtype=guess.dtype)
        p_upper = self.grid.pmax_tensor.to(device=guess.device, dtype=guess.dtype)
        v_lower = self.grid.vmin_tensor.to(device=guess.device, dtype=guess.dtype)
        v_upper = self.grid.vmax_tensor.to(device=guess.device, dtype=guess.dtype)

        pv_p_unit = z_unit[:, :pv_count]
        pv_gen_bus = pv_in_spv_map.to(device=guess.device, dtype=torch.int64)
        pv_p_mw = pv_p_unit * p_upper.index_select(0, pv_gen_bus) + (1.0 - pv_p_unit) * p_lower.index_select(0, pv_gen_bus)

        spv_vm_unit = z_unit[:, pv_count:pv_count + spv_count]
        spv_bus = spv_bus_ids.to(device=guess.device, dtype=torch.int64)
        spv_vm_pu = spv_vm_unit * v_upper.index_select(0, spv_bus) + (1.0 - spv_vm_unit) * v_lower.index_select(0, spv_bus)

        z_projected = torch.cat([pv_p_mw, spv_vm_pu], dim=1)
        return PhysicsLayerFn.apply(inputs, z_projected, self.grid, self.config, self.newton_solver)