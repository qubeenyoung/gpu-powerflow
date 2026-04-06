# Utils/loss_fn.py
from __future__ import annotations

from typing import Tuple

import torch

from gnn_solver.grid_constants import GridConstants  


def split_state(
    state: torch.Tensor,
    *,
    ngen: int,
    nbus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pg = state[:, :ngen]
    qg = state[:, ngen:2 * ngen]
    vm = state[:, -2 * nbus:-nbus]
    va = state[:, -nbus:]
    return pg, qg, vm, va


def compute_power_balance_mismatch(
    inputs: torch.Tensor,
    state: torch.Tensor,
    *,
    grid: GridConstants,
) -> torch.Tensor:
    nbus = grid.nbus
    ngen = grid.ngen

    device = state.device
    dtype = state.dtype

    pg, qg, vm, va = split_state(state, ngen=ngen, nbus=nbus)

    vr = vm * torch.cos(va)
    vi = vm * torch.sin(va)
    vz = torch.complex(vr, vi)  # (B, nbus), complex

    # I = Ybus * V
    I = torch.sparse.mm(grid.Ybus_tensor.to(device=device), vz.T)  # (nbus, B), complex

    # S = V * conj(I)
    S = vz * torch.conj(I.T)  # (B, nbus), complex
    P_inj = S.real.to(dtype=dtype)
    Q_inj = S.imag.to(dtype=dtype)

    pg_expand = torch.zeros((pg.shape[0], nbus), device=device, dtype=dtype)
    pg_expand.index_copy_(1, grid.spv_tensor.to(device=device), pg)

    qg_expand = torch.zeros((qg.shape[0], nbus), device=device, dtype=dtype)
    qg_expand.index_copy_(1, grid.spv_tensor.to(device=device), qg)

    p_mismatch = (pg_expand - inputs[:, :nbus].to(dtype=dtype, device=device)) - P_inj
    q_mismatch = (qg_expand - inputs[:, nbus:].to(dtype=dtype, device=device)) - Q_inj

    return torch.cat([p_mismatch, q_mismatch], dim=1)


def compute_inequality_residuals(
    state: torch.Tensor,
    *,
    grid: GridConstants,
) -> torch.Tensor:
    nbus = grid.nbus
    ngen = grid.ngen

    pg, qg, vm, va = split_state(state, ngen=ngen, nbus=nbus)

    vr = vm * torch.cos(va)
    vi = vm * torch.sin(va)
    vz = torch.complex(vr, vi)  # (B, nbus), complex

    # Branch currents
    If = torch.sparse.mm(grid.Yf_tensor, vz.T)  # (nbranch, B), complex
    It = torch.sparse.mm(grid.Yt_tensor, vz.T)  # (nbranch, B), complex

    fbus = grid.fbus_tensor
    tbus = grid.tbus_tensor

    # Branch complex powers (from/to)
    Sf = vz.index_select(1, fbus) * torch.conj(If.T)  # (B, nbranch), complex
    St = vz.index_select(1, tbus) * torch.conj(It.T)  # (B, nbranch), complex

    Sff = Sf * torch.conj(Sf)  # |Sf|^2
    Stt = St * torch.conj(St)  # |St|^2

    slack_in_spv = torch.as_tensor(grid.indices.slack_in_spv, dtype=torch.long, device=torch.device)

    pmax = grid.pmax_tensor
    pmin = grid.pmin_tensor
    qmax = grid.qmax_tensor
    qmin = grid.qmin_tensor
    vmax = grid.vmax_tensor
    vmin = grid.vmin_tensor
    line_limit = grid.line_limit_tensor

    return torch.cat(
        [
            pg.index_select(1, slack_in_spv) - pmax.index_select(0, slack_in_spv),
            pmin.index_select(0, slack_in_spv) - pg.index_select(1, slack_in_spv),
            qg - qmax,
            qmin - qg,
            vm - vmax,
            vmin - vm,
            Sff.real - line_limit,
            Stt.real - line_limit,
        ],
        dim=1,
    )


def compute_inequality_violations(
    inputs: torch.Tensor,
    state: torch.Tensor,
    *,
    grid: GridConstants,
) -> torch.Tensor:
    return torch.clamp(
        compute_inequality_residuals(inputs, state, grid=grid),
        0.0,
    )

def compute_cost(
    state: torch.Tensor,
    *,
    quad_costs: torch.Tensor,
    lin_costs: torch.Tensor,
    const_cost: torch.Tensor | float,
    genbase: float,
    ngen: int,
    nbus: int,
) -> torch.Tensor:
    """
    Objective cost consistent with ACOPFProblem.obj_fn (per-sample).
    """
    device = state.device
    dtype = state.dtype

    pg, _, _, _ = split_state(state, ngen=ngen, nbus=nbus)

    genbase_t = torch.tensor(genbase, device=device, dtype=dtype)
    pg_mw = pg * genbase_t

    quad_costs = quad_costs.to(device=device, dtype=dtype)
    lin_costs = lin_costs.to(device=device, dtype=dtype)
    const_cost_t = (
        const_cost.to(device=device, dtype=dtype)
        if isinstance(const_cost, torch.Tensor)
        else torch.tensor(float(const_cost), device=device, dtype=dtype)
    )

    cost = (quad_costs * (pg_mw ** 2)).sum(dim=1) + (lin_costs * pg_mw).sum(dim=1) + const_cost_t
    return cost / (genbase_t * genbase_t)


def compute_loss(
    inputs: torch.Tensor,
    state: torch.Tensor,
    *,
    grid: GridConstants,
    quad_costs: torch.Tensor,
    lin_costs: torch.Tensor,
    const_cost: torch.Tensor | float,
    genbase: float,
    LagM_slack_p_gen: torch.Tensor,
    LagM_gen: torch.Tensor,
    LagM_bus: torch.Tensor,
    LagM_line: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss consistent with your total_loss():
      total = obj_cost + sum(LagM_* * ineq_violation_parts)

    Returns:
      total:    (B,)
      obj_cost: (B,)
      ineq_violation: (B, nineq)
      eq_mismatch:    (B, 2*nbus)
    """

    obj_cost = compute_cost(
        state,
        quad_costs=quad_costs,
        lin_costs=lin_costs,
        const_cost=const_cost,
        genbase=genbase,
        ngen=grid.ngen,
        nbus=grid.nbus,
    )

    ineq_violation = compute_inequality_violations(inputs, state, grid=grid)

    # Partition (matches ACOPFProblem.ineq_resid layout)
    # slack Pg: 2 * nslack
    nslack = int(grid.indices.nslack)
    ngen = grid.ngen
    nbus = grid.nbus
    nbranch = grid.nbranch

    i0 = 0
    i1 = i0 + 2 * nslack
    i2 = i1 + 2 * ngen
    i3 = i2 + 2 * nbus
    i4 = i3 + 2 * nbranch

    ineq_spg = ineq_violation[:, i0:i1]
    ineq_qg = ineq_violation[:, i1:i2]
    ineq_vm = ineq_violation[:, i2:i3]
    ineq_line = ineq_violation[:, i3:i4]

    ineq_cost = torch.cat(
        [
            LagM_slack_p_gen * ineq_spg,
            LagM_gen * ineq_qg,
            LagM_bus * ineq_vm,
            LagM_line * ineq_line,
        ],
        dim=1,
    )

    return obj_cost + ineq_cost.sum(dim=1)