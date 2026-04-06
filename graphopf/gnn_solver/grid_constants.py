from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pypower.api import ext2int, loadcase, makeYbus
from pypower import idx_brch, idx_bus, idx_gen
from pypower.idx_brch import F_BUS, T_BUS


@dataclass(frozen=True)
class GridIndices:
    slack: np.ndarray
    pv: np.ndarray
    spv: np.ndarray
    pq: np.ndarray
    nonslack_bus: np.ndarray
    slack_in_spv: np.ndarray
    pv_in_spv: np.ndarray
    nslack: int
    npv: int


class GridConstants:
    def __init__(
        self,
        case_path: str,
        *,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device

        ppc = loadcase(case_path)
        self.ppc = ext2int(ppc)

        self.genbase = float(self.ppc["baseMVA"])

        bus = self.ppc["bus"]
        gen = self.ppc["gen"]
        branch = self.ppc["branch"]

        self.nbus = int(bus.shape[0])
        self.ngen = int(gen.shape[0])
        self.nbranch = int(branch.shape[0])

        # int32
        slack = np.where(bus[:, idx_bus.BUS_TYPE] == 3)[0].astype(np.int32, copy=False)
        pv = np.where(bus[:, idx_bus.BUS_TYPE] == 2)[0].astype(np.int32, copy=False)

        spv = np.concatenate([slack, pv]).astype(np.int32, copy=False)
        spv.sort()

        pq = np.setdiff1d(np.arange(self.nbus, dtype=np.int32), spv, assume_unique=False).astype(np.int32, copy=False)
        nonslack_bus = np.sort(np.concatenate([pq, pv]).astype(np.int32, copy=False))

        slack_in_spv = np.array([np.where(spv == x)[0][0] for x in slack], dtype=np.int32)
        pv_in_spv = np.array([np.where(spv == x)[0][0] for x in pv], dtype=np.int32)

        self.indices = GridIndices(
            slack=slack,
            pv=pv,
            spv=spv,
            pq=pq,
            nonslack_bus=nonslack_bus,
            slack_in_spv=slack_in_spv,
            pv_in_spv=pv_in_spv,
            nslack=int(slack.size),
            npv=int(pv.size),
        )

        # float64
        self.pmax = (gen[:, idx_gen.PMAX] / self.genbase).astype(np.float64, copy=False)
        self.pmin = (gen[:, idx_gen.PMIN] / self.genbase).astype(np.float64, copy=False)
        self.qmax = (gen[:, idx_gen.QMAX] / self.genbase).astype(np.float64, copy=False)
        self.qmin = (gen[:, idx_gen.QMIN] / self.genbase).astype(np.float64, copy=False)
        self.vmax = bus[:, idx_bus.VMAX].astype(np.float64, copy=False)
        self.vmin = bus[:, idx_bus.VMIN].astype(np.float64, copy=False)

        self.slack_va = np.deg2rad(bus[slack.astype(np.int64), idx_bus.VA]).astype(np.float64, copy=False)

        rate_a_pu = (branch[:, idx_brch.RATE_A] / self.genbase).astype(np.float64, copy=False)
        line_limit = np.square(rate_a_pu, dtype=np.float64)
        self.line_limit = np.where(line_limit == 0.0, np.inf, line_limit).astype(np.float64, copy=False)

        Ybus, Yf, Yt = makeYbus(self.genbase, bus, branch)
        self.Ybus = Ybus.tocsr().astype(np.complex128, copy=False)
        self.Yf = Yf.tocsr().astype(np.complex128, copy=False)
        self.Yt = Yt.tocsr().astype(np.complex128, copy=False)

        # ---- torch tensors ----
        # indices: int32
        self.spv_tensor = torch.as_tensor(self.indices.spv, dtype=torch.int32, device=device)
        self.slack_tensor = torch.as_tensor(self.indices.slack, dtype=torch.int32, device=device)
        self.slack_in_spv_tensor = torch.as_tensor(self.indices.slack_in_spv, dtype=torch.int32, device=device)

        # scalars/bounds: float64
        self.genbase_tensor = torch.tensor(self.genbase, dtype=torch.float64, device=device)

        self.pmax_tensor = torch.as_tensor(self.pmax, dtype=torch.float64, device=device)
        self.pmin_tensor = torch.as_tensor(self.pmin, dtype=torch.float64, device=device)
        self.qmax_tensor = torch.as_tensor(self.qmax, dtype=torch.float64, device=device)
        self.qmin_tensor = torch.as_tensor(self.qmin, dtype=torch.float64, device=device)
        self.vmax_tensor = torch.as_tensor(self.vmax, dtype=torch.float64, device=device)
        self.vmin_tensor = torch.as_tensor(self.vmin, dtype=torch.float64, device=device)
        self.line_limit_tensor = torch.as_tensor(self.line_limit, dtype=torch.float64, device=device)

        # dense real/imag: float64
        Ybus_dense = torch.as_tensor(self.Ybus.toarray(), dtype=torch.complex128, device=device)
        self.Ybus_real_tensor = Ybus_dense.real.to(dtype=torch.float64)
        self.Ybus_imag_tensor = Ybus_dense.imag.to(dtype=torch.float64)

        # CSR sparse: crow/col must be int32, values complex128
        self.Ybus_tensor = torch.sparse_csr_tensor(
            crow=torch.as_tensor(self.Ybus.indptr, dtype=torch.int32, device=device),
            col=torch.as_tensor(self.Ybus.indices, dtype=torch.int32, device=device),
            value=torch.as_tensor(self.Ybus.data, dtype=torch.complex128, device=device),
            size=self.Ybus.shape,
        )
        self.Yf_tensor = torch.sparse_csr_tensor(
            crow=torch.as_tensor(self.Yf.indptr, dtype=torch.int32, device=device),
            col=torch.as_tensor(self.Yf.indices, dtype=torch.int32, device=device),
            value=torch.as_tensor(self.Yf.data, dtype=torch.complex128, device=device),
            size=self.Yf.shape,
        )
        self.Yt_tensor = torch.sparse_csr_tensor(
            crow=torch.as_tensor(self.Yt.indptr, dtype=torch.int32, device=device),
            col=torch.as_tensor(self.Yt.indices, dtype=torch.int32, device=device),
            value=torch.as_tensor(self.Yt.data, dtype=torch.complex128, device=device),
            size=self.Yt.shape,
        )

        # branch endpoints: int32
        self.fbus_tensor = torch.as_tensor(
            branch[:, F_BUS].astype(np.int32, copy=False),
            dtype=torch.int32,
            device=device,
        )
        self.tbus_tensor = torch.as_tensor(
            branch[:, T_BUS].astype(np.int32, copy=False),
            dtype=torch.int32,
            device=device,
        )
