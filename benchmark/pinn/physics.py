"""Power-flow physics for the PINN benchmark.

State-prediction PINN: a network predicts the bus voltage state V, and the loss is
the power-flow residual  F(V) = V ⊙ conj(Ybus·V) − Sbus  (zero at a solution).
The residual is a differentiable function of V (plain complex linear algebra), so it
trains the network without needing any solver — the solvers (backends) instead play
the physics-oracle / differentiable-layer role (see backends.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.common import matpower_data  # noqa: E402


def load_case(case_path, device="cpu"):
    """Load a MATPOWER case into the tensors the PINN benchmark needs."""
    c = matpower_data.load_case(case_path)
    n = c.ybus.shape[0]
    return {
        "case": c,
        "name": c.case_name,
        "n": n,
        # Dense complex Ybus is fine for the modest cases this benchmark targets.
        "Ybus": torch.tensor(c.ybus.toarray(), dtype=torch.complex128, device=device),
        "Sbus": torch.tensor(c.sbus, dtype=torch.complex128, device=device),
        "V0": torch.tensor(c.v0, dtype=torch.complex128, device=device),
        "pv": c.pv.astype(np.int64),
        "pq": c.pq.astype(np.int64),
    }


def residual(V: torch.Tensor, ybus: torch.Tensor, sbus: torch.Tensor) -> torch.Tensor:
    """Complex power-flow mismatch F(V) for V[B,n] (or [n]). Returns same shape.

    (Ybus·V)_i = Σ_j Ybus_ij V_j, so for a batch I = V @ Ybusᵀ."""
    inj = V @ ybus.transpose(-2, -1)          # Ybus · V, batched
    return V * torch.conj(inj) - sbus


def residual_loss(V: torch.Tensor, ybus: torch.Tensor, sbus: torch.Tensor) -> torch.Tensor:
    """Mean squared power-flow residual — the PINN physics loss."""
    f = residual(V, ybus, sbus)
    return (f.real ** 2 + f.imag ** 2).mean()
