"""The trainable "dummy layer" in front of every backend.

A small MLP that maps a load scenario (Sbus real/imag features) to a predicted
voltage state V = Vm·exp(i·Va). It starts at a flat profile (Vm≈1, Va≈0) so the
first epoch begins from the usual power-flow initial guess.
"""
from __future__ import annotations

import torch
from torch import nn


class StatePredictor(nn.Module):
    """Load features [B, 2n] -> voltage state V [B, n] (complex)."""

    def __init__(self, n_bus: int, hidden: int = 128):
        super().__init__()
        self.n = n_bus
        self.net = nn.Sequential(
            nn.Linear(2 * n_bus, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * n_bus),
        )
        # Start near a flat voltage profile: zero the last layer so the bias sets it.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias[: n_bus] = 1.0   # Vm ≈ 1.0 p.u.
            last.bias[n_bus:] = 0.0    # Va ≈ 0 rad

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.net(features)
        vm = out[:, : self.n]
        va = out[:, self.n:]
        return torch.polar(vm.abs(), va)   # complex V = |Vm|·exp(i·Va)
