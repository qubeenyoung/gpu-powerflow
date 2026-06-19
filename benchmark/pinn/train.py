#!/usr/bin/env python3
"""PINN benchmark: train a dummy layer in front of each power-flow backend.

For every backend we run the same state-prediction PINN (decision 1a):

    load scenario --[dummy MLP]--> predicted V ;  loss = ‖F(V)‖²   (physics residual)

Only the dummy MLP trains (decision 2a); the backend plays the physics role:
it runs a forward Newton solve as an oracle (reference V_ref + solve time), and the
differentiable backend (cupf-torch) additionally runs a forward+backward demo.

    python3 -m benchmark.pinn.train --case case118 --epochs 5
    python3 -m benchmark.pinn.train --backends pypower cupf-torch --case case300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark import backends  # noqa: E402  (the relocated per-backend runners + adapter)
from benchmark.pinn import physics  # noqa: E402
from benchmark.pinn.model import StatePredictor  # noqa: E402
from benchmark.common import matpower_data  # noqa: E402


def make_scenarios(ct, batch, seed=0):
    """`batch` load scenarios = base Sbus scaled by U[0.8, 1.2]; features = [Re, Im]."""
    g = torch.Generator().manual_seed(seed)
    scales = 0.8 + 0.4 * torch.rand(batch, 1, generator=g, dtype=torch.float64)
    sbus = ct["Sbus"].cpu().unsqueeze(0) * scales      # [B, n] complex128
    features = torch.cat([sbus.real, sbus.imag], dim=1)  # [B, 2n] float64
    return features, sbus


def train_one(name, ct, args):
    print(f"\n===== backend: {name} (differentiable={backends.is_differentiable(name)}) =====",
          flush=True)

    # 1) physics oracle: forward Newton solve (reference V_ref + solve time)
    try:
        v_ref, solve_s = backends.reference_solve(name, ct)
    except Exception as exc:
        print(f"  [oracle] unavailable: {exc}")
        v_ref, solve_s = None, None
    if solve_s is not None:
        print(f"  [oracle] forward solve: {solve_s * 1e3:.2f} ms"
              + ("" if v_ref is not None else " (V_ref not extracted)"))
    else:
        print("  [oracle] skipped (toolchain unavailable)")

    # 2) PINN: train the dummy MLP on the physics residual
    ybus, sbus_base = ct["Ybus"].cpu(), ct["Sbus"].cpu()
    features, sbus_batch = make_scenarios(ct, args.batch, args.seed)
    model = StatePredictor(ct["n"], args.hidden).double()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.perf_counter()
    losses = []
    for epoch in range(args.epochs):
        opt.zero_grad()
        v_pred = model(features)                                  # [B, n] complex
        loss = physics.residual_loss(v_pred, ybus, sbus_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        print(f"  epoch {epoch + 1}/{args.epochs}: residual_loss={loss.item():.6e}", flush=True)
    train_s = time.perf_counter() - t0

    # 3) validation against the oracle (when V_ref is available)
    val_mae = None
    if v_ref is not None:
        with torch.no_grad():
            f0 = torch.cat([sbus_base.real, sbus_base.imag]).unsqueeze(0)
            v0_pred = model(f0)[0].numpy()
        val_mae = float(np.mean(np.abs(v0_pred - v_ref)))

    # 4) differentiable backend: forward+backward through the solver
    diff = None
    if backends.is_differentiable(name):
        try:
            diff = backends.diff_solve_demo(ct)
        except Exception as exc:
            print(f"  [autograd] demo unavailable: {exc}")
    if diff is not None:
        print(f"  [autograd] solve fwd={diff[0]:.2f} ms, bwd={diff[1]:.2f} ms")

    return {
        "backend": name, "case": ct["name"], "n_bus": ct["n"],
        "oracle_solve_ms": None if solve_s is None else round(solve_s * 1e3, 3),
        "epochs": args.epochs, "loss_first": losses[0], "loss_last": losses[-1],
        "train_ms": round(train_s * 1e3, 3),
        "val_mae_vs_oracle": None if val_mae is None else round(val_mae, 6),
        "diff_fwd_ms": None if diff is None else round(diff[0], 3),
        "diff_bwd_ms": None if diff is None else round(diff[1], 3),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="PINN benchmark across power-flow backends.")
    p.add_argument("--backends", nargs="*", default=list(backends.BACKENDS),
                   choices=list(backends.BACKENDS))
    p.add_argument("--case", default="case118")
    p.add_argument("--dataset-root", type=Path, default=matpower_data.DEFAULT_DATASET_ROOT)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    case_path = matpower_data.resolve_case_paths(args.dataset_root, [args.case])[0]
    ct = physics.load_case(case_path)
    print(f"PINN benchmark | case={ct['name']} n_bus={ct['n']} epochs={args.epochs} batch={args.batch}")

    rows = [train_one(name, ct, args) for name in args.backends]

    print("\n==== summary ====")
    cols = ["backend", "oracle_solve_ms", "loss_first", "loss_last", "train_ms",
            "val_mae_vs_oracle", "diff_fwd_ms", "diff_bwd_ms"]
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r.get(c)) for c in cols))


if __name__ == "__main__":
    main()
