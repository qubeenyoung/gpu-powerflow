# benchmark — PINN training across power-flow backends

A physics-informed (PINN) training benchmark: a trainable **dummy layer** sits in
front of every power-flow backend, and a small network is trained for a few epochs
so each backend can be compared *as used inside an ML training loop*.

## What it does

State-prediction PINN: a load scenario is fed through a dummy MLP that predicts the
bus voltage state **V**, and the loss is the power-flow residual

```
F(V) = V ⊙ conj(Ybus·V) − Sbus            loss = mean‖F(V)‖²
```

which is zero at a power-flow solution. Only the dummy MLP trains; each **backend**
plays the *physics* role:

| backend | role | differentiable |
|---|---|---|
| `pypower`, `pandapower` | CPU Newton (pandapower stack) — forward oracle | no |
| `cupf-pybind` | cuPF `_cupf` NewtonSolver (CUDA cuDSS) — forward oracle | no |
| `cupf-torch` | cuPF torch autograd wrapper — **gradient through the NR solve** | **yes** |
| `exapf` | ExaPF.jl (Julia) — forward solve timing (best effort) | no |
| `matpower` | MATPOWER (MATLAB) — forward solve timing (best effort) | no |

The oracle gives a reference `V_ref` (for a validation MAE) and a forward solve time.
The differentiable backend (`cupf-torch`) additionally runs a forward+backward demo,
measuring the gradient passed back through the solver (adjoint / implicit-function).

## Run

```bash
# all backends, 5 epochs, case118
python3 -m benchmark.pinn.train --case case118 --epochs 5

# pick backends / case
python3 -m benchmark.pinn.train --backends pypower cupf-torch --case case300
```

Options: `--backends`, `--case`, `--dataset-root` (default `/datasets/matpower`),
`--epochs` (5), `--batch` (64), `--hidden`, `--lr`, `--seed`.

`exapf`/`matpower` need the Julia/MATLAB toolchains; if absent their oracle is
skipped (the PINN still trains — the residual loss needs only Ybus/Sbus).

## Layout

```
benchmark/
  backends/              relocated backend runner code + the PINN adapter
    exapf/               ExaPF.jl (Julia) runner
    matlab/              MATPOWER (MATLAB) runner — run_matpower_case.*, login_online.bash
    pypower.py           pandapower / SciPy NR baseline runner
    pandapower.py        pandapower baseline (shim over pypower)
    matpower.py          MATLAB MATPOWER baseline runner
    __init__.py          per-backend physics oracle + cupf-torch differentiable demo
  pinn/
    physics.py           case tensors + residual F(V) and the PINN loss
    model.py             StatePredictor — the trainable dummy MLP (flat-start V)
    train.py             the 5-epoch training loop + summary table
```

cuPF stays in `cuPF/` (library, `run_cupf.py`, torch wrapper); the adapter imports it.
