# AMGX v2

This directory is a readability-first refactor target for the experimental AMGX JFNK code in `../amgx/cpp/jfnk_amgx_gpu.cu`.

The current implementation builds an augmented 2-slot-per-bus Newton system directly in bus-local ordering:

```text
x = [bus0.theta, bus0.Vm, bus1.theta, bus1.Vm, ...]
```

Slack rows and PV `Vm` rows are represented as fixed identity rows so the augmented matrix remains square while the corresponding update is forced to zero.

## Directory Layout

- `src/model`: bus graph ordering, bus-local indexing, and augmented CSR pattern definition.
- `src/assembly`: device-side Jacobian assembly, residual assembly, and voltage update.
- `src/linear`: AMGX preconditioner, CSR SpMV, vector utilities, and FGMRES.
- `src/solver`: top-level Newton/AMGX solver and shared solver options/stats.
- `src/tools`: dump-case command line runners.

## Current Implementation Status

Implemented:

- Natural and reverse Cuthill-McKee bus ordering.
- Bus-local indexing with two formal slots per bus: `[theta_i, Vm_i]`.
- PV buses use active `P/theta` and fixed `Q/Vm` slots.
- PQ buses use active `P/theta` and `Q/Vm` slots.
- Slack buses use fixed identity slots.
- Bus-local augmented CSR pattern generation.
- Device-side analytic Jacobian value assembly for the augmented CSR.
- Device-side residual assembly for active slots, with zeros in fixed slots.
- Device-side voltage update from active augmented slots.
- Device-side CSR SpMV for the assembled bus-local Jacobian.
- FGMRES implementation with callback-based operator and preconditioner hooks.
- AMGX preconditioner wrapper with device-pointer matrix/vector setup.
- Top-level nonlinear Newton loop with nonlinear residual tolerance.
- Command line probe for the existing `dump_*` case directories.

Still missing:

- Numerical validation against the old `amgx_jfnk_probe` and cuDSS/Newton baselines.
- Per-outer residual trace CSV and detailed timing breakdown.

## Build

```bash
cmake -S exp/20260414/amgx_v2 -B exp/20260414/amgx_v2/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build exp/20260414/amgx_v2/build -j
```

## Run

```bash
./exp/20260414/amgx_v2/build/amgx_v2_probe \
  --case case_ACTIVSg200 \
  --ordering natural \
  --nonlinear-tol 1e-8 \
  --linear-tol 1e-2 \
  --inner-max-iter 500 \
  --gmres-restart 200
```

The default dataset root is `exp/20260414/amgx/cupf_dumps`. Use `--list-cases` to list available dumps.
