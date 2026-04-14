# GPU AMGX FGMRES + AMG JFNK Experiment

Date: 2026-04-14

## Scope

- Implemented a C++/CUDA nonlinear JFNK probe under `exp/20260414/amgx`.
- The Krylov matvec is Jacobian-free: the current device voltage state is perturbed on GPU, a CUDA mismatch kernel computes `F(x + eps v)`, and a CUDA kernel forms `(F(x + eps v) - F(x)) / eps`. This experiment does not read or solve saved `J.csr` linear systems.
- FGMRES vectors, Arnoldi basis vectors, residuals, `dx`, and Jv scratch buffers are device resident. The host still drives scalar control flow and the small GMRES least-squares solve.
- AMGX is used as the `amg_fd` right preconditioner for FGMRES. The preconditioner matrix values are assembled on GPU using colored finite-difference Jv probes; only the fixed sparsity/coloring metadata is built on CPU and uploaded.
- AMGX runs in device double precision integer mode, `AMGX_mode_dDDI`. AMGX preconditioner application uses device pointers for rhs/x, so inner iterations do not round-trip Krylov vectors through host memory.
- Dataset source: `/workspace/datasets/texas_univ_cases/pf_dataset`.
- Converted cuPF dumps: `/workspace/exp/20260414/amgx/cupf_dumps`.

## Build

```bash
cmake -S /workspace/exp/20260414/amgx \
  -B /workspace/exp/20260414/amgx/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build /workspace/exp/20260414/amgx/build \
  --target amgx_jfnk_probe \
  -j
```

## Evaluation Setup

- Solver: `fgmres`
- Preconditioner: `amg_fd`
- Nonlinear tolerance: `1e-8`
- Linear tolerance: `1e-2`
- GMRES restart: `30`
- Nonlinear max iterations: `20`
- FD epsilon: `auto`
- AMGX version from run log: `2.4.0`
- CUDA architecture: `sm_86`

The first full run used `inner_max_iter=500` on all 12 Texas University cases. The 9 failures were rerun with `inner_max_iter=2000`.

Raw result files:

- `/workspace/exp/20260414/amgx/results/texas_univ_fgmres_amgx_amg_inner500.csv`
- `/workspace/exp/20260414/amgx/results/texas_univ_fgmres_amgx_amg_inner2000_failures.csv`

## Summary

- `inner_max_iter=500`: 3 / 12 converged to `1e-8`; summed per-case runtime `10.891 s`.
- `inner_max_iter=2000` rerun on failures: 1 / 9 recovered; summed per-case runtime `24.492 s`.
- Effective result after rerun: 4 / 12 converged to `1e-8`; summed per-case runtime `26.602 s`.
- Dominant failure reason: `linear_max_inner_iterations`.
- `Texas7k_20220923` improved from `1.567e-02` at 500 to `1.175e-04` at 2000, but still did not reach `1e-8`.

## Effective Results

For cases that passed at 500, the 500 result is used. For 500 failures, the 2000 rerun result is used.

| case | converged | inner cap | outer iters | final mismatch | total inner | max inner | Jv calls | sec | failure reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case_ACTIVSg200 | true | 500 | 4 | 4.608e-09 | 76 | 36 | 137 | 0.795 | none |
| case_ACTIVSg500 | true | 500 | 5 | 6.649e-09 | 82 | 27 | 202 | 0.409 | none |
| MemphisCase2026_Mar7 | true | 2000 | 4 | 2.506e-09 | 2030 | 1591 | 2195 | 1.996 | none |
| case_ACTIVSg2000 | true | 500 | 6 | 8.820e-10 | 536 | 206 | 656 | 0.906 | none |
| Base_Florida_42GW | false | 2000 | 2 | 1.400e+01 | 2012 | 2000 | 2143 | 1.890 | linear_max_inner_iterations |
| Texas7k_20220923 | false | 2000 | 3 | 1.175e-04 | 2657 | 2000 | 2823 | 2.899 | linear_max_inner_iterations |
| Base_Texas_66GW | false | 2000 | 1 | 9.903e+01 | 2000 | 2000 | 2091 | 2.173 | linear_max_inner_iterations |
| Base_MIOHIN_76GW | false | 2000 | 2 | 8.361e+00 | 2104 | 2000 | 2238 | 2.539 | linear_max_inner_iterations |
| Base_West_Interconnect_121GW | false | 2000 | 1 | 3.205e+01 | 2000 | 2000 | 2126 | 2.394 | linear_max_inner_iterations |
| case_ACTIVSg25k | false | 2000 | 1 | 4.578e+01 | 2000 | 2000 | 2091 | 2.699 | linear_max_inner_iterations |
| case_ACTIVSg70k | false | 2000 | 1 | 1.301e+02 | 2000 | 2000 | 2102 | 3.773 | linear_max_inner_iterations |
| Base_Eastern_Interconnect_515GW | false | 2000 | 1 | 5.137e+02 | 2000 | 2000 | 2099 | 4.129 | linear_max_inner_iterations |

## Conclusion

The implemented GPU FGMRES + AMGX AMG preconditioner now keeps the expensive JFNK inner path on device. The convergence picture is mostly unchanged from the earlier hybrid prototype: it solves the small cases and Memphis with a larger inner cap, while the larger cases hit the linear iteration limit before useful Newton progress.

Next tuning targets are AMGX AMG configuration, preconditioner reuse policy, a stronger FD Jacobian scaling strategy, and comparing against the existing ILU(0)/ILUT JFNK baselines on the same converted Texas University dumps.
