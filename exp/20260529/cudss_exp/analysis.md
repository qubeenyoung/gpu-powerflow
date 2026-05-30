# cuDSS Flat-Time Diagnosis

## Setup

- Instrumented binary: `/tmp/custom_linear_solver_cudss_profile/cudss_run`
- Source runner: `custom_linear_solver/scripts/run_cudss_solver.cpp`
- Instrumentation: NVTX ranges around read, upload, setup, analysis, each
  factorize iteration, each solve iteration, and residual check.
- nsys cases: `case5`, `case118`, `case1197`, `case6468rte`,
  `case_ACTIVSg25k`, `case_ACTIVSg70k`, `case_SyntheticUSA`
- nsys repeat: 5, to expose first-call versus warm-call behavior.
- ncu scope: first `cudss_factorize_0` and first `cudss_solve_0` ranges only.

## Main Finding

The flat cuDSS `factorize` and `solve` times in the original independent-process
benchmark are mostly first-use CUDA/cuDSS launch overhead, not numerical kernel
time.

In the original benchmark, each raw row launched a fresh process with
`--repeat 1`, so every measurement included the first `CUDSS_PHASE_FACTORIZATION`
and first `CUDSS_PHASE_SOLVE` after analysis. nsys shows those first calls spend
most of their wall time inside `cudaLaunchKernel_v7000`, while the GPU kernels
themselves are tiny on small cases.

## nsys Evidence

First `factorize` and `solve` ranges:

| case | n | phase | wall ms | CUDA runtime API ms | GPU kernel ms | kernel count |
|---|---:|---|---:|---:|---:|---:|
| case5 | 5 | factorize_0 | 15.451 | 15.411 | 0.022 | 6 |
| case5 | 5 | solve_0 | 3.531 | 3.503 | 0.017 | 4 |
| case1197 | 2392 | factorize_0 | 16.583 | 16.537 | 0.120 | 7 |
| case1197 | 2392 | solve_0 | 3.612 | 3.573 | 0.082 | 7 |
| case_ACTIVSg25k | 47246 | factorize_0 | 16.939 | 16.893 | 1.784 | 7 |
| case_ACTIVSg25k | 47246 | solve_0 | 4.188 | 4.152 | 0.714 | 7 |
| case_SyntheticUSA | 156255 | factorize_0 | 19.415 | 19.366 | 4.629 | 7 |
| case_SyntheticUSA | 156255 | solve_0 | 5.104 | 5.061 | 1.468 | 7 |

The runtime breakdown makes the first-call fixed cost explicit:

| case | range | dominant CUDA runtime calls |
|---|---|---|
| case5 | factorize_0 | `cudaLaunchKernel`: 15.386 ms over 6 launches; sync: 0.011 ms |
| case5 | factorize_1 | `cudaLaunchKernel`: 0.022 ms over 6 launches; sync: 0.012 ms |
| case5 | solve_0 | `cudaLaunchKernel`: 3.465 ms over 4 launches; sync: 0.003 ms |
| case5 | solve_1 | `cudaLaunchKernel`: 0.013 ms over 4 launches; sync: 0.011 ms |
| case_SyntheticUSA | factorize_0 | `cudaLaunchKernel`: 14.644 ms over 7 launches; sync: 4.450 ms |
| case_SyntheticUSA | factorize_1 | `cudaLaunchKernel`: 0.023 ms over 7 launches; sync: 4.647 ms |
| case_SyntheticUSA | solve_0 | `cudaLaunchKernel`: 3.649 ms over 7 launches; sync: 1.361 ms |
| case_SyntheticUSA | solve_1 | `cudaLaunchKernel`: 0.022 ms over 7 launches; sync: 1.452 ms |

Interpretation:

- The first `factorize` has a roughly 15 ms launch-side cost across all case
  sizes.
- The first `solve` has a roughly 3.5 ms launch-side cost across all case sizes.
- After the first call, `cudaLaunchKernel` cost drops to about 0.02 ms total.
- Warm factorize/solve then track actual GPU kernel time and scale with case
  size.

## CUDA Module Loading Check

`CUDA_MODULE_LOADING=EAGER` moves the first-use loading cost into setup. This
collapses first factorize/solve time:

| case | default factor ms | EAGER factor ms | default solve ms | EAGER solve ms | EAGER setup ms |
|---|---:|---:|---:|---:|---:|
| case5 | 12.07 | 0.067 | 2.89 | 0.058 | 675.82 |
| case118 | 11.92 | 0.117 | 2.86 | 0.089 | 675.69 |
| case1197 | 12.03 | 0.242 | 2.94 | 0.127 | 678.71 |
| case6468rte | 12.55 | 0.713 | 3.11 | 0.306 | 685.06 |
| case_ACTIVSg25k | 13.39 | 1.925 | 3.58 | 0.762 | 673.40 |
| case_ACTIVSg70k | 15.17 | 4.705 | 4.18 | 1.402 | 675.93 |
| case_SyntheticUSA | 15.11 | 4.923 | 4.36 | 1.514 | 677.05 |

This confirms the flat floor is primarily lazy module loading / first kernel
launch cost. It is not a true factorization or triangular-solve numerical cost.

## ncu Evidence

ncu was collected only inside `cudss_factorize_0` and `cudss_solve_0` NVTX
ranges. ncu adds heavy replay overhead to application wall time, so use the
kernel metrics, not the runner-reported phase time under ncu.

| case | n | phase | kernel invocations | kernel duration sum ms | mean achieved occupancy % | max grid size |
|---|---:|---|---:|---:|---:|---:|
| case5 | 5 | factorize | 6 | 0.033 | 6.28 | 3 |
| case5 | 5 | solve | 4 | 0.027 | 5.12 | 3 |
| case1197 | 2392 | factorize | 6 | 0.156 | 19.82 | 1598 |
| case1197 | 2392 | solve | 6 | 0.100 | 15.57 | 1004 |
| case_ACTIVSg25k | 47246 | factorize | 6 | 2.182 | 40.48 | 23633 |
| case_ACTIVSg25k | 47246 | solve | 6 | 0.495 | 26.64 | 22342 |
| case_SyntheticUSA | 156255 | factorize | 6 | 5.506 | 45.78 | 75561 |
| case_SyntheticUSA | 156255 | solve | 6 | 1.160 | 41.15 | 75561 |

Interpretation:

- cuDSS uses a small, nearly fixed number of kernels for factorize/solve.
- Tiny cases launch grids of size only 1-3, so GPU utilization is naturally low.
- Kernel duration and occupancy grow with matrix size, which is the expected
  numerical scaling.
- The original flat wall-time curve hid this scaling behind first-launch fixed
  overhead.

## Conclusion

The cause is not that cuDSS numerical factorization/solve is inherently constant
time. The original measurement repeatedly captured cold first-use overhead:

1. independent process per raw row,
2. `--repeat 1`,
3. first `CUDSS_PHASE_FACTORIZATION`,
4. first `CUDSS_PHASE_SOLVE`,
5. CUDA lazy module loading charged to the first `cudaLaunchKernel` calls.

After warm-up or with `CUDA_MODULE_LOADING=EAGER`, factorize/solve scale with
case size:

- `case5`: factorize about 0.07 ms, solve about 0.06 ms
- `case_SyntheticUSA`: factorize about 4.9 ms, solve about 1.5 ms

For fair solver comparison, the benchmark should report both:

- cold process first-call time, if startup behavior matters;
- warmed steady-state time, if repeated Newton iterations are the target.

For cuPF Newton-Raphson, warmed steady-state is more relevant after initialization.

## Artifacts

- `results/nsys/*_repeat5.nsys-rep`: Nsight Systems reports
- `results/nsys/*_repeat5.sqlite`: exported SQLite traces
- `results/nsys/nsys_phase_summary.csv`: every NVTX phase range
- `results/nsys/nsys_phase_summary_agg.csv`: phase averages
- `results/nsys/nsys_first_vs_warm.csv`: first call versus warm-call summary
- `results/nsys/nsys_runtime_call_summary.csv`: CUDA runtime call breakdown
- `results/ncu/*_factorize0_basic.csv`: ncu basic metrics for first factorize
- `results/ncu/*_solve0_basic.csv`: ncu basic metrics for first solve
- `results/ncu/ncu_phase_kernel_summary.csv`: per-kernel ncu metrics
- `results/ncu/ncu_phase_summary.csv`: phase-level ncu summary
- `results/eager/eager_single_run.csv`: `CUDA_MODULE_LOADING=EAGER` runs
- `results/eager/module_loading_comparison.csv`: default versus EAGER comparison

