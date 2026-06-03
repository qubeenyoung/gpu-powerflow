# custom_linear_solver — Documentation Index

GPU-resident sparse direct solver (multifrontal LU) specialized for power-flow
Newton–Raphson Jacobians, with a cuDSS-style `analyze → factorize → solve` API.

**Start here:** [`main-report.md`](main-report.md) — overview, pipeline, measured
performance, and the techniques behind them.

## Current reference

| document | what it covers |
|---|---|
| [`main-report.md`](main-report.md) | **The report.** What the solver is, how it works, results, key techniques, limitations. |
| [`api-and-build-design.md`](api-and-build-design.md) | Public API shape, phase responsibilities, build targets/flags, cuPF integration points. |
| [`related-work-and-contribution.md`](related-work-and-contribution.md) | GPU sparse-solver landscape, why general solvers are slow on power-grid Jacobians, and an honest novelty assessment. |

## History (per-cycle reports & design deep-dives)

Chronological records of how the optimizations were found and measured. Kept for
provenance; the conclusions are folded into `main-report.md`.

| document | what it covers |
|---|---|
| [`history/fsa-optimization-report.md`](history/fsa-optimization-report.md) | factorize/solve/analyze optimization frontier: GPU symbolic build, parallel ND, partitioned-inverse solve, batching, where 30 % is and isn't reachable. |
| [`history/analyze-bottleneck-and-optimization.md`](history/analyze-bottleneck-and-optimization.md) | Analyze-phase bottleneck breakdown (METIS separator dominance) and what moved the needle. |
| [`history/fp32-batched-factor-solve-optimization.md`](history/fp32-batched-factor-solve-optimization.md) | FP32-native batched factor+solve; tensor cores vs. the real (latency) bottleneck. |
| [`history/tensor-core-factorize-design.md`](history/tensor-core-factorize-design.md) | Tensor-core batched multifrontal factorize design; amalgamation to grow fronts past the thin-K limit. |
| [`history/b1-single-system-optimization.md`](history/b1-single-system-optimization.md) | B=1 single-system fp32/fp64 study: the critical-path floor, what was tried, and the precision lever. |
| [`history/warm-cache-stack-port-expectations.md`](history/warm-cache-stack-port-expectations.md) | Expectations when porting `perf/warm-cache-stack` (mysolver) techniques into this solver / cuPF. |

## Reproducing the numbers

Build (METIS + CUDA required; build out-of-tree):

```bash
cmake -S custom_linear_solver -B /tmp/clsbuild \
  -DCLS_BUILD_CUDA_OPS=ON -DCLS_BUILD_SCRIPTS=ON -DCLS_CUDA_ARCHITECTURES=86
cmake --build /tmp/clsbuild -j
```

Run one Newton linear system (kernel timing on, median of repeats):

```bash
CLS_KERNEL_TIME=1 /tmp/clsbuild/custom_linear_solver_run \
  /workspace/cls_linsys/case9241pegase --repeat 80 --single-precision fp64
# --single-precision fp64 | fp32 | mixed     --batch B   (uniform-batch experiment)
```

Research knobs: `CLS_KERNEL_TIME`, `CLS_DUMP` (front/level structure),
`CLS_CAP` (amalgamation cap), `CLS_SHCNT` (FP32 shared-front level-count gate),
`CLS_COOP_SOLVE` (cooperative solve — documented negative result), `CLS_FT`
(mid-spine factor threads).
