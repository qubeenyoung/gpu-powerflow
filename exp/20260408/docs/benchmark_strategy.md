# cuPF Benchmark IO / Timing Strategy

## Directory Layout

`exp/20260408/` below is reserved for this experiment set.

- `docs/`: experiment design, run notes, interpretation notes
- `scripts/`: benchmark runners, manifest generators, post-processing helpers
- `results/`: raw outputs from each run

## Target Input Cases

The shortest input path for `cuPF` is the existing v1 dump format that `cuPF`
already reads in [dump_case_loader.cpp](/workspace/cuPF/tests/cpp/dump_case_loader.cpp).

Current target dump directories in this workspace:

- `/workspace/v1/core/dumps/case118_ieee`
- `/workspace/v1/core/dumps/case1354_pegase`
- `/workspace/v1/core/dumps/case6468_rte`
- `/workspace/v1/core/dumps/case13659_pegase`

Note:

- The repository currently contains `case1354_pegase`, not `case1454`.
- Benchmark input should reuse the existing dump files directly:
  `dump_Ybus.mtx`, `dump_Sbus.txt`, `dump_V.txt`, `dump_pv.txt`, `dump_pq.txt`

## Measurement Policy

The first timing insertion point is the common NR loop in
[newton_solver.cpp](/workspace/cuPF/cpp/src/newton_solver/core/newton_solver.cpp).
This gives one timing policy for both CPU and CUDA.

Measured scopes:

- `NR.analyze.total`
- `NR.solve.total`
- `NR.solve.initialize`
- `NR.iteration.total`
- `NR.computeMismatch`
- `NR.updateJacobian`
- `NR.solveLinearSystem`
- `NR.updateVoltage`
- `NR.solve.downloadV`
- `NR.batch.solve.total`
- `NR.batch.initialize`
- `NR.batch.iteration.total`
- `NR.batch.computeMismatch`
- `NR.batch.updateJacobian`
- `NR.batch.solveLinearSystem`
- `NR.batch.updateVoltage`
- `NR.batch.downloadV`

## CUDA Timing Rule

CUDA work is asynchronous, so wall-clock timing from `NewtonSolver` is only
useful if the backend flushes outstanding work before each timer scope ends.

For that reason:

- the backend interface exposes `synchronizeForTiming()`
- CPU keeps the default no-op behavior
- CUDA calls `cudaDeviceSynchronize()` only when the core timing path requests it

This keeps the timing policy centralized in the NR loop while avoiding
permanent synchronization in normal non-timed builds.

## Logging / Compile-Time Switches

The inserted timing and log code follows the existing compile-time switches:

- `CUPF_ENABLE_LOG`
- `CUPF_ENABLE_TIMING`
- `CUPF_ENABLE_DUMP`

Recommended build modes:

- performance run: `CUPF_ENABLE_TIMING` on, dump off, log optional
- validation run: timing on or off, dump on, log optional

## Result Storage Plan

`results/` should eventually store one directory per experiment run, for example:

- `results/<run_name>/manifest.json`
- `results/<run_name>/runs.csv`
- `results/<run_name>/<case>/<backend>/summary.json`
- `results/<run_name>/<case>/<backend>/dumps/`

The code inserted in this step only adds common log/timer scopes.
Result-file writing should stay in a dedicated benchmark runner, not inside the
solver core.
