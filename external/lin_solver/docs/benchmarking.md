# Benchmarking

This project standardizes all solver experiments as a sparse linear system:

```text
A * x = rhs
```

Every benchmark case must provide:

| File | Meaning |
|---|---|
| `*.mtx`, `J.mtx`, or another Matrix Market matrix | Sparse matrix `A` |
| `rhs.mtx` | Right-hand-side vector |
| `x_true.mtx` | Reference solution vector |

The benchmark runner reads `rhs.mtx` and checks that `x_true.mtx` exists. It
does not generate fallback random vectors. Numerical quality should be measured
with the `compute_error` C++ API after a solver writes a candidate solution
vector.

## Dataset Sources

| Dataset family | Matrix file | `rhs.mtx` | `x_true.mtx` |
|---|---|---|---|
| SuiteSparse Matrix Collection | Extracted matrix, for example `/datasets/benchmark_matrices/matrices/memplus/memplus.mtx` | Computed as `A * x_true` | Fixed-seed random vector from `U(-1, 1)` |
| MATPOWER Newton-Raphson | `/datasets/power_system/nr_linear_systems/<case>/J.mtx` | Copy of `F.mtx` | Solution of `J * x_true = F`, computed with SuiteSparse KLU |

The default SuiteSparse random seed is:

```text
LINEAR_SYSTEM_RANDOM_SEED=20260521
```

## Error Metrics

`compute_error` reports exactly two metrics:

| Metric | Definition | Purpose |
|---|---|---|
| `berr` | `max_i |(A*x-rhs)_i| / (|A|*|x| + |rhs|)_i` | Componentwise backward error with row-wise scaling |
| `absolute_error` | `||x - x_true||_2` | Absolute solution-vector error against the reference solution |

If a row denominator in `berr` is zero and the corresponding residual is
nonzero, the reported `berr` is infinity.

Path-based example:

```cpp
#include "tools/compute_error.hpp"

const sparse_direct::error::ErrorMetrics metrics =
    sparse_direct::error::compute_error(
        "/datasets/power_system/nr_linear_systems/case30/J.mtx",
        "/datasets/power_system/nr_linear_systems/case30/rhs.mtx",
        "/path/to/solver_output_x.mtx",
        "/datasets/power_system/nr_linear_systems/case30/x_true.mtx");
```

In-memory example:

```cpp
#include "tools/compute_error.hpp"

const sparse_direct::error::ErrorMetrics metrics =
    sparse_direct::error::compute_error(matrix, rhs, solver_x, x_true);
```

Use `x_true` as the candidate solution to sanity-check a generated case. Both
metrics should be near zero, subject to the numerical properties of the system.

## Solver Evaluation Output

The benchmark executable is implemented in
`src/benchmark/third_party_solvers.cpp`. It writes CSV by default:

```text
report/benchmark/third_party_solvers.csv
```

Each solver/case pair appends one row with:

```text
matrix,group,solver,rows,cols,nnz,density,analysis_ms,factor_ms,solve_ms,berr,absolute_error,success,message
```

Example:

```bash
benchmark \
  --matrix-set smoke \
  --solver klu,umfpack \
  --output report/benchmark/smoke.csv
```

Use `--append` to append rows to an existing CSV. The header is written when
the output file is created or truncated.

## Profiling Blocks

The benchmark code provides block profiling through `src/tools/profiling.hpp`.
The implementation is split by responsibility:

| File | Responsibility |
|---|---|
| `src/tools/timer.hpp`, `src/tools/timer.cpp` | CPU wall-clock timing and CUDA event timing |
| `src/tools/nvtx_profiler.hpp`, `src/tools/nvtx_profiler.cpp` | NVTX ranges and marks for Nsight tools |
| `src/tools/profiling.hpp`, `src/tools/profiling.cpp` | Public scope macros that combine timer and NVTX behavior |

Use these macros in benchmark or solver code:

| Macro | Behavior when profiling is enabled | Behavior when disabled |
|---|---|---|
| `PROFILE_SCOPE("name")` | Opens a scoped CPU timer/NVTX range until the current C++ block exits | Expands to no-op |
| `PROFILE_CUDA_SCOPE("name")` | Opens a scoped CPU timer/NVTX range and, when enabled, a CUDA event timer | Expands to no-op |
| `PROFILE_FUNCTION()` | Uses the current function name as the scope name | Expands to no-op |
| `PROFILE_CUDA_FUNCTION()` | Uses the current function name with CUDA event timing | Expands to no-op |
| `PROFILE_MARK("name")` | Emits an instant marker | Expands to no-op |

The current benchmark emits these ranges:

| Range | Meaning |
|---|---|
| `benchmark.total` | Full benchmark process |
| `case.<group>.<matrix>` | One selected benchmark matrix/case |
| `solve.<matrix>.<solver>` | One solver call, emitted with `PROFILE_CUDA_SCOPE` |

Profiling is compile-time controlled:

```bash
cmake -S . -B /tmp/profile-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TIMER=ON \
  -DENABLE_CUDA_TIMER=ON \
  -DENABLE_NVTX=ON

cmake --build /tmp/profile-build --target benchmark -j
```

| CMake option | Default | Effect |
|---|---:|---|
| `ENABLE_TIMER` | `OFF` | Prints CPU wall-clock `cpu_ms` for each scope |
| `ENABLE_CUDA_TIMER` | `OFF` | Prints CUDA event elapsed time `cuda_ms` for each scope |
| `ENABLE_NVTX` | `OFF` | Emits NVTX ranges and marks for Nsight tools |

With all profiling options off, the macros do not evaluate their arguments and
compile to no-op.

Timer output with CPU and CUDA timing enabled looks like:

```text
[timer] solve.case30.cudss-gpu cpu_ms=12.345 cuda_ms=10.921
```

CUDA timing uses events on the default stream. It is most useful around CUDA
solver calls such as cuDSS. Use `PROFILE_SCOPE` for CPU-only regions and
`PROFILE_CUDA_SCOPE` for regions that may launch CUDA work. CUDA scopes can
report `cuda_ms=unavailable` when CUDA events cannot be created in the current
runtime environment.

## Nsight Usage

Nsight Systems timeline capture:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output report/nsys/benchmark \
  /tmp/profile-build/benchmark --matrix-set smoke --solver cudss-gpu
```

Nsight Compute can use NVTX ranges to focus on a solver block:

```bash
ncu \
  --nvtx \
  --nvtx-include 'solve.*.cudss-gpu' \
  --target-processes all \
  /tmp/profile-build/benchmark --matrix-set smoke --solver cudss-gpu
```

The NVTX ranges are labels for Nsight. Kernel, CUDA API, and memory activity
are still measured by Nsight Systems or Nsight Compute.
