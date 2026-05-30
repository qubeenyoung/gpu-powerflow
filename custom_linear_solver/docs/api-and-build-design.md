# custom_linear_solver API and Build Design

## Goal

Provide a cuDSS-like sparse direct solver API backed by the copied CUDA
multifrontal code from `external/lin_solver`.

The solver should be GPU-resident in the cuPF path:

```text
cuPF GPU Jacobian/RHS buffers
  -> custom_linear_solver set_data/set_rhs/set_solution
  -> analyze once
  -> factorize/solve every Newton iteration
  -> cuPF GPU voltage update
```

No benchmark wrapper or host Matrix Market path should be required.

## Public API Shape

The public C++ surface is declared in:

```text
src/solver.hpp
src/matrix/view.hpp
```

Planned user flow:

```cpp
custom_linear_solver::Solver solver;

solver.set_data({
    .nrows = n,
    .ncols = n,
    .nnz = nnz,
    .index_type = custom_linear_solver::IndexType::Int32,
    .location = custom_linear_solver::DataLocation::Device,
    .row_offsets = d_csr_row_ptr,
    .col_indices = d_csr_col_idx,
    .values = d_csr_values,
});
solver.set_rhs({.size = n, .location = DataLocation::Device, .values = d_rhs});
solver.set_solution({.size = n, .location = DataLocation::Device, .values = d_x});

solver.analyze();

for each Newton iteration:
    update d_csr_values and d_rhs on the GPU
    solver.factorize();
    solver.solve();
```

The wrapper should accept device CSR because cuPF already owns GPU-resident
Jacobian/RHS buffers. Internally, the solver may build CSC/permuted structures,
but that must be hidden behind `analyze()` and device kernels.

## Phase Responsibilities

### set_data

Registers the sparse matrix descriptor.

Initial target:

- square matrix
- FP64 values
- zero-based CSR
- `int32_t` indices first, `int64_t` later if needed
- device pointers as the primary path

Host pointers are useful for tests, but the cuPF integration path should use
device pointers.

### analyze

Runs one-time value-independent work for a fixed sparsity pattern:

- validate CSR descriptor
- build or upload CSR-to-CSC/permutation maps
- compute fill-reducing ordering with METIS ND
- build symmetric pattern, elimination tree, filled pattern
- build `custom_linear_solver::plan::MultifrontalPlan`
- capture factor and solve CUDA graphs

This corresponds to cuDSS analysis and should happen from cuPF initialize.

### factorize

Runs value-dependent numeric work:

- transform current CSR values into the ordered CSC/value array expected by
  `MultifrontalPlan`
- scatter values into the multifrontal arena
- replay the factor CUDA graph
- detect zero pivot / singular flag
- optionally run shift retry if enabled

This corresponds to cuDSS factorization/refactorization and should happen every
Newton iteration after cuPF fills the Jacobian.

### solve

Runs the solve graph against the currently registered RHS:

- apply any row/column scaling or permutations
- copy/transform RHS into the plan's device solve buffer
- replay the solve CUDA graph
- inverse-permute/unscale into the user-provided solution buffer

The final cuPF path should not copy RHS or solution through host memory.

### get_data

Returns the registered descriptors and current state metadata. This is mainly
for debugging and integration checks, matching the "get data" style API the
user requested.

## Copied Source Inventory

Copied operation sources:

```text
src/solver.hpp
src/matrix/view.hpp
src/plan/multifrontal_plan.cu
src/plan/multifrontal_plan.hpp
src/factorize/multifrontal.cu
src/factorize/multifrontal.hpp
src/solve/multifrontal.cu
src/solve/multifrontal.hpp
src/symbolic/elimination_tree.cpp
src/symbolic/elimination_tree.hpp
src/symbolic/supernode.cpp
src/symbolic/supernode.hpp
src/symbolic/multifrontal.cpp
src/symbolic/multifrontal.hpp
src/reordering/metis_nd.cpp
src/reordering/metis_nd.hpp
```

Why these files:

- `solver.hpp`: cuDSS-like phase API (`set_data`, `analyze`, `factorize`, `solve`).
- `matrix/view.hpp`: GPU/host CSR and dense vector descriptors.
- `plan/multifrontal_plan.*`: shared multifrontal execution state, CUDA graph
  handles, device arena ownership, and move/lifetime management.
- `factorize/multifrontal.*`: numeric multifrontal analyze/factorize path,
  CUDA factor graph, and front arena.
- `solve/multifrontal.*`: multifrontal forward/backward solve kernels, solve
  CUDA graph capture, and solve graph replay.
- `symbolic/*`: symmetric pattern, etree, fill pattern, relaxed panels,
  multifrontal extend-add maps.
- `metis_nd.*`: nested-dissection ordering.

Not copied into the build:

- benchmark drivers
- benchmark Matrix Market tools; `scripts/io.*` keeps only the minimal runner IO
- third-party solver adapters
- old GPU factor/solve/spmv paths
- CPU own pipeline implementations
- MC64 matching experiments
- GPU nested-dissection experiments

## Current Build Target

`custom_linear_solver/CMakeLists.txt` defines:

```text
custom_linear_solver_api   interface target for public headers
custom_linear_solver_ops   static target for copied GPU operation code
custom_linear_solver_run   script executable for Matrix Market smoke runs
```

External dependencies for the operation target:

```text
CUDA runtime
METIS
```

CUDA architecture defaults to `86` because the copied multifrontal kernels use
`atomicAdd(double)`, which requires sm_60 or newer. Override with:

```bash
cmake -S custom_linear_solver -B build/custom_linear_solver \
  -DCLS_CUDA_ARCHITECTURES=80
```

The full `external/lin_solver/CMakeLists.txt` is intentionally not reused
because it pulls in benchmark-only dependencies such as MPI, MKL, MUMPS,
PaStiX, STRUMPACK, GLU, PanguLU, cuDSS, and benchmark tools.

Build the script runner with:

```bash
cmake -S custom_linear_solver -B build/custom_linear_solver \
  -DCLS_BUILD_CUDA_OPS=ON \
  -DCLS_BUILD_SCRIPTS=ON
cmake --build build/custom_linear_solver -j
```

Run a generated Newton linear system with:

```bash
build/custom_linear_solver/custom_linear_solver_run \
  /datasets/matpower_linear_systems/case30 \
  --solution-out /tmp/case30_cls_solution.mtx
```

The script runner is a harness only: it reads Matrix Market on host, uploads
CSR/RHS to device, calls `set_data`, `set_rhs`, `set_solution`, `analyze`,
`factorize`, and `solve`, then downloads the solution for residual checking.
The cuPF integration path should keep Jacobian/RHS/solution ownership in cuPF
device buffers and call the same solver API directly.

## Device-Resident Runtime Path

The cuPF-facing `Solver` path keeps the Newton-iteration work on device:

- `factorize()` calls `factorize_multifrontal_device()`.
- `solve()` calls `solve_multifrontal_device()`.
- Matrix values, RHS, and solution are expected to be device pointers.
- Pattern copies and map uploads are confined to `analyze()`.

Current device runtime entry points:

```cpp
bool factorize_multifrontal_device(
    custom_linear_solver::plan::MultifrontalPlan& plan,
    const double* d_csr_values,
    const int* d_ordered_value_to_csr,
    double* kernel_ms = nullptr);

bool solve_multifrontal_device(
    custom_linear_solver::plan::MultifrontalPlan& plan,
    const double* d_rhs,
    double* d_solution,
    const int* d_perm,
    double* kernel_ms = nullptr);
```

The remaining host/device transfers are analyze-only:

- device CSR pattern to host for CPU METIS/symbolic
- host-built symbolic/factor/solve maps to device

For current analyze bottlenecks and optimization status, see
`docs/analyze-bottleneck-and-optimization.md`.

## cuPF Integration Point

The cuPF adapter should mirror `CudaLinearSolveCuDSS`:

```text
initialize -> Solver::set_data + Solver::analyze
prepare_rhs -> Solver::set_rhs / update RHS pointer if needed
factorize  -> Solver::factorize
solve      -> Solver::solve
```

The first cuPF target should be:

- CUDA FP64
- single matrix / single RHS
- forward Newton solve only

Batch, FP32/Mixed, and adjoint/transpose solve should be added after the
forward path is correct.
