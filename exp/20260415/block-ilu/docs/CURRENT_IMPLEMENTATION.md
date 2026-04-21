# Current Implementation

This document describes the current implementation in
`exp/20260415/block-ilu`. It follows the active Schur-reduced Newton path:

```text
outer Newton state: FP64
inner Schur linear solve: FP32
J11 solver: partitioned block dense LU
Schur preconditioner: J22 partitioned block dense LU
linear solver: BiCGSTAB by default, restarted GMRES optionally
```

The main executable is:

```text
exp/20260415/block-ilu/build/block_ilu_probe
```

## Reduced Jacobian Layout

The code uses the cuPF reduced power-flow variable layout:

```text
pvpq = pv || pq
n_pvpq = len(pvpq)
n_pq = len(pq)
dim = n_pvpq + n_pq
```

Rows:

```text
[0, n_pvpq)    P mismatch rows for pvpq buses
[n_pvpq, dim)  Q mismatch rows for pq buses
```

Columns:

```text
[0, n_pvpq)    theta variables for pvpq buses
[n_pvpq, dim)  Vm variables for pq buses
```

The Newton Jacobian is split as:

```text
J = [ J11  J12 ]
    [ J21  J22 ]

J11 = dP / dtheta
J12 = dP / dVm
J21 = dQ / dtheta
J22 = dQ / dVm
```

The implementation builds CSR patterns for the full reduced Jacobian and for
each block:

```text
full J: dim     x dim
J11:    n_pvpq x n_pvpq
J12:    n_pvpq x n_pq
J21:    n_pq   x n_pvpq
J22:    n_pq   x n_pq
```

Relevant files:

```text
src/model/reduced_jacobian.hpp
src/model/reduced_jacobian.cpp
src/assembly/reduced_jacobian_assembler.hpp
src/assembly/reduced_jacobian_assembler.cu
```

## Newton Solver

The top-level solver is:

```text
PowerFlowBlockIluSolver
```

Relevant files:

```text
src/solver/powerflow_block_ilu_solver.hpp
src/solver/powerflow_block_ilu_solver.cu
```

At each Newton iteration it performs:

```text
1. Compute mismatch F(V) and infinity norm.
2. Stop if ||F||_inf <= nonlinear_tolerance.
3. Assemble current Jacobian values.
4. Initialize Schur operator and block solvers if this is the first iteration.
5. Refresh FP32 Schur/block values from FP64 Jacobian values.
6. Numerically factor J11 block dense LU.
7. Numerically factor J22 block dense LU if enabled as Schur preconditioner.
8. Form rhs = -F.
9. Solve the Schur reduced linear system with the selected Krylov solver.
10. Recover full dx = [dtheta; dVm].
11. Apply voltage update, optionally with backtracking line search.
```

The voltage state and mismatch calculation remain FP64. The active inner
linear solve path stores Schur matrices and Krylov vectors in FP32.

## Implicit Schur System

The active linear system is the Schur complement in `dVm`:

```text
S dVm = rhs_s

S x = J22 x - J21 (J11^{-1} (J12 x))
```

The Schur RHS is:

```text
rhs_s = rhs_q - J21 (J11^{-1} rhs_p)
```

After solving for `dVm`, the solver recovers `dtheta`:

```text
dtheta = J11^{-1} (rhs_p - J12 dVm)
```

Relevant files:

```text
src/linear/implicit_schur_operator_f32.hpp
src/linear/implicit_schur_operator_f32.cu
```

The Schur matvec uses two CUDA streams:

```text
j22_stream:
  J22 x

chain_stream:
  J12 x -> J11 solve -> J21 result
```

The final output is:

```text
y = J22 x - J21 (J11^{-1} (J12 x))
```

## J11 Block Dense LU

The current J11 solver is:

```text
PartitionedDenseLuJ11BlockF32
```

Despite the class name, the implementation is a generic square CSR block dense
LU solver and is also reused for J22.

Relevant files:

```text
src/linear/partitioned_dense_lu_j11_f32.hpp
src/linear/partitioned_dense_lu_j11_f32.cu
```

Analysis builds a block diagonal dense approximation from the sparse block:

```text
1. Build row/column permutation.
2. Partition the graph into blocks.
3. Build maps from CSR entries to dense block slots.
4. Build device arrays of matrix and RHS pointers for batched dense operations.
```

Current production settings use:

```text
reorder:    AMD
partition:  METIS
block size: CLI value, e.g. 16 or 64
backend:    cuBLAS batched getrf/getrs
```

For every Newton iteration, values are refreshed and each dense block is
factorized numerically:

```text
CSR J11 values -> dense block values -> cublasSgetrfBatched
```

Applying `J11^{-1}` gathers the RHS into dense block order, applies batched
LU solve, and scatters the solution back:

```text
rhs_old -> dense_rhs -> cublasSgetrsBatched -> out_old
```

## J22 Schur Preconditioner

The active Schur preconditioner is:

```text
M ~= blockdiag(J22 blocks)
z = M^{-1} r
```

CLI option:

```text
--schur-preconditioner j22-block-dense-lu
```

The code reuses `PartitionedDenseLuJ11BlockF32` for J22:

```text
j22_dense_.analyze(
  J22 FP32 CSR,
  host J22 pattern,
  J11 reorder option,
  J11 block size option,
  cuBLAS dense backend,
  J11 partition option
)
```

The J22 block dense preconditioner therefore uses the same block size and graph
partition mode as the J11 dense solve.

The preconditioner is applied inside BiCGSTAB as right preconditioning:

```text
p_hat = M^{-1} p
v     = S p_hat

s     = r - alpha v
s_hat = M^{-1} s
t     = S s_hat

dVm += alpha p_hat + omega s_hat
r    = s - omega t
```

The implementation also contains an optional J22 ILU0 Schur preconditioner:

```text
--schur-preconditioner j22-ilu0
```

That path uses a separate FP32 J22 value buffer because cuSPARSE ILU0 overwrites
CSR values in place. The block dense LU path does not overwrite the J22 SpMV
CSR value buffer.

## Krylov Solvers

The default FP32 Schur Krylov solver is:

```text
SchurBicgstabSolverF32
```

The optional FP32 Schur GMRES solver is:

```text
SchurGmresSolverF32
```

Relevant files:

```text
src/linear/schur_bicgstab_solver_f32.hpp
src/linear/schur_bicgstab_solver_f32.cu
src/linear/schur_gmres_solver_f32.hpp
src/linear/schur_gmres_solver_f32.cu
```

Both solvers:

```text
1. Builds rhs_s through the Schur operator.
2. Initializes dVm = 0.
3. Runs the selected Krylov iteration on the implicit Schur operator.
4. Applies optional right preconditioning through J22 block dense LU.
5. Refreshes the true Schur residual after the iteration loop.
6. Calls Schur recovery to write full FP64 dx.
```

GMRES uses restarted Arnoldi with the restart length from `--gmres-restart`.
The solver selection CLI is:

```text
--linear-solver bicgstab|gmres
```

Scalar reductions use cuBLAS with device pointer mode and custom CUDA kernels
for fused vector updates and norm reductions where implemented. The Krylov
vectors are FP32.

## Precision And Data Movement

The outer Newton state is FP64:

```text
voltage_re / voltage_im
mismatch
full dx
Jacobian values from assembly
```

The active Schur solve copies numeric values to FP32 device buffers:

```text
J11 FP64 values -> J11 FP32 values
J12 FP64 values -> J12 FP32 values
J21 FP64 values -> J21 FP32 values
J22 FP64 values -> J22 FP32 values
```

The Schur solve itself uses FP32 matrices and vectors:

```text
rhs_s
dVm
r, r_hat, p, v, s, t
J11 dense blocks
J22 dense preconditioner blocks
```

After the solve:

```text
dtheta FP32 -> dx FP64
dVm    FP32 -> dx FP64
```

The `exact-klu` J11 mode is an oracle path. It copies RHS and solution through
host memory and is not the active GPU production path.

## CLI Shape

The current block64 experiment command is:

```bash
./exp/20260415/block-ilu/build/block_ilu_probe \
  --case case_ACTIVSg2000 \
  --max-outer 20 \
  --inner-max-iter 300 \
  --linear-tol 1e-3 \
  --inner-precision fp32 \
  --j11-solver partition-dense-lu \
  --j11-reorder amd \
  --j11-partition metis \
  --j11-block-size 64 \
  --j11-dense-backend cublas \
  --line-search \
  --schur-preconditioner j22-block-dense-lu
```

Important options:

```text
--inner-precision fp32
  Uses FP32 Schur matrices, Krylov vectors, reductions, and dense block LU.

--j11-solver partition-dense-lu
  Uses graph-partitioned block dense LU for J11^{-1}.

--j11-reorder amd
  Applies AMD ordering before graph partitioning.

--j11-partition metis
  Uses METIS graph partitioning for dense blocks.

--j11-block-size N
  Sets the dense block size for both J11 and J22 block dense LU.

--j11-dense-backend cublas
  Uses cuBLAS batched dense LU for J11.

--schur-preconditioner j22-block-dense-lu
  Applies block dense LU(J22) as right preconditioner for Schur BiCGSTAB.
```

## Output Lines

The probe prints structured lines:

```text
BLOCK_ILU_PATTERN
  Case dimensions and CSR nonzero counts.

BLOCK_ILU_VALUES
  Frobenius norms of J11/J12/J21/J22 values.

BLOCK_ILU_FACTORIZATION
  Standalone J11/J22 factorization smoke status.

BLOCK_ILU_SOLVE
  Whole Newton solve summary.

BLOCK_ILU_OUTER
  One Newton iteration summary.

BLOCK_ILU_INNER
  One Schur BiCGSTAB solve summary inside a Newton iteration.

BLOCK_ILU_STEP
  Combined per-step line used by older scripts.

DIRECTION_COMPARE
  Optional first-step direction comparison when --compare-direction is used.
```

## Main Source Map

```text
src/tools/block_ilu_probe.cpp
  CLI, case loading, result printing, optional direction comparison.

src/solver/powerflow_block_ilu_solver.cu
  Newton loop, mismatch, Jacobian assembly call, factorization call,
  linear solve call, voltage update, line search.

src/assembly/reduced_jacobian_assembler.cu
  cuPF-layout reduced Jacobian value assembly.

src/linear/implicit_schur_operator_f32.cu
  FP32 implicit Schur operator, RHS build, Schur matvec, solution recovery,
  J22 Schur preconditioner dispatch.

src/linear/partitioned_dense_lu_j11_f32.cu
  AMD/COLAMD ordering, METIS/BFS partitioning, dense block map construction,
  cuBLAS/cuSolver dense factorization and solve.

src/linear/schur_bicgstab_solver_f32.cu
  FP32 Schur BiCGSTAB with optional right preconditioning.

src/linear/schur_gmres_solver_f32.cu
  FP32 restarted Schur GMRES with optional right preconditioning.

src/linear/csr_spmv.cu
  CSR SpMV kernels for FP64 and FP32 paths.

src/linear/cusparse_ilu0_block_f32.cu
  Optional FP32 cuSPARSE ILU0 block solve path.

src/linear/exact_klu_j11_f32.cpp
  CPU KLU J11 exact oracle path.
```
