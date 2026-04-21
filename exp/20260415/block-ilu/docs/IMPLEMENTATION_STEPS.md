# Implementation Steps

## Step 1: Case Loader And Reduced Index

Reuse `cuPF/tests/dump_case_loader.cpp`.

Build a small reduced-index model:

- `pvpq = pv || pq`
- `theta_slot[bus] = pvpq position or -1`
- `vm_slot[bus] = n_pvpq + pq position or -1`
- `dimF = n_pvpq + n_pq`

Validation target:

- `case_ACTIVSg200`: `n_pvpq = 199`, `n_pq = 162`, `dimF = 361`

## Step 2: Full cuPF Jacobian Pattern And Value Assembly

Implement the same reduced four-block pattern as cuPF's `JacobianBuilder`.

The full matrix is:

```text
J = [J11 J12; J21 J22]
```

Build maps:

- `mapJ11`, `mapJ12`, `mapJ21`, `mapJ22`: one entry per Ybus nonzero
- `diagJ11`, `diagJ12`, `diagJ21`, `diagJ22`: one entry per bus

The CUDA value kernel mirrors `cuPF/cpp/src/newton_solver/ops/jacobian/cuda_edge_fp64.cu`.

Validation target:

- compare block norms against `exp/20260415/jacobian_analysis/results/block_norms.csv`
- use CPU debug download for the first case

## Step 3: Diagonal Block Patterns

Construct separate CSR matrices:

```text
J11: n_pvpq x n_pvpq
J22: n_pq   x n_pq
```

Build separate maps:

- `mapB11`, `diagB11` for J11
- `mapB22`, `diagB22` for J22

These block matrices should have the same numeric values as the corresponding
sub-blocks inside full `J`, but their column indices are local to the block.

Validation target:

- `||J11_full_subblock - J11_block||_max = 0`
- `||J22_full_subblock - J22_block||_max = 0`

## Step 4: cuSPARSE ILU0 Wrapper For One CSR Block

Create a readable RAII wrapper:

```text
CusparseIlu0Block
  analyze(row_ptr, col_idx)
  factorize(values)
  solve(rhs, out)
```

For the first pass use:

- `cusparseDcsrilu02_analysis`
- `cusparseDcsrilu02`
- `cusparseSpSV_analysis`
- `cusparseSpSV_solve`

The wrapper owns:

- matrix descriptor for ILU0
- `csrilu02Info_t`
- ILU scratch buffer
- `cusparseSpMatDescr_t` for the factor matrix
- dense vector descriptors for temporary RHS/output
- lower and upper `cusparseSpSVDescr_t`
- SpSV scratch buffers

Validation target:

- solve `J11 z = r` and `J22 z = r` smoke vectors after factorization
- fail explicitly on zero pivot

## Step 5: Block-Diagonal ILU Preconditioner

Create:

```text
BlockIluPreconditioner
  setup_patterns(J11_pattern, J22_pattern)
  factorize(J11_values, J22_values)
  apply(r, z)
```

`apply` splits the input vector:

```text
r_p = r[0:n_pvpq]
r_q = r[n_pvpq:dimF]
```

and writes:

```text
z[0:n_pvpq]      = ILU(J11)^-1 r_p
z[n_pvpq:dimF]  = ILU(J22)^-1 r_q
```

## Step 6: Full-J BiCGSTAB

Implement right-preconditioned BiCGSTAB:

```text
v = J * (M^-1 p)
s = r - alpha v
t = J * (M^-1 s)
x += alpha M^-1 p + omega M^-1 s
```

Use full `J` CSR SpMV for every operator application.

Track:

- inner iterations
- full-J SpMV calls
- preconditioner applies
- final true residual `||b - Jx|| / ||b||`
- breakdown reason

## Step 7: Newton Probe

Create `block_ilu_probe` CLI:

```bash
./block_ilu_probe \
  --case case_ACTIVSg200 \
  --max-outer 10 \
  --inner-max-iter 500 \
  --linear-tol 1e-6 \
  --nonlinear-tol 1e-8
```

Output one summary line and optional CSV:

- case
- converged
- outer iterations
- final mismatch
- total inner iterations
- total SpMV calls
- total preconditioner applies
- ILU zero-pivot status for J11/J22

## Step 8: First Smoke Criteria

The first implementation is acceptable when:

- `case_ACTIVSg200` builds and runs
- full Jacobian block norms match `jacobian_analysis`
- ILU factorization succeeds for both `J11` and `J22`
- BiCGSTAB produces a finite `dx`
- the Newton mismatch decreases after the first update

Full convergence on every case is not required for the first smoke.

## Step 9: Schur-Side J22 ILU0 Preconditioner

Add an optional Schur preconditioner for the implicit Schur BiCGSTAB path:

```text
M ~= J22
z = ILU0(J22)^-1 r
```

Use right-preconditioned BiCGSTAB on the Schur operator:

```text
v = S * (M^-1 p)
s = r - alpha v
t = S * (M^-1 s)
dVm += alpha M^-1 p + omega M^-1 s
```

Important implementation rule:

- keep one FP32 value buffer for `J22*x` SpMV
- keep a separate FP32 value buffer for `ILU0(J22)` because cuSPARSE ILU0
  overwrites CSR values in place

CLI:

```bash
./exp/20260415/block-ilu/build/block_ilu_probe \
  --case case_ACTIVSg2000 \
  --max-outer 1 \
  --inner-max-iter 300 \
  --linear-tol 1e-3 \
  --inner-precision fp32 \
  --j11-solver partition-dense-lu \
  --j11-reorder amd \
  --j11-partition metis \
  --j11-block-size 16 \
  --j11-dense-backend cublas \
  --line-search \
  --schur-preconditioner j22-ilu0
```

Current smoke result: factorization succeeds, but BiCGSTAB breaks down on the
tested ACTIVS cases. This says raw `J22` ILU0 is not a usable Schur
preconditioner as-is.

## Step 10: Schur-Side J22 Block Dense LU Preconditioner

Add an optional Schur preconditioner:

```text
M ~= blockdiag(J22 blocks)
z = M^-1 r
```

Use the same right-preconditioned BiCGSTAB placement as Step 9:

```text
v = S * (M^-1 p)
s = r - alpha v
t = S * (M^-1 s)
dVm += alpha M^-1 p + omega M^-1 s
```

CLI:

```bash
--schur-preconditioner j22-block-dense-lu
```

The implementation reuses the existing FP32 partitioned dense LU code with
AMD reordering, METIS partitioning, and cuBLAS batched LU.

Validation target:

- `j22_zero_pivot = -1`
- no BiCGSTAB breakdown on `case_ACTIVSg200`, `case_ACTIVSg500`,
  `case_ACTIVSg2000`
- first Newton linear solve reaches `linear_tol=1e-3`
- compare first-outer inner iterations and solve time against no Schur
  preconditioner

Current result: block16 reduces first-outer iterations from `59` to `12` on
`case_ACTIVSg2000`, and first linear solve time from roughly `18.6 ms` to
`9.7 ms`.
