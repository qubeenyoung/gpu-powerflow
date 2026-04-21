# Status

Created: 2026-04-15

## Done

- Created `exp/20260415/block-ilu`
- Fixed experiment definition:
  - full operator: cuPF reduced `J = [J11 J12; J21 J22]`
  - preconditioner: `blockdiag(ILU(J11), ILU(J22))`
  - initial solver: right-preconditioned BiCGSTAB
  - current solver: right-preconditioned BiCGSTAB
- Documented design and staged implementation plan
- Checked local CUDA headers for cuSPARSE ILU0 and SpSV API availability
- Implemented reduced cuPF Jacobian index and full/J11/J22 CSR patterns
- Implemented CUDA FP64 value assembly for full `J`, `J11`, and `J22`
- Verified `case_ACTIVSg200` block norms against `jacobian_analysis`
- Implemented cuSPARSE ILU0 block wrapper for `J11` and `J22`
- Implemented right-preconditioned BiCGSTAB and a CUDA Newton probe
- Added right-preconditioned restarted GMRES
- Parallelized independent `J11` and `J22` triangular solves with separate
  cuSPARSE streams
- Parallelized full-`J` SpMV row blocks with separate CUDA streams; one SpMV
  runs the P rows and Q rows independently, split at `n_pvpq`
- Switched GMRES dot/norm reductions to cuBLAS device pointer mode and batches
  the small Hessenberg column transfer once per Arnoldi step
- Added optional timing breakdown fields for preconditioner, SpMV, reduction,
  vector update, small solve, and true residual refresh
- Added separate outer Newton-loop and inner linear-solver trace lines:
  `BLOCK_ILU_OUTER` and `BLOCK_ILU_INNER`
- Added implicit Schur-reduced BiCGSTAB:
  `Sx = J22*x - J21*(J11^{-1}*(J12*x))`
- Smoke-ran `case_ACTIVSg200`

## Current: Schur J22 ILU0 Attempt

Implemented optional Schur-side `J22` ILU0 preconditioning for the FP32 implicit
Schur BiCGSTAB path.

CLI switch:

```bash
--schur-preconditioner j22-ilu0
```

Implementation detail that matters: `J22*x` SpMV and `ILU0(J22)` do not share
the same value buffer. cuSPARSE ILU0 overwrites CSR values in place, so the
factorization uses a separate FP32 `J22` value buffer.

First-outer smoke, `linear_tol=1e-3`, `METIS + AMD + cuBLAS block16`:

| case | J22 ILU0 result | inner iters | linear relres | linear solve ms |
|---|---:|---:|---:|---:|
| case_ACTIVSg200 | rho_breakdown | 71 | 1.112e+00 | 24.737 |
| case_ACTIVSg500 | rho_breakdown | 122 | 1.614e+00 | 38.922 |
| case_ACTIVSg2000 | rho_breakdown | 59 | 1.257e+00 | 23.442 |

Conclusion: factorization itself succeeds (`j22_zero_pivot=-1`), but raw
`ILU0(J22)` is a bad Schur preconditioner here. It destabilizes BiCGSTAB
instead of reducing the Schur iteration count.

## Current: Schur J22 Block Dense LU Attempt

Implemented optional Schur-side block diagonal dense LU for `J22`:

```bash
--schur-preconditioner j22-block-dense-lu
```

It reuses the existing FP32 partitioned dense LU machinery with `AMD + METIS`
and the same block size used by the J11 path. For this J22 Schur preconditioner
the dense backend is cuBLAS batched LU.

First-outer smoke, `linear_tol=1e-3`, `METIS + AMD + cuBLAS block16`:

| case | baseline inner | J22 dense inner | baseline linear ms | J22 dense linear ms | after mismatch |
|---|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | 61 | 9 | 18.678 | 8.975 | 7.140e-04 |
| case_ACTIVSg500 | 72 | 22 | 20.740 | 12.430 | 4.691e-02 |
| case_ACTIVSg2000 | 59 | 12 | 18.595 | 9.714 | 8.930e-01 |

`case_ACTIVSg2000`, 20 outer iterations:

```text
final_mismatch = 8.518e-02
total_inner_iterations = 292
total_linear_solve_sec = 77.628 ms
avg completed inner iteration = 0.233 ms
```

Conclusion: this is the first Schur-side preconditioner that actually reduces
the Krylov iteration count into the 10-ish range on the first 4k-sized linear
system. It improves linear solve time, but the outer Newton convergence is still
slow, so the next question is direction quality rather than raw inner speed.

### Block64 All-Case Newton Run

Command shape:

```bash
./exp/20260415/block-ilu/build/block_ilu_probe \
  --case <case> \
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

Result files:

```text
results/j11_j22_dense_b64_tol1e-3_line_search_all_cases.log
results/j11_j22_dense_b64_tol1e-3_line_search_summary.csv
```

Summary:

| case | failure | outer | final mismatch | total inner | linear ms |
|---|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | max_outer_iterations | 20 | 7.042e-01 | 852 | 840.025 |
| Base_Florida_42GW | max_outer_iterations | 20 | 3.376e-02 | 582 | 301.858 |
| Base_MIOHIN_76GW | max_outer_iterations | 20 | 8.108e-02 | 648 | 380.104 |
| Base_Texas_66GW | line_search_failed | 12 | 9.406e-01 | 271 | 159.343 |
| Base_West_Interconnect_121GW | line_search_failed | 11 | 4.557e-02 | 447 | 291.176 |
| MemphisCase2026_Mar7 | line_search_failed | 19 | 4.300e-05 | 323 | 159.273 |
| Texas7k_20220923 | max_outer_iterations | 20 | 3.143e-02 | 481 | 253.924 |
| case_ACTIVSg200 | max_outer_iterations | 20 | 4.872e-06 | 111 | 60.783 |
| case_ACTIVSg2000 | max_outer_iterations | 20 | 1.723e-02 | 240 | 119.833 |
| case_ACTIVSg25k | linear_rho_breakdown | 1 | 4.578e+01 | 96 | 66.484 |
| case_ACTIVSg500 | line_search_failed | 8 | 3.357e-03 | 107 | 59.707 |
| case_ACTIVSg70k | linear_rho_breakdown | 1 | 1.301e+02 | 71 | 66.438 |

## Previous BiCGSTAB Smoke Result

Command:

```bash
./exp/20260415/block-ilu/build/block_ilu_probe \
  --case case_ACTIVSg200 \
  --max-outer 10 \
  --inner-max-iter 500
```

Result:

```text
J11/J22 ILU0 factorization: zero pivot = -1 for both blocks
BiCGSTAB: did not converge in 500 inner iterations
linear_relative_residual: 1.545461888448e+01
dx not applied in strict mode
```

With `--continue-on-linear-failure`, the Newton mismatch increases:

```text
outer 1: 8.843803310369e-03 -> 1.511784717161e-02
outer 2: 1.511784717161e-02 -> 2.035240386825e-01
outer 3: 2.035240386825e-01 -> 5.548003002687e-01
```

So the first `blockdiag(ILU0(J11), ILU0(J22))` attempt is implemented, but the
first smoke case indicates that this preconditioner is not stable enough as-is.

## Other Cases

Command:

```bash
for case_dir in exp/20260414/amgx/cupf_dumps/*; do
  case_name=$(basename "$case_dir")
  [ "$case_name" = "case_ACTIVSg200" ] && continue
  ./exp/20260415/block-ilu/build/block_ilu_probe \
    --case "$case_name" \
    --max-outer 10 \
    --inner-max-iter 500
done
```

Strict mode result: every tested case failed in the first linear solve. ILU0
factorization itself succeeded for both `J11` and `J22` in every case
(`zero_pivot = -1`).

| case | failure | inner iters | linear relres |
|---|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | rho_breakdown | 92 | 3.844e+02 |
| Base_Florida_42GW | rho_breakdown | 93 | 4.685e+02 |
| Base_MIOHIN_76GW | rho_breakdown | 50 | 1.392e+01 |
| Base_Texas_66GW | rho_breakdown | 286 | 3.963e+00 |
| Base_West_Interconnect_121GW | rho_breakdown | 108 | 7.625e+02 |
| MemphisCase2026_Mar7 | alpha_breakdown | 74 | 3.383e+04 |
| Texas7k_20220923 | rho_breakdown | 74 | 1.397e+01 |
| case_ACTIVSg2000 | rho_breakdown | 129 | 6.054e+00 |
| case_ACTIVSg25k | rho_breakdown | 178 | 8.307e+01 |
| case_ACTIVSg500 | max_iterations | 500 | 1.568e+01 |
| case_ACTIVSg70k | rho_breakdown | 65 | 2.997e+02 |

Full log:

```text
results/all_cases_strict_inner500.log
```

## Previous BiCGSTAB Linear Iteration Timing

Instrumented `BicgstabSolver` to report the BiCGSTAB loop-body time per
completed inner iteration. The number excludes Jacobian assembly and ILU0
factorization. One BiCGSTAB iteration includes the two full-Jacobian SpMVs, two
block ILU triangular-solve applications, and the cuBLAS vector reductions and
updates inside the iteration.

Command shape:

```bash
./exp/20260415/block-ilu/build/block_ilu_probe \
  --case <case> \
  --max-outer 1 \
  --inner-max-iter 500
```

Result from one run over all dump cases:

| case | dim | inner iters | avg ms / BiCGSTAB iter | linear solve ms | failure |
|---|---:|---:|---:|---:|---|
| case_ACTIVSg200 | 361 | 500 | 0.326 | 163.305 | max_iterations |
| case_ACTIVSg500 | 943 | 422 | 0.366 | 154.355 | rho_breakdown |
| MemphisCase2026_Mar7 | 1797 | 58 | 0.440 | 25.577 | rho_breakdown |
| case_ACTIVSg2000 | 3607 | 92 | 0.523 | 48.174 | rho_breakdown |
| Texas7k_20220923 | 12843 | 182 | 0.678 | 123.475 | rho_breakdown |
| Base_Texas_66GW | 14431 | 120 | 0.861 | 103.358 | rho_breakdown |
| Base_Florida_42GW | 11210 | 500 | 0.860 | 430.112 | max_iterations |
| Base_MIOHIN_76GW | 20186 | 43 | 1.145 | 49.299 | rho_breakdown |
| Base_West_Interconnect_121GW | 40748 | 64 | 1.526 | 97.796 | rho_breakdown |
| case_ACTIVSg25k | 47246 | 500 | 0.594 | 297.126 | max_iterations |
| case_ACTIVSg70k | 134104 | 239 | 0.740 | 176.946 | rho_breakdown |
| Base_Eastern_Interconnect_515GW | 154916 | 47 | 1.059 | 49.931 | rho_breakdown |

## GMRES Optimization Pass

Current command shape:

```bash
./exp/20260415/block-ilu/build/block_ilu_probe \
  --case <case> \
  --max-outer 1 \
  --inner-max-iter <n> \
  --gmres-restart <m> \
  --residual-check-interval 5
```

For hot-loop timing without per-stage synchronization, add:

```bash
--no-timing-breakdown
```

Smoke and sanitizer checks:

```text
cmake --build exp/20260415/block-ilu/build -j: passed
compute-sanitizer memcheck, case_ACTIVSg200, inner=2, restart=2: 0 errors
```

Representative first-linear-system results:

| case | restart | inner iters | avg ms / GMRES iter | final linear relres | note |
|---|---:|---:|---:|---:|---|
| case_ACTIVSg200 | 30 | 100 | 0.230 | 7.136e-01 | timing breakdown on |
| case_ACTIVSg200 | 30 | 500 | 0.808 | 7.134e-01 | timing breakdown off |
| case_ACTIVSg200 | 10 | 500 | 0.487 | 8.597e-01 | faster, weaker |
| case_ACTIVSg500 | 30 | 500 | 0.821 | 6.085e-01 | timing breakdown off |
| case_ACTIVSg2000 | 30 | 500 | 0.839 | 7.583e-01 | timing breakdown off |
| case_ACTIVSg25k | 30 | 200 | 0.365 | 9.984e-01 | timing breakdown off |
| case_ACTIVSg70k | 30 | 200 | 0.500 | 9.999e-01 | timing breakdown off |
| Base_Eastern_Interconnect_515GW | 30 | 200 | 0.561 | 9.917e-01 | timing breakdown off |

Timing breakdown for `case_ACTIVSg200`, GMRES restart 30, inner 100:

```text
preconditioner: 4.663 ms
SpMV:           0.867 ms
reduction/MGS: 16.814 ms
vector update:  0.705 ms
true residual:  0.199 ms
```

Interpretation: GMRES removes BiCGSTAB breakdown and can reduce the residual
more smoothly on small cases, but the block-diagonal ILU preconditioner is still
too weak. Larger cases barely move from relative residual `~1.0` within 200
inner iterations.

## Previous GMRES Outer vs Inner Loop Trace

Output is now split by loop level:

- `BLOCK_ILU_OUTER`: one Newton step, including mismatch, Jacobian assembly,
  ILU factorization, voltage update, and after-update mismatch timing.
- `BLOCK_ILU_INNER`: one GMRES solve inside that Newton step, including inner
  iterations, restart cycles, SpMV/preconditioner counts, and linear residual.

Strict mode command shape:

```bash
./exp/20260415/block-ilu/build/block_ilu_probe \
  --case <case> \
  --max-outer 3 \
  --inner-max-iter 100 \
  --gmres-restart 30 \
  --residual-check-interval 5 \
  --no-timing-breakdown
```

Strict mode result: the first GMRES solve still hits the inner iteration limit,
so no Newton update is applied (`dx_was_applied=false`).

| case | outer reached | before mismatch | after mismatch | inner iters | avg ms / GMRES iter | linear relres | failure |
|---|---:|---:|---:|---:|---:|---:|---|
| case_ACTIVSg200 | 1 | 8.844e-03 | 8.844e-03 | 100 | 0.206 | 7.136e-01 | linear_max_iterations |
| case_ACTIVSg500 | 1 | 1.437e+00 | 1.437e+00 | 100 | 0.225 | 6.110e-01 | linear_max_iterations |
| case_ACTIVSg2000 | 1 | 2.030e+01 | 2.030e+01 | 100 | 0.286 | 7.765e-01 | linear_max_iterations |

With timing breakdown enabled for `case_ACTIVSg200`, outer 1 was:

```text
outer_iteration:          24.489 ms
  mismatch:                0.058 ms
  Jacobian assembly:        0.046 ms
  ILU factorization:        0.324 ms
  GMRES inner solve:       23.768 ms
  voltage update:           0.000 ms
  after mismatch:           0.000 ms
```

The inner GMRES breakdown for that same run was:

```text
inner iterations:         100
restart cycles:             4
SpMV calls:               104
preconditioner applies:   100
avg GMRES iter:         0.218 ms
linear relative residual: 7.136e-01

preconditioner apply:      4.517 ms
SpMV:                      0.832 ms
reduction/MGS:            15.897 ms
vector update:             0.658 ms
true residual refresh:     0.199 ms
```

Forced mode command shape:

```bash
./exp/20260415/block-ilu/build/block_ilu_probe \
  --case case_ACTIVSg200 \
  --max-outer 5 \
  --inner-max-iter 100 \
  --gmres-restart 30 \
  --residual-check-interval 5 \
  --continue-on-linear-failure \
  --no-timing-breakdown
```

Forced mode applies the failed GMRES update and is therefore only diagnostic.
It reaches five Newton steps, but every inner solve still fails:

| outer | before mismatch | after mismatch | inner iters | avg ms / GMRES iter | linear relres |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.844e-03 | 5.544e-03 | 100 | 0.207 | 7.136e-01 |
| 2 | 5.544e-03 | 5.553e-03 | 100 | 0.201 | 9.998e-01 |
| 3 | 5.553e-03 | 5.550e-03 | 100 | 0.201 | 1.000e+00 |
| 4 | 5.550e-03 | 5.550e-03 | 100 | 0.200 | 1.000e+00 |
| 5 | 5.550e-03 | 5.550e-03 | 100 | 0.201 | 1.000e+00 |

Interpretation: the outer Newton loop is currently blocked by the inner linear
solve quality. Separating the traces makes it clear that the nonlinear loop is
not the immediate bottleneck; the first GMRES solve fails before a strict Newton
step can apply `dx`.

### Previous GMRES 4k-Class Focus: `case_ACTIVSg2000`

There is no dump named 4k in the current dataset. The closest available
4k-class reduced linear system is `case_ACTIVSg2000`:

```text
n_bus=2000
n_pv=391
n_pq=1608
n_pvpq=1999
dim=3607
full_nnz=26345
j11_nnz=7331
j22_nnz=6040
```

Strict mode, restart 30, inner max 100, timing breakdown off:

```text
outer 1:
  before mismatch:        2.030443204550e+01
  after mismatch:         2.030443204550e+01
  dx applied:             false
  outer time:             31.626 ms
  GMRES solve time:       30.647 ms
  avg GMRES iteration:     0.287 ms
  GMRES inner iterations: 100
  linear relres:          7.764512723910e-01
```

Strict mode, restart 30, inner max 500, timing breakdown off:

```text
outer 1:
  before mismatch:        2.030443204550e+01
  after mismatch:         2.030443204550e+01
  dx applied:             false
  outer time:            147.533 ms
  GMRES solve time:      146.539 ms
  avg GMRES iteration:     0.287 ms
  GMRES inner iterations: 500
  linear relres:          7.582864156784e-01
```

Timing breakdown on, restart 30, inner max 100:

```text
outer iteration:          32.834 ms
  mismatch:                0.062 ms
  Jacobian assembly:        0.045 ms
  ILU factorization:        0.434 ms
  GMRES inner solve:       31.861 ms

GMRES inner:
  preconditioner apply:    10.794 ms
  SpMV:                     0.920 ms
  reduction/MGS:           17.533 ms
  vector update:            0.658 ms
  true residual refresh:    0.264 ms
```

Forced mode, restart 30, inner max 100, timing breakdown off:

| outer | before mismatch | after mismatch | avg ms / GMRES iter | linear relres |
|---:|---:|---:|---:|---:|
| 1 | 2.030e+01 | 1.567e+01 | 0.282 | 7.765e-01 |
| 2 | 1.567e+01 | 1.565e+01 | 0.277 | 9.943e-01 |
| 3 | 1.565e+01 | 1.543e+01 | 0.278 | 9.765e-01 |
| 4 | 1.543e+01 | 1.543e+01 | 0.277 | 9.939e-01 |
| 5 | 1.543e+01 | 1.526e+01 | 0.278 | 9.830e-01 |

Interpretation for the 4k-class case: increasing GMRES from 100 to 500 inner
iterations barely improves the first linear residual (`0.776 -> 0.758`). This
confirms that the current `blockdiag(ILU0(J11), ILU0(J22))` preconditioner is
not giving a usable linear solve, not merely that the outer Newton loop needs
more iterations.

## BiCGSTAB Rollback and SpMV Parallelization

GMRES remains in the source tree as a previous comparison path, but the Newton
solver now uses right-preconditioned BiCGSTAB again.

SpMV parallelization added in `CsrSpmv`:

- each full-`J` SpMV is split into top P rows and bottom Q rows
- the split point is `n_pvpq`
- the two row blocks run on separate nonblocking CUDA streams
- the two BiCGSTAB SpMVs inside one iteration are still sequential because
  `A * s_hat` depends on `s = r - alpha * A * p_hat`

4k-class case:

```text
case=case_ACTIVSg2000
dim=3607
full_nnz=26345
j11_nnz=7331
j22_nnz=6040
```

Strict mode, timing breakdown on, inner max 500:

```text
linear_solver:          bicgstab
failure:                linear_rho_breakdown
inner iterations:       145
linear relres:          5.997255212001e+00
linear solve time:      68.467 ms
avg BiCGSTAB iter:       0.472 ms
SpMV calls:             291
preconditioner applies: 290
reduction calls:        873

preconditioner apply:   30.796 ms
parallel row SpMV:       3.217 ms
reduction:              30.910 ms
vector update:           2.579 ms
residual refresh:        0.088 ms
```

Strict mode without timing breakdown is faster per iteration but still breaks
down. Repeated runs showed BiCGSTAB's expected sensitivity to tiny numeric
differences from the assembled/factorized Jacobian:

```text
inner iterations before rho breakdown: 85 to 155
avg BiCGSTAB iteration:                about 0.42 ms
linear relres at failure:              about 7.0 to 11.1
```

Forced mode remains diagnostic only. On `case_ACTIVSg2000`, applying failed
BiCGSTAB updates makes the nonlinear mismatch explode:

| outer | before mismatch | after mismatch | inner iters | linear relres |
|---:|---:|---:|---:|---:|
| 1 | 2.030e+01 | 1.456e+03 | 82 | 9.482e+00 |
| 2 | 1.456e+03 | 1.269e+06 | 500 | 6.554e+00 |
| 3 | 1.269e+06 | 1.586e+11 | 500 | 8.815e+00 |
| 4 | 1.586e+11 | 3.778e+21 | 500 | 1.340e+01 |
| 5 | 3.778e+21 | 4.467e+33 | 426 | 1.711e+00 |

Sanity check:

```text
compute-sanitizer memcheck, case_ACTIVSg2000, inner=2: 0 errors
```

## Implicit Schur Implementation

The current solver path is the Schur-reduced linear system:

```text
J11*dtheta + J12*dvm = b1
J21*dtheta + J22*dvm = b2

rhs_s = b2 - J21*(J11^{-1}*b1)
S*dvm = rhs_s
dtheta = J11^{-1}*(b1 - J12*dvm)

S*x = J22*x - J21*(J11^{-1}*(J12*x))
```

`J11^{-1}` is approximated by the cuSPARSE ILU0 factors of `J11`; no dense
inverse is formed.

Implementation changes:

- `ReducedJacobianPatterns` now builds `J12` and `J21` CSR patterns in addition
  to full `J`, `J11`, and `J22`.
- `ReducedJacobianAssembler` updates `J11`, `J12`, `J21`, and `J22` values
  inside the Newton loop.
- `CsrSpmv` supports rectangular matrices and async stream launches.
- `ImplicitSchurOperator` evaluates `S*x` without materializing `S`.
- `SchurBicgstabSolver` solves the reduced `Vm` system and recovers the full
  `dx = [dtheta; dvm]`.

Async Schur matvec layout:

```text
stream A:
  y0 = J22*x

stream B:
  tmp1 = J12*x
  tmp2 = J11^{-1}*tmp1
  tmp3 = J21*tmp2

default stream:
  wait(A, B)
  y = y0 - tmp3
```

4k-class smoke case:

```text
case=case_ACTIVSg2000
dim=3607
n_pvpq=1999
n_pq=1608
full_nnz=26345
j11_nnz=7331
j12_nnz=6487
j21_nnz=6487
j22_nnz=6040
```

Sanity check:

```text
compute-sanitizer memcheck, case_ACTIVSg2000, inner=2: 0 errors
```

Hot-loop timing, strict mode, inner max 10, timing breakdown off:

```text
linear_solver:          implicit_schur_bicgstab
linear solve time:       6.27 to 6.36 ms
avg Schur BiCGSTAB iter: 0.434 to 0.439 ms
inner iterations:       10
Schur linear relres:     2.986e-02
Schur matvec calls:     21
low-level SpMV calls:   65
J11 solve calls:        23
reduction calls:        62
```

Timing breakdown on, inner max 10:

```text
linear solve time:       7.302 ms
avg iteration:           0.524 ms

reduction:               4.189 ms
J11 triangular solves:   1.838 ms
J12/J21/J22 SpMVs:       0.524 ms
vector update/combine:   0.332 ms
residual refresh:        0.199 ms
rhs_s build:             0.123 ms
solution recover:        0.098 ms
Schur matvec section:    2.531 ms
```

### Device-Scalar/Fused Reduction Pass

Applied to the implicit Schur BiCGSTAB path:

- cuBLAS handle uses `CUBLAS_POINTER_MODE_DEVICE`
- `rho`, `alpha`, `beta`, and `omega` are kept in device scalar storage
- vector update kernels read device scalars directly
- fused kernels now combine:
  - `s = r - alpha*v` with `norm(s)`
  - `dot(t,t)` with `dot(t,s)`
  - `x/r` update with `norm(r)`

The logical reduction count is still the same, but the number of host scalar
synchronization points and separate vector/reduction kernels is lower.

4k-class `case_ACTIVSg2000`, strict mode, inner max 10, timing breakdown off:

```text
before: linear solve 6.27-6.36 ms, avg iter 0.434-0.439 ms
after:  linear solve 3.54-3.57 ms, avg iter 0.316-0.319 ms
Schur relres after 10 iters: 2.986e-02
```

Timing breakdown on, inner max 10:

```text
before:
  linear solve:      7.302 ms
  reduction:         4.189 ms
  J11 solves:        1.838 ms
  SpMV:              0.524 ms

after:
  linear solve:      4.510 ms
  reduction:         1.370 ms
  J11 solves:        1.836 ms
  SpMV:              0.527 ms
```

Convergence with inner max 300, timing breakdown off:

```text
outer 1:
  before optimization: linear solve 73-84 ms, avg iter ~0.386 ms
  after optimization:  linear solve 54 ms,    avg iter ~0.263 ms
  linear relres:       9.312e-07
```

Interpretation: after the reduction pass, the largest hot-loop cost is no
longer reduction; `J11` triangular solves are now the dominant measured kernel
bucket for the 10-iteration run.

### Linear Tolerance 1e-2

4k-class `case_ACTIVSg2000`, implicit Schur BiCGSTAB, timing breakdown off:

```text
linear_tolerance=1e-2
max_outer=10
max_inner=300
```

Result:

```text
outer iterations:          10
nonlinear converged:       false
final mismatch:            1.407500399184e+00
total inner iterations:    725
total linear solve time:   192.202 ms
avg linear iteration:      0.260 ms
failure:                   max_outer_iterations
```

Outer-level linear solve times:

| outer | inner iters | linear solve ms | linear relres | mismatch after |
|---:|---:|---:|---:|---:|
| 1 | 22 | 6.595 | 9.883e-03 | 1.434e+00 |
| 2 | 76 | 20.112 | 9.565e-03 | 1.414e+00 |
| 3 | 109 | 28.687 | 9.842e-03 | 1.413e+00 |
| 4 | 83 | 21.884 | 9.787e-03 | 1.413e+00 |
| 5 | 78 | 20.550 | 9.972e-03 | 1.412e+00 |
| 6 | 68 | 17.928 | 9.057e-03 | 1.411e+00 |
| 7 | 75 | 19.774 | 9.217e-03 | 1.410e+00 |
| 8 | 70 | 18.569 | 9.925e-03 | 1.409e+00 |
| 9 | 72 | 19.128 | 9.617e-03 | 1.408e+00 |
| 10 | 72 | 18.976 | 9.962e-03 | 1.408e+00 |

Timing breakdown on, outer 1 only:

```text
linear solve:             8.629 ms
J11 triangular solves:    3.882 ms
reduction:                2.267 ms
J12/J21/J22 SpMVs:        1.147 ms
vector update/combine:    0.421 ms
Schur matvec section:     5.400 ms
```

Interpretation: relaxing the linear tolerance from `1e-6` to `1e-2` cuts the
linear solve cost substantially, but it does not fix the nonlinear stall. The
final mismatch after 10 outer steps remains about `1.407`.

Convergence with inner max 300:

```text
outer 1:
  mismatch:      2.030443e+01 -> 1.427964e+00
  inner iters:   184 to 213 in observed runs
  linear relres: ~1e-6
  solve time:    73 to 84 ms

outer 10:
  final mismatch after 10 outer steps: 1.407486e+00
```

Interpretation: the implicit Schur linear system is much better conditioned
than the full-J block-ILU BiCGSTAB path. It reaches `~3e-2` Schur residual in
10 iterations and `~1e-6` in roughly `200-250` iterations on the 4k-class case.
The remaining issue is not linear breakdown anymore; the Newton update stalls
after the first large mismatch reduction.

### First-Step Direction Check

Added `block_ilu_probe --compare-direction --max-outer 1`.

The reference direction is not read from a dump file. The existing dump does
not store cuPF's first `dx`, so the probe reconstructs the same first Newton
linear system on the host and solves:

```text
J0 * dx_ref = -F0
```

with Eigen KLU. The current implicit-Schur BiCGSTAB direction is then compared
against `dx_ref` on the same initial voltage.

4k-class `case_ACTIVSg2000`, timing breakdown off:

| linear tol | inner iters | solve ms | cos(dx) | angle(dx) | rel dx diff | cos(theta) | cos(Vm) | full-J relres | mismatch after direct | mismatch after iter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-2` | 22 | 6.612 | 0.710 | 44.79 deg | 0.730 | -0.012 | 0.858 | 6.090e-2 | 2.692e-1 | 1.434e+0 |
| `1e-6` | 201 | 53.474 | 0.826 | 34.31 deg | 0.565 | 0.022 | 0.999 | 6.006e-2 | 2.692e-1 | 1.428e+0 |

Interpretation: tightening BiCGSTAB tolerance improves the `Vm` component, but
the full Newton direction is still far from the KLU direct direction. The
theta block is nearly orthogonal to the direct theta direction, and the full-J
linear residual remains about `6e-2` even when the Schur solver reports
`~1e-6`. This points at the approximate `J11^{-1}` inside the implicit Schur
operator, not just the BiCGSTAB stopping tolerance.

All other dump cases with `linear_tol=1e-6`, `max_outer=1`,
`max_inner=300`, timing breakdown off:

| case | dim | inner/ms | angle | theta cos | Vm cos | KLU after | iter after |
|---|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | 361 | 114 / 19.370 | 14.63 | 0.945 | 1.000 | 2.363e-6 | 7.405e-4 |
| case_ACTIVSg500 | 943 | 125 / 21.285 | 12.23 | 0.004 | 1.000 | 4.711e-2 | 5.883e-2 |
| MemphisCase2026_Mar7 | 1797 | 231 / 49.720 | 69.99 | 0.203 | 0.900 | 1.586e-7 | 1.068e-3 |
| Base_Florida_42GW | 11210 | 300 / 129.349 | 65.57 | 0.269 | 0.999 | 1.416e1 | 1.416e1 |
| Texas7k_20220923 | 12843 | 300 / 103.190 | 41.07 | 0.112 | 0.993 | 1.562e-2 | 1.458e-1 |
| Base_Texas_66GW | 14431 | 300 / 128.029 | 71.58 | 0.002 | 0.993 | 2.191e0 | 6.133e0 |
| Base_MIOHIN_76GW | 20186 | 300 / 168.282 | 82.01 | 0.177 | 0.999 | 8.411e0 | 8.406e0 |
| Base_West_Interconnect_121GW | 40748 | 300 / 227.333 | 85.55 | -0.042 | 0.986 | 4.927e-1 | 5.590e-1 |
| case_ACTIVSg25k | 47246 | 300 / 85.238 | 89.61 | -0.009 | 0.427 | 3.105e0 | 2.020e1 |
| case_ACTIVSg70k | 134104 | 300 / 108.130 | 90.36 | -0.013 | 0.166 | 1.384e1 | 2.564e1 |
| Base_Eastern_Interconnect_515GW | 154916 | 300 / 154.962 | 89.42 | 0.087 | 0.967 | 1.573e1 | 1.571e1 |

Across cases, `theta` is the unstable component. The `Vm` direction is often
close to KLU, but `theta` is usually low-cosine or nearly orthogonal. On the
large ACTIVS cases, `Vm` also degrades.

### Exact J11 Recover Check

Added `J11_EXACT_RECOVER_COMPARE`.

This is the first isolation test for `J11^{-1}`. It keeps the `dVm` produced by
the current implicit-Schur BiCGSTAB solve, but recomputes only:

```text
dtheta = J11^{-1} * (rhs_P - J12*dVm)
```

using an exact Eigen KLU solve on `J11`. This does not change the Schur
operator used inside BiCGSTAB; it only tests whether the final theta recovery
is the source of the direction error.

With `linear_tol=1e-6`, `max_outer=1`, `max_inner=300`:

| case | dim | angle ILU | angle exact J11 | theta cos ILU -> exact | mismatch ILU -> exact |
|---|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | 361 | 14.63 | 0.42 | 0.945 -> 1.000 | 7.405e-4 -> 8.666e-5 |
| case_ACTIVSg500 | 943 | 12.23 | 1.26 | 0.004 -> 1.000 | 5.883e-2 -> 5.227e-2 |
| MemphisCase2026_Mar7 | 1797 | 69.98 | 9.90 | 0.203 -> 0.998 | 1.068e-3 -> 6.525e-5 |
| case_ACTIVSg2000 | 3607 | 34.31 | 2.88 | 0.022 -> 0.999 | 1.428e0 -> 2.978e-1 |
| Base_Florida_42GW | 11210 | 65.57 | 1.19 | 0.270 -> 1.000 | 1.416e1 -> 1.420e1 |
| Texas7k_20220923 | 12843 | 41.07 | 6.45 | 0.112 -> 0.998 | 1.458e-1 -> 8.330e-2 |
| Base_Texas_66GW | 14431 | 71.58 | 2.48 | 0.000 -> 1.000 | 6.133e0 -> 2.176e0 |
| Base_MIOHIN_76GW | 20186 | 82.01 | 0.33 | 0.178 -> 1.000 | 8.406e0 -> 8.404e0 |
| Base_West_Interconnect_121GW | 40748 | 85.54 | 2.09 | -0.043 -> 0.999 | 5.591e-1 -> 5.549e-1 |
| case_ACTIVSg25k | 47246 | 88.51 | 4.19 | 0.006 -> 0.999 | 2.647e1 -> 1.053e1 |
| case_ACTIVSg70k | 134104 | 90.32 | 0.36 | -0.012 -> 1.000 | 5.962e0 -> 1.553e1 |
| Base_Eastern_Interconnect_515GW | 154916 | 89.41 | 0.17 | 0.093 -> 1.000 | 1.570e1 -> 1.570e1 |

Interpretation: exact `J11` recovery almost completely fixes the direction
error in `theta`. On the 4k case, the full direction angle improves from
`34.31 deg` to `2.88 deg`, and mismatch after the update improves from
`1.428e0` to `2.978e-1`, close to the full KLU direct update
`2.692e-1`. Therefore the dominant first-step direction error is the ILU0
approximation in the final `J11^{-1}` theta recovery. For cases where
BiCGSTAB hit `max_inner=300`, the remaining mismatch is also affected by the
quality of the `dVm` solve.

### J11 Reordering Check

Added `--j11-reorder none|amd|colamd|rcm`.

The implementation uses a symmetric permutation for `J11` only:

```text
J11_perm = P * J11 * P^T
rhs_perm = P * rhs
x_perm   ~= ILU0(J11_perm)^{-1} * rhs_perm
x        = P^T * x_perm
```

The same wrapper is used in Schur RHS construction, Schur matvecs, and final
theta recovery. `J12`, `J21`, and `J22` are not reordered.

`case_ACTIVSg2000`, `linear_tol=1e-6`, `max_outer=1`, `max_inner=300`,
timing breakdown off:

| J11 reorder | converged | inner/ms | Schur relres | angle | theta cos | Vm cos | full-J relres | mismatch after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | true | 209 / 55.410 | 8.75e-7 | 34.31 | 0.022 | 0.999 | 6.006e-2 | 1.428e0 |
| AMD | true | 188 / 45.669 | 9.50e-7 | 34.29 | 0.061 | 0.999 | 6.010e-2 | 1.428e0 |
| COLAMD | true | 195 / 46.721 | 9.85e-7 | 34.33 | -0.009 | 0.999 | 6.002e-2 | 1.428e0 |
| RCM | true | 210 / 79.414 | 4.81e-7 | 34.28 | 0.067 | 0.999 | 6.006e-2 | 1.428e0 |

AMD/COLAMD reduce the first-step inner solve time on this case, but they do
not fix the Newton direction. The `theta` component is still almost
orthogonal to the KLU direct direction, and the full-J residual remains around
`6e-2`. A short `compute-sanitizer` memcheck on `case_ACTIVSg200` with AMD
reported `ERROR SUMMARY: 0 errors`.

### AMD Graph Partition + Batched Dense LU

Added:

```text
--j11-solver partition-dense-lu --j11-reorder amd --j11-block-size N
```

Implementation:

```text
1. Compute AMD order for J11.
2. Build the J11 graph in AMD-permuted coordinates.
3. Seed BFS graph partitions in AMD order, capped by --j11-block-size.
4. Assemble each diagonal partition block as padded column-major dense.
5. Factor all same-size padded blocks with cublasDgetrfBatched.
6. For every J11 solve:
   rhs_old -> P rhs -> block RHS -> batched getrs -> P^T solution.
```

Only the diagonal partition blocks are kept; off-partition couplings are
ignored, so this is a graph-partition block Jacobi approximation to
`J11^{-1}`.

`case_ACTIVSg2000`, first Newton iteration, `linear_tol=1e-6`,
`max_inner=300`, timing breakdown off:

| J11 solver | block | inner/ms | angle | theta cos | Vm cos | full-J relres | mismatch after |
|---|---:|---:|---:|---:|---:|---:|---:|
| ILU0 + AMD | - | 188 / 45.669 | 34.29 | 0.061 | 0.999 | 6.010e-2 | 1.428e0 |
| dense LU + AMD graph | 16 | 199 / 42.911 | 33.00 | 0.265 | 0.999 | 4.153e-2 | 7.990e-1 |
| dense LU + AMD graph | 32 | 191 / 60.558 | 33.46 | 0.204 | 0.999 | 4.151e-2 | 7.682e-1 |
| dense LU + AMD graph | 64 | 192 / 111.670 | 32.08 | 0.337 | 0.999 | 3.504e-2 | 8.117e-1 |
| dense LU + AMD graph | 128 | 225 / 275.677 | 31.39 | 0.384 | 0.999 | 2.307e-2 | 6.056e-1 |
| dense LU + AMD graph | 256 | 203 / 447.058 | 32.36 | 0.310 | 0.999 | 2.515e-2 | 5.129e-1 |
| dense LU + AMD graph | 512 | 199 / 840.891 | 31.54 | 0.380 | 0.999 | 1.689e-2 | 2.754e-1 |

Timing breakdown examples:

| J11 solver | block | linear ms | J11 solve ms | SpMV ms | reduction ms | J11 factor ms |
|---|---:|---:|---:|---:|---:|---:|
| ILU0 + AMD | - | 55.092 | 26.578 | 8.466 | 12.688 | 0.244 |
| dense LU + AMD graph | 16 | 57.892 | 18.651 | 9.905 | 14.896 | 0.570 |
| dense LU + AMD graph | 512 | 887.623 | 818.786 | 11.592 | 16.047 | 41.411 |

Interpretation: graph-partition dense LU improves the first-step update quality
relative to ILU0, especially for larger blocks. At block 512, the first update
mismatch is close to the full KLU direct update (`2.754e-1` vs `2.692e-1`),
but the J11 solve cost is far too high. Small blocks are much cheaper and
already improve mismatch (`1.428e0 -> ~0.8e0`), but they do not repair the
theta direction enough.

Validation: `compute-sanitizer --tool memcheck` on `case_ACTIVSg200`,
AMD graph dense LU block 16, short inner run: `ERROR SUMMARY: 0 errors`.

### FP32 Inner Loop Check

Added:

```text
--inner-precision fp32
```

Current scope:

```text
outer Newton voltage/mismatch/Jacobian assembly: FP64
inner Schur values J11/J12/J21/J22:         FP32 copies
inner RHS and Krylov vectors:               FP32
inner reductions and dot products:          FP32
J12/J21/J22 SpMV:                           FP32
J11 partition dense LU:                     cublasSgetrfBatched/SgetrsBatched
returned Newton dx:                         converted back to FP64
```

This mode currently requires:

```text
--j11-solver partition-dense-lu
```

`case_ACTIVSg2000`, first Newton iteration, AMD graph dense LU:

| precision | block | converged | inner/ms | Schur relres | angle | theta cos | full-J relres | mismatch after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP64 inner | 16 | true | 218 / 46.672 | 9.58e-7 | 33.00 | 0.265 | 4.153e-2 | 7.990e-1 |
| FP32 inner | 16 | false | 246 / 35.912 | 1.32e-6 | 33.00 | 0.265 | 4.153e-2 | 7.990e-1 |
| FP64 inner | 128 | true | 200 / 238.945 | 9.66e-7 | 31.39 | 0.384 | 2.307e-2 | 6.056e-1 |
| FP32 inner | 128 | false | 227 / 161.806 | 1.32e-6 | 31.39 | 0.384 | 2.307e-2 | 6.056e-1 |
| FP64 inner | 512 | true | 201 / 859.599 | 9.90e-7 | 31.54 | 0.380 | 1.689e-2 | 2.754e-1 |
| FP32 inner | 512 | false | 231 / 473.821 | 1.67e-6 | 31.54 | 0.380 | 1.689e-2 | 2.754e-1 |

The FP32 run reports `linear_converged=false` here because the final refreshed
Schur residual lands slightly above the strict `1e-6` threshold. The actual
Newton direction and post-update mismatch are effectively the same as FP64 for
these cases.

Timing breakdown:

| precision | block | linear ms | J11 solve ms | SpMV ms | reduction ms | J11 factor ms |
|---|---:|---:|---:|---:|---:|---:|
| FP64 inner | 16 | 63.499 | 20.620 | 10.982 | 16.588 | 0.570 |
| FP32 inner | 16 | 55.151 | 10.986 | 11.364 | 16.774 | 0.565 |
| FP64 inner | 512 | 922.253 | 852.378 | 11.854 | 16.396 | 41.460 |
| FP32 inner | 512 | 479.331 | 395.730 | 12.269 | 16.907 | 15.481 |

Interpretation: FP32 inner roughly halves the J11 dense solve/factor cost on
large blocks without changing the first-step direction metrics. It does not
repair the theta direction by itself; it only makes the dense-block
approximation cheaper.

Tensor Core review:

- cuBLAS documents `cublas<t>getrfBatched()` and `cublas<t>getrsBatched()` as
  batched LU/solve APIs, but these APIs do not expose a compute type or math
  mode knob like GEMM does.
- cuBLAS documentation notes that `NVIDIA_TF32_OVERRIDE=0` disables Tensor Core
  acceleration for FP32 computations in NVIDIA libraries, but this applies to
  library operations that have Tensor Core-enabled kernels. The batched LU API
  itself does not provide an explicit Tensor Core path.
- Therefore the current `cublasSgetrfBatched/SgetrsBatched` dense-LU path
  should be treated as FP32 CUDA-core LU, not a guaranteed Tensor Core path.
  To use Tensor Cores, we would need a blocked/no-pivot or panel+GEMM LU
  implementation where the trailing matrix updates are expressed as TF32/HMMA
  GEMMs, or use mixed-precision iterative refinement around such an
  approximate block solve.

Validation: `compute-sanitizer --tool memcheck` on `case_ACTIVSg200`, FP32
inner, AMD graph dense LU block 16, short inner run:
`ERROR SUMMARY: 0 errors`.

### Tensor Core Dense LU Backend

Added:

```text
--j11-dense-backend cublas|tc
```

For the probe CLI, `--inner-precision fp32 --j11-solver partition-dense-lu`
now selects `tc` by default. Use `--j11-dense-backend cublas` to force the
old pivoted cuBLAS FP32 path for comparison.

Current TC backend scope:

```text
--inner-precision fp32
--j11-solver partition-dense-lu
--j11-reorder amd
```

Implementation:

```text
1. Assemble the same AMD graph-partitioned dense J11 blocks in FP32.
2. Factor each block with a custom blocked no-pivot LU.
3. Factor panel: custom CUDA kernel.
4. Solve U12 and L21 panel coupling: custom CUDA kernels.
5. Trailing update A22 -= L21*U12:
   cublasGemmStridedBatchedEx(..., CUBLAS_COMPUTE_32F_FAST_TF32,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP)
6. Apply solves with batched lower/upper triangular solves and no pivots.
```

This is Tensor Core-eligible for the trailing GEMM update. It is not the same
as pivoted `cublasSgetrfBatched`, and the triangular solve phase is not a
Tensor Core operation.

Validation:

```text
compute-sanitizer memcheck, case_ACTIVSg200, fp32 inner, TC block 16,
inner=2: ERROR SUMMARY: 0 errors
```

`case_ACTIVSg2000`, first Newton iteration, `linear_tol=1e-6`,
`max_inner=300`, timing breakdown off:

| backend | block | inner/ms | Schur relres | angle | theta cos | full-J relres | mismatch after |
|---|---:|---:|---:|---:|---:|---:|---:|
| cublas getrf | 16 | 246 / 35.996 | 1.32e-6 | 33.00 | 0.265 | 4.153e-2 | 7.990e-1 |
| TC no-pivot | 16 | 246 / 34.798 | 1.28e-6 | 33.00 | 0.265 | 4.153e-2 | 7.990e-1 |
| cublas getrf | 128 | 227 / 151.731 | 1.32e-6 | 31.39 | 0.384 | 2.307e-2 | 6.056e-1 |
| TC no-pivot | 128 | 246 / 116.902 | 1.12e-6 | 31.39 | 0.384 | 2.307e-2 | 6.056e-1 |
| cublas getrf | 512 | 231 / 461.731 | 1.67e-6 | 31.54 | 0.380 | 1.689e-2 | 2.754e-1 |
| TC no-pivot | 512 | 220 / 374.175 | 1.31e-6 | 31.54 | 0.380 | 1.689e-2 | 2.754e-1 |

Timing breakdown on:

| backend | block | linear ms | J11 solve ms | SpMV ms | reduction ms |
|---|---:|---:|---:|---:|---:|
| cublas getrf | 16 | 55.263 | 10.974 | 11.406 | 16.810 |
| TC no-pivot | 16 | 53.175 | 9.216 | 11.387 | 16.594 |
| cublas getrf | 128 | 170.391 | 89.411 | 11.957 | 16.291 |
| TC no-pivot | 128 | 136.409 | 85.923 | 12.967 | 17.613 |
| cublas getrf | 512 | 481.285 | 398.031 | 12.179 | 16.831 |
| TC no-pivot | 512 | 394.983 | 347.035 | 11.647 | 16.362 |

Interpretation: the TC/no-pivot backend works on the tested cases and reduces
the dense-block J11 bucket, especially for large blocks. It does not change
the first-step Newton direction because it is the same graph-partition block
Jacobi approximation to `J11^{-1}`. The overall path is still dominated by
hundreds of Schur BiCGSTAB iterations and repeated J11 block solves.

### METIS Partition and cuSolver Backend

Added:

```text
--j11-partition bfs|metis
--j11-dense-backend cusolver
```

Implementation details:

- `--j11-partition metis` builds the J11 graph in the selected reorder
  coordinate system, currently AMD in the main experiments.
- METIS is called only from analyze, so the graph partition is computed once.
  Newton iterations still refresh only the numeric J11 values.
- The requested block cap is enforced after METIS by splitting oversized
  partitions into deterministic chunks of at most `--j11-block-size`.
- CUDA 12.8 provides cuSolver dense `getrf`, but not dense
  `getrfBatched`. Therefore `--j11-dense-backend cusolver` runs
  `cusolverDnSgetrf` once per dense partition block and then reuses the
  existing batched cuBLAS `getrs` solve path.

Validation:

```text
cmake --build exp/20260415/block-ilu/build -j: passed
compute-sanitizer memcheck, case_ACTIVSg200, METIS+TC block16, inner=2:
  ERROR SUMMARY: 0 errors
compute-sanitizer memcheck, case_ACTIVSg200, METIS+cuSolver block16, inner=2:
  ERROR SUMMARY: 0 errors
```

`case_ACTIVSg2000`, first Newton linear solve, timing breakdown on:

| partition | backend | block | linear ms | factor ms | J11 solve ms | calls | us/call |
|---|---|---:|---:|---:|---:|---:|---:|
| BFS | TC no-pivot | 16 | 53.552 | 0.081 | 9.199 | 494 | 18.6 |
| METIS | TC no-pivot | 16 | 45.621 | 0.078 | 7.062 | 424 | 16.7 |
| BFS | TC no-pivot | 128 | 135.754 | 38.052 | 85.652 | 494 | 173.4 |
| METIS | TC no-pivot | 128 | 112.375 | 37.547 | 66.181 | 480 | 137.9 |
| BFS | cuSolver getrf | 16 | 54.929 | 14.858 | 10.907 | 495 | 22.0 |
| METIS | cuSolver getrf | 16 | 47.779 | 6.956 | 8.727 | 427 | 20.4 |
| BFS | cuSolver getrf | 128 | 179.897 | 45.224 | 95.923 | 492 | 195.0 |
| METIS | cuSolver getrf | 128 | 175.662 | 5.994 | 86.764 | 568 | 152.8 |

Interpretation for acceleration experiments: METIS improves the J11 apply
cost in these runs. For block 16, the J11 apply drops from about `18.6 us` to
`16.7 us` per call on the TC backend. The cuSolver backend is useful as a
pivoted factorization comparison point, but it is not a true batched getrf
path on this CUDA version and is not faster than TC for the repeated apply
cost.

## Next Step

- Add CSV output for summary and step trace
- Add an identity/Jacobi comparison mode to isolate whether the issue is
  GMRES mechanics or the block-ILU preconditioner quality
- Try reorder/pivot-boost variants for `J11` and `J22`
