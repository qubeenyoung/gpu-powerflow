# Newton-Krylov Experiment Plan

Date: 2026-04-14

## Goal

This experiment tests a Jacobian-free Newton-Krylov power-flow solver in C++.

The production Newton loop currently runs:

```text
mismatch -> jacobian -> linear_solve -> voltage_update
```

This experiment removes the explicit Jacobian stage and keeps only:

```text
mismatch -> matrix-free linear_solve -> voltage_update
```

The matrix-free linear solve computes Jacobian-vector products with finite
differences of the mismatch operator:

```text
J(V) * v ~= (F(V + eps * v) - F(V)) / eps
```

The Newton equation stays:

```text
J(V) * dx = -F(V)
```

## Solver Decision From `exp/20260413/iterative`

Relevant result:

| solver | solved / 67 at residual <= 1e-8 | solved / 67 at residual <= 1e-3 | note |
| --- | ---: | ---: | --- |
| BiCGSTAB + ILUT | 64 | 63 | best robustness and speed in the explicit-J snapshot harness |
| BiCGSTAB + ILU(0) | 54 | 57 | many more Krylov iterations |
| BiCGSTAB + ILU(1) | not in 1e-8 broad table | 56 | did not improve over ILU(0) |
| BiCGSTAB + Block-Jacobi b32 | 27 | 41 | too weak |
| BiCGSTAB + HYPRE BoomerAMG | not primary | 40 | poor broad replacement |
| FGMRES + HYPRE BoomerAMG | not primary | 35 | worse than BiCGSTAB + AMG and far below ILUT |

Decision:

1. Use `BiCGSTAB` as the first matrix-free Krylov core.
2. Do not use ILUT/ILU/AMG in the Jacobian-free path, because they require an
   assembled matrix and violate this experiment's constraint.
3. Start with no preconditioner. If it stalls or breaks down, add a
   matrix-free `GMRES(m)` fallback in the same harness rather than trying AMG.
4. Use `linear_tol = 1e-3` for the first end-to-end pass, because the
   20260413 loose-tolerance run is the closest prior signal for an inexact
   Newton solve. Then sweep `1e-2`, `1e-3`, `1e-4`.

## Linear Solver And Preconditioner Candidates

### Strict Jacobian-Free Candidates

These candidates are allowed in the first pass because they do not assemble the
Newton Jacobian and use only matrix-free `Jv` products inside `linear_solve`.

| priority | linear solver | preconditioner | why | risk |
| --- | --- | --- | --- | --- |
| P0 | `BiCGSTAB` | none | Non-symmetric matrix support, low memory, consistent with the 20260413 BiCGSTAB signal | May break down or oscillate without ILUT |
| P1 | restarted `GMRES(30)` | none | More robust residual control than BiCGSTAB for hard non-symmetric systems | More memory and orthogonalization cost per inner iteration |
| P1 | restarted `GMRES(50)` | none | Same as `GMRES(30)`, but gives the Krylov basis more room before restart | Higher memory and more dot products |
| P2 | `FGMRES(30)` | variable/inexact identity | Useful if we later add damping or nonlinear/inexact preconditioning | Not useful before a real variable preconditioner exists |

Default order:

```text
bicgstab_none -> gmres30_none -> gmres50_none
```

The `FGMRES` branch should wait until the preconditioner is genuinely variable
between inner iterations. The 20260413 HYPRE FGMRES result did not justify using
FGMRES as the first broad run by itself.

### Strict Preconditioner Candidates

These are still compatible with the first-pass rule that the main solve path
does not call a Jacobian builder or Jacobian op.

| priority | preconditioner | implementation idea | why | risk |
| --- | --- | --- | --- | --- |
| P0 | none | `z = r` | Establish the pure JFNK baseline | Might not converge on large cases |
| P1 | scalar diagonal scaling | use `1 / max(||F0||_inf, 1)` or row/global residual scaling only | Cheap sanity option; no extra operator | Weak, mostly conditioning hygiene |
| P2 | finite-difference diagonal | estimate selected diagonal entries with coordinate perturbations and mismatch calls | Still Jacobian-free in spirit; can stabilize BiCGSTAB | Costs O(dimF) mismatch calls per outer step if done fully |
| P2 | block finite-difference diagonal | estimate only bus-local 2-by-2 or 1-by-1 blocks for pq/pv slots | Cheaper and more PF-aware than full diagonal probing | More code and still approximate |

For the first implementation, only `none` should be built. Add finite-difference
diagonal only after the no-preconditioner run shows whether the failure mode is
inner Krylov stagnation rather than nonlinear step acceptance.

### Relaxed Preconditioner Candidates

These are useful if we decide that "Jacobian-free" means matrix-free Krylov for
`Jv`, while allowing an approximate/prebuilt matrix only as a preconditioner.
They should be placed behind an explicit CLI flag such as
`--allow-assembled-preconditioner`, because they weaken the strict experiment
claim.

| priority | preconditioner | source | why | risk |
| --- | --- | --- | --- | --- |
| R0 | stale ILUT | assemble explicit `J` once at `iter_000`, then reuse ILUT for all outer steps | Best prior solver signal: BiCGSTAB + ILUT solved 64/67 at `1e-8` | Violates strict no-Jacobian path; setup cost can be high |
| R1 | refresh ILUT every `k` outer steps | assemble explicit `J` every `k=3` or only on stagnation | Stronger nonlinear robustness than stale ILUT | Moves toward ordinary Newton with skipped Jacobians |
| R2 | ILU(0) / ILU(1) | same assembled approximate `J` path | Cheaper setup than ILUT | 20260413 results were much weaker than ILUT |
| R3 | fast-decoupled PF blocks `B'`, `B''` | build from `-imag(Ybus)` and solve angle/magnitude blocks separately | PF-structured and cheaper than full Jacobian | May be weak for high R/X or stressed cases |
| R4 | HYPRE BoomerAMG | use an approximate assembled block or normal-equation preconditioner | Easy comparison to 20260413 AMG experiments | Prior broad result was weak: 35-40/67 at `1e-3` |

Recommended relaxed experiment only after strict P0/P1:

```text
BiCGSTAB + stale ILUT(iter_000)
GMRES(30) + stale ILUT(iter_000)
BiCGSTAB + fast-decoupled B'/B''
```

If the relaxed ILUT variant converges and the strict variant does not, report it
as a preconditioned JFNK ablation rather than as the main Jacobian-free result.

### GPU Extension Recommendation

For the GPU path, prefer solvers whose hot loop is only:

```text
Jv callback + dot + axpy + norm + small host-side convergence check
```

This maps well to existing CUDA mismatch and voltage-update ops, and avoids
explicit sparse factorization.

Recommended GPU order:

| priority | solver/preconditioner | GPU fit | recommendation |
| --- | --- | --- | --- |
| G0 | `BiCGSTAB + none` | Best memory footprint; two matrix-free `Jv` calls per iteration; simple vector kernels/cuBLAS ops | First GPU port after CPU JFNK works |
| G1 | `GMRES(30) + none` | More stable than BiCGSTAB but needs basis storage and orthogonalization reductions | First fallback if BiCGSTAB breaks down |
| G2 | `GMRES(50) + none` | Same as `GMRES(30)`, more robust but heavier | Use only for hard cases or sensitivity run |
| G3 | `BiCGSTAB + analytic/FD diagonal scaling` | Lightweight if diagonal can be computed cheaply on device | Add only after no-preconditioner failure is characterized |
| G4 | `FGMRES(30) + variable preconditioner` | Technically suitable, but only useful with a real variable preconditioner | Defer |

Best first GPU target:

```text
cuda_jfnk_bicgstab_none_fp64_mismatch
```

Use `CudaMixedStorage` only after the FP64 version is numerically understood.
The mixed path is attractive because the existing fast profile uses FP64
mismatch with FP32 solve/update buffers, but matrix-free finite differences are
more sensitive to perturbation size than direct cuDSS solves. A good sequence is:

```text
CPU FP64 JFNK
CUDA FP64 JFNK
CUDA mixed JFNK
```

GPU `Jv` design:

```text
base device state: V, F0
scratch device state: V_scratch, F_scratch, dx_scratch

for Jv(v):
  dx_scratch = eps * v
  copy base V -> scratch V
  voltage_update(scratch)
  mismatch(scratch)
  out = (F_scratch - F0) / eps
```

Avoid copying scratch state through host memory. The first CUDA JFNK prototype
should add device-to-device state copy helpers or a dedicated scratch storage
rather than round-tripping `V`/`F` to CPU.

GPU preconditioner ranking:

| priority | preconditioner | GPU note | recommendation |
| --- | --- | --- | --- |
| G0 | none | No setup, no sparse solve, pure matrix-free | Build first |
| G1 | scalar/residual scaling | One or two reductions; cheap | Fine as a diagnostic, not a real preconditioner |
| G2 | diagonal from analytic local approximation | Can be one kernel over buses using `Ybus` diagonal and current `V` | Best lightweight preconditioner candidate if strict no-explicit-J is relaxed slightly |
| G3 | finite-difference diagonal | Too many mismatch calls if fully probed; only viable with coloring/batching | Avoid broad use initially |
| G4 | block-Jacobi 2x2 bus-local | GPU-friendly apply step, but setup needs either analytic blocks or many probes | Good second-stage research candidate |
| G5 | ILUT/ILU/AMG | Requires assembled matrix and sparse triangular/multigrid setup; prior CPU AMG was weak | Do not use in the first GPU JFNK path |

Relaxed GPU preconditioner candidate:

```text
BiCGSTAB/GMRES + analytic fast-decoupled B'/B''
```

This is more GPU-friendly than ILUT because it can be block-structured and based
on `Ybus`-derived sparse matrices, but it should be labeled as relaxed
preconditioning rather than pure Jacobian-free solving.

Do not prioritize GPU ILUT for this experiment. The 20260413 explicit-J results
make ILUT attractive algorithmically, but GPU ILUT would pull the experiment
toward assembled sparse factorization again and away from the stated
`mismatch -> linear_solve -> voltage_update` goal.

### Selected Solver/Preconditioner Set

Use `BiCGSTAB + ILUT` as the explicit-J baseline. It is not the strict JFNK
path, but it is the reference point that every relaxed preconditioned variant
must beat or explain.

Candidate grid:

```text
solvers:         BiCGSTAB, FGMRES
preconditioners: Block-Jacobi, ILU(0), AMG
baseline:        BiCGSTAB + ILUT
```

Recommended shortlist:

| priority | combination | role | reason |
| --- | --- | --- | --- |
| S0 | `BiCGSTAB + ILUT` | baseline | Best 20260413 explicit-J result: 64/67 at `1e-8`, 63/67 at `1e-3` |
| S1 | `BiCGSTAB + ILU(0)` | first candidate | Best non-ILUT candidate already measured: 54/67 at `1e-8`, 57/67 at `1e-3`; cheaper setup than ILUT |
| S2 | `FGMRES + ILU(0)` | stability check | Same cheaper preconditioner as S1, but GMRES-style residual control may avoid BiCGSTAB breakdowns |
| S3 | `BiCGSTAB + AMG` | AMG reference | Already measured at 40/67 for `1e-3`; keep as a negative/contrast baseline for AMG |
| S4 | `FGMRES + AMG` | AMG fallback check | Already measured at 35/67 for `1e-3`; lower priority than BiCGSTAB+AMG, but useful to close the AMG question |
| S5 | `GMRES(30) + bus-local Block-Jacobi FD` | JFNK/GPU-structured candidate | New 20260414 JFNK smoke result: 5/8 on benchmark dumps, best strict matrix-free preconditioned path so far |

Deprioritize:

| combination | reason |
| --- | --- |
| `BiCGSTAB + contiguous Block-Jacobi b32` | 20260413 contiguous block-Jacobi b32 solved only 27/67 at `1e-8`, 41/67 at `1e-3` |
| `FGMRES + contiguous Block-Jacobi b32` | Worth trying only after block definition becomes PF-structured; contiguous blocks are too weak |

If we only run three non-baseline combinations, run:

```text
BiCGSTAB + ILU(0)
FGMRES + ILU(0)
BiCGSTAB + AMG
```

If GPU expansion is the deciding factor, add a second-stage PF-structured
`Block-Jacobi 2x2` experiment, not the old contiguous block-size-32 variant.

Implementation note:

```text
--preconditioner bus_block_jacobi_fd
```

uses a finite-difference bus-local block setup:

- PV bus: 1x1 block for `P/theta`
- PQ bus: 2x2 block for `[P, Q] x [theta, Vm]`

It does not call an explicit Jacobian op. It samples local columns with the
same matrix-free `Jv` callback used by the Krylov solver. This is useful for
correctness and convergence screening, but the GPU-oriented version should use
analytic local blocks to avoid O(dimF) setup `Jv` calls per outer iteration.

## Initial Scope

Implement only a CPU FP64 harness first:

```text
CpuFp64Storage
CpuMismatchOpF64
JfnkLinearSolveBiCGSTAB
JfnkLinearSolve GMRES fallback
Bus-local block-Jacobi FD preconditioner
CpuVoltageUpdateF64
```

Do not instantiate or call:

```text
CpuJacobianOpF64
CudaJacobianOp*
JacobianBuilder in the main JFNK solve path
cuDSS direct solve
```

The main JFNK path should pass empty `JacobianMaps` and an empty
`JacobianStructure` into `CpuFp64Storage::prepare()`. That storage path already
derives `dimF` from `pv/pq`, and skips Jacobian matrix allocation when the
structure is empty. Explicit dumped `J` values are allowed only in validation
gates, never inside the matrix-free linear solve.

CUDA is a phase-2 target after the CPU algorithm is stable, because JFNK matvecs
need repeated scratch-state perturbation and mismatch evaluation. CPU storage is
much easier to clone safely for the first pass.

## Planned Directory Layout

```text
/workspace/exp/20260414/newton-krylov
  PLAN.md
  CMakeLists.txt
  cpp/
    CMakeLists.txt
    newton_krylov_probe.cpp
    jfnk_bicgstab.hpp
    jfnk_bicgstab.cpp
  results/
```

## Core Loop

The probe should mirror `dump_linear_systems.cpp`, but with the Jacobian stage
removed:

```text
prepare storage
upload case data

for outer_iter in 0..max_iter:
  mismatch.run(ctx)
  if converged: break

  linear_solve.run(ctx)
    - stores base F0 = storage.F
    - solves A(dx) = -F0 with BiCGSTAB
    - A(v) calls matrix-free Jv
    - writes dx into storage.dx

  voltage_update.run(ctx)
```

The CSV output should record both Newton and inner Krylov behavior:

```text
case,solver,converged,outer_iterations,final_mismatch,
linear_tol,fd_eps_rule,total_jv_calls,total_inner_iterations,
max_inner_iterations,linear_failures,total_sec,mismatch_sec,
linear_solve_sec,voltage_update_sec,jv_mismatch_sec,jv_update_sec
```

## Matrix-Free Jv Design

Use a scratch CPU storage object for each linear solve. Do not perturb the main
storage in place.

For one Jv evaluation:

```text
scratch = current nonlinear state
scratch.dx = eps * v
scratch_voltage_update.run(scratch_ctx)
scratch_mismatch.run(scratch_ctx)
jv = (scratch.F - base_F) / eps
```

This keeps the public stage vocabulary to mismatch, linear solve, and voltage
update. The finite-difference perturbation uses `CpuVoltageUpdateF64`, so the
state-vector packing matches the production Newton correction:

```text
dx[0:n_pv]              -> theta[pv]
dx[n_pv:n_pv+n_pq]      -> theta[pq]
dx[n_pv+n_pq:dimF]      -> Vm[pq]
```

Initial epsilon rule:

```text
eps = sqrt(machine_epsilon) * (1 + ||x_state||_inf) / max(||v||_inf, tiny)
```

Sweep these fixed values for sensitivity:

```text
1e-5, 1e-6, 1e-7, 1e-8
```

## Validation Gates

### Gate 1: Jv sanity on one Newton point

Use small and medium cases where an explicit-J snapshot already exists:

```text
/workspace/exp/20260413/iterative/pf_dumps/<case>/cuda_mixed_edge/iter_000/J.csr
```

For `iter_000`, the case's `dump_V.txt` is the matching state. Compare:

```text
matrix_free_Jv = (F(V + eps*v) - F(V)) / eps
explicit_Jv    = J_dump * v
```

Metrics:

```text
relative_inf_error = ||matrix_free_Jv - explicit_Jv||_inf / max(||explicit_Jv||_inf, tiny)
```

This is a diagnostic only; the dumped `J` may be mixed FP32-origin for
`cuda_mixed_edge`, while the first JFNK harness is CPU FP64.

### Gate 2: One-step linear solve sanity

At `iter_000`, solve:

```text
JFNK(dx) = -F(V0)
```

Compare against the dumped direct solution when available:

```text
x_delta_direct_inf = ||dx_jfnk - x_direct||_inf
```

Also record the true nonlinear residual after one update:

```text
||F(V0 + dx_jfnk)||_inf
```

### Gate 3: End-to-end Newton convergence

Start with the benchmark dump set:

```text
/workspace/datasets/pglib-opf/cuPF_benchmark_dumps
case30_ieee
case118_ieee
case793_goc
case1354_pegase
case2746wop_k
case4601_goc
case8387_pegase
case9241_pegase
```

Then extend to the 67 PF converted cases already generated for the iterative
experiment:

```text
/workspace/exp/20260413/iterative/pf_cupf_dumps
```

Success criterion:

```text
final_mismatch <= 1e-8
no NaN/Inf
outer_iterations <= 50
```

## Experiment Matrix

First pass:

| label | Krylov | preconditioner | linear tol | fd eps |
| --- | --- | --- | --- | --- |
| `bicgstab_none_tol1e-3_autoeps` | BiCGSTAB | none | `1e-3` | auto |
| `bicgstab_none_tol1e-4_autoeps` | BiCGSTAB | none | `1e-4` | auto |
| `bicgstab_none_tol1e-3_eps1e-6` | BiCGSTAB | none | `1e-3` | `1e-6` |

Fallback pass only if BiCGSTAB stalls/breaks down:

| label | Krylov | preconditioner | linear tol | restart |
| --- | --- | --- | --- | --- |
| `gmres30_none_tol1e-3_autoeps` | GMRES | none | `1e-3` | 30 |
| `gmres50_none_tol1e-3_autoeps` | GMRES | none | `1e-3` | 50 |

## Baselines To Report

Use existing artifacts as context:

```text
/workspace/exp/20260413/iterative/ITERATIVE_SOLVER_EXPERIMENT_RESULTS.md
/workspace/exp/20260413/iterative/ITERATIVE_SOLVER_TOL1E3_RESULTS.md
/workspace/exp/20260413/iterative/CPU_AMG_EXPERIMENT_RESULTS.md
/workspace/exp/20260413/iterative/CPU_AMG_FGMRES_EXPERIMENT_RESULTS.md
```

The key comparison is not raw speed against cuDSS yet. The first question is:

```text
Can a Jacobian-free loop converge with only mismatch, linear solve, and voltage update?
```

Report:

- convergence count
- final mismatch distribution
- outer Newton iterations
- total Jv calls
- inner Krylov iterations per outer step
- failure reason: breakdown, max inner iter, max outer iter, non-finite mismatch

## Expected Risks

- Without ILUT, BiCGSTAB may require too many Jv calls or hit breakdown.
- Finite-difference `eps` can dominate accuracy. The epsilon sweep is mandatory.
- Full Newton steps may be too aggressive. If convergence oscillates, add a
  damping/line-search pass that still uses only mismatch and voltage update on
  scratch states.
- CPU scratch-copy Jv is intentionally simple but expensive. Performance claims
  should wait until the CUDA/scratch-state design is settled.

## Build And Run Sketch

```bash
cmake -S /workspace/exp/20260414/newton-krylov \
  -B /workspace/exp/20260414/newton-krylov/build \
  -GNinja

cmake --build /workspace/exp/20260414/newton-krylov/build \
  --target newton_krylov_probe -j

/workspace/exp/20260414/newton-krylov/build/newton_krylov_probe \
  --dataset-root /workspace/datasets/pglib-opf/cuPF_benchmark_dumps \
  --cases case30_ieee case118_ieee case793_goc case1354_pegase case2746wop_k case4601_goc case8387_pegase case9241_pegase \
  --solver bicgstab_none \
  --linear-tol 1e-3 \
  --fd-eps auto \
  --tolerance 1e-8 \
  --max-iter 50 \
  --output-csv /workspace/exp/20260414/newton-krylov/results/bicgstab_none_tol1e-3_autoeps.csv
```
