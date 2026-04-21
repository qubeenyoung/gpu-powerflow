# 2026-04-15 Block-ILU Experiment

First implementation attempt for a cuSPARSE ILU preconditioner on the diagonal
blocks of the cuPF Newton Jacobian.

The full Krylov operator remains the cuPF reduced Jacobian:

```text
J = [ J11  J12 ] = [ dP/dtheta  dP/dVm ]
    [ J21  J22 ]   [ dQ/dtheta  dQ/dVm ]
```

The first preconditioner was:

```text
M = [ ILU(J11)     0   ]
    [    0      ILU(J22)]
```

The Krylov solver still multiplies by the full `J`, including `J12` and `J21`,
but each preconditioner application solves the two independent triangular
systems coming from `J11` and `J22`.

This is not the earlier bus-local `2 x 2` block Jacobi experiment.

The current solver path is an implicit Schur-reduced solve:

```text
S x = J22*x - J21*(J11^{-1}*(J12*x))
```

`J11^{-1}` is applied by the cuSPARSE ILU0 triangular solves for `J11`; the
inverse is not formed explicitly.

## Scope

The first pass targets:

- FP64 CUDA path only
- cuPF reduced Jacobian layout: `dimF = n_pvpq + n_pq`
- full `J` CSR values updated inside the Newton loop
- `J11`, `J12`, `J21`, and `J22` CSR values updated inside the same loop
- cuSPARSE ILU0 on `J11`
- implicit Schur BiCGSTAB on the `Vm` block
- async Schur matvec streams:
  `J22*x` runs in parallel with `J12*x -> J11^{-1} -> J21`
- device-scalar BiCGSTAB updates and fused reduction/update kernels
- one case at a time from `exp/20260414/amgx/cupf_dumps`

## Implementation Documents

- [docs/CURRENT_IMPLEMENTATION.md](docs/CURRENT_IMPLEMENTATION.md)
- [docs/IMPLEMENTATION_STEPS.md](docs/IMPLEMENTATION_STEPS.md)
- [docs/DESIGN.md](docs/DESIGN.md)
- [docs/API_NOTES.md](docs/API_NOTES.md)
- [docs/TODO.md](docs/TODO.md)

## Current Status

The first CUDA implementation is in place:

- cuPF reduced Jacobian pattern/value assembly
- separate `J11`, `J12`, `J21`, and `J22` device CSR values updated in the
  Newton loop
- cuSPARSE ILU0 on `J11`
- implicit Schur BiCGSTAB on `S`
- optional implicit Schur GMRES via `--linear-solver gmres`
- cuBLAS device pointer mode plus fused BiCGSTAB reduction/update kernels
- CUDA mismatch and voltage-update loop
- separated Newton outer-loop and linear inner-loop reporting
- first-step direction comparison against an Eigen KLU direct solve via
  `block_ilu_probe --compare-direction --max-outer 1`
- optional `J11`-only ILU0 reordering via
  `--j11-reorder none|amd|colamd|rcm`; all modes use `P J11 P^T`,
  `P rhs`, then `P^T x` recovery
- optional AMD-seeded graph partitioned dense block-LU for `J11` via
  `--j11-solver partition-dense-lu --j11-reorder amd --j11-block-size N`
- optional FP32 inner loop mode via
  `--inner-precision fp32`; outer Newton state/Jacobian assembly stays FP64,
  while Schur matrix values, Krylov vectors, reductions, SpMVs, and dense
  block LU use FP32. This mode currently requires partition dense LU for `J11`.
- optional Tensor Core-eligible no-pivot FP32 dense-LU backend via
  `--j11-dense-backend tc`; the panel LU is custom CUDA and the trailing
  block updates use TF32 `cublasGemmStridedBatchedEx`. This applies only to
  `--inner-precision fp32 --j11-solver partition-dense-lu`, and is the default
  backend for that CLI mode unless `--j11-dense-backend cublas` is specified.
- optional METIS graph partitioning via `--j11-partition metis`. The partition
  is built once in analyze, then reused while only matrix values are refreshed
  in later Newton iterations.
- optional cuSolver FP32 dense-LU factorization backend via
  `--j11-dense-backend cusolver`. CUDA 12.8 exposes cuSolver dense `getrf`,
  not dense `getrfBatched`, so this backend runs `cusolverDnSgetrf` per dense
  partition block and reuses the existing batched cuBLAS triangular solve.

The Schur linear solve now converges on the 4k-class smoke case, but the Newton
outer loop stalls after the first large mismatch reduction. See
[STATUS.md](STATUS.md) for timing and next debugging steps.
