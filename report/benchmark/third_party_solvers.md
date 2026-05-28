# Third-Party Solver Report

This report records the benchmark runner settings for installed third-party
direct sparse solvers and summarizes the latest full benchmark output.

## Output Contract

Benchmark CSV output:

```text
report/benchmark/third_party_solvers.csv
```

CSV columns:

```text
matrix,group,solver,rows,cols,nnz,density,analysis_ms,factor_ms,solve_ms,berr,absolute_error,success,message
```

The timed phases are solver phases only:

| Column | Meaning |
|---|---|
| `analysis_ms` | Ordering, symbolic analysis, or solver reordering phase |
| `factor_ms` | Numeric factorization phase |
| `solve_ms` | Triangular solve or direct solve phase |

Quality metrics:

| Column | Meaning |
|---|---|
| `berr` | Componentwise backward error, `max_i |(A*x-rhs)_i| / (|A|*|x| + |rhs|)_i` |
| `absolute_error` | `||x - x_true||_2` |

## Solver Settings

| Solver | Runtime name | Main settings |
|---|---|---|
| SuiteSparse KLU | `klu` | CSC input, `klu_defaults`, `klu_analyze`, `klu_factor`, `klu_solve`. |
| SuiteSparse UMFPACK | `umfpack` | CSC input, `umfpack_di_defaults`, symbolic/numeric/solve API split. |
| SuperLU | `superlu` | CSC input, `DOFACT`, default column permutation from SuperLU, explicit `get_perm_c + sp_preorder`, `dgstrf`, `dgstrs`. |
| oneMKL PARDISO | `pardiso` | CSR input, real unsymmetric `mtype=11`, METIS ordering, nonsymmetric scaling, weighted matching, pivot perturbation `13`, one-based indices. |
| MUMPS CPU | `mumps-cpu` | Unsymmetric assembled triplet input, MPI world communicator, `ICNTL(51)=0`, job `1/2/3` split. |
| MUMPS GPU | `mumps-gpu` | Same triplet path, `ICNTL(51)=1`, async GPU memory pinning disabled with `KEEP(422)=0` because this image's MUMPS 5.9 async pinning path segfaults on null-size pin requests. |
| PaStiX CPU | `pastix-cpu` | CSC input normalized with `spmCheckAndCorrect`, LU factorization, METIS ordering, dynamic CPU scheduler, `GPU_NBR=0`, optional `PASTIX_EPSILON_MAGN_CTRL` override. |
| PaStiX GPU | `pastix-gpu` | Same matrix normalization and METIS/LU settings, StarPU scheduler, `GPU_NBR=1`. |
| STRUMPACK CPU | `strumpack-cpu` | MT interface, direct solve, no compression, METIS ordering with `METIS_NodeNDP`, max-diagonal-product matching, GPU disabled. |
| STRUMPACK GPU | `strumpack-gpu` | Same MT/direct/METIS settings as CPU wrapper, CUDA path enabled. |
| cuDSS GPU | `cudss-gpu` | CSR input with 64-bit indices, general/full matrix view, matching enabled, OpenMP threading layer loaded, `CUDSS_CONFIG_HOST_NTHREADS` set from `CUDSS_HOST_NTHREADS` or hardware concurrency, synchronized analysis/factor/solve phases. |
| GLU 3.0 GPU | `glu3-gpu` | NICSLU preprocessing from CSC is outside timed phases; analysis is `Symbolic_Matrix` construction/fill/csr/predictLU/leveling; factorization is CUDA LU; solve is GLU solve. |
| PanguLU GPU | `pangulu-gpu` | CSC input, `pangulu_init/gstrf/gstrs`, GPU enabled, warp/block settings `4/4`, `nthread=1`, `reordering_nthread=4`, default `nb=n` with `PANGULU_NB` override. |

Notes:

- STRUMPACK uses METIS in the MT sequential wrapper because that interface rejects parallel matrix reorderings such as PARMETIS.
- PanguLU with `nb < n` triggers an out-of-bounds shared-memory read in `tstrf_cuda` on this single-process wrapper. `nb=n` is the only validated stable setting in this environment.
- GLU 3.0 is single precision in this installation, so its `absolute_error` is expected to be worse than double-precision solvers.
- PaStiX GPU and STRUMPACK GPU include their runtime scheduling overhead inside solver phase timings.

## Latest Full Benchmark Summary

Commands:

```bash
STARPU_SILENT=1 STARPU_CALIBRATE=0 \
  /tmp/sparse_direct_solver_phase_consistent_build/benchmark \
  --matrix-set all \
  --solver cpu \
  --output /tmp/third_party_cpu.csv

for solver in mumps-gpu pastix-gpu strumpack-gpu cudss-gpu glu3-gpu; do
  STARPU_SILENT=1 STARPU_CALIBRATE=0 \
    /tmp/sparse_direct_solver_phase_consistent_build/benchmark \
    --matrix-set all \
    --solver "$solver" \
    --warmup-gpu \
    --output "/tmp/third_party_gpu_${solver}.csv"
done

for matrix in memplus rajat27 wang3 onetone2 rajat15 case30 case118 case1197 \
  case_ACTIVSg2000 case3012wp case6468rte case8387pegase \
  case_ACTIVSg25k case_SyntheticUSA; do
  STARPU_SILENT=1 STARPU_CALIBRATE=0 \
    /tmp/sparse_direct_solver_phase_consistent_build/benchmark \
    --matrix-set all \
    --matrix "$matrix" \
    --solver pangulu \
    --output "/tmp/third_party_pangulu_${matrix}.csv"
done
```

The latest run separates CPU, GPU, and PanguLU measurements. GPU solvers are
measured solver-by-solver with `--warmup-gpu`, which runs one unrecorded solve
on the same matrix before writing the measured row. PanguLU stays matrix-by-
matrix because two large cases exit inside the library.

The latest CSV contains 182 matrix-solver rows: 177 successful rows and 5
failure rows.

Benchmark matrices:

| Matrix | Group | Rows | NNZ |
|---|---|---:|---:|
| `memplus` | suitesparse | 17758 | 126150 |
| `rajat27` | suitesparse | 20640 | 99777 |
| `wang3` | suitesparse | 26064 | 177168 |
| `onetone2` | suitesparse | 36057 | 227628 |
| `rajat15` | suitesparse | 37261 | 443573 |
| `case30` | matpower_nr | 53 | 361 |
| `case118` | matpower_nr | 181 | 1051 |
| `case1197` | matpower_nr | 2392 | 14344 |
| `case_ACTIVSg2000` | matpower_nr | 3607 | 26345 |
| `case3012wp` | matpower_nr | 5725 | 36263 |
| `case6468rte` | matpower_nr | 12643 | 87845 |
| `case8387pegase` | matpower_nr | 14908 | 110572 |
| `case_ACTIVSg25k` | matpower_nr | 47246 | 318672 |
| `case_SyntheticUSA` | matpower_nr | 156255 | 1052085 |

Representative CPU solver: SuiteSparse KLU (`klu`).

| Matrix | Analysis ms | Factorize ms | Solve ms | berr | absolute error |
|---|---:|---:|---:|---:|---:|
| `memplus` | 5.851 | 2.812 | 0.140 | 8.000e-16 | 1.126e-11 |
| `rajat27` | 7.546 | 3.027 | 0.244 | 1.532e-05 | 3.026e-05 |
| `wang3` | 10.655 | 3180.366 | 9.242 | 5.147e-15 | 4.271e-13 |
| `onetone2` | 55.025 | 177.722 | 1.782 | 1.312e-07 | 1.674e-04 |
| `rajat15` | 26.976 | 81.559 | 1.806 | 2.177e-09 | 9.314e-07 |
| `case30` | 0.020 | 0.016 | 0.001 | 1.108e-16 | 0.000e+00 |
| `case118` | 0.053 | 0.042 | 0.003 | 1.956e-16 | 0.000e+00 |
| `case1197` | 0.227 | 0.204 | 0.018 | 4.075e-15 | 0.000e+00 |
| `case_ACTIVSg2000` | 0.849 | 1.732 | 0.075 | 5.450e-16 | 0.000e+00 |
| `case3012wp` | 1.168 | 1.335 | 0.075 | 3.891e-16 | 0.000e+00 |
| `case6468rte` | 2.742 | 3.193 | 0.171 | 7.102e-16 | 0.000e+00 |
| `case8387pegase` | 3.992 | 4.447 | 0.229 | 3.632e-15 | 0.000e+00 |
| `case_ACTIVSg25k` | 9.889 | 19.699 | 0.736 | 1.990e-14 | 0.000e+00 |
| `case_SyntheticUSA` | 32.714 | 94.500 | 4.760 | 2.003e-14 | 0.000e+00 |

Representative GPU solver: NVIDIA cuDSS (`cudss-gpu`).

| Matrix | Analysis ms | Factorize ms | Solve ms | berr | absolute error |
|---|---:|---:|---:|---:|---:|
| `memplus` | 98.913 | 1.140 | 0.420 | 1.132e-15 | 1.238e-11 |
| `rajat27` | 80.063 | 1.005 | 0.387 | 8.279e-12 | 4.067e-08 |
| `wang3` | 124.561 | 33.442 | 1.999 | 1.740e-15 | 2.557e-13 |
| `onetone2` | 126.759 | 7.652 | 1.492 | 9.557e-11 | 1.476e-09 |
| `rajat15` | 151.035 | 3.365 | 0.847 | 4.141e-13 | 5.247e-09 |
| `case30` | 30.980 | 0.081 | 0.058 | 2.062e-16 | 8.257e-17 |
| `case118` | 44.056 | 0.097 | 0.070 | 1.922e-16 | 1.334e-18 |
| `case1197` | 67.473 | 0.223 | 0.107 | 2.138e-15 | 1.663e-14 |
| `case_ACTIVSg2000` | 73.647 | 0.630 | 0.266 | 3.032e-16 | 2.381e-14 |
| `case3012wp` | 69.479 | 0.466 | 0.218 | 2.439e-14 | 5.672e-16 |
| `case6468rte` | 80.106 | 0.629 | 0.284 | 6.151e-14 | 1.181e-16 |
| `case8387pegase` | 86.902 | 1.110 | 0.359 | 7.031e-11 | 4.498e-16 |
| `case_ACTIVSg25k` | 103.361 | 1.859 | 0.673 | 1.157e-14 | 1.109e-12 |
| `case_SyntheticUSA` | 226.699 | 5.123 | 1.441 | 1.218e-14 | 6.446e-10 |

Failure rows:

| Matrix | Solver | Failure |
|---|---|---|
| `onetone2` | `pastix-cpu` | Error metric failed because the output vector contains non-finite values. |
| `onetone2` | `pastix-gpu` | Error metric failed because the output vector contains non-finite values. |
| `onetone2` | `pangulu-gpu` | Error metric failed because the output vector contains non-finite values. |
| `case_ACTIVSg25k` | `pangulu-gpu` | Process exited with CUDA illegal memory access in PanguLU preprocessing. |
| `case_SyntheticUSA` | `pangulu-gpu` | Process exited because requested PanguLU shared memory exceeds the GPU limit. |

## Open Issues

| Solver | Issue | Current handling |
|---|---|---|
| PaStiX CPU/GPU | `onetone2` returns non-finite values even though PaStiX reports success. Raising `PASTIX_EPSILON_MAGN_CTRL` can make the vector finite, but the solution is inaccurate (`berr≈1`). PaStiX GPU also aborts in the StarPU registration path when too many matrices are run in one process with warmup. | Keep default direct solve settings and mark rows as `success=false` when error computation detects non-finite output. For GPU warmup timing, measure the later large matrices in separate processes. |
| PanguLU GPU | `PANGULU_NB < n` triggers an out-of-bounds shared-memory read in `tstrf_cuda` under this single-process wrapper. `nb=n` avoids that path on small and medium cases, but `onetone2` returns non-finite values and the two largest MATPOWER cases exit inside PanguLU. | Keep failure rows in the CSV. The current wrapper is usable only for the passing matrix subset until the PanguLU block-size/shared-memory path is fixed. |
| MUMPS GPU | MUMPS 5.9 async GPU memory pinning path emits null-size `cudaHostRegister` requests and can segfault on finalization. | Keep GPU offload enabled with `ICNTL(51)=1`, but disable async pinning via `KEEP(422)=0`. |
