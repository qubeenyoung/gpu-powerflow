# Linear Solver Benchmark Report v2

## 1. What Changed From v1

- Ginkgo was installed locally with CUDA support under `third_party/ginkgo/install/`; both GMRES+Jacobi and BiCGSTAB+Jacobi wrappers ran on the CUDA executor.
- SuperLU_DIST was built locally with CUDA and MPI, but the fp64 ABglobal benchmark wrapper failed at runtime for every system with `Invalid ISPEC at line 556 in .../get_perm_c.c`; fp32 is explicitly unsupported by this wrapper.
- STRUMPACK was built locally with CUDA and MPI. The first CUDA install linked, but wrapper linking exposed MPICH LTO/CUDA fatbin issues, so a second no-LTO install was created under `third_party/strumpack/install_nolto/`. STRUMPACK np=1 ran all dumped systems; np=2 and np=4 synthetic sanity attempts hung after initialization and were interrupted.
- The original v1 report was not overwritten; this report uses `summary_second_pass.csv` and second-pass raw JSON files with `_second_pass` suffixes.

## 2. Environment

| item | value |
| --- | --- |
| hostname | 7d44605f53c6 |
| OS | Linux-6.8.0-110-generic-x86_64-with-glibc2.35 |
| CPU | AMD Ryzen 5 5600 6-Core Processor |
| RAM GB | 31.26 |
| GPU | NVIDIA GeForce RTX 3090 (24576 MiB) |
| NVIDIA driver | 580.126.09 |
| CUDA toolkit / nvcc | CUDA 12.8 (nvcc V12.8.93) |
| Python | 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0] |
| gcc | gcc (Ubuntu 12.3.0-1ubuntu1~22.04.2) 12.3.0 |
| g++ | g++ (Ubuntu 12.3.0-1ubuntu1~22.04.2) 12.3.0 |
| CMake | cmake version 3.22.1 |
| MPI after retry | MPICH local plus system MPICH/OpenMPI packages available |

## 3. Dataset Table

| case | iteration | num_bus | num_pv | num_pq | matrix_size | nnz | rhs_norm | pattern_hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_validation | 0 | 0 | 0 | 0 | 5x5 | 16 | 2.424 | a3dfe5a39556fae42897c146fe3b2aa2b3b067d77f46671f919fae733870690b |
| case14 | 0 | 14 | 4 | 9 | 22x22 | 146 | 0.0455 | a6d546a32c79e8b98c75a8bc0769e85d04c9453d3f89e3178d6a8cf82e29d781 |
| case14 | 1 | 14 | 4 | 9 | 22x22 | 146 | 5.76e-05 | a6d546a32c79e8b98c75a8bc0769e85d04c9453d3f89e3178d6a8cf82e29d781 |
| case14 | 2 | 14 | 4 | 9 | 22x22 | 146 | 1.34e-10 | a6d546a32c79e8b98c75a8bc0769e85d04c9453d3f89e3178d6a8cf82e29d781 |
| case118 | 0 | 118 | 53 | 64 | 181x181 | 1051 | 1.421 | 96d96782c0b757d0909fe72d53ef727cecfef18a6fb0f60d1323c0b610be5958 |
| case118 | 1 | 118 | 53 | 64 | 181x181 | 1051 | 0.0245 | 96d96782c0b757d0909fe72d53ef727cecfef18a6fb0f60d1323c0b610be5958 |
| case118 | 3 | 118 | 53 | 64 | 181x181 | 1051 | 1.6e-12 | 96d96782c0b757d0909fe72d53ef727cecfef18a6fb0f60d1323c0b610be5958 |
| case300 | 0 | 300 | 68 | 231 | 530x530 | 3736 | 13.15 | 9c1ad799d742df265bace14f66edf0d569114a33e7b61c6cd6228e533ac8543f |
| case300 | 1 | 300 | 68 | 231 | 530x530 | 3736 | 4.02 | 9c1ad799d742df265bace14f66edf0d569114a33e7b61c6cd6228e533ac8543f |
| case300 | 5 | 300 | 68 | 231 | 530x530 | 3736 | 2.6e-12 | 9c1ad799d742df265bace14f66edf0d569114a33e7b61c6cd6228e533ac8543f |
| case1354pegase | 0 | 1354 | 259 | 1094 | 2447x2447 | 15803 | 29.83 | 802788188a259605181b5fda6aafd2dd99a8118e42573efd1a1eef19b6e196bc |
| case1354pegase | 1 | 1354 | 259 | 1094 | 2447x2447 | 15803 | 1.083 | 802788188a259605181b5fda6aafd2dd99a8118e42573efd1a1eef19b6e196bc |
| case1354pegase | 4 | 1354 | 259 | 1094 | 2447x2447 | 15803 | 1.9e-11 | 802788188a259605181b5fda6aafd2dd99a8118e42573efd1a1eef19b6e196bc |
| case2869pegase | 0 | 2869 | 509 | 2359 | 5227x5227 | 36591 | 88.89 | df9b9e249b1666581f9eb66abc52e8d49cb4b330c45e67ae6afc2b2a975993ee |
| case2869pegase | 1 | 2869 | 509 | 2359 | 5227x5227 | 36591 | 89.43 | df9b9e249b1666581f9eb66abc52e8d49cb4b330c45e67ae6afc2b2a975993ee |
| case2869pegase | 6 | 2869 | 509 | 2359 | 5227x5227 | 36591 | 3.06e-09 | df9b9e249b1666581f9eb66abc52e8d49cb4b330c45e67ae6afc2b2a975993ee |
| case9241pegase | 0 | 9241 | 1444 | 7796 | 17036x17036 | 129412 | 108.64 | e7e4c226d863b9c0b629df7caddc6667b11b35e086a349a13a01b052bb6a9873 |
| case9241pegase | 1 | 9241 | 1444 | 7796 | 17036x17036 | 129412 | 88.41 | e7e4c226d863b9c0b629df7caddc6667b11b35e086a349a13a01b052bb6a9873 |
| case9241pegase | 6 | 9241 | 1444 | 7796 | 17036x17036 | 129412 | 3.02e-09 | e7e4c226d863b9c0b629df7caddc6667b11b35e086a349a13a01b052bb6a9873 |

## 4. Solver Availability

| solver | type | build_status | version | CUDA_enabled | MPI_required | GPU_resident_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ginkgo | GPU iterative Krylov | installed/runnable | 2.0.0 develop | yes | no | GPU resident for matrix/vector operations | CUDA executor built locally; GMRES and BiCGSTAB with Jacobi were benchmarked. |
| SuperLU_DIST | distributed sparse direct | installed_cuda / runtime_failed | 9.2.1 | yes | yes | not observed | CUDA build succeeded, but fp64 ABglobal wrapper failed at runtime: Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/2026051... |
| STRUMPACK | distributed sparse direct / multifrontal | installed_cuda_nolto / runnable np=1 | 8.0.0 | yes | yes | host input/output with internal GPU offload | Rebuilt with MPI compile options set to -fno-lto; np=1 ran all systems. np=2/np=4 synthetic sanity runs hung after initialization. Startu... |

## 5. Second-Pass Result Counts

| solver | status | converged | rows |
| --- | --- | --- | --- |
| Ginkgo-BiCGSTAB-Jacobi | ok | False | 21 |
| Ginkgo-BiCGSTAB-Jacobi | ok | True | 17 |
| Ginkgo-GMRES-Jacobi | ok | False | 17 |
| Ginkgo-GMRES-Jacobi | ok | True | 21 |
| STRUMPACK | ok | False | 9 |
| STRUMPACK | ok | True | 29 |
| STRUMPACK | timeout | False | 2 |
| SuperLU_DIST | runtime_failed | False | 19 |
| SuperLU_DIST | unsupported | False | 19 |

## 6. Aggregate Timing Summary

| solver | rows | converged_rows | median_analysis | median_factor | median_solve | median_solver |
| --- | --- | --- | --- | --- | --- | --- |
| cuDSS | 38 | 38 | 13.96 | 0.124 | 0.0769 | 14.1 |
| cuSolverSP | 38 | 38 | 0 | 0 | 5.118 | 5.118 |
| AMGx | 38 | 20 | 14.01 | 0 | 62.04 | 78.38 |
| Ginkgo-GMRES-Jacobi | 38 | 21 | 15.05 | 0 | 442.54 | 457.52 |
| Ginkgo-BiCGSTAB-Jacobi | 38 | 17 | 15.01 | 0 | 94.51 | 109.47 |
| STRUMPACK | 40 | 29 | 1.606 | 11.24 | 1.141 | 15.73 |
| SuperLU_DIST | 38 | 0 | n/a | n/a | n/a | n/a |

## 7. New Performance Table

| case | iter | solver | dtype | status | analysis_ms | factor_ms | solve_ms | solver_ms | end_to_end_ms | rel_res_2 | rel_err_2 | conv | iters | residency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_validation | 0 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 13.99 | 0 | 0.184 | 14.17 | 218.64 | 5.61e-08 | 8.14e-08 | True | 1 | yes_for_matrix_vector_and_itera... |
| synthetic_validation | 0 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 14.13 | 0 | 0.22 | 14.35 | 216.95 | 1.37e-16 | 1.16e-16 | True | 0 | yes_for_matrix_vector_and_itera... |
| synthetic_validation | 0 | Ginkgo-GMRES-Jacobi | fp32 | ok | 14.09 | 0 | 1.514 | 15.6 | 248.93 | 2.34e-07 | 3.42e-07 | True | 14 | yes_for_matrix_vector_and_itera... |
| synthetic_validation | 0 | Ginkgo-GMRES-Jacobi | fp64 | ok | 26.85 | 0 | 0.226 | 27.08 | 315.74 | 2.26e-16 | 2.79e-16 | True | 1 | yes_for_matrix_vector_and_itera... |
| synthetic_validation | 0 | STRUMPACK np=1 | fp32 | ok | 0.517 | 10.06 | 0.453 | 11.03 | 322.79 | 5.47e-08 | 7.76e-08 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| synthetic_validation | 0 | STRUMPACK np=1 | fp64 | ok | 0.655 | 10.94 | 2.301 | 13.89 | 1311.1 | 4.58e-17 | 9.94e-17 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| synthetic_validation | 0 | STRUMPACK np=2 | fp64 | timeout | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False | -1 | unknown |
| synthetic_validation | 0 | STRUMPACK np=4 | fp64 | timeout | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False | -1 | unknown |
| synthetic_validation | 0 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| synthetic_validation | 0 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 124.81 | n/a | n/a | False | -1 | unknown |
| case14 | 0 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 14.08 | 0 | 0.182 | 14.27 | 214.32 | 1.07e-07 | 2.68e-07 | True | 1 | yes_for_matrix_vector_and_itera... |
| case14 | 0 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 14.15 | 0 | 0.221 | 14.37 | 214.71 | 4.45e-16 | 9.61e-16 | True | 0 | yes_for_matrix_vector_and_itera... |
| case14 | 0 | Ginkgo-GMRES-Jacobi | fp32 | ok | 14.1 | 0 | 2.393 | 16.5 | 250.56 | 3.23e-07 | 6.23e-07 | True | 19 | yes_for_matrix_vector_and_itera... |
| case14 | 0 | Ginkgo-GMRES-Jacobi | fp64 | ok | 14.26 | 0 | 0.228 | 14.49 | 217.37 | 6e-16 | 8.55e-16 | True | 1 | yes_for_matrix_vector_and_itera... |
| case14 | 0 | STRUMPACK np=1 | fp32 | ok | 0.735 | 9.309 | 0.554 | 10.6 | 309.49 | 7.32e-08 | 4.2e-07 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case14 | 0 | STRUMPACK np=1 | fp64 | ok | 0.848 | 9.368 | 0.626 | 10.84 | 1149.5 | 2.04e-16 | 8.79e-16 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case14 | 0 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case14 | 0 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 114.71 | n/a | n/a | False | -1 | unknown |
| case14 | 1 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 14.14 | 0 | 0.184 | 14.32 | 218.31 | 1.34e-07 | 3.3e-07 | True | 1 | yes_for_matrix_vector_and_itera... |
| case14 | 1 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 14.18 | 0 | 0.221 | 14.4 | 211.27 | 6.97e-16 | 5.04e-16 | True | 0 | yes_for_matrix_vector_and_itera... |
| case14 | 1 | Ginkgo-GMRES-Jacobi | fp32 | ok | 14.09 | 0 | 12.7 | 26.79 | 395.72 | 6.14e-07 | 2.27e-07 | True | 51 | yes_for_matrix_vector_and_itera... |
| case14 | 1 | Ginkgo-GMRES-Jacobi | fp64 | ok | 14.24 | 0 | 0.229 | 14.47 | 221.84 | 8.98e-16 | 4.75e-16 | True | 1 | yes_for_matrix_vector_and_itera... |
| case14 | 1 | STRUMPACK np=1 | fp32 | ok | 0.704 | 9.542 | 0.487 | 10.73 | 306.46 | 1.28e-07 | 2.04e-07 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case14 | 1 | STRUMPACK np=1 | fp64 | ok | 0.715 | 11.88 | 4.973 | 17.57 | 500.23 | 1.36e-16 | 8.23e-16 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case14 | 1 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case14 | 1 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 111.76 | n/a | n/a | False | -1 | unknown |
| case14 | 2 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 13.86 | 0 | 0.182 | 14.04 | 217.07 | 1.26e-07 | 2.75e-07 | True | 1 | yes_for_matrix_vector_and_itera... |
| case14 | 2 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 13.9 | 0 | 0.23 | 14.13 | 216.02 | 1.07e-15 | 7.93e-16 | True | 0 | yes_for_matrix_vector_and_itera... |
| case14 | 2 | Ginkgo-GMRES-Jacobi | fp32 | ok | 13.85 | 0 | 0.334 | 14.18 | 222.27 | 2.42e-07 | 2.95e-07 | True | 4 | yes_for_matrix_vector_and_itera... |
| case14 | 2 | Ginkgo-GMRES-Jacobi | fp64 | ok | 14.14 | 0 | 0.239 | 14.38 | 220.63 | 8.06e-16 | 3.78e-16 | True | 1 | yes_for_matrix_vector_and_itera... |
| case14 | 2 | STRUMPACK np=1 | fp32 | ok | 0.869 | 9.358 | 0.333 | 10.56 | 299.84 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case14 | 2 | STRUMPACK np=1 | fp64 | ok | 0.23 | 9.521 | 0.452 | 10.2 | 296.29 | 3.21e-16 | 4.98e-16 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case14 | 2 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case14 | 2 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 113.86 | n/a | n/a | False | -1 | unknown |
| case118 | 0 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 14.85 | 0 | 5.37 | 20.22 | 279.34 | 3.59e-07 | 3.68e-06 | True | 56 | yes_for_matrix_vector_and_itera... |
| case118 | 0 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 14.93 | 0 | 8.41 | 23.34 | 323.82 | 7.8e-12 | 2.43e-10 | True | 56 | yes_for_matrix_vector_and_itera... |
| case118 | 0 | Ginkgo-GMRES-Jacobi | fp32 | ok | 14.86 | 0 | 55.39 | 70.24 | 979.25 | 1.31e-07 | 1.12e-06 | True | 147 | yes_for_matrix_vector_and_itera... |
| case118 | 0 | Ginkgo-GMRES-Jacobi | fp64 | ok | 14.99 | 0 | 26.03 | 41.01 | 581.46 | 7.46e-11 | 7.61e-10 | True | 72 | yes_for_matrix_vector_and_itera... |
| case118 | 0 | STRUMPACK np=1 | fp32 | ok | 0.894 | 9.979 | 0.757 | 11.63 | 315.88 | 1.52e-07 | 3.19e-06 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case118 | 0 | STRUMPACK np=1 | fp64 | ok | 0.79 | 9.406 | 1.194 | 11.39 | 317.61 | 2.11e-16 | 1.13e-15 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case118 | 0 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case118 | 0 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 114.51 | n/a | n/a | False | -1 | unknown |
| case118 | 1 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 14.73 | 0 | 5.848 | 20.58 | 284.80 | 9.74e-07 | 2.58e-06 | True | 61 | yes_for_matrix_vector_and_itera... |
| case118 | 1 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 15 | 0 | 8.397 | 23.4 | 325.15 | 2.38e-11 | 6.48e-11 | True | 56 | yes_for_matrix_vector_and_itera... |
| case118 | 1 | Ginkgo-GMRES-Jacobi | fp32 | ok | 14.91 | 0 | 53.91 | 68.83 | 958.42 | 2e-07 | 1.01e-06 | True | 144 | yes_for_matrix_vector_and_itera... |
| case118 | 1 | Ginkgo-GMRES-Jacobi | fp64 | ok | 15.01 | 0 | 26.47 | 41.48 | 587.91 | 7.26e-11 | 4.69e-10 | True | 73 | yes_for_matrix_vector_and_itera... |
| case118 | 1 | STRUMPACK np=1 | fp32 | ok | 0.425 | 10.03 | 0.526 | 10.98 | 429.04 | 2.48e-07 | 1.52e-06 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case118 | 1 | STRUMPACK np=1 | fp64 | ok | 1.021 | 9.236 | 0.691 | 10.95 | 326.07 | 5.57e-16 | 2.14e-15 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case118 | 1 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case118 | 1 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 112.39 | n/a | n/a | False | -1 | unknown |
| case118 | 3 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 14.76 | 0 | 5.668 | 20.43 | 281.25 | 1.1e-06 | 1.46e-06 | True | 59 | yes_for_matrix_vector_and_itera... |
| case118 | 3 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 14.94 | 0 | 9.984 | 24.93 | 348.47 | 8.84e-11 | 7.75e-11 | True | 55 | yes_for_matrix_vector_and_itera... |
| case118 | 3 | Ginkgo-GMRES-Jacobi | fp32 | ok | 14.94 | 0 | 55.9 | 70.84 | 1026.9 | 2.56e-07 | 3.57e-07 | True | 148 | yes_for_matrix_vector_and_itera... |
| case118 | 3 | Ginkgo-GMRES-Jacobi | fp64 | ok | 14.97 | 0 | 28.18 | 43.15 | 614.04 | 8.71e-11 | 4.84e-10 | True | 75 | yes_for_matrix_vector_and_itera... |
| case118 | 3 | STRUMPACK np=1 | fp32 | ok | 1.059 | 9.874 | 0.359 | 11.29 | 315.34 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case118 | 3 | STRUMPACK np=1 | fp64 | ok | 0.952 | 9.273 | 0.457 | 10.68 | 303.21 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case118 | 3 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case118 | 3 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 115.03 | n/a | n/a | False | -1 | unknown |
| case300 | 0 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 14.87 | 0 | 94.59 | 109.47 | 1528.6 | 21761170.2 | 3559992656.4 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case300 | 0 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 14.95 | 0 | 26.6 | 41.55 | 574.87 | 4.02e-11 | 2.93e-09 | True | 164 | yes_for_matrix_vector_and_itera... |
| case300 | 0 | Ginkgo-GMRES-Jacobi | fp32 | ok | 15 | 0 | 442.45 | 457.45 | 6398.5 | 2.76e-07 | 6.53e-06 | True | 1000 | yes_for_matrix_vector_and_itera... |
| case300 | 0 | Ginkgo-GMRES-Jacobi | fp64 | ok | 14.98 | 0 | 274.31 | 289.29 | 4061.6 | 8.6e-11 | 9.68e-09 | True | 585 | yes_for_matrix_vector_and_itera... |
| case300 | 0 | STRUMPACK np=1 | fp32 | ok | 1.501 | 9.787 | 0.865 | 12.15 | 324.26 | 3.63e-07 | 6.28e-06 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case300 | 0 | STRUMPACK np=1 | fp64 | ok | 1.439 | 10.3 | 1.045 | 12.78 | 325.99 | 9.9e-16 | 3.03e-14 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case300 | 0 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case300 | 0 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 120.20 | n/a | n/a | False | -1 | unknown |
| case300 | 1 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 14.79 | 0 | 94.68 | 109.48 | 1534.7 | 2367321452640993792.0 | 38345300194559172608.0 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case300 | 1 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 14.96 | 0 | 28.72 | 43.68 | 614.31 | 4.83e-11 | 1.55e-09 | True | 173 | yes_for_matrix_vector_and_itera... |
| case300 | 1 | Ginkgo-GMRES-Jacobi | fp32 | ok | 14.97 | 0 | 442.63 | 457.60 | 6406.9 | 1.79e-07 | 1.82e-06 | True | 1000 | yes_for_matrix_vector_and_itera... |
| case300 | 1 | Ginkgo-GMRES-Jacobi | fp64 | ok | 15.02 | 0 | 215.01 | 230.04 | 3233.9 | 9.76e-11 | 7.02e-09 | True | 466 | yes_for_matrix_vector_and_itera... |
| case300 | 1 | STRUMPACK np=1 | fp32 | ok | 1.484 | 9.846 | 0.679 | 12.01 | 322.61 | 5.15e-07 | 4.18e-06 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case300 | 1 | STRUMPACK np=1 | fp64 | ok | 1.425 | 10.21 | 1.089 | 12.72 | 325.42 | 4.34e-16 | 3.48e-15 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case300 | 1 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case300 | 1 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 117.37 | n/a | n/a | False | -1 | unknown |
| case300 | 5 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 14.97 | 0 | 94.8 | 109.77 | 1532.4 | 0.0192 | 0.337 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case300 | 5 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 15.03 | 0 | 29.49 | 44.52 | 617.48 | 6.32e-11 | 1.07e-08 | True | 144 | yes_for_matrix_vector_and_itera... |
| case300 | 5 | Ginkgo-GMRES-Jacobi | fp32 | ok | 15.03 | 0 | 441.95 | 456.98 | 6401.1 | 8.81e-07 | 3.34e-06 | True | 1000 | yes_for_matrix_vector_and_itera... |
| case300 | 5 | Ginkgo-GMRES-Jacobi | fp64 | ok | 14.95 | 0 | 266.30 | 281.25 | 3952.1 | 9.87e-11 | 1.56e-08 | True | 573 | yes_for_matrix_vector_and_itera... |
| case300 | 5 | STRUMPACK np=1 | fp32 | ok | 1.338 | 10.21 | 0.501 | 12.05 | 323.88 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case300 | 5 | STRUMPACK np=1 | fp64 | ok | 1.712 | 10.08 | 0.618 | 12.41 | 352.30 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case300 | 5 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case300 | 5 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 115.03 | n/a | n/a | False | -1 | unknown |
| case1354pegase | 0 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 15.01 | 0 | 94.63 | 109.64 | 1534.3 | 5700721411051301888.0 | 516754124023197335552.0 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 0 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 15.64 | 0 | 184.32 | 199.96 | 2802.1 | 1.84e-07 | 7.56e-06 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 0 | Ginkgo-GMRES-Jacobi | fp32 | ok | 15.06 | 0 | 442.96 | 458.02 | 6425.3 | 3.16e-05 | 0.0167 | True | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 0 | Ginkgo-GMRES-Jacobi | fp64 | ok | 15.24 | 0 | 493.29 | 508.53 | 7133.6 | 3.14e-05 | 0.0168 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 0 | STRUMPACK np=1 | fp32 | ok | 5.549 | 12.88 | 7.072 | 25.5 | 508.70 | 5.52e-07 | 4.38e-06 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case1354pegase | 0 | STRUMPACK np=1 | fp64 | ok | 5.86 | 11.98 | 5.598 | 23.44 | 469.72 | 1.06e-15 | 1.37e-14 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case1354pegase | 0 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case1354pegase | 0 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 119.97 | n/a | n/a | False | -1 | unknown |
| case1354pegase | 1 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 15.04 | 0 | 94.68 | 109.72 | 1534.4 | 0.0114 | 0.579 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 1 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 15.16 | 0 | 181.94 | 197.10 | 2762.6 | 0.000102 | 0.00112 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 1 | Ginkgo-GMRES-Jacobi | fp32 | ok | 15.13 | 0 | 446.00 | 461.13 | 6454.5 | 0.000231 | 0.0324 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 1 | Ginkgo-GMRES-Jacobi | fp64 | ok | 15.21 | 0 | 491.94 | 507.15 | 7100.9 | 0.00023 | 0.0323 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 1 | STRUMPACK np=1 | fp32 | ok | 5.671 | 11.81 | 1.996 | 19.48 | 422.32 | 3.33e-06 | 9.48e-06 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case1354pegase | 1 | STRUMPACK np=1 | fp64 | ok | 6.036 | 19.37 | 3.915 | 29.32 | 526.77 | 7.37e-15 | 2.38e-14 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case1354pegase | 1 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case1354pegase | 1 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 120.26 | n/a | n/a | False | -1 | unknown |
| case1354pegase | 4 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 16.59 | 0 | 94.87 | 111.47 | 1550.6 | 508308.1 | 12261150.9 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 4 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 15.22 | 0 | 223.50 | 238.72 | 3338.5 | 0.000169 | 0.00282 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 4 | Ginkgo-GMRES-Jacobi | fp32 | ok | 15.12 | 0 | 451.78 | 466.90 | 6562.9 | 0.000968 | 0.0577 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 4 | Ginkgo-GMRES-Jacobi | fp64 | ok | 15.17 | 0 | 487.20 | 502.37 | 7052.7 | 0.000972 | 0.0582 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case1354pegase | 4 | STRUMPACK np=1 | fp32 | ok | 9.036 | 16.16 | 1.567 | 26.76 | 529.00 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case1354pegase | 4 | STRUMPACK np=1 | fp64 | ok | 6.109 | 11.54 | 2.112 | 19.76 | 436.49 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case1354pegase | 4 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case1354pegase | 4 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 121.40 | n/a | n/a | False | -1 | unknown |
| case2869pegase | 0 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 15.3 | 0 | 94.43 | 109.73 | 1541.6 | n/a | n/a | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 0 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 15.47 | 0 | 183.62 | 199.09 | 2795.0 | 6.73e-05 | 0.0502 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 0 | Ginkgo-GMRES-Jacobi | fp32 | ok | 15.32 | 0 | 448.00 | 463.32 | 6496.5 | 0.000267 | 0.188 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 0 | Ginkgo-GMRES-Jacobi | fp64 | ok | 15.73 | 0 | 492.59 | 508.32 | 7129.5 | 0.000267 | 0.187 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 0 | STRUMPACK np=1 | fp32 | ok | 12.26 | 13.39 | 3.76 | 29.41 | 628.82 | 2.64e-06 | 8.75e-05 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case2869pegase | 0 | STRUMPACK np=1 | fp64 | ok | 13.31 | 12.52 | 7.253 | 33.07 | 630.27 | 6e-15 | 2.74e-13 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case2869pegase | 0 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case2869pegase | 0 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 135.18 | n/a | n/a | False | -1 | unknown |
| case2869pegase | 1 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 15.3 | 0 | 93.89 | 109.19 | 1535.0 | 0.0127 | 0.683 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 1 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 15.59 | 0 | 183.81 | 199.40 | 2791.1 | 8.44e-05 | 0.182 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 1 | Ginkgo-GMRES-Jacobi | fp32 | ok | 15.24 | 0 | 444.36 | 459.60 | 6447.1 | 0.000163 | 0.375 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 1 | Ginkgo-GMRES-Jacobi | fp64 | ok | 15.53 | 0 | 488.62 | 504.15 | 7080.0 | 0.000163 | 0.374 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 1 | STRUMPACK np=1 | fp32 | ok | 16.67 | 13.57 | 3.906 | 34.15 | 633.96 | 8.66e-07 | 0.000142 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case2869pegase | 1 | STRUMPACK np=1 | fp64 | ok | 17.68 | 13.04 | 10.84 | 41.56 | 707.96 | 1.96e-15 | 1.47e-13 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case2869pegase | 1 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case2869pegase | 1 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 129.04 | n/a | n/a | False | -1 | unknown |
| case2869pegase | 6 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 15.39 | 0 | 94.63 | 110.02 | 1550.1 | 0.737 | 52.64 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 6 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 15.55 | 0 | 226.66 | 242.22 | 3391.2 | 0.000225 | 0.11 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 6 | Ginkgo-GMRES-Jacobi | fp32 | ok | 15.3 | 0 | 446.76 | 462.06 | 6472.8 | 0.000287 | 0.258 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 6 | Ginkgo-GMRES-Jacobi | fp64 | ok | 15.62 | 0 | 493.69 | 509.31 | 7138.6 | 0.000286 | 0.258 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case2869pegase | 6 | STRUMPACK np=1 | fp32 | ok | 21.95 | 16.71 | 0.478 | 39.14 | 740.61 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case2869pegase | 6 | STRUMPACK np=1 | fp64 | ok | 15.67 | 16.84 | 7.68 | 40.19 | 869.72 | 2.93e-15 | 1.15e-14 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case2869pegase | 6 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case2869pegase | 6 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 127.78 | n/a | n/a | False | -1 | unknown |
| case9241pegase | 0 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 16.6 | 0 | 101.46 | 118.06 | 1679.2 | 106.78 | 17.55 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 0 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 16.86 | 0 | 225.02 | 241.88 | 3407.7 | 11.49 | 7.375 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 0 | Ginkgo-GMRES-Jacobi | fp32 | ok | 16.73 | 0 | 443.11 | 459.85 | 6471.4 | 0.000846 | 0.914 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 0 | Ginkgo-GMRES-Jacobi | fp64 | ok | 17.58 | 0 | 502.40 | 519.98 | 7337.5 | 0.000846 | 0.914 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 0 | STRUMPACK np=1 | fp32 | ok | 54.12 | 18.45 | 12.75 | 85.32 | 1421.9 | 7.95e-06 | 0.000624 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case9241pegase | 0 | STRUMPACK np=1 | fp64 | ok | 54.12 | 26.03 | 25.91 | 106.07 | 1641.1 | 1.65e-14 | 1.44e-12 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case9241pegase | 0 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case9241pegase | 0 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 169.82 | n/a | n/a | False | -1 | unknown |
| case9241pegase | 1 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 16.72 | 0 | 102.86 | 119.58 | 1696.4 | 27323.2 | 14953.8 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 1 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 16.91 | 0 | 226.45 | 243.36 | 3426.9 | 10.64 | 4.912 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 1 | Ginkgo-GMRES-Jacobi | fp32 | ok | 16.76 | 0 | 443.36 | 460.12 | 6471.4 | 0.000414 | 0.92 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 1 | Ginkgo-GMRES-Jacobi | fp64 | ok | 17.24 | 0 | 501.11 | 518.34 | 7319.4 | 0.000413 | 0.92 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 1 | STRUMPACK np=1 | fp32 | ok | 53.51 | 23.49 | 15.67 | 92.67 | 1448.6 | 3.11e-06 | 0.0013 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case9241pegase | 1 | STRUMPACK np=1 | fp64 | ok | 54.9 | 27.7 | 35.39 | 118.00 | 1717.2 | 6.72e-15 | 9.49e-13 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case9241pegase | 1 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case9241pegase | 1 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 172.14 | n/a | n/a | False | -1 | unknown |
| case9241pegase | 6 | Ginkgo-BiCGSTAB-Jacobi | fp32 | ok | 16.63 | 0 | 102.22 | 118.85 | 1686.1 | 20.26 | 11.95 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 6 | Ginkgo-BiCGSTAB-Jacobi | fp64 | ok | 16.98 | 0 | 262.82 | 279.80 | 3937.2 | 0.00309 | 0.688 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 6 | Ginkgo-GMRES-Jacobi | fp32 | ok | 16.69 | 0 | 446.55 | 463.25 | 6511.2 | 0.000889 | 0.606 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 6 | Ginkgo-GMRES-Jacobi | fp64 | ok | 17.18 | 0 | 499.86 | 517.04 | 7284.4 | 0.000889 | 0.606 | False | 1000 | yes_for_matrix_vector_and_itera... |
| case9241pegase | 6 | STRUMPACK np=1 | fp32 | ok | 52.4 | 35.2 | 0.25 | 87.85 | 1403.4 | 1 | 1 | False | 1 | cpu_gpu_hybrid_mpi_dist |
| case9241pegase | 6 | STRUMPACK np=1 | fp64 | ok | 66.66 | 31.84 | 36.42 | 134.92 | 1936.0 | 3.47e-15 | 1.97e-13 | True | 1 | cpu_gpu_hybrid_mpi_dist |
| case9241pegase | 6 | SuperLU_DIST | fp32 | unsupported | 0 | 0 | 0 | 0 | 0 | n/a | n/a | False | -1 | unsupported |
| case9241pegase | 6 | SuperLU_DIST | fp64 | runtime_failed | n/a | n/a | n/a | n/a | 170.86 | n/a | n/a | False | -1 | unknown |

## 8. Runtime Failures

| case | iter | solver | dtype | status | attempted_command | error_tail |
| --- | --- | --- | --- | --- | --- | --- |
| case118 | 0 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case118 | 1 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case118 | 3 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case1354pegase | 0 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case1354pegase | 1 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case1354pegase | 4 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case14 | 0 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case14 | 1 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case14 | 2 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case2869pegase | 0 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case2869pegase | 1 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case2869pegase | 6 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case300 | 0 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case300 | 1 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case300 | 5 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case9241pegase | 0 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case9241pegase | 1 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| case9241pegase | 6 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |
| synthetic_validation | 0 | STRUMPACK | fp64 | timeout | /usr/bin/mpirun.mpich -np 2 exp/20260510/lin_sol/solvers/strumpack/build/strumpack_benchmark... | # Initializing STRUMPACK; WARNING: SLATE is required for full GPU support. Configure with -DTPL_ENABLE_SLATE=ON. Matr... |
| synthetic_validation | 0 | STRUMPACK | fp64 | timeout | /usr/bin/mpirun.mpich -np 4 exp/20260510/lin_sol/solvers/strumpack/build/strumpack_benchmark... | # Initializing STRUMPACK; WARNING: SLATE is required for full GPU support. Configure with -DTPL_ENABLE_SLATE=ON. Matr... |
| synthetic_validation | 0 | SuperLU_DIST | fp64 | runtime_failed | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun -np 1 /wo... | Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec... |

## 9. Direct Solver Comparison

cuDSS remains the cleanest direct-solver default from the combined evidence. It ran all v1 systems in FP64 and FP32, exposed separate analysis/factorization/solve phases, kept a GPU-resident workflow, and directly targets general sparse systems suitable for power-flow Jacobians.

cuSolverSP also ran the v1 systems accurately through the QR path, but it is a monolithic API path in this benchmark and does not provide the same reusable direct factorization structure that cuPF wants for repeated Newton Jacobian solves. The CUDA 12.8 headers observed in v1 also marked older sparse LU/Cholesky paths as deprecated with cuDSS as the replacement, so cuSolverSP is useful NVIDIA context rather than the best integration target.

SuperLU_DIST is still a candidate as an external distributed sparse direct baseline, but not as successful GPU evidence in this pass. The CUDA build completed, MPI was available, and the wrapper built, but every fp64 run failed immediately with the `get_perm_c.c` `Invalid ISPEC` runtime error. Because no solve completed, there is no residual or timing evidence for cuPF suitability beyond installation complexity and the failed ABglobal path.

STRUMPACK is much stronger than v1: the CUDA/MPI build and np=1 wrapper ran all dumped systems. However, the run is host-input/output with internal GPU offload rather than a simple GPU-resident cuPF-style interface, startup warns that SLATE is required for full GPU support, and np=2/np=4 synthetic runs hung. Its median second-pass solver time is competitive in this small single-node run, but the setup complexity and MPI/hybrid data path make it a better external direct-solver baseline than a default embedded cuPF solver.

## 10. Iterative Solver Comparison

Ginkgo produced 76 ok rows across GMRES+Jacobi and BiCGSTAB+Jacobi. It converged on all validation and smaller systems, but convergence degraded on larger PEGASE cases: GMRES converged in 21 of 38 rows, and BiCGSTAB converged in 17 of 38 rows. Several large-case rows reached the configured 1000-iteration limit with residuals too large for Newton linear-solve evidence.

AMGx from v1 showed the same broad pattern: useful small-case convergence, but larger Jacobians were sensitive to the fixed configuration and often hit the iteration limit. These iterative libraries are not rejected simply for speed; the issue is that standalone Newton-Raphson linear solves need consistent residual quality across changing nonsymmetric Jacobians, and the tested fixed preconditioned Krylov configurations did not provide that consistency.

## 11. Final Interpretation

The second pass improves the evidence base without overturning the v1 conclusion. Ginkgo is now a working CUDA iterative baseline, and STRUMPACK is now a working CUDA/MPI direct baseline for single-rank runs. They are valuable comparison points, but neither has the same combination of direct-solver stability, general nonsymmetric sparse Jacobian support, GPU execution model, reusable analysis/factorization structure, and low integration complexity that cuDSS offers for cuPF.

Therefore, cuDSS remains the best default sparse linear solver library for cuPF. STRUMPACK should remain in future reports as an external distributed direct-solver baseline, SuperLU_DIST should remain a candidate only after its runtime driver issue is resolved, and Ginkgo/AMGx should remain iterative-library baselines for cases where preconditioning strategy and convergence policy are the subject of a separate cuITER-style study.

## 12. Reproduction Commands

```bash
cd /workspace/gpu-powerflow
LIN_SOL_WARMUP=3 LIN_SOL_REPEATS=10 LIN_SOL_TIMEOUT=240 python3 exp/20260510/lin_sol/scripts/run_second_pass.py
python3 exp/20260510/lin_sol/scripts/make_report_v2.py
```
