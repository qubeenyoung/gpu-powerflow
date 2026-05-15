# Linear Solver Benchmark Report

## 1. Experiment Purpose

This experiment independently benchmarks GPU sparse linear solver libraries on Newton-Raphson power-flow Jacobian systems of the form `J dx = -F`. The Jacobians are generated from MATPOWER/PGLIB-format cases using the standard PYPOWER/MATPOWER construction: `Ybus`, `Sbus`, PV/PQ bus partitioning, mismatch vector, and the four Jacobian blocks `dP/dVa`, `dP/dVm`, `dQ/dVa`, and `dQ/dVm`.

The purpose is to identify the most suitable sparse linear solver library for cuPF. cuDSS is treated as the main sparse direct GPU baseline because the power-flow Newton step needs robust solves for general nonsymmetric Jacobian matrices and benefits from GPU residency plus reusable symbolic analysis when the sparsity pattern is stable.

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
| MPI C++ | not found |

## 3. Dataset Table

| case | iteration | num_bus | num_pv | num_pq | matrix_size | nnz | rhs_norm | pattern_hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case118 | 0 | 118 | 53 | 64 | 181x181 | 1051 | 1.421 | 96d96782c0b757d0 |
| case118 | 1 | 118 | 53 | 64 | 181x181 | 1051 | 0.02451 | 96d96782c0b757d0 |
| case118 | 3 | 118 | 53 | 64 | 181x181 | 1051 | 1.605e-12 | 96d96782c0b757d0 |
| case1354pegase | 0 | 1354 | 259 | 1094 | 2447x2447 | 15803 | 29.83 | 802788188a259605 |
| case1354pegase | 1 | 1354 | 259 | 1094 | 2447x2447 | 15803 | 1.083 | 802788188a259605 |
| case1354pegase | 4 | 1354 | 259 | 1094 | 2447x2447 | 15803 | 1.897e-11 | 802788188a259605 |
| case14 | 0 | 14 | 4 | 9 | 22x22 | 146 | 0.04547 | a6d546a32c79e8b9 |
| case14 | 1 | 14 | 4 | 9 | 22x22 | 146 | 5.762e-05 | a6d546a32c79e8b9 |
| case14 | 2 | 14 | 4 | 9 | 22x22 | 146 | 1.344e-10 | a6d546a32c79e8b9 |
| case2869pegase | 0 | 2869 | 509 | 2359 | 5227x5227 | 36591 | 88.89 | df9b9e249b166658 |
| case2869pegase | 1 | 2869 | 509 | 2359 | 5227x5227 | 36591 | 89.43 | df9b9e249b166658 |
| case2869pegase | 6 | 2869 | 509 | 2359 | 5227x5227 | 36591 | 3.055e-09 | df9b9e249b166658 |
| case300 | 0 | 300 | 68 | 231 | 530x530 | 3736 | 13.15 | 9c1ad799d742df26 |
| case300 | 1 | 300 | 68 | 231 | 530x530 | 3736 | 4.02 | 9c1ad799d742df26 |
| case300 | 5 | 300 | 68 | 231 | 530x530 | 3736 | 2.596e-12 | 9c1ad799d742df26 |
| case9241pegase | 0 | 9241 | 1444 | 7796 | 17036x17036 | 129412 | 108.6 | e7e4c226d863b9c0 |
| case9241pegase | 1 | 9241 | 1444 | 7796 | 17036x17036 | 129412 | 88.41 | e7e4c226d863b9c0 |
| case9241pegase | 6 | 9241 | 1444 | 7796 | 17036x17036 | 129412 | 3.022e-09 | e7e4c226d863b9c0 |
| synthetic_validation | 0 | 0 | 0 | 0 | 5x5 | 16 | 2.424 | a3dfe5a39556fae4 |

## 4. Solver Availability

| solver | type | build_status | version | CUDA_enabled | GPU_resident_status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| cuDSS | GPU sparse direct | ok | 0.7.1 | yes | yes | cuDSS sparse direct LU path; analysis is run once and reused across repeated numeric factorization/solve calls for the same sparsity pattern. |
| cuSolverSP | NVIDIA sparse direct QR | ok | 11.7.3.90 | yes | yes_for_QR_API | Uses cuSolverSP csrlsvqr for general sparse systems. The CUDA 12.8 LU and Cholesky sparse solve APIs are marked deprecated in headers with cuDSS as the replacement; QR is monoli... |
| AMGx | GPU iterative AMG/GMRES | ok | 2.4.0 api 1.0 | yes | yes | AMGx iterative GMRES/AMG configuration; setup is reported as analysis_ms and solve_ms is repeated solve time from zero initial guess. |
| Ginkgo | GPU iterative Krylov | unavailable | unavailable | unknown | unavailable | Ginkgo CUDA benchmark wrapper placeholder. Ginkgo was not found as an installed CUDA-enabled library in this environment. Install log: /workspace/gpu-powerflow/exp/20260510/lin_... |
| SuperLU_DIST | distributed sparse direct | unavailable | unavailable | requires CUDA build | unavailable | SuperLU_DIST GPU-capable sparse direct candidate is unavailable until a CUDA-enabled SuperLU_DIST build is present. Install log: /workspace/gpu-powerflow/exp/20260510/lin_sol/lo... |
| STRUMPACK | sparse direct / multifrontal | unavailable | unavailable | requires CUDA build | unavailable | STRUMPACK sparse direct candidate is unavailable until a CUDA-enabled STRUMPACK build is present. Install log: /workspace/gpu-powerflow/exp/20260510/lin_sol/logs/install_strumpa... |

### Install/Build Logs

| log | status | tail |
| --- | --- | --- |
| install_amgx.log | available | [amgx] prefix: /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/amgx/install [amgx] status=available [amgx] include=/usr/local/include/amgx_c.h -rw-r--r-- 1 root root 5... |
| install_ginkgo.log | unavailable | [ginkgo] started 2026-05-10T08:52:59Z [ginkgo] prefix: /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/ginkgo/install [ginkgo] status=unavailable [ginkgo] failure=no e... |
| install_superlu_dist.log | build_failed | [superlu_dist] status=build_failed [superlu_dist] failure=MPI C/C++ compilers not found on PATH [superlu_dist] missing_dependency=mpicc/mpicxx [superlu_dist] likely_system_comma... |
| install_strumpack.log | build_failed | [strumpack] status=build_failed [strumpack] failure=MPI C/C++ compilers not found on PATH [strumpack] missing_dependency=mpicc/mpicxx [strumpack] likely_system_command=sudo apt-... |

## 5. Main Performance Table

| case | iteration | solver | dtype | analysis_ms | factorization_ms | solve_ms | total_solver_ms | total_end_to_end_ms | relative_residual_2 | relative_error_to_x_ref_2 | converged | num_iterations | gpu_resident_after_initial_load |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_validation | 0 | cuDSS | fp64 | 13.42 | 0.0337 | 0.03017 | 13.48 | 261.6 | 4.58e-17 | 9.96e-17 | True | 1 | yes |
| synthetic_validation | 0 | cuDSS | fp32 | 13.43 | 0.03142 | 0.02707 | 13.48 | 178.9 | 2.78e-08 | 2.84e-08 | True | 1 | yes |
| synthetic_validation | 0 | cuSolverSP | fp64 | 0 | 0 | 0.2276 | 0.2276 | 155.9 | 3.46e-16 | 3.2e-16 | True | 1 | yes_for_QR_API |
| synthetic_validation | 0 | cuSolverSP | fp32 | 0 | 0 | 0.1874 | 0.1874 | 155.6 | 1.99e-07 | 2.52e-07 | True | 1 | yes_for_QR_API |
| synthetic_validation | 0 | AMGx | fp64 | 5.058 | 0 | 0.179 | 5.237 | 271.6 | 2.67e-16 | 3.69e-16 | True | 1 | yes |
| synthetic_validation | 0 | AMGx | fp32 | 5.061 | 0 | 0.1646 | 5.225 | 254.7 | 1.89e-07 | 1.76e-07 | True | 1 | yes |
| synthetic_validation | 0 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| synthetic_validation | 0 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| synthetic_validation | 0 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| synthetic_validation | 0 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| synthetic_validation | 0 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| synthetic_validation | 0 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 0 | cuDSS | fp64 | 13.94 | 0.08019 | 0.06235 | 14.08 | 186.8 | 1.89e-16 | 8.9e-16 | True | 1 | yes |
| case118 | 0 | cuDSS | fp32 | 13.92 | 0.06574 | 0.05328 | 14.04 | 180.7 | 1.22e-07 | 1.75e-06 | True | 1 | yes |
| case118 | 0 | cuSolverSP | fp64 | 0 | 0 | 2.334 | 2.334 | 184.3 | 9.34e-16 | 1.94e-14 | True | 1 | yes_for_QR_API |
| case118 | 0 | cuSolverSP | fp32 | 0 | 0 | 1.22 | 1.22 | 171.1 | 2.75e-07 | 2.2e-06 | True | 1 | yes_for_QR_API |
| case118 | 0 | AMGx | fp64 | 7.868 | 0 | 0.302 | 8.17 | 259.1 | 7.9e-16 | 4.68e-15 | True | 1 | yes |
| case118 | 0 | AMGx | fp32 | 8.639 | 0 | 0.2428 | 8.882 | 253.5 | 3.5e-07 | 5.54e-06 | True | 1 | yes |
| case118 | 0 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 0 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 0 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 0 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 0 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 0 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 1 | cuDSS | fp64 | 13.97 | 0.08051 | 0.06234 | 14.11 | 181.6 | 5.92e-16 | 2.57e-15 | True | 1 | yes |
| case118 | 1 | cuDSS | fp32 | 13.96 | 0.06451 | 0.05335 | 14.07 | 185.3 | 2.37e-07 | 6.4e-07 | True | 1 | yes |
| case118 | 1 | cuSolverSP | fp64 | 0 | 0 | 2.319 | 2.319 | 183.2 | 1.18e-15 | 2.86e-14 | True | 1 | yes_for_QR_API |
| case118 | 1 | cuSolverSP | fp32 | 0 | 0 | 1.225 | 1.225 | 168.4 | 8.1e-07 | 1.12e-05 | True | 1 | yes_for_QR_API |
| case118 | 1 | AMGx | fp64 | 9.608 | 0 | 0.3018 | 9.909 | 256.9 | 6.33e-16 | 7.7e-15 | True | 1 | yes |
| case118 | 1 | AMGx | fp32 | 6.3 | 0 | 0.2447 | 6.544 | 254.6 | 6.08e-07 | 2.11e-06 | True | 1 | yes |
| case118 | 1 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 1 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 1 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 1 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 1 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 1 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 3 | cuDSS | fp64 | 13.89 | 0.08031 | 0.06213 | 14.04 | 180.8 | 9.02e-16 | 3.95e-15 | True | 1 | yes |
| case118 | 3 | cuDSS | fp32 | 13.9 | 0.06513 | 0.05306 | 14.02 | 186.1 | 3.22e-07 | 3.59e-06 | True | 1 | yes |
| case118 | 3 | cuSolverSP | fp64 | 0 | 0 | 2.328 | 2.328 | 183.7 | 1.64e-15 | 1.66e-14 | True | 1 | yes_for_QR_API |
| case118 | 3 | cuSolverSP | fp32 | 0 | 0 | 1.224 | 1.224 | 169.7 | 1.28e-06 | 1.84e-05 | True | 1 | yes_for_QR_API |
| case118 | 3 | AMGx | fp64 | 9.656 | 0 | 0.3011 | 9.957 | 256.5 | 1.05e-15 | 5.07e-15 | True | 1 | yes |
| case118 | 3 | AMGx | fp32 | 7.881 | 0 | 0.2439 | 8.125 | 260.4 | 4.43e-07 | 6.95e-07 | True | 1 | yes |
| case118 | 3 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 3 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 3 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 3 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 3 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case118 | 3 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 0 | cuDSS | fp64 | 19.06 | 0.2416 | 0.1496 | 19.45 | 201.1 | 1.09e-15 | 6.17e-14 | True | 1 | yes |
| case1354pegase | 0 | cuDSS | fp32 | 19.05 | 0.1698 | 0.1143 | 19.33 | 195.1 | 3.99e-07 | 1.07e-05 | True | 1 | yes |
| case1354pegase | 0 | cuSolverSP | fp64 | 0 | 0 | 21.96 | 21.96 | 451 | 2.28e-15 | 7.02e-14 | True | 1 | yes_for_QR_API |
| case1354pegase | 0 | cuSolverSP | fp32 | 0 | 0 | 12.48 | 12.48 | 311.6 | 1.33e-06 | 2.51e-05 | True | 1 | yes_for_QR_API |
| case1354pegase | 0 | AMGx | fp64 | 20.61 | 0 | 97.27 | 117.9 | 1524 | 0.000532 | 0.139 | False | 200 | yes |
| case1354pegase | 0 | AMGx | fp32 | 14.4 | 0 | 66.42 | 80.82 | 1121 | 0.000532 | 0.139 | False | 200 | yes |
| case1354pegase | 0 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 0 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 0 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 0 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 0 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 0 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 1 | cuDSS | fp64 | 17.64 | 0.2093 | 0.1296 | 17.98 | 183 | 6.05e-15 | 9.38e-15 | True | 1 | yes |
| case1354pegase | 1 | cuDSS | fp32 | 17.59 | 0.148 | 0.09891 | 17.83 | 180.5 | 2.3e-06 | 6.09e-06 | True | 1 | yes |
| case1354pegase | 1 | cuSolverSP | fp64 | 0 | 0 | 19.84 | 19.84 | 415.1 | 1.79e-14 | 2.64e-13 | True | 1 | yes_for_QR_API |
| case1354pegase | 1 | cuSolverSP | fp32 | 0 | 0 | 13.11 | 13.11 | 329.5 | 9.02e-06 | 0.000148 | True | 1 | yes_for_QR_API |
| case1354pegase | 1 | AMGx | fp64 | 13.75 | 0 | 97.64 | 111.4 | 1527 | 0.00428 | 0.251 | False | 200 | yes |
| case1354pegase | 1 | AMGx | fp32 | 16.33 | 0 | 66.31 | 82.64 | 1132 | 0.00428 | 0.25 | False | 200 | yes |
| case1354pegase | 1 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 1 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 1 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 1 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 1 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 1 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 4 | cuDSS | fp64 | 17.4 | 0.209 | 0.13 | 17.74 | 189.7 | 1.89e-14 | 7.37e-14 | True | 1 | yes |
| case1354pegase | 4 | cuDSS | fp32 | 17.44 | 0.1484 | 0.09875 | 17.68 | 180.5 | 1.02e-05 | 7.77e-05 | True | 1 | yes |
| case1354pegase | 4 | cuSolverSP | fp64 | 0 | 0 | 19.82 | 19.82 | 413 | 5.48e-14 | 1.42e-13 | True | 1 | yes_for_QR_API |
| case1354pegase | 4 | cuSolverSP | fp32 | 0 | 0 | 12.49 | 12.49 | 312.3 | 2.93e-05 | 0.000179 | True | 1 | yes_for_QR_API |
| case1354pegase | 4 | AMGx | fp64 | 21.62 | 0 | 97.62 | 119.2 | 1543 | 0.0178 | 0.502 | False | 200 | yes |
| case1354pegase | 4 | AMGx | fp32 | 23.5 | 0 | 66.29 | 89.79 | 1131 | 0.0178 | 0.502 | False | 200 | yes |
| case1354pegase | 4 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 4 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 4 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 4 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 4 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case1354pegase | 4 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 0 | cuDSS | fp64 | 12.68 | 0.04452 | 0.03502 | 12.76 | 177.3 | 2.19e-16 | 9.15e-16 | True | 1 | yes |
| case14 | 0 | cuDSS | fp32 | 12.49 | 0.03799 | 0.03113 | 12.56 | 169.7 | 3.23e-08 | 4.39e-07 | True | 1 | yes |
| case14 | 0 | cuSolverSP | fp64 | 0 | 0 | 0.4263 | 0.4263 | 148 | 2.81e-16 | 1.07e-15 | True | 1 | yes_for_QR_API |
| case14 | 0 | cuSolverSP | fp32 | 0 | 0 | 0.2794 | 0.2794 | 145.5 | 1.42e-07 | 5.44e-07 | True | 1 | yes_for_QR_API |
| case14 | 0 | AMGx | fp64 | 5.126 | 0 | 0.185 | 5.311 | 242.8 | 8.29e-16 | 7.19e-16 | True | 1 | yes |
| case14 | 0 | AMGx | fp32 | 5.122 | 0 | 0.172 | 5.294 | 261.1 | 3.38e-07 | 8.11e-07 | True | 1 | yes |
| case14 | 0 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 0 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 0 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 0 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 0 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 0 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 1 | cuDSS | fp64 | 13.47 | 0.04813 | 0.03934 | 13.56 | 180.1 | 3.22e-16 | 4.33e-16 | True | 1 | yes |
| case14 | 1 | cuDSS | fp32 | 13.48 | 0.0416 | 0.0354 | 13.56 | 183.1 | 1.55e-07 | 1.95e-07 | True | 1 | yes |
| case14 | 1 | cuSolverSP | fp64 | 0 | 0 | 0.468 | 0.468 | 164.9 | 8.67e-16 | 3.06e-15 | True | 1 | yes_for_QR_API |
| case14 | 1 | cuSolverSP | fp32 | 0 | 0 | 0.3018 | 0.3018 | 157.2 | 6.28e-07 | 6.5e-07 | True | 1 | yes_for_QR_API |
| case14 | 1 | AMGx | fp64 | 5.102 | 0 | 0.1882 | 5.29 | 253.8 | 6.55e-16 | 1.4e-15 | True | 1 | yes |
| case14 | 1 | AMGx | fp32 | 5.048 | 0 | 0.1692 | 5.217 | 264 | 2.38e-07 | 3.43e-07 | True | 1 | yes |
| case14 | 1 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 1 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 1 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 1 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 1 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 1 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 2 | cuDSS | fp64 | 13.46 | 0.04895 | 0.03901 | 13.55 | 179.5 | 3.52e-16 | 1.06e-15 | True | 1 | yes |
| case14 | 2 | cuDSS | fp32 | 13.52 | 0.04148 | 0.03518 | 13.59 | 179.5 | 1.81e-07 | 4.05e-07 | True | 1 | yes |
| case14 | 2 | cuSolverSP | fp64 | 0 | 0 | 0.4652 | 0.4652 | 164.6 | 2.18e-15 | 4.29e-15 | True | 1 | yes_for_QR_API |
| case14 | 2 | cuSolverSP | fp32 | 0 | 0 | 0.3032 | 0.3032 | 161.3 | 7.39e-07 | 2.03e-06 | True | 1 | yes_for_QR_API |
| case14 | 2 | AMGx | fp64 | 5.151 | 0 | 0.189 | 5.34 | 254.8 | 6.47e-16 | 4.32e-16 | True | 1 | yes |
| case14 | 2 | AMGx | fp32 | 5.077 | 0 | 0.1703 | 5.247 | 260.8 | 2.86e-07 | 2.85e-07 | True | 1 | yes |
| case14 | 2 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 2 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 2 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 2 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 2 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case14 | 2 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 0 | cuDSS | fp64 | 23.99 | 0.2958 | 0.1802 | 24.47 | 210.1 | 4.98e-15 | 3.12e-13 | True | 1 | yes |
| case2869pegase | 0 | cuDSS | fp32 | 24.01 | 0.2132 | 0.1326 | 24.35 | 207.2 | 2.65e-06 | 7.12e-05 | True | 1 | yes |
| case2869pegase | 0 | cuSolverSP | fp64 | 0 | 0 | 50.98 | 50.98 | 845.2 | 1.15e-14 | 1.26e-12 | True | 1 | yes_for_QR_API |
| case2869pegase | 0 | cuSolverSP | fp32 | 0 | 0 | 31.27 | 31.27 | 563.9 | 6.84e-06 | 0.000129 | True | 1 | yes_for_QR_API |
| case2869pegase | 0 | AMGx | fp64 | 13.4 | 0 | 102.2 | 115.6 | 1596 | 0.00267 | 0.549 | False | 200 | yes |
| case2869pegase | 0 | AMGx | fp32 | 21.33 | 0 | 72.54 | 93.86 | 1230 | 0.00267 | 0.549 | False | 200 | yes |
| case2869pegase | 0 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 0 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 0 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 0 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 0 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 0 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 1 | cuDSS | fp64 | 22.66 | 0.2591 | 0.1567 | 23.07 | 203.6 | 2.11e-15 | 3.57e-13 | True | 1 | yes |
| case2869pegase | 1 | cuDSS | fp32 | 22.72 | 0.1869 | 0.1159 | 23.03 | 200.6 | 7.72e-07 | 3.16e-05 | True | 1 | yes |
| case2869pegase | 1 | cuSolverSP | fp64 | 0 | 0 | 49.77 | 49.77 | 809.6 | 4.68e-15 | 3.83e-13 | True | 1 | yes_for_QR_API |
| case2869pegase | 1 | cuSolverSP | fp32 | 0 | 0 | 31.29 | 31.29 | 564.2 | 2.38e-06 | 0.000323 | True | 1 | yes_for_QR_API |
| case2869pegase | 1 | AMGx | fp64 | 23.56 | 0 | 102.3 | 125.8 | 1594 | 0.00094 | 0.607 | False | 200 | yes |
| case2869pegase | 1 | AMGx | fp32 | 21.17 | 0 | 72.09 | 93.26 | 1213 | 0.00094 | 0.607 | False | 200 | yes |
| case2869pegase | 1 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 1 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 1 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 1 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 1 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 1 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 6 | cuDSS | fp64 | 22.6 | 0.2583 | 0.1572 | 23.01 | 196.9 | 2.43e-15 | 1.57e-14 | True | 1 | yes |
| case2869pegase | 6 | cuDSS | fp32 | 22.64 | 0.1877 | 0.1155 | 22.95 | 198.9 | 1.36e-06 | 1.14e-05 | True | 1 | yes |
| case2869pegase | 6 | cuSolverSP | fp64 | 0 | 0 | 49.75 | 49.75 | 804.9 | 7.37e-15 | 5.86e-13 | True | 1 | yes_for_QR_API |
| case2869pegase | 6 | cuSolverSP | fp32 | 0 | 0 | 31.26 | 31.26 | 565.6 | 4.1e-06 | 0.000642 | True | 1 | yes_for_QR_API |
| case2869pegase | 6 | AMGx | fp64 | 17.02 | 0 | 102.4 | 119.4 | 1603 | 0.00185 | 0.572 | False | 200 | yes |
| case2869pegase | 6 | AMGx | fp32 | 14.31 | 0 | 71.61 | 85.92 | 1208 | 0.00185 | 0.572 | False | 200 | yes |
| case2869pegase | 6 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 6 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 6 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 6 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 6 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case2869pegase | 6 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 0 | cuDSS | fp64 | 13.81 | 0.1234 | 0.0769 | 14.01 | 182.1 | 8.91e-16 | 7.81e-15 | True | 1 | yes |
| case300 | 0 | cuDSS | fp32 | 13.7 | 0.094 | 0.05949 | 13.86 | 171.8 | 2.02e-07 | 2.22e-05 | True | 1 | yes |
| case300 | 0 | cuSolverSP | fp64 | 0 | 0 | 5.107 | 5.107 | 215.6 | 1.66e-15 | 1e-13 | True | 1 | yes_for_QR_API |
| case300 | 0 | cuSolverSP | fp32 | 0 | 0 | 2.736 | 2.736 | 187.7 | 8.02e-07 | 5.7e-05 | True | 1 | yes_for_QR_API |
| case300 | 0 | AMGx | fp64 | 21.3 | 0 | 60.87 | 82.17 | 1060 | 9.71e-09 | 1.02e-06 | True | 137 | yes |
| case300 | 0 | AMGx | fp32 | 14.34 | 0 | 24.58 | 38.92 | 581 | 9.69e-07 | 0.000145 | True | 83 | yes |
| case300 | 0 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 0 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 0 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 0 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 0 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 0 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 1 | cuDSS | fp64 | 13.71 | 0.1241 | 0.0768 | 13.91 | 175.8 | 5.68e-16 | 1.04e-15 | True | 1 | yes |
| case300 | 1 | cuDSS | fp32 | 13.71 | 0.09379 | 0.0596 | 13.87 | 184.3 | 3e-07 | 5.14e-06 | True | 1 | yes |
| case300 | 1 | cuSolverSP | fp64 | 0 | 0 | 5.115 | 5.115 | 209.6 | 1.02e-15 | 7.41e-14 | True | 1 | yes_for_QR_API |
| case300 | 1 | cuSolverSP | fp32 | 0 | 0 | 2.74 | 2.74 | 182.5 | 9.94e-07 | 3.11e-05 | True | 1 | yes_for_QR_API |
| case300 | 1 | AMGx | fp64 | 12.32 | 0 | 53.19 | 65.52 | 967 | 7.97e-09 | 4.7e-07 | True | 116 | yes |
| case300 | 1 | AMGx | fp32 | 11.47 | 0 | 19.5 | 30.97 | 530.1 | 9.47e-07 | 4.87e-05 | True | 66 | yes |
| case300 | 1 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 1 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 1 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 1 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 1 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 1 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 5 | cuDSS | fp64 | 13.6 | 0.1238 | 0.0769 | 13.8 | 172.2 | 2.94e-15 | 1.14e-14 | True | 1 | yes |
| case300 | 5 | cuDSS | fp32 | 13.7 | 0.09383 | 0.0596 | 13.85 | 171.9 | 6.06e-07 | 5.09e-06 | True | 1 | yes |
| case300 | 5 | cuSolverSP | fp64 | 0 | 0 | 5.121 | 5.121 | 210.5 | 6.06e-15 | 7.67e-13 | True | 1 | yes_for_QR_API |
| case300 | 5 | cuSolverSP | fp32 | 0 | 0 | 2.736 | 2.736 | 180 | 1.84e-06 | 0.000367 | True | 1 | yes_for_QR_API |
| case300 | 5 | AMGx | fp64 | 12.73 | 0 | 63.22 | 75.94 | 1105 | 9.55e-09 | 9.9e-07 | True | 141 | yes |
| case300 | 5 | AMGx | fp32 | 14.27 | 0 | 27.59 | 41.86 | 617.8 | 1.52e-06 | 3.02e-05 | True | 90 | yes |
| case300 | 5 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 5 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 5 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 5 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 5 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case300 | 5 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 0 | cuDSS | fp64 | 46.19 | 0.7876 | 0.3041 | 47.28 | 264.8 | 1.28e-14 | 6.86e-13 | True | 1 | yes |
| case9241pegase | 0 | cuDSS | fp32 | 46.35 | 0.4914 | 0.1964 | 47.04 | 260.3 | 6.47e-06 | 7.92e-05 | True | 1 | yes |
| case9241pegase | 0 | cuSolverSP | fp64 | 0 | 0 | 191.9 | 191.9 | 2702 | 4.08e-14 | 3.49e-12 | True | 1 | yes_for_QR_API |
| case9241pegase | 0 | cuSolverSP | fp32 | 0 | 0 | 135.3 | 135.3 | 1958 | 2.17e-05 | 0.00327 | True | 1 | yes_for_QR_API |
| case9241pegase | 0 | AMGx | fp64 | 23.69 | 0 | 120.6 | 144.3 | 1894 | 0.102 | 0.963 | False | 200 | yes |
| case9241pegase | 0 | AMGx | fp32 | 17.04 | 0 | 97.3 | 114.3 | 1577 | 0.102 | 0.963 | False | 200 | yes |
| case9241pegase | 0 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 0 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 0 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 0 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 0 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 0 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 1 | cuDSS | fp64 | 45.92 | 0.781 | 0.3021 | 47.01 | 266.1 | 5.12e-15 | 4.74e-13 | True | 1 | yes |
| case9241pegase | 1 | cuDSS | fp32 | 46.09 | 0.4857 | 0.1946 | 46.77 | 258.2 | 2.57e-06 | 0.000159 | True | 1 | yes |
| case9241pegase | 1 | cuSolverSP | fp64 | 0 | 0 | 191.9 | 191.9 | 2696 | 1.71e-14 | 1.06e-12 | True | 1 | yes_for_QR_API |
| case9241pegase | 1 | cuSolverSP | fp32 | 0 | 0 | 135.2 | 135.2 | 1971 | 9.21e-06 | 0.00722 | True | 1 | yes_for_QR_API |
| case9241pegase | 1 | AMGx | fp64 | 22.69 | 0 | 114.2 | 136.8 | 1813 | 0.0203 | 0.958 | False | 200 | yes |
| case9241pegase | 1 | AMGx | fp32 | 23.04 | 0 | 95.04 | 118.1 | 1548 | 0.0204 | 0.958 | False | 200 | yes |
| case9241pegase | 1 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 1 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 1 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 1 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 1 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 1 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 6 | cuDSS | fp64 | 46.08 | 0.7826 | 0.3022 | 47.16 | 267.2 | 3.37e-15 | 1.21e-13 | True | 1 | yes |
| case9241pegase | 6 | cuDSS | fp32 | 46.32 | 0.4852 | 0.1957 | 47 | 261 | 1.55e-06 | 2.46e-05 | True | 1 | yes |
| case9241pegase | 6 | cuSolverSP | fp64 | 0 | 0 | 191.7 | 191.7 | 2705 | 9.18e-15 | 2.99e-12 | True | 1 | yes_for_QR_API |
| case9241pegase | 6 | cuSolverSP | fp32 | 0 | 0 | 135.1 | 135.1 | 1957 | 4.88e-06 | 0.00204 | True | 1 | yes_for_QR_API |
| case9241pegase | 6 | AMGx | fp64 | 22.67 | 0 | 113.6 | 136.3 | 1785 | 0.0577 | 0.683 | False | 200 | yes |
| case9241pegase | 6 | AMGx | fp32 | 18.43 | 0 | 96.22 | 114.6 | 1560 | 0.0577 | 0.683 | False | 200 | yes |
| case9241pegase | 6 | Ginkgo | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 6 | Ginkgo | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 6 | SuperLU_DIST | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 6 | SuperLU_DIST | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 6 | STRUMPACK | fp64 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |
| case9241pegase | 6 | STRUMPACK | fp32 | nan | nan | nan | nan | nan | nan | nan | False | -1 | unavailable |

## 6. Direct Solver Comparison

cuDSS produced converged direct-solver rows and exposes separate analysis, factorization, and solve phases. cuSolverSP QR produced converged rows where reported, but the tested API is monolithic and does not expose a reusable factorization phase. The distributed direct alternatives were not runnable in this environment: STRUMPACK, SuperLU_DIST.

Direct-solver interpretation should focus on stability, support for general nonsymmetric sparse systems, GPU residency after initial matrix upload, setup complexity, factorization/solve phase visibility, and integration cost for cuPF. The local CUDA 12.8 headers mark the cuSolverSP sparse LU host path and cuSolverRF analyze/refactor functions as deprecated with cuDSS indicated as the replacement path, so cuSolverSP/RF claims in this report are limited to observed API availability and the QR wrapper result.

## 7. Iterative Solver Comparison

AMGx converged on 20/38 tested rows with the fixed GMRES+AMG configuration. Ginkgo statuses: unavailable.

AMGx and Ginkgo are not rejected merely for speed. The evidence to check is convergence status, iteration count, final residual, sensitivity to the fixed configuration, and end-to-end setup time. A Newton-Raphson production solver needs reliable residual quality across changing Jacobians; any iterative method that is configuration-sensitive or fails on representative systems remains a candidate for separate cuITER-style experiments rather than the default cuPF sparse direct backend.

## 8. Final Interpretation

The current evidence selects cuDSS when its rows are available and converged because it gives the best integration shape for cuPF: direct factorization for general sparse Jacobians, GPU-resident data after upload, explicit analysis/factorization/solve phases, and symbolic-analysis reuse for repeated systems with the same sparsity pattern.

SuperLU_DIST and STRUMPACK remain direct-solver alternatives for future work, but in this environment their availability is blocked by local build/runtime dependencies recorded above. Their likely integration overhead includes MPI launch/runtime management and host/device or hybrid data-residency decisions that must be measured before considering them as cuPF defaults.

cuSolverSP/RF remains an NVIDIA alternative only to the extent supported by the local CUDA Toolkit APIs. The benchmark uses cuSolverSP QR for general CSR systems and records the lack of separated factorization timing in that path; deprecated or unsupported Cholesky paths are not used for nonsymmetric Jacobians.

## 9. Documentation Checked

- NVIDIA cuDSS documentation: https://docs.nvidia.com/cuda/cudss/index.html
- NVIDIA CUDA 12.8 release notes for cuSolverSP/RF deprecations: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
- NVIDIA cuSOLVER documentation: https://docs.nvidia.com/cuda/archive/12.8.2/cusolver/index.html

No fabricated numbers are included. Missing or failed solver rows are reported with status, logs, and notes instead of being silently omitted.
