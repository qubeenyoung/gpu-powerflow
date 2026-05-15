# Linear Solver Large-Case SuperLU_DIST Diagnosis v4

## Scope

This v4 pass reruns the representative FP64 comparison on the dumped power-flow Jacobian systems with matrix size at least 5K. It focuses on `case2869pegase` and `case9241pegase`, and diagnoses why the previously fixed SuperLU_DIST path is slow.

No production cuPF source code was modified. New outputs were written under `exp/20260510/lin_sol/measurement_audit/results/` and this report.

## Case Selection

`case1354pegase` is not used as a representative large case because its iteration-0 Jacobian is only 2447 x 2447 with 15803 nonzeros, below the requested 5K matrix-size threshold.

The representative cases are:

| case | matrix size | nnz | reason |
| --- | ---: | ---: | --- |
| case2869pegase | 5227 x 5227 | 36591 | First dumped case above 5K; primary table. |
| case9241pegase | 17036 x 17036 | 129412 | Larger confirmation case; secondary table. |

## MPI Consistency

SuperLU_DIST v4 runs used the local MPICH compiler and launcher pair that matches the linked SuperLU_DIST build.

- `mpicc`: `gcc -I/workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/build/../install/include -L/workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/build/../install/lib -Wl,-rpath -Wl,/workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/build/../install/lib -Wl,--enable-new-dtags -lmpi`
- `mpicxx`: `g++ -I/workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/build/../install/include -L/workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/build/../install/lib -lmpicxx -Wl,-rpath -Wl,/workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/build/../install/lib -Wl,--enable-new-dtags -lmpi`
- `mpirun`: `HYDRA build details:`
- `ldd_superlu_phase`: `linux-vdso.so.1 (0x00007ffc541a3000)`

## Primary Representative Table

`case2869pegase`, iteration 0, FP64. For Ginkgo and SuperLU_DIST, the table selects the best rerun configuration by convergence, scaled residual acceptability, then solver time.

| solver | config | analysis ms | factor ms | solve ms | solver ms | end-to-end ms | conv | iters | rel resid | abs resid | scaled resid | rel err | phase visibility | GPU residency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cuDSS | default | 24.059 | 2.958e-01 | 1.798e-01 | 24.534 | 299.008 | yes | 1.000 | 4.449e-15 | 3.955e-13 | 4.449e-15 | 1.507e-13 | analysis_factor_solve | yes |
| cuSolverSP | default | 0.000e+00 | 0.000e+00 | 50.028 | 50.028 | 836.876 | yes | 1.000 | 1.149e-14 | 1.021e-12 | 1.149e-14 | 1.258e-12 | monolithic_qr | yes_for_QR_API |
| AMGx | GMRES+AMG(BlockJacobi) | 16.032 | 0.000e+00 | 101.806 | 117.838 | 1,591.0 | no | 200.000 | 2.671e-03 | 2.374e-01 | 2.671e-03 | 5.486e-01 | setup_solve_iterative | yes |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab | 15.343 | 0.000e+00 | 182.938 | 198.281 | 2,786.0 | no | 1,000.0 | 6.728e-05 | 5.981e-03 | 6.728e-05 | 5.019e-02 | setup_solve_iterative | yes_for_matrix_vector_and_iterations |
| STRUMPACK | np=1 | 11.162 | 12.211 | 9.093 | 32.466 | 584.769 | yes | 1.000 | 6.001e-15 | 5.335e-13 | 6.001e-15 | 2.741e-13 | analysis_factor_solve_wall | cpu_gpu_hybrid_mpi_dist |
| SuperLU_DIST | MMD_AT_PLUS_A+LargeDiag_MC64 np=1 | 11.448 | 159.880 | 5.766e-01 | 171.988 | 197.054 | yes | 1.000 | 4.534e-15 | 4.030e-13 | 4.534e-15 | 4.519e-13 | SuperLUStat_t_internal_plus_wrapper_wall | cpu_gpu_hybrid_abglobal |

## Direct Solver Comparison

| solver | config | analysis ms | factor ms | solve ms | solver ms | rel resid | scaled resid | phase visibility | GPU residency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cuDSS | default | 24.059 | 2.958e-01 | 1.798e-01 | 24.534 | 4.449e-15 | 4.449e-15 | analysis_factor_solve | yes |
| cuSolverSP | default | 0.000e+00 | 0.000e+00 | 50.028 | 50.028 | 1.149e-14 | 1.149e-14 | monolithic_qr | yes_for_QR_API |
| STRUMPACK | np=1 | 11.162 | 12.211 | 9.093 | 32.466 | 6.001e-15 | 6.001e-15 | analysis_factor_solve_wall | cpu_gpu_hybrid_mpi_dist |
| SuperLU_DIST | MMD_AT_PLUS_A+LargeDiag_MC64 np=1 | 11.448 | 159.880 | 5.766e-01 | 171.988 | 4.534e-15 | 4.534e-15 | SuperLUStat_t_internal_plus_wrapper_wall | cpu_gpu_hybrid_abglobal |

cuDSS remains the cleanest direct GPU baseline for cuPF-style repeated Newton solves because it exposes analysis, factorization, and solve phases and keeps the sparse solve path GPU-oriented. cuSolverSP is a valid NVIDIA monolithic sparse QR comparison, but it is not equivalent to reusable numeric factorization. STRUMPACK is a valid external MPI/hybrid direct baseline at `np=1`; it has host/distributed inputs with internal GPU offload. SuperLU_DIST now solves accurately with supported permutations. Its best MMD ordering is a meaningful external direct-solver baseline, while the old NATURAL ordering is the slow diagnostic path.

## Iterative Solver Comparison

| solver | config | setup ms | solve ms | solver ms | conv | iters | rel resid | scaled resid | rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMGx | GMRES+AMG(BlockJacobi) | 16.032 | 101.806 | 117.838 | no | 200.000 | 2.671e-03 | 2.671e-03 | 5.486e-01 |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab | 15.343 | 182.938 | 198.281 | no | 1,000.0 | 6.728e-05 | 6.728e-05 | 5.019e-02 |

AMGx and Ginkgo were rerun as iterative library candidates, not as custom cuSPARSE Krylov solvers. On this primary case the recorded convergence and scaled residuals are the deciding evidence, not speed alone. If a row is unconverged, it is not a reliable Newton linear solve for annual-report claims even if the wall time is moderate.

## Large-Case Confirmation

`case9241pegase`, iteration 0, FP64:

| solver | config | analysis ms | factor ms | solve ms | solver ms | end-to-end ms | conv | iters | rel resid | scaled resid | phase visibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cuDSS | default | 46.009 | 7.781e-01 | 2.994e-01 | 47.087 | 272.581 | yes | 1.000 | 1.249e-14 | 1.249e-14 | analysis_factor_solve |
| cuSolverSP | default | 0.000e+00 | 0.000e+00 | 191.983 | 191.983 | 2,715.0 | yes | 1.000 | 4.075e-14 | 4.075e-14 | monolithic_qr |
| AMGx | GMRES+AMG(BlockJacobi) | 16.281 | 0.000e+00 | 120.451 | 136.732 | 1,878.4 | no | 200.000 | 1.017e-01 | 1.017e-01 | setup_solve_iterative |
| Ginkgo-GMRES-Jacobi | gmres | 16.670 | 0.000e+00 | 498.862 | 515.533 | 7,242.4 | no | 1,000.0 | 8.464e-04 | 8.464e-04 | setup_solve_iterative |
| STRUMPACK | np=1 | 50.411 | 18.924 | 28.978 | 98.313 | 1,487.5 | yes | 1.000 | 1.674e-14 | 1.674e-14 | analysis_factor_solve_wall |
| SuperLU_DIST | MMD_AT_PLUS_A+NOROWPERM np=1 | 36.384 | 163.682 | 1.868 | 202.199 | 290.845 | yes | 1.000 | 1.394e-14 | 1.394e-14 | SuperLUStat_t_internal_plus_wrapper_wall |

## SuperLU_DIST Failure/Fix History

The earlier `get_perm_c.c Invalid ISPEC` failure was not a numerical performance result. It was caused by requesting `METIS_AT_PLUS_A` while the installed SuperLU_DIST build did not have ParMETIS enabled. The v4 diagnostic wrapper uses exact enum names from the installed headers and avoids `METIS_AT_PLUS_A`.

The original repeated wrapper remains invalid for repeated timing because the ABglobal driver mutates SuperLU_DIST matrix/solver state. v4 therefore reports one-shot in-process timings and marks repeated ABglobal reuse as not valid unless the wrapper reconstructs or deep-copies the input state for every call.

## SuperLU_DIST Slow-Time Diagnosis

Dominant measured source:

- `case2869pegase`: numeric factorization
- `case9241pegase`: numeric factorization

Phase breakdown for the default v4 SuperLU_DIST configuration (`NATURAL + LargeDiag_MC64`, `np=1`):

| case | np | load ms | convert ms | grid ms | construct ms | rowperm ms | colperm ms | symbolic ms | dist ms | factor ms | solve ms | cleanup ms | solver ms | external ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case2869pegase | 1.000 | 23.359 | 8.314 | 3.683e-02 | 1.481e-01 | 9.159e-01 | 5.974e-01 | 29.973 | 91.927 | 4,661.6 | 14.649 | 3.361 | 4,800.3 | 5,004.8 |
| case9241pegase | 1.000 | 81.098 | 32.243 | 3.943e-02 | 5.080e-01 | 3.033 | 2.060 | 248.304 | 909.220 | 135,703.5 | 119.799 | 41.646 | 136,988.6 | 137,321.7 |

The SuperLU_DIST phase timers show that the slow NATURAL path is not primarily MatrixMarket loading, MPI process launch, or triangular solve. For both large cases, numerical factorization is the dominant component in this ABglobal configuration. Matrix distribution and symbolic analysis are visible but smaller. Process launch overhead is measurable with the `mpirun -np 1 /bin/true` baseline, but it is negligible compared with factorization on the NATURAL large-case runs.

The configuration sweep changes the interpretation: supported MMD orderings avoid the catastrophic NATURAL factorization time. The slow result is therefore best described as a poor/default ordering choice in the diagnostic wrapper, not an intrinsic SuperLU_DIST inability to solve these matrices.

If a phase is blank in the CSV, it was not exposed by the current wrapper/API or the run failed before that phase was available. The public ABglobal call still hides finer subphases inside numerical factorization.

## SuperLU_DIST Configuration Sweep

The sweep used the supported installed-header enum names: `NATURAL`, `MMD_AT_PLUS_A`, `MMD_ATA`, `LargeDiag_MC64`, and `NOROWPERM`. `METIS_AT_PLUS_A` was intentionally not used because ParMETIS is disabled.

| case | colperm | rowperm | status | conv | analysis ms | factor ms | solve ms | solver ms | scaled resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case2869pegase | MMD_ATA | LargeDiag_MC64 | ok | yes | 14.139 | 161.450 | 8.135e-01 | 176.496 | 4.357e-15 |
| case2869pegase | MMD_ATA | NOROWPERM | ok | yes | 13.394 | 161.384 | 8.195e-01 | 175.695 | 5.446e-15 |
| case2869pegase | MMD_AT_PLUS_A | LargeDiag_MC64 | ok | yes | 11.448 | 159.880 | 5.766e-01 | 171.988 | 4.534e-15 |
| case2869pegase | MMD_AT_PLUS_A | NOROWPERM | ok | yes | 10.564 | 161.182 | 5.728e-01 | 172.407 | 5.403e-15 |
| case2869pegase | NATURAL | LargeDiag_MC64 | ok | yes | 124.046 | 4,454.9 | 14.790 | 4,594.1 | 7.875e-15 |
| case2869pegase | NATURAL | NOROWPERM | ok | yes | 123.129 | 4,409.0 | 14.600 | 4,547.2 | 6.951e-15 |
| case9241pegase | MMD_ATA | LargeDiag_MC64 | ok | yes | 52.463 | 174.901 | 2.910 | 230.623 | 1.410e-14 |
| case9241pegase | MMD_ATA | NOROWPERM | ok | yes | 50.711 | 170.734 | 2.948 | 224.702 | 1.365e-14 |
| case9241pegase | MMD_AT_PLUS_A | LargeDiag_MC64 | ok | yes | 39.005 | 162.118 | 1.871 | 203.276 | 1.341e-14 |
| case9241pegase | MMD_AT_PLUS_A | NOROWPERM | ok | yes | 36.384 | 163.682 | 1.868 | 202.199 | 1.394e-14 |
| case9241pegase | NATURAL | LargeDiag_MC64 | ok | yes | 1,150.3 | 134,414.1 | 117.589 | 135,684.0 | 2.738e-14 |
| case9241pegase | NATURAL | NOROWPERM | ok | yes | 1,164.3 | 133,544.9 | 118.636 | 134,830.0 | 2.958e-14 |

Full sweep output: `measurement_audit/results/superlu_dist_config_sweep.csv`.

## MPI Rank Sweep

The rank sweep used `OMP_NUM_THREADS=1`, local MPICH `mpirun`, and the default `NATURAL + LargeDiag_MC64` SuperLU_DIST configuration.

| case | np | status | conv | analysis ms | factor ms | solve ms | solver ms | scaled resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case2869pegase | 1.000 | ok | yes | 125.129 | 4,491.9 | 15.357 | 4,632.8 | 7.875e-15 |
| case2869pegase | 2.000 | ok | yes | 94.139 | 2,731.8 | 11.739 | 2,838.0 | 7.588e-15 |
| case2869pegase | 4.000 | ok | yes | 78.842 | 2,168.7 | 10.371 | 2,258.0 | 7.230e-15 |
| case9241pegase | 1.000 | ok | yes | 1,143.9 | 135,358.2 | 119.405 | 136,623.5 | 2.738e-14 |
| case9241pegase | 2.000 | ok | yes | 788.912 | 87,281.7 | 93.709 | 88,164.9 | 2.566e-14 |
| case9241pegase | 4.000 | ok | yes | 663.331 | 69,342.6 | 82.130 | 70,084.1 | 1.893e-14 |
| synthetic_validation | 1.000 | ok | yes | 4.201e-02 | 153.826 | 4.994e-02 | 153.929 | 6.871e-17 |
| synthetic_validation | 2.000 | ok | yes | 1.653e-01 | 224.813 | 5.764e-02 | 225.042 | 6.871e-17 |
| synthetic_validation | 4.000 | ok | yes | 2.932e-01 | 358.960 | 5.807e-02 | 359.315 | 6.871e-17 |

Full rank output: `measurement_audit/results/superlu_dist_mpi_rank_sweep.csv`.

## CPU-Only vs CUDA Build

The existing local SuperLU_DIST installation used by v4 is CUDA-enabled and linked against CUDA runtime, cuBLAS, cuSolver, and cuSPARSE. A separate CPU-only SuperLU_DIST installation was not present in the benchmark workspace, so v4 does not fabricate a CPU-only comparison. The correct future comparison is to build a second isolated CPU-only prefix and run the same phase wrapper against it.

## Process-Level vs In-Process Timing

v4 provides an in-process single-solve wrapper and records Python external elapsed time for the surrounding `mpirun` command. The repeated ABglobal wrapper is still not valid for performance comparison because ABglobal mutates input/solver state. Therefore SuperLU_DIST timing is valid as a one-shot external MPI/hybrid baseline, but not as a reusable repeated Newton factorization timing comparable to cuDSS analysis reuse.

## Failed or Timed-Out Runs

No v4 command failures were recorded.

## Output Files

- `measurement_audit/results/large_case_solver_comparison.csv`
- `measurement_audit/results/superlu_dist_phase_breakdown.csv`
- `measurement_audit/results/superlu_dist_config_sweep.csv`
- `measurement_audit/results/superlu_dist_mpi_rank_sweep.csv`
- `measurement_audit/results/raw_json/`

## Annual-Report Interpretation

The large-case v4 evidence supports using cuDSS as the default cuPF sparse linear solver because it combines direct-solver robustness for general nonsymmetric Jacobians, GPU execution, explicit phase visibility, and reusable analysis/factorization structure that matches repeated Newton solves with stable sparsity.

SuperLU_DIST should be described as an external distributed direct-solver baseline rather than a cuPF-default candidate in this evidence set. It now runs accurately after the permutation fix, and supported MMD orderings are far faster than the NATURAL diagnostic path. The remaining integration caveats are MPI launch/setup, host/distributed input, one-shot ABglobal timing, and lack of a validated reusable in-process timing path comparable to cuDSS analysis reuse.

STRUMPACK remains useful as an MPI/hybrid direct baseline, but its `np=1` success and prior multi-rank instability make it less straightforward as an embedded cuPF default. AMGx and Ginkgo remain iterative-library candidates; their suitability depends on convergence and residual quality on these Newton Jacobians, not only elapsed time.
