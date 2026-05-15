# Integrated Linear Solver Measurement Validity Report v5

## Purpose

This report integrates the v3 measurement audit, v4 large-case rerun, SuperLU_DIST ordering diagnosis, and the latest solver optimality checks. The goal is not to crown the fastest number in isolation, but to decide which measurements are valid evidence for choosing a sparse linear solver library for cuPF power-flow Jacobian systems.

Representative large cases are `case2869pegase` and `case9241pegase`, iteration 0, FP64.

## Main Large-Case Summary

| case | solver | config | conv | analysis | factor | solve | solver ms | reuse ms | scaled resid | GPU/residency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case2869pegase | cuDSS | general CSR LU, analysis reused | yes | 24.059 | 2.958e-01 | 1.798e-01 | 24.534 | 4.756e-01 | 4.449e-15 | yes |
| case2869pegase | cuSolverSP | csrlsvqr monolithic QR | yes | 0.000e+00 | 0.000e+00 | 50.028 | 50.028 | n/a | 1.149e-14 | yes_for_QR_API |
| case2869pegase | AMGx | fgmres_amg_block_jacobi_1000 | no | 13.176 | 0.000e+00 | 526.680 | 539.857 | n/a | 6.393e-04 | yes |
| case2869pegase | Ginkgo | bicgstab | no | 15.343 | 0.000e+00 | 182.938 | 198.281 | n/a | 6.728e-05 | yes_for_matrix_vector_and_iterations |
| case2869pegase | STRUMPACK | MPIDist np=1 default no-compression | yes | 11.162 | 12.211 | 9.093 | 32.466 | n/a | 6.001e-15 | cpu_gpu_hybrid_mpi_dist |
| case2869pegase | SuperLU_DIST | MMD_AT_PLUS_A best observed; acc_offload=0 | yes | 10.505 | 4.104 | 5.737e-01 | 15.266 | n/a | 5.403e-15 | cpu_gpu_hybrid_abglobal |
| case2869pegase | SuperLU_DIST | MMD_AT_PLUS_A best with acc_offload=1 | yes | 11.028 | 142.055 | 5.956e-01 | 153.750 | n/a | 5.900e-15 | cpu_gpu_hybrid_abglobal |
| case9241pegase | cuDSS | general CSR LU, analysis reused | yes | 46.009 | 7.781e-01 | 2.994e-01 | 47.087 | 1.078 | 1.249e-14 | yes |
| case9241pegase | cuSolverSP | csrlsvqr monolithic QR | yes | 0.000e+00 | 0.000e+00 | 191.983 | 191.983 | n/a | 4.075e-14 | yes_for_QR_API |
| case9241pegase | AMGx | gmres_amg_block_jacobi_1000 | no | 16.713 | 0.000e+00 | 611.217 | 627.930 | n/a | 9.958e-02 | yes |
| case9241pegase | Ginkgo | gmres | no | 16.670 | 0.000e+00 | 498.862 | 515.533 | n/a | 8.464e-04 | yes_for_matrix_vector_and_iterations |
| case9241pegase | STRUMPACK | MPIDist np=1 default no-compression | yes | 50.411 | 18.924 | 28.978 | 98.313 | n/a | 1.674e-14 | cpu_gpu_hybrid_mpi_dist |
| case9241pegase | SuperLU_DIST | MMD_AT_PLUS_A best observed; acc_offload=0 | yes | 37.060 | 12.692 | 2.043 | 52.050 | n/a | 1.394e-14 | cpu_gpu_hybrid_abglobal |
| case9241pegase | SuperLU_DIST | MMD_AT_PLUS_A best with acc_offload=1 | yes | 37.270 | 153.057 | 1.872 | 192.462 | n/a | 1.394e-14 | cpu_gpu_hybrid_abglobal |

`reuse ms` is only populated when the wrapper exposes a meaningful repeated Newton-style factorization-plus-solve number. For cuDSS this is the strongest cuPF-relevant timing because symbolic analysis is reusable for a fixed sparsity pattern. For monolithic or one-shot MPI wrappers, blank reuse timing means the current wrapper does not validate that use case.

## Solver Validity Classification

| solver | measurement status | best-effort status | GPU evidence | cuPF relevance | remaining issue |
| --- | --- | --- | --- | --- | --- |
| cuDSS | valid best GPU baseline | strong | yes, GPU-resident cuDSS phases timed with CUDA events | high | Need production integration tuning, but benchmark measurement is sound. |
| cuSolverSP/RF | valid monolithic QR | reasonable for available raw CSR API | yes for cuSolverSP QR path | medium/low | QR is monolithic; RF needs supplied LU factors and is not a drop-in Jacobian solver here. |
| AMGx | valid iterative evidence | limited finite grid | yes, GPU iterative library | low as standalone Newton solve | GMRES/FGMRES AMG BlockJacobi did not converge on >=5K cases even at 1000 iterations. |
| Ginkgo | valid as Jacobi-only wrapper | incomplete | yes, CUDA executor used | low until advanced preconditioners wired | Wrapper only tests GMRES/BiCGSTAB with Jacobi; no ILU/ParILU/ISAI path. |
| STRUMPACK | valid external MPI/hybrid np=1 | reasonable external baseline | weak/qualified; CUDA build but SLATE warning means not full GPU | medium as external baseline, low as default | np>1 instability in prior audit; no SLATE full-GPU path. |
| SuperLU_DIST | fixed and reclassified | reasonable for ABglobal after MMD/offload sweep | no for best result; offload-off is fastest, offload-on is hybrid and slower | medium as CPU/MPI external baseline, low as GPU default | High-level ABglobal one-shot; ParMETIS/COLAMD absent; reusable lower-level path not validated. |

## Direct Solver Findings

| case | solver | config | solver ms | reuse ms | scaled resid | phase visibility | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| case2869pegase | cuDSS | general CSR LU, analysis reused | 24.534 | 4.756e-01 | 4.449e-15 | analysis_factor_solve_reusable | valid GPU direct baseline and cuPF integration baseline |
| case2869pegase | cuSolverSP | csrlsvqr monolithic QR | 50.028 | n/a | 1.149e-14 | monolithic_qr | valid NVIDIA monolithic QR comparison, not reusable factorization |
| case2869pegase | STRUMPACK | MPIDist np=1 default no-compression | 32.466 | n/a | 6.001e-15 | analysis_factor_solve_host_mpi | valid external MPI/hybrid direct baseline; not full-GPU evidence |
| case2869pegase | SuperLU_DIST | MMD_AT_PLUS_A best observed; acc_offload=0 | 15.266 | n/a | 5.403e-15 | SuperLUStat_t_internal_plus_wrapper_wall | best observed is CPU-dominant ABglobal, not GPU evidence |
| case2869pegase | SuperLU_DIST | MMD_AT_PLUS_A best with acc_offload=1 | 153.750 | n/a | 5.900e-15 | SuperLUStat_t_internal_plus_wrapper_wall | CUDA-enabled hybrid ABglobal; slower than offload-off |
| case9241pegase | cuDSS | general CSR LU, analysis reused | 47.087 | 1.078 | 1.249e-14 | analysis_factor_solve_reusable | valid GPU direct baseline and cuPF integration baseline |
| case9241pegase | cuSolverSP | csrlsvqr monolithic QR | 191.983 | n/a | 4.075e-14 | monolithic_qr | valid NVIDIA monolithic QR comparison, not reusable factorization |
| case9241pegase | STRUMPACK | MPIDist np=1 default no-compression | 98.313 | n/a | 1.674e-14 | analysis_factor_solve_host_mpi | valid external MPI/hybrid direct baseline; not full-GPU evidence |
| case9241pegase | SuperLU_DIST | MMD_AT_PLUS_A best observed; acc_offload=0 | 52.050 | n/a | 1.394e-14 | SuperLUStat_t_internal_plus_wrapper_wall | best observed is CPU-dominant ABglobal, not GPU evidence |
| case9241pegase | SuperLU_DIST | MMD_AT_PLUS_A best with acc_offload=1 | 192.462 | n/a | 1.394e-14 | SuperLUStat_t_internal_plus_wrapper_wall | CUDA-enabled hybrid ABglobal; slower than offload-off |

Key points:

- `cuDSS` is the cleanest GPU direct-solver evidence: general CSR LU, explicit analysis/factor/solve phases, CUDA-event timing, and validated analysis reuse. Its one-shot time is competitive, and its repeated Newton factor+solve time is sub-millisecond to about 1 ms on the two large cases.
- `cuSolverSP` is valid as a monolithic QR comparison, but it is not a reusable factorization comparison. `cuSolverRF` remains outside this wrapper because it requires externally supplied LU factors.
- `STRUMPACK` solves accurately and is useful as an external MPI/hybrid direct baseline. Prior logs warn that SLATE is required for full GPU support, so it should not be treated as fully GPU-resident evidence.
- `SuperLU_DIST` needed reclassification. NATURAL ordering was not best effort. MMD ordering fixed the catastrophic factorization time. The fastest observed SuperLU_DIST result disables SuperLU GPU offload, so it is a CPU-dominant MPI/ABglobal baseline, not GPU sparse direct evidence.

## Iterative Solver Findings

| case | solver | config | conv | iters | solver ms | scaled resid | rel err | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case2869pegase | AMGx | fgmres_amg_block_jacobi_1000 | no | 1,000.0 | 539.857 | 6.393e-04 | 1.616e-01 | large-grid-tested; did not converge on representative large cases |
| case2869pegase | Ginkgo | bicgstab | no | 1,000.0 | 198.281 | 6.728e-05 | 5.019e-02 | CUDA executor but Jacobi-only wrapper; did not converge on representative large cases |
| case9241pegase | AMGx | gmres_amg_block_jacobi_1000 | no | 1,000.0 | 627.930 | 9.958e-02 | 9.622e-01 | large-grid-tested; did not converge on representative large cases |
| case9241pegase | Ginkgo | gmres | no | 1,000.0 | 515.533 | 8.464e-04 | 9.144e-01 | CUDA executor but Jacobi-only wrapper; did not converge on representative large cases |

AMGx was rerun on the large cases with GMRES/FGMRES AMG BlockJacobi at 200 and 1000 iterations. The 1000-iteration runs reduce residuals but still do not converge. Ginkgo large-case evidence remains limited because the existing wrapper only wires Jacobi-preconditioned GMRES/BiCGSTAB, and those did not converge on the large representative systems.

This does not prove AMGx or Ginkgo are poor libraries. It shows that the tested standalone iterative configurations are not robust drop-in Newton linear solvers for these Jacobians.

## SuperLU_DIST Correction

The SuperLU_DIST story changed materially:

1. `METIS_AT_PLUS_A` caused the earlier `Invalid ISPEC` because ParMETIS is not enabled.
2. `NATURAL` ordering runs were valid but not best effort; they caused huge numeric LU factorization time.
3. `MMD_AT_PLUS_A` / `MMD_ATA` made the solve accurate and much faster.
4. `np=1` was best on these single-node matrix sizes; larger MPI rank counts and alternate process-grid shapes were slower.
5. `superlu_acc_offload=0` was fastest:
   - `case2869pegase`: about 15.3 ms solver time.
   - `case9241pegase`: about 52.1 ms solver time.

Therefore SuperLU_DIST should be reported as a valid external CPU/MPI direct-solver baseline after configuration repair, not as a GPU-resident sparse direct solver baseline.

## Updated Annual-Report Interpretation

The evidence for selecting cuDSS should be phrased carefully:

cuDSS remains the most suitable default sparse linear solver for cuPF because it is a direct solver for general nonsymmetric sparse Jacobians, exposes and reuses analysis/factorization structure in a way that matches repeated Newton power-flow solves, runs through a GPU-resident NVIDIA library interface, and has low integration complexity compared with MPI/hybrid external solvers.

The conclusion should not claim that every alternative is simply slower. The better evidence is:

- cuSolverSP is monolithic QR and not the reusable LU/refactorization path cuPF wants.
- AMGx and Ginkgo did not show robust standalone convergence under the finite tested configurations.
- STRUMPACK is accurate but external MPI/hybrid and not full-GPU in this build.
- SuperLU_DIST can be fast after MMD/offload tuning, but the fastest path is CPU-dominant and ABglobal one-shot, not cuDSS-equivalent GPU evidence.

CSV summary: `measurement_audit/results/integrated_solver_optimality_summary.csv`
