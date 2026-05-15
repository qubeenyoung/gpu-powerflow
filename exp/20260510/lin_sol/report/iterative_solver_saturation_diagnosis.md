# AMGx and Ginkgo Saturation Diagnosis

This note extends the linear solver measurement audit by checking where the iterative GPU solvers stop making useful residual progress on the large power-flow Jacobian systems. It reuses the existing dumped systems and wrappers, and it does not touch production cuPF source code.

## Method

- Cases: `case2869pegase` iteration 0 and `case9241pegase` iteration 0.
- dtype: FP64.
- AMGx configs: GMRES/FGMRES with AMG aggregation and BlockJacobi smoother.
- Ginkgo configs: GMRES/BiCGSTAB with Jacobi, matching the current wrapper capability.
- Iteration caps: 50, 100, 200, 400, 800, 1000, 1500, 2000.
- Saturation is reported as a practical plateau when all remaining tested caps improve residual by less than 20%; otherwise the largest tested cap is reported as not saturated. If residual worsens substantially after the best point, the limit is classified as instability after best residual.

## Saturation Summary

| case | solver/config | limit type | iter cap | solve ms | residual at limit | best residual | note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| case2869pegase | amgx_fgmres_amg_block_jacobi | not_saturated_by_2000 | 2000 | 1074.0 | 5.470e-05 | 5.470e-05 | Residual is still improving at the largest tested cap, but has not reached tolerance. |
| case2869pegase | amgx_gmres_amg_block_jacobi | not_saturated_by_2000 | 2000 | 1045.9 | 5.240e-05 | 5.240e-05 | Residual is still improving at the largest tested cap, but has not reached tolerance. |
| case2869pegase | ginkgo_bicgstab_jacobi | unstable_after_best_residual | 1000 | 190.246 | 6.728e-05 | 6.728e-05 | Residual worsens substantially after the best cap; extra iterations are numerically unhelpful. |
| case2869pegase | ginkgo_gmres_jacobi | not_saturated_by_2000 | 2000 | 983.644 | 9.041e-05 | 9.041e-05 | Residual is still improving at the largest tested cap, but has not reached tolerance. |
| case9241pegase | amgx_fgmres_amg_block_jacobi | practical_plateau | 100 | 72.459 | 0.108 | 0.099 | Remaining sweep improves residual by less than 20%. |
| case9241pegase | amgx_gmres_amg_block_jacobi | practical_plateau | 100 | 66.952 | 0.108 | 0.099 | Remaining sweep improves residual by less than 20%. |
| case9241pegase | ginkgo_bicgstab_jacobi | unstable_after_best_residual | 100 | 28.869 | 0.066 | 0.066 | Residual worsens substantially after the best cap; extra iterations are numerically unhelpful. |
| case9241pegase | ginkgo_gmres_jacobi | practical_plateau | 1500 | 757.164 | 6.583e-04 | 6.055e-04 | Remaining sweep improves residual by less than 20%. |

## What Saturates

For `case9241pegase`, AMGx reaches a practical plateau very early, around 100 to 200 iterations. The residual is about `1.08e-1` at 100 iterations and about `1.02e-1` at 200 iterations, but only reaches about `9.92e-2` at 2000 iterations while solve time grows from roughly 67-141 ms to more than 1.2 s. This is a convergence-quality limit, not a raw throughput limit.

For `case2869pegase`, AMGx continues to improve through 2000 iterations, reaching about `5.2e-5`, but it still does not meet the FP64 tolerance and takes about 1.0 s of solve time. It is not saturated by 2000 iterations, but the time-to-accuracy tradeoff is poor for Newton linear solves.

Ginkgo GMRES with Jacobi improves more smoothly. On `case9241pegase` it becomes a practical plateau around 1500 iterations: residual improves from about `6.58e-4` to `6.06e-4` by 2000, while solve time rises from about 757 ms to about 1009 ms. On `case2869pegase`, it is still improving at 2000 iterations but remains above the requested tolerance.

Ginkgo BiCGSTAB with Jacobi is unstable on these Jacobians. On `case2869pegase`, the best observed residual occurs near 1000 iterations and then worsens. On `case9241pegase`, it improves at 100 iterations but then diverges badly as the cap increases.

## Bottleneck Evidence

Nsight Compute launch sampling was used only for operation identification, not timing, because profiler overhead changes the measured solve time.

AMGx setup/profile-start captures aggregation hierarchy construction: `size2_selector::*`, CUB scans/sorts, and device fills. The solve-window captures repeated `amgx::csrmv`, BlockJacobi presmoothing, `aggregation::restrictResidualKernel`, AXPBY vector updates, and CUB reductions. The bottleneck is the repeated AMG V-cycle/Krylov iteration work, dominated by sparse matrix-vector products, BlockJacobi smoothing, residual restriction, and vector reductions. On `case9241pegase`, those iterations keep consuming time after the residual has essentially plateaued.

Ginkgo setup/profile-start captures CSR/COO conversion and cuSPARSE scan/sort kernels plus Jacobi setup. The solve-window captures `csr::abstract_classical_spmv`, Jacobi `kernel::apply`, residual-norm kernels, cuBLAS `dot`/`nrm2` reductions, dense vector updates, and GMRES Hessenberg QR kernels. The bottleneck is the Krylov iteration loop: sparse matvec plus Jacobi application plus global reduction/orthogonalization work. For GMRES, the reduction/orthogonalization component is visible in the dot/nrm2 and Hessenberg QR kernels.

## H2D and D2H

The wrappers do measure explicit host/device transfer windows. For `case9241pegase` FP64:

| solver/config | iter cap | h2d ms | d2h ms | note |
| --- | ---: | ---: | ---: | --- |
| AMGx GMRES/AMG BlockJacobi | 200 | 71.962 | 0.026 | H2D includes AMGX initialize/config/resource creation plus matrix/vector upload, so it is not a pure memcpy measurement. |
| AMGx GMRES/AMG BlockJacobi | 2000 | 78.970 | 0.026 | Same caveat; solve then remains GPU-resident. |
| Ginkgo GMRES/Jacobi | 200 | 0.671 | 0.118 | H2D is Ginkgo CSR/vector clone to CUDA executor. |
| Ginkgo GMRES/Jacobi | 2000 | 0.681 | 0.125 | D2H is final solution clone back to host. |
| Ginkgo BiCGSTAB/Jacobi | 100 | 0.669 | 0.130 | Same Ginkgo transfer path. |

Internal library transfers may still occur during setup or solve, especially in AMG hierarchy construction and Ginkgo format/conversion/setup paths. They are not separately exposed by the current wrappers; they are included in `analysis_ms` or `solve_ms`. Nsight Compute launch sampling identified kernels but was not used to quantify `cudaMemcpy` time. A precise internal transfer breakdown would require Nsight Systems/CUPTI tracing or explicit wrapper instrumentation around library-visible copy calls.

## Limitation

The current Ginkgo wrapper only wires Jacobi preconditioning even though the config parser accepts a preconditioner field. Therefore this diagnosis is a fair saturation audit for the implemented Ginkgo path, not a full Ginkgo best-preconditioner study. AMGx residual history was not available from the wrapper because `AMGX_solver_get_iteration_residual` reported that residual history was not recorded, so saturation was inferred by cap sweep.

## Files

- Sweep CSV: `measurement_audit/results/iterative_saturation_sweep.csv`
- Summary CSV: `measurement_audit/results/iterative_saturation_summary.csv`
- Profile CSV: `measurement_audit/results/iterative_bottleneck_profile.csv`
