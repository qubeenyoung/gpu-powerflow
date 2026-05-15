# SuperLU_DIST Optimality Check

This follow-up checks whether the SuperLU_DIST large-case runs were reasonable best-effort executions. It varies the main knobs exposed by the current ABglobal diagnostic wrapper:

- fill-reducing column ordering fixed to `MMD_AT_PLUS_A`
- row permutation: `LargeDiag_MC64` and `NOROWPERM`
- GPU offload enabled/disabled through `superlu_acc_offload`
- process grid shape for `np=2` and `np=4`
- selected solver options: equilibration, iterative refinement, tiny-pivot replacement

The check still uses `pdgssvx_ABglobal`, so it is a one-shot MPI/hybrid baseline rather than a validated reusable Newton factorization path.

## Best Observed Runs

| label | case | np | grid | rowperm | conv | factor ms | solve ms | solver ms | gpu MB | scaled resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offload_off | case2869pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 4.104 | 5.737e-01 | 15.266 | 0.000e+00 | 5.403e-15 |
| offload_off | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 4.110 | 5.720e-01 | 16.047 | 0.000e+00 | 4.534e-15 |
| option_equil_off | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 142.055 | 5.956e-01 | 153.750 | 1.096e-01 | 5.900e-15 |
| option_par_symb | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 143.079 | 5.816e-01 | 155.011 | 1.096e-01 | 4.534e-15 |
| option_replace_tiny | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 143.676 | 5.813e-01 | 155.693 | 1.096e-01 | 4.534e-15 |
| offload_off | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 12.692 | 2.043 | 52.050 | 0.000e+00 | 1.394e-14 |
| offload_off | case9241pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 13.015 | 1.854 | 54.988 | 0.000e+00 | 1.341e-14 |
| option_replace_tiny | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 153.057 | 1.872 | 192.462 | 2.017e-01 | 1.394e-14 |
| option_equil_off | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 155.209 | 1.855 | 192.982 | 2.017e-01 | 1.377e-14 |
| option_refine_double | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 149.617 | 1.651 | 194.409 | 2.017e-01 | 1.639e-14 |

## GPU Offload Check

| label | case | np | grid | rowperm | conv | factor ms | solve ms | solver ms | gpu MB | scaled resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offload_on | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 237.177 | 5.810e-01 | 249.104 | 1.096e-01 | 4.534e-15 |
| offload_off | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 4.110 | 5.720e-01 | 16.047 | 0.000e+00 | 4.534e-15 |
| offload_on | case2869pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 153.971 | 5.788e-01 | 165.174 | 1.096e-01 | 5.403e-15 |
| offload_off | case2869pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 4.104 | 5.737e-01 | 15.266 | 0.000e+00 | 5.403e-15 |
| offload_on | case9241pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 161.331 | 1.954 | 203.408 | 2.017e-01 | 1.341e-14 |
| offload_off | case9241pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 13.015 | 1.854 | 54.988 | 0.000e+00 | 1.341e-14 |
| offload_on | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 161.534 | 1.901 | 201.564 | 2.017e-01 | 1.394e-14 |
| offload_off | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 12.692 | 2.043 | 52.050 | 0.000e+00 | 1.394e-14 |

The CUDA-enabled build is real, but the best MMD runs use very small GPU buffers. Disabling SuperLU_DIST GPU offload did not make the large cases slower in this ABglobal path; in some runs it was slightly faster. Therefore the current best SuperLU_DIST numbers should be interpreted as MPI/CPU-dominant hybrid timings, not as strong GPU-resident sparse direct-solver evidence.

## Process Grid Check

| label | case | np | grid | rowperm | conv | factor ms | solve ms | solver ms | gpu MB | scaled resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid_np2 | case2869pegase | 2.000 | 1.000x2.000 | LargeDiag_MC64 | yes | 231.062 | 1.029 | 246.982 | 1.096e-01 | 5.152e-15 |
| grid_np2 | case2869pegase | 2.000 | 2.000x1.000 | LargeDiag_MC64 | yes | 204.695 | 1.027 | 215.623 | 1.053e-01 | 4.657e-15 |
| grid_np4 | case2869pegase | 4.000 | 1.000x4.000 | LargeDiag_MC64 | yes | 353.761 | 9.224e-01 | 368.653 | 1.096e-01 | 5.307e-15 |
| grid_np4 | case2869pegase | 4.000 | 2.000x2.000 | LargeDiag_MC64 | yes | 345.553 | 9.895e-01 | 356.780 | 1.053e-01 | 5.177e-15 |
| grid_np4 | case2869pegase | 4.000 | 4.000x1.000 | LargeDiag_MC64 | yes | 347.909 | 1.068 | 360.939 | 7.764e-02 | 4.704e-15 |
| grid_np2 | case9241pegase | 2.000 | 1.000x2.000 | NOROWPERM | yes | 229.616 | 4.254 | 271.099 | 2.017e-01 | 1.461e-14 |
| grid_np2 | case9241pegase | 2.000 | 2.000x1.000 | NOROWPERM | yes | 230.794 | 2.197 | 264.068 | 2.017e-01 | 1.377e-14 |
| grid_np4 | case9241pegase | 4.000 | 1.000x4.000 | NOROWPERM | yes | 394.253 | 2.280 | 438.761 | 2.017e-01 | 1.396e-14 |
| grid_np4 | case9241pegase | 4.000 | 2.000x2.000 | NOROWPERM | yes | 350.460 | 2.485 | 383.808 | 2.017e-01 | 1.427e-14 |
| grid_np4 | case9241pegase | 4.000 | 4.000x1.000 | NOROWPERM | yes | 351.473 | 2.146 | 382.907 | 2.017e-01 | 1.383e-14 |

The earlier `1 x np` grid was not the only shape tested here. For these single-node cases, increasing ranks or using a square-ish `2 x 2` grid did not beat `np=1`; communication/setup overhead and the small matrix sizes relative to distributed direct-solver overhead make single-rank best for this dataset.

## Option Check

| label | case | np | grid | rowperm | conv | factor ms | solve ms | solver ms | gpu MB | scaled resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| option_equil_off | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 142.055 | 5.956e-01 | 153.750 | 1.096e-01 | 5.900e-15 |
| option_replace_tiny | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 143.676 | 5.813e-01 | 155.693 | 1.096e-01 | 4.534e-15 |
| option_refine_double | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 153.719 | 5.011e-01 | 167.344 | 1.096e-01 | 6.477e-15 |
| option_par_symb | case2869pegase | 1.000 | 1.000x1.000 | LargeDiag_MC64 | yes | 143.079 | 5.816e-01 | 155.011 | 1.096e-01 | 4.534e-15 |
| option_equil_off | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 155.209 | 1.855 | 192.982 | 2.017e-01 | 1.377e-14 |
| option_replace_tiny | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 153.057 | 1.872 | 192.462 | 2.017e-01 | 1.394e-14 |
| option_refine_double | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 149.617 | 1.651 | 194.409 | 2.017e-01 | 1.639e-14 |
| option_par_symb | case9241pegase | 1.000 | 1.000x1.000 | NOROWPERM | yes | 157.263 | 1.883 | 195.916 | 2.017e-01 | 1.394e-14 |

No tested option materially beats the best `MMD_AT_PLUS_A` one-shot configuration. Iterative refinement adds cost and is unnecessary because the no-refinement residuals are already near FP64 reference quality. Tiny-pivot replacement and equilibration changes did not produce a better default.

## Judgment

The original NATURAL ordering result was not best effort. The later MMD ordering results are a reasonable best-effort ABglobal configuration for this installed SuperLU_DIST build. However, they are not a proof of optimal SuperLU_DIST overall because:

- the test uses the high-level ABglobal driver, not a lower-level reusable factor/solve API
- ParMETIS and COLAMD are not enabled in this build
- GPU usage in best MMD runs is minimal
- 3D/SLATE-style fully GPU-oriented paths were not validated here

For cuPF annual-report wording, SuperLU_DIST should be reported as a valid external MPI/hybrid direct-solver baseline with best-effort MMD ordering, but not as a fully GPU-resident or cuPF-integration-equivalent alternative to cuDSS.

CSV: `measurement_audit/results/superlu_dist_optimality_check.csv`
