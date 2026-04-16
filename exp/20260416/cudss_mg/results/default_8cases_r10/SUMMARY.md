# cuDSS MG Newton Solver Benchmark `default_8cases_r10`

## Setup

- Created UTC: 2026-04-16T03:59:02.029925+00:00
- Dataset root: `/workspace/datasets/pglib-opf/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS MG indices: `0,1`
- cuDSS reordering: `DEFAULT`
- cuDSS MT: False
- cuDSS host threads: `AUTO`
- cuDSS ND_NLEVELS: `AUTO`
- CUDA_VISIBLE_DEVICES: ``

## end2end

| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| case118_ieee | cuda_edge | 11.728 | 20.074 | 0.584x | 0.830 | 2.946 | 0.282x | True/True | 5.0/5.0 |
| case1354_pegase | cuda_edge | 18.926 | 30.477 | 0.621x | 2.032 | 7.081 | 0.287x | True/True | 6.0/6.0 |
| case2746wop_k | cuda_edge | 24.612 | 37.249 | 0.661x | 2.228 | 7.295 | 0.305x | True/True | 5.0/5.0 |
| case30_ieee | cuda_edge | 10.928 | 19.224 | 0.568x | 0.691 | 2.457 | 0.281x | True/True | 5.0/5.0 |
| case4601_goc | cuda_edge | 36.355 | 52.399 | 0.694x | 4.145 | 11.599 | 0.357x | True/True | 6.0/6.0 |
| case793_goc | cuda_edge | 15.728 | 25.435 | 0.618x | 1.529 | 4.945 | 0.309x | True/True | 5.8/5.8 |
| case8387_pegase | cuda_edge | 53.827 | 72.567 | 0.742x | 6.004 | 15.219 | 0.395x | True/True | 7.0/7.0 |
| case9241_pegase | cuda_edge | 59.192 | 80.742 | 0.733x | 7.135 | 18.754 | 0.380x | True/True | 8.0/8.0 |

## operators

| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| case118_ieee | cuda_edge | 11.855 | 20.144 | 0.589x | 0.947 | 3.012 | 0.314x | True/True | 5.0/5.0 |
| case1354_pegase | cuda_edge | 19.355 | 30.435 | 0.636x | 2.337 | 7.087 | 0.330x | True/True | 6.0/6.0 |
| case2746wop_k | cuda_edge | 25.330 | 37.204 | 0.681x | 2.645 | 7.286 | 0.363x | True/True | 5.0/5.0 |
| case30_ieee | cuda_edge | 11.077 | 19.038 | 0.582x | 0.759 | 2.444 | 0.311x | True/True | 5.0/5.0 |
| case4601_goc | cuda_edge | 37.486 | 52.541 | 0.713x | 4.785 | 11.612 | 0.412x | True/True | 6.0/6.0 |
| case793_goc | cuda_edge | 15.858 | 25.558 | 0.620x | 1.681 | 5.058 | 0.332x | True/True | 5.8/5.8 |
| case8387_pegase | cuda_edge | 55.568 | 72.742 | 0.764x | 7.167 | 15.274 | 0.469x | True/True | 7.0/7.0 |
| case9241_pegase | cuda_edge | 61.428 | 80.667 | 0.761x | 8.623 | 18.650 | 0.462x | True/True | 8.0/8.0 |

## Operator Metrics

| case | profile | metric | baseline ms | MG ms | speedup | delta ms |
|---|---|---|---:|---:|---:|---:|
| case118_ieee | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 8.923 | 14.945 | 0.597x | 6.022 |
| case118_ieee | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.673 | 1.862 | 0.898x | 0.189 |
| case118_ieee | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.081 | 0.329 | 0.247x | 0.248 |
| case118_ieee | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.224 | 0.921 | 0.243x | 0.697 |
| case118_ieee | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.047 | 0.048 | 0.973x | 0.001 |
| case118_ieee | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.267 | 1.369 | 0.195x | 1.102 |
| case118_ieee | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.040 | 0.039 | 1.003x | -0.000 |
| case118_ieee | cuda_edge | `NR.analyze.linear_solve.total_sec` | 10.598 | 16.810 | 0.630x | 6.212 |
| case118_ieee | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.267 | 0.279 | 0.957x | 0.012 |
| case118_ieee | cuda_edge | `NR.analyze.total.total_sec` | 10.906 | 17.130 | 0.637x | 6.224 |
| case118_ieee | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.050 | 0.052 | 0.958x | 0.002 |
| case118_ieee | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.624 | 2.674 | 0.233x | 2.050 |
| case118_ieee | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.113 | 0.122 | 0.933x | 0.008 |
| case118_ieee | cuda_edge | `NR.iteration.total.total_sec` | 0.864 | 2.925 | 0.295x | 2.062 |
| case118_ieee | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.069 | 0.072 | 0.964x | 0.003 |
| case118_ieee | cuda_edge | `NR.solve.download.total_sec` | 0.018 | 0.019 | 0.979x | 0.000 |
| case118_ieee | cuda_edge | `NR.solve.total.total_sec` | 0.946 | 3.011 | 0.314x | 2.065 |
| case118_ieee | cuda_edge | `NR.solve.upload.total_sec` | 0.061 | 0.064 | 0.958x | 0.003 |
| case118_ieee | cuda_edge | `analyze_sec` | 10.907 | 17.131 | 0.637x | 6.224 |
| case118_ieee | cuda_edge | `elapsed_sec` | 11.855 | 20.144 | 0.589x | 8.289 |
| case118_ieee | cuda_edge | `solve_sec` | 0.947 | 3.012 | 0.314x | 2.065 |
| case1354_pegase | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 14.612 | 20.749 | 0.704x | 6.138 |
| case1354_pegase | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.662 | 1.838 | 0.904x | 0.176 |
| case1354_pegase | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.309 | 0.885 | 0.349x | 0.576 |
| case1354_pegase | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.723 | 2.499 | 0.289x | 1.776 |
| case1354_pegase | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.106 | 0.109 | 0.973x | 0.003 |
| case1354_pegase | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.641 | 3.018 | 0.213x | 2.376 |
| case1354_pegase | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.418 | 0.424 | 0.985x | 0.006 |
| case1354_pegase | cuda_edge | `NR.analyze.linear_solve.total_sec` | 16.277 | 22.590 | 0.721x | 6.314 |
| case1354_pegase | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.320 | 0.330 | 0.971x | 0.009 |
| case1354_pegase | cuda_edge | `NR.analyze.total.total_sec` | 17.017 | 23.346 | 0.729x | 6.330 |
| case1354_pegase | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.062 | 0.065 | 0.953x | 0.003 |
| case1354_pegase | cuda_edge | `NR.iteration.linear_solve.total_sec` | 1.789 | 6.520 | 0.274x | 4.732 |
| case1354_pegase | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.215 | 0.226 | 0.951x | 0.011 |
| case1354_pegase | cuda_edge | `NR.iteration.total.total_sec` | 2.180 | 6.928 | 0.315x | 4.748 |
| case1354_pegase | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.106 | 0.107 | 0.988x | 0.001 |
| case1354_pegase | cuda_edge | `NR.solve.download.total_sec` | 0.029 | 0.029 | 0.986x | 0.000 |
| case1354_pegase | cuda_edge | `NR.solve.total.total_sec` | 2.335 | 7.086 | 0.330x | 4.750 |
| case1354_pegase | cuda_edge | `NR.solve.upload.total_sec` | 0.123 | 0.125 | 0.982x | 0.002 |
| case1354_pegase | cuda_edge | `analyze_sec` | 17.018 | 23.347 | 0.729x | 6.330 |
| case1354_pegase | cuda_edge | `elapsed_sec` | 19.355 | 30.435 | 0.636x | 11.080 |
| case1354_pegase | cuda_edge | `solve_sec` | 2.337 | 7.087 | 0.330x | 4.750 |
| case2746wop_k | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 19.845 | 26.777 | 0.741x | 6.932 |
| case2746wop_k | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.647 | 1.882 | 0.875x | 0.236 |
| case2746wop_k | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.414 | 1.101 | 0.376x | 0.687 |
| case2746wop_k | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.840 | 2.552 | 0.329x | 1.712 |
| case2746wop_k | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.132 | 0.136 | 0.971x | 0.004 |
| case2746wop_k | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.636 | 2.860 | 0.222x | 2.225 |
| case2746wop_k | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.797 | 0.828 | 0.962x | 0.032 |
| case2746wop_k | cuda_edge | `NR.analyze.linear_solve.total_sec` | 21.494 | 28.663 | 0.750x | 7.168 |
| case2746wop_k | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.391 | 0.423 | 0.925x | 0.032 |
| case2746wop_k | cuda_edge | `NR.analyze.total.total_sec` | 22.684 | 29.916 | 0.758x | 7.232 |
| case2746wop_k | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.052 | 0.055 | 0.954x | 0.003 |
| case2746wop_k | cuda_edge | `NR.iteration.linear_solve.total_sec` | 2.030 | 6.657 | 0.305x | 4.627 |
| case2746wop_k | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.232 | 0.238 | 0.974x | 0.006 |
| case2746wop_k | cuda_edge | `NR.iteration.total.total_sec` | 2.407 | 7.045 | 0.342x | 4.638 |
| case2746wop_k | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.085 | 0.087 | 0.979x | 0.002 |
| case2746wop_k | cuda_edge | `NR.solve.download.total_sec` | 0.040 | 0.041 | 0.983x | 0.001 |
| case2746wop_k | cuda_edge | `NR.solve.total.total_sec` | 2.644 | 7.285 | 0.363x | 4.641 |
| case2746wop_k | cuda_edge | `NR.solve.upload.total_sec` | 0.193 | 0.196 | 0.985x | 0.003 |
| case2746wop_k | cuda_edge | `analyze_sec` | 22.685 | 29.917 | 0.758x | 7.232 |
| case2746wop_k | cuda_edge | `elapsed_sec` | 25.330 | 37.204 | 0.681x | 11.873 |
| case2746wop_k | cuda_edge | `solve_sec` | 2.645 | 7.286 | 0.363x | 4.642 |
| case30_ieee | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 8.484 | 14.476 | 0.586x | 5.992 |
| case30_ieee | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.596 | 1.831 | 0.871x | 0.236 |
| case30_ieee | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.066 | 0.271 | 0.244x | 0.205 |
| case30_ieee | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.178 | 0.762 | 0.234x | 0.584 |
| case30_ieee | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.038 | 0.046 | 0.829x | 0.008 |
| case30_ieee | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.205 | 1.051 | 0.195x | 0.845 |
| case30_ieee | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.014 | 0.015 | 0.903x | 0.002 |
| case30_ieee | cuda_edge | `NR.analyze.linear_solve.total_sec` | 10.082 | 16.311 | 0.618x | 6.228 |
| case30_ieee | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.219 | 0.264 | 0.827x | 0.046 |
| case30_ieee | cuda_edge | `NR.analyze.total.total_sec` | 10.317 | 16.592 | 0.622x | 6.275 |
| case30_ieee | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.040 | 0.046 | 0.856x | 0.007 |
| case30_ieee | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.493 | 2.135 | 0.231x | 1.643 |
| case30_ieee | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.092 | 0.108 | 0.851x | 0.016 |
| case30_ieee | cuda_edge | `NR.iteration.total.total_sec` | 0.692 | 2.365 | 0.292x | 1.674 |
| case30_ieee | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.061 | 0.068 | 0.886x | 0.008 |
| case30_ieee | cuda_edge | `NR.solve.download.total_sec` | 0.015 | 0.017 | 0.874x | 0.002 |
| case30_ieee | cuda_edge | `NR.solve.total.total_sec` | 0.758 | 2.443 | 0.310x | 1.685 |
| case30_ieee | cuda_edge | `NR.solve.upload.total_sec` | 0.048 | 0.057 | 0.836x | 0.009 |
| case30_ieee | cuda_edge | `analyze_sec` | 10.318 | 16.593 | 0.622x | 6.276 |
| case30_ieee | cuda_edge | `elapsed_sec` | 11.077 | 19.038 | 0.582x | 7.961 |
| case30_ieee | cuda_edge | `solve_sec` | 0.759 | 2.444 | 0.311x | 1.685 |
| case4601_goc | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 29.045 | 36.974 | 0.786x | 7.929 |
| case4601_goc | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.654 | 1.919 | 0.862x | 0.266 |
| case4601_goc | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.588 | 1.461 | 0.402x | 0.873 |
| case4601_goc | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 1.851 | 4.987 | 0.371x | 3.136 |
| case4601_goc | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.248 | 0.244 | 1.015x | -0.004 |
| case4601_goc | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.106 | 3.979 | 0.278x | 2.873 |
| case4601_goc | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 1.502 | 1.504 | 0.999x | 0.002 |
| case4601_goc | cuda_edge | `NR.analyze.linear_solve.total_sec` | 30.702 | 38.896 | 0.789x | 8.194 |
| case4601_goc | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.494 | 0.524 | 0.943x | 0.030 |
| case4601_goc | cuda_edge | `NR.analyze.total.total_sec` | 32.700 | 40.927 | 0.799x | 8.227 |
| case4601_goc | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.072 | 0.074 | 0.978x | 0.002 |
| case4601_goc | cuda_edge | `NR.iteration.linear_solve.total_sec` | 3.806 | 10.686 | 0.356x | 6.880 |
| case4601_goc | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.383 | 0.383 | 1.001x | -0.000 |
| case4601_goc | cuda_edge | `NR.iteration.total.total_sec` | 4.375 | 11.259 | 0.389x | 6.884 |
| case4601_goc | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.107 | 0.107 | 0.996x | 0.000 |
| case4601_goc | cuda_edge | `NR.solve.download.total_sec` | 0.055 | 0.056 | 0.995x | 0.000 |
| case4601_goc | cuda_edge | `NR.solve.total.total_sec` | 4.783 | 11.611 | 0.412x | 6.828 |
| case4601_goc | cuda_edge | `NR.solve.upload.total_sec` | 0.349 | 0.293 | 1.193x | -0.056 |
| case4601_goc | cuda_edge | `analyze_sec` | 32.701 | 40.928 | 0.799x | 8.227 |
| case4601_goc | cuda_edge | `elapsed_sec` | 37.486 | 52.541 | 0.713x | 15.055 |
| case4601_goc | cuda_edge | `solve_sec` | 4.785 | 11.612 | 0.412x | 6.828 |
| case793_goc | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 12.024 | 18.096 | 0.664x | 6.071 |
| case793_goc | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.627 | 1.861 | 0.874x | 0.234 |
| case793_goc | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.246 | 0.685 | 0.359x | 0.439 |
| case793_goc | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.454 | 1.649 | 0.276x | 1.194 |
| case793_goc | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.083 | 0.085 | 0.971x | 0.003 |
| case793_goc | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.428 | 2.148 | 0.199x | 1.720 |
| case793_goc | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.228 | 0.235 | 0.966x | 0.008 |
| case793_goc | cuda_edge | `NR.analyze.linear_solve.total_sec` | 13.654 | 19.960 | 0.684x | 6.306 |
| case793_goc | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.292 | 0.302 | 0.968x | 0.010 |
| case793_goc | cuda_edge | `NR.analyze.total.total_sec` | 14.176 | 20.499 | 0.692x | 6.323 |
| case793_goc | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.060 | 0.063 | 0.952x | 0.003 |
| case793_goc | cuda_edge | `NR.iteration.linear_solve.total_sec` | 1.218 | 4.574 | 0.266x | 3.357 |
| case793_goc | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.171 | 0.182 | 0.942x | 0.011 |
| case793_goc | cuda_edge | `NR.iteration.total.total_sec` | 1.558 | 4.931 | 0.316x | 3.373 |
| case793_goc | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.102 | 0.104 | 0.976x | 0.003 |
| case793_goc | cuda_edge | `NR.solve.download.total_sec` | 0.024 | 0.025 | 0.959x | 0.001 |
| case793_goc | cuda_edge | `NR.solve.total.total_sec` | 1.680 | 5.057 | 0.332x | 3.377 |
| case793_goc | cuda_edge | `NR.solve.upload.total_sec` | 0.095 | 0.097 | 0.977x | 0.002 |
| case793_goc | cuda_edge | `analyze_sec` | 14.176 | 20.500 | 0.692x | 6.324 |
| case793_goc | cuda_edge | `elapsed_sec` | 15.858 | 25.558 | 0.620x | 9.700 |
| case793_goc | cuda_edge | `solve_sec` | 1.681 | 5.058 | 0.332x | 3.377 |
| case8387_pegase | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 42.945 | 51.741 | 0.830x | 8.796 |
| case8387_pegase | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.664 | 1.861 | 0.895x | 0.196 |
| case8387_pegase | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.708 | 1.642 | 0.431x | 0.934 |
| case8387_pegase | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 2.876 | 7.063 | 0.407x | 4.187 |
| case8387_pegase | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.440 | 0.438 | 1.004x | -0.002 |
| case8387_pegase | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.478 | 4.584 | 0.322x | 3.106 |
| case8387_pegase | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 2.980 | 3.041 | 0.980x | 0.061 |
| case8387_pegase | cuda_edge | `NR.analyze.linear_solve.total_sec` | 44.613 | 53.605 | 0.832x | 8.992 |
| case8387_pegase | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.804 | 0.818 | 0.983x | 0.014 |
| case8387_pegase | cuda_edge | `NR.analyze.total.total_sec` | 48.399 | 57.466 | 0.842x | 9.067 |
| case8387_pegase | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.100 | 0.101 | 0.990x | 0.001 |
| case8387_pegase | cuda_edge | `NR.iteration.linear_solve.total_sec` | 5.523 | 13.748 | 0.402x | 8.225 |
| case8387_pegase | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.705 | 0.681 | 1.035x | -0.024 |
| case8387_pegase | cuda_edge | `NR.iteration.total.total_sec` | 6.469 | 14.673 | 0.441x | 8.204 |
| case8387_pegase | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.130 | 0.133 | 0.978x | 0.003 |
| case8387_pegase | cuda_edge | `NR.solve.download.total_sec` | 0.086 | 0.086 | 0.999x | 0.000 |
| case8387_pegase | cuda_edge | `NR.solve.total.total_sec` | 7.166 | 15.272 | 0.469x | 8.106 |
| case8387_pegase | cuda_edge | `NR.solve.upload.total_sec` | 0.606 | 0.509 | 1.191x | -0.097 |
| case8387_pegase | cuda_edge | `analyze_sec` | 48.400 | 57.467 | 0.842x | 9.067 |
| case8387_pegase | cuda_edge | `elapsed_sec` | 55.568 | 72.742 | 0.764x | 17.174 |
| case8387_pegase | cuda_edge | `solve_sec` | 7.167 | 15.274 | 0.469x | 8.107 |
| case9241_pegase | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 46.858 | 55.759 | 0.840x | 8.901 |
| case9241_pegase | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.669 | 1.927 | 0.866x | 0.258 |
| case9241_pegase | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.720 | 1.753 | 0.411x | 1.033 |
| case9241_pegase | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 3.721 | 9.107 | 0.409x | 5.386 |
| case9241_pegase | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.587 | 0.567 | 1.036x | -0.021 |
| case9241_pegase | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.736 | 5.401 | 0.321x | 3.665 |
| case9241_pegase | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 3.404 | 3.422 | 0.995x | 0.018 |
| case9241_pegase | cuda_edge | `NR.analyze.linear_solve.total_sec` | 48.530 | 57.689 | 0.841x | 9.159 |
| case9241_pegase | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.867 | 0.903 | 0.960x | 0.036 |
| case9241_pegase | cuda_edge | `NR.analyze.total.total_sec` | 52.803 | 62.016 | 0.851x | 9.213 |
| case9241_pegase | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.134 | 0.133 | 1.005x | -0.001 |
| case9241_pegase | cuda_edge | `NR.iteration.linear_solve.total_sec` | 6.791 | 16.856 | 0.403x | 10.065 |
| case9241_pegase | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.876 | 0.842 | 1.041x | -0.034 |
| case9241_pegase | cuda_edge | `NR.iteration.total.total_sec` | 7.968 | 17.998 | 0.443x | 10.030 |
| case9241_pegase | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.156 | 0.156 | 1.000x | 0.000 |
| case9241_pegase | cuda_edge | `NR.solve.download.total_sec` | 0.093 | 0.091 | 1.022x | -0.002 |
| case9241_pegase | cuda_edge | `NR.solve.total.total_sec` | 8.622 | 18.648 | 0.462x | 10.026 |
| case9241_pegase | cuda_edge | `NR.solve.upload.total_sec` | 0.555 | 0.554 | 1.002x | -0.001 |
| case9241_pegase | cuda_edge | `analyze_sec` | 52.805 | 62.017 | 0.851x | 9.213 |
| case9241_pegase | cuda_edge | `elapsed_sec` | 61.428 | 80.667 | 0.761x | 19.240 |
| case9241_pegase | cuda_edge | `solve_sec` | 8.623 | 18.650 | 0.462x | 10.027 |

## Files

- `manifest.json`: configuration, commands, and environment
- `summary.csv`: one row per repeat
- `aggregates.csv`: grouped timing statistics
- `mg_comparison.csv`: top-level baseline vs MG timing
- `operator_comparison.csv`: all collected timers, baseline vs MG
- `raw/`: per-repeat parsed payload and stdout
