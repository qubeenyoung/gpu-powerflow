# cuDSS MG Newton Solver Benchmark `texas_univ_12cases_r10_gpu23`

## Setup

- Created UTC: 2026-04-16T04:24:17.969995+00:00
- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`
- Cases: case_ACTIVSg200, case_ACTIVSg500, MemphisCase2026_Mar7, case_ACTIVSg2000, Base_Florida_42GW, Texas7k_20220923, Base_Texas_66GW, Base_MIOHIN_76GW, Base_West_Interconnect_121GW, case_ACTIVSg25k, case_ACTIVSg70k, Base_Eastern_Interconnect_515GW
- Profiles: cuda_edge
- Modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS MG indices: `0,1`
- cuDSS reordering: `DEFAULT`
- cuDSS MT: False
- cuDSS host threads: `AUTO`
- cuDSS ND_NLEVELS: `AUTO`
- CUDA_VISIBLE_DEVICES: `2,3`

## end2end

| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Base_Eastern_Interconnect_515GW | cuda_edge | 399.269 | 457.003 | 0.874x | 39.952 | 72.316 | 0.552x | True/True | 6.7/6.9 |
| Base_Florida_42GW | cuda_edge | 40.908 | 55.116 | 0.742x | 4.082 | 10.045 | 0.406x | True/True | 5.0/5.0 |
| Base_MIOHIN_76GW | cuda_edge | 68.344 | 79.069 | 0.864x | 6.539 | 10.024 | 0.652x | True/True | 5.0/5.0 |
| Base_Texas_66GW | cuda_edge | 52.335 | 65.171 | 0.803x | 5.876 | 11.184 | 0.525x | True/True | 5.0/5.0 |
| Base_West_Interconnect_121GW | cuda_edge | 107.208 | 122.819 | 0.873x | 9.224 | 15.145 | 0.609x | True/True | 5.8/5.8 |
| MemphisCase2026_Mar7 | cuda_edge | 16.369 | 24.216 | 0.676x | 1.089 | 2.691 | 0.405x | True/True | 3.0/3.0 |
| Texas7k_20220923 | cuda_edge | 43.296 | 56.822 | 0.762x | 3.632 | 8.761 | 0.415x | True/True | 4.0/4.0 |
| case_ACTIVSg200 | cuda_edge | 11.811 | 19.298 | 0.612x | 0.580 | 1.774 | 0.327x | True/True | 3.0/3.0 |
| case_ACTIVSg2000 | cuda_edge | 22.547 | 32.546 | 0.693x | 2.172 | 5.659 | 0.384x | True/True | 4.0/4.0 |
| case_ACTIVSg25k | cuda_edge | 119.364 | 134.240 | 0.889x | 8.650 | 13.401 | 0.645x | True/True | 5.0/5.0 |
| case_ACTIVSg500 | cuda_edge | 13.706 | 22.071 | 0.621x | 1.013 | 3.106 | 0.326x | True/True | 4.0/4.0 |
| case_ACTIVSg70k | cuda_edge | 322.791 | 342.624 | 0.942x | 29.989 | 37.068 | 0.809x | True/True | 7.1/7.0 |

## operators

| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Base_Eastern_Interconnect_515GW | cuda_edge | 397.912 | 454.654 | 0.875x | 38.931 | 78.336 | 0.497x | True/True | 6.4/6.6 |
| Base_Florida_42GW | cuda_edge | 41.037 | 55.138 | 0.744x | 4.230 | 10.083 | 0.419x | True/True | 5.0/5.0 |
| Base_MIOHIN_76GW | cuda_edge | 67.667 | 79.289 | 0.853x | 6.450 | 10.162 | 0.635x | True/True | 5.0/5.0 |
| Base_Texas_66GW | cuda_edge | 50.542 | 61.395 | 0.823x | 5.177 | 8.635 | 0.600x | True/True | 5.0/5.0 |
| Base_West_Interconnect_121GW | cuda_edge | 110.694 | 136.253 | 0.812x | 11.436 | 25.239 | 0.453x | True/True | 5.7/5.9 |
| MemphisCase2026_Mar7 | cuda_edge | 16.445 | 24.287 | 0.677x | 1.162 | 2.724 | 0.427x | True/True | 3.0/3.0 |
| Texas7k_20220923 | cuda_edge | 43.415 | 57.010 | 0.762x | 3.727 | 8.795 | 0.424x | True/True | 4.0/4.0 |
| case_ACTIVSg200 | cuda_edge | 11.885 | 18.781 | 0.633x | 0.631 | 1.639 | 0.385x | True/True | 3.0/3.0 |
| case_ACTIVSg2000 | cuda_edge | 22.737 | 32.620 | 0.697x | 2.309 | 5.710 | 0.404x | True/True | 4.0/4.0 |
| case_ACTIVSg25k | cuda_edge | 122.871 | 146.336 | 0.840x | 10.995 | 22.116 | 0.497x | True/True | 5.0/5.0 |
| case_ACTIVSg500 | cuda_edge | 13.804 | 22.195 | 0.622x | 1.115 | 3.229 | 0.345x | True/True | 4.0/4.0 |
| case_ACTIVSg70k | cuda_edge | 336.060 | 389.648 | 0.862x | 37.463 | 73.724 | 0.508x | True/True | 7.0/7.0 |

## Operator Metrics

| case | profile | metric | baseline ms | MG ms | speedup | delta ms |
|---|---|---|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 321.476 | 339.466 | 0.947x | 17.989 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.840 | 2.071 | 0.888x | 0.232 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 3.623 | 9.873 | 0.367x | 6.250 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 15.363 | 42.679 | 0.360x | 27.316 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 3.201 | 3.180 | 1.006x | -0.020 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 5.555 | 11.722 | 0.474x | 6.167 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 29.168 | 28.867 | 1.010x | -0.301 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 323.327 | 341.543 | 0.947x | 18.217 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 6.476 | 5.898 | 1.098x | -0.579 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.analyze.total.total_sec` | 358.977 | 376.314 | 0.954x | 17.337 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.436 | 0.456 | 0.958x | 0.019 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 27.899 | 67.610 | 0.413x | 39.711 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 4.379 | 4.468 | 0.980x | 0.090 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.total.total_sec` | 33.026 | 72.872 | 0.453x | 39.846 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.302 | 0.327 | 0.924x | 0.025 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.solve.download.total_sec` | 0.656 | 0.648 | 1.012x | -0.008 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.solve.total.total_sec` | 38.928 | 78.333 | 0.497x | 39.405 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.solve.upload.total_sec` | 5.240 | 4.807 | 1.090x | -0.433 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `analyze_sec` | 358.980 | 376.317 | 0.954x | 17.337 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `elapsed_sec` | 397.912 | 454.654 | 0.875x | 56.742 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `solve_sec` | 38.931 | 78.336 | 0.497x | 39.405 |
| Base_Florida_42GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 32.639 | 40.712 | 0.802x | 8.072 |
| Base_Florida_42GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.616 | 1.810 | 0.893x | 0.195 |
| Base_Florida_42GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.577 | 1.573 | 0.367x | 0.996 |
| Base_Florida_42GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 1.427 | 4.020 | 0.355x | 2.593 |
| Base_Florida_42GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.238 | 0.246 | 0.969x | 0.008 |
| Base_Florida_42GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.923 | 3.216 | 0.287x | 2.293 |
| Base_Florida_42GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 1.935 | 1.916 | 1.010x | -0.020 |
| Base_Florida_42GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 34.258 | 42.525 | 0.806x | 8.267 |
| Base_Florida_42GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.610 | 0.610 | 1.000x | 0.000 |
| Base_Florida_42GW | cuda_edge | `NR.analyze.total.total_sec` | 36.805 | 45.053 | 0.817x | 8.248 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.065 | 0.067 | 0.976x | 0.002 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 3.179 | 9.069 | 0.350x | 5.891 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.371 | 0.375 | 0.988x | 0.005 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.total.total_sec` | 3.706 | 9.604 | 0.386x | 5.898 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.085 | 0.087 | 0.975x | 0.002 |
| Base_Florida_42GW | cuda_edge | `NR.solve.download.total_sec` | 0.064 | 0.085 | 0.748x | 0.021 |
| Base_Florida_42GW | cuda_edge | `NR.solve.total.total_sec` | 4.228 | 10.082 | 0.419x | 5.853 |
| Base_Florida_42GW | cuda_edge | `NR.solve.upload.total_sec` | 0.455 | 0.389 | 1.168x | -0.066 |
| Base_Florida_42GW | cuda_edge | `analyze_sec` | 36.807 | 45.055 | 0.817x | 8.248 |
| Base_Florida_42GW | cuda_edge | `elapsed_sec` | 41.037 | 55.138 | 0.744x | 14.101 |
| Base_Florida_42GW | cuda_edge | `solve_sec` | 4.230 | 10.083 | 0.419x | 5.854 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 55.173 | 63.330 | 0.871x | 8.158 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.656 | 1.780 | 0.930x | 0.124 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.863 | 1.575 | 0.548x | 0.712 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 2.341 | 4.043 | 0.579x | 1.702 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.390 | 0.171 | 2.276x | -0.219 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.211 | 3.444 | 0.352x | 2.233 |
| Base_MIOHIN_76GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 3.517 | 3.487 | 1.008x | -0.030 |
| Base_MIOHIN_76GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 56.831 | 65.113 | 0.873x | 8.281 |
| Base_MIOHIN_76GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.864 | 0.522 | 1.655x | -0.342 |
| Base_MIOHIN_76GW | cuda_edge | `NR.analyze.total.total_sec` | 61.215 | 69.125 | 0.886x | 7.910 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.079 | 0.073 | 1.088x | -0.006 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 4.824 | 9.251 | 0.521x | 4.427 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.605 | 0.315 | 1.920x | -0.290 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.total.total_sec` | 5.604 | 9.731 | 0.576x | 4.126 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.089 | 0.084 | 1.056x | -0.005 |
| Base_MIOHIN_76GW | cuda_edge | `NR.solve.download.total_sec` | 0.103 | 0.046 | 2.206x | -0.056 |
| Base_MIOHIN_76GW | cuda_edge | `NR.solve.total.total_sec` | 6.449 | 10.160 | 0.635x | 3.712 |
| Base_MIOHIN_76GW | cuda_edge | `NR.solve.upload.total_sec` | 0.738 | 0.380 | 1.945x | -0.359 |
| Base_MIOHIN_76GW | cuda_edge | `analyze_sec` | 61.216 | 69.126 | 0.886x | 7.910 |
| Base_MIOHIN_76GW | cuda_edge | `elapsed_sec` | 67.667 | 79.289 | 0.853x | 11.622 |
| Base_MIOHIN_76GW | cuda_edge | `solve_sec` | 6.450 | 10.162 | 0.635x | 3.712 |
| Base_Texas_66GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 40.677 | 48.161 | 0.845x | 7.484 |
| Base_Texas_66GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.617 | 1.748 | 0.925x | 0.132 |
| Base_Texas_66GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.705 | 1.282 | 0.550x | 0.578 |
| Base_Texas_66GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 1.832 | 3.284 | 0.558x | 1.452 |
| Base_Texas_66GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.290 | 0.138 | 2.106x | -0.152 |
| Base_Texas_66GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.069 | 3.150 | 0.339x | 2.081 |
| Base_Texas_66GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 2.399 | 2.389 | 1.004x | -0.010 |
| Base_Texas_66GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 42.297 | 49.913 | 0.847x | 7.615 |
| Base_Texas_66GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.664 | 0.455 | 1.462x | -0.210 |
| Base_Texas_66GW | cuda_edge | `NR.analyze.total.total_sec` | 45.363 | 52.759 | 0.860x | 7.396 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.066 | 0.061 | 1.084x | -0.005 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 3.909 | 7.869 | 0.497x | 3.960 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.440 | 0.242 | 1.816x | -0.198 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.total.total_sec` | 4.509 | 8.263 | 0.546x | 3.754 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.087 | 0.084 | 1.031x | -0.003 |
| Base_Texas_66GW | cuda_edge | `NR.solve.download.total_sec` | 0.080 | 0.040 | 1.975x | -0.039 |
| Base_Texas_66GW | cuda_edge | `NR.solve.total.total_sec` | 5.176 | 8.633 | 0.600x | 3.458 |
| Base_Texas_66GW | cuda_edge | `NR.solve.upload.total_sec` | 0.584 | 0.327 | 1.789x | -0.258 |
| Base_Texas_66GW | cuda_edge | `analyze_sec` | 45.364 | 52.760 | 0.860x | 7.396 |
| Base_Texas_66GW | cuda_edge | `elapsed_sec` | 50.542 | 61.395 | 0.823x | 10.853 |
| Base_Texas_66GW | cuda_edge | `solve_sec` | 5.177 | 8.635 | 0.600x | 3.457 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 89.008 | 100.771 | 0.883x | 11.763 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.777 | 1.868 | 0.952x | 0.090 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 1.219 | 3.550 | 0.343x | 2.332 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 4.252 | 12.242 | 0.347x | 7.990 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.864 | 0.874 | 0.988x | 0.010 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 2.049 | 5.427 | 0.378x | 3.378 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 6.945 | 6.845 | 1.015x | -0.100 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 90.788 | 102.643 | 0.885x | 11.854 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 1.521 | 1.520 | 1.001x | -0.001 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.analyze.total.total_sec` | 99.256 | 111.011 | 0.894x | 11.755 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.118 | 0.124 | 0.948x | 0.006 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 8.419 | 22.132 | 0.380x | 13.713 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 1.252 | 1.272 | 0.985x | 0.020 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.total.total_sec` | 9.958 | 23.686 | 0.420x | 13.728 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.161 | 0.149 | 1.078x | -0.012 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.solve.download.total_sec` | 0.193 | 0.191 | 1.010x | -0.002 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.solve.total.total_sec` | 11.434 | 25.237 | 0.453x | 13.802 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.solve.upload.total_sec` | 1.279 | 1.356 | 0.943x | 0.077 |
| Base_West_Interconnect_121GW | cuda_edge | `analyze_sec` | 99.257 | 111.013 | 0.894x | 11.756 |
| Base_West_Interconnect_121GW | cuda_edge | `elapsed_sec` | 110.694 | 136.253 | 0.812x | 25.559 |
| Base_West_Interconnect_121GW | cuda_edge | `solve_sec` | 11.436 | 25.239 | 0.453x | 13.803 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 13.130 | 19.186 | 0.684x | 6.056 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.576 | 1.764 | 0.894x | 0.188 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.321 | 0.746 | 0.429x | 0.426 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.209 | 0.541 | 0.386x | 0.332 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.039 | 0.039 | 1.000x | 0.000 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.264 | 1.063 | 0.248x | 0.799 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.292 | 0.304 | 0.963x | 0.011 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 14.709 | 20.953 | 0.702x | 6.244 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.278 | 0.303 | 0.918x | 0.025 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.analyze.total.total_sec` | 15.282 | 21.562 | 0.709x | 6.280 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.026 | 0.027 | 0.967x | 0.001 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.836 | 2.392 | 0.349x | 1.557 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.106 | 0.108 | 0.981x | 0.002 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.total.total_sec` | 1.013 | 2.573 | 0.394x | 1.560 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.042 | 0.043 | 0.986x | 0.001 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.solve.download.total_sec` | 0.026 | 0.026 | 1.004x | -0.000 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.solve.total.total_sec` | 1.161 | 2.723 | 0.426x | 1.562 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.solve.upload.total_sec` | 0.119 | 0.121 | 0.982x | 0.002 |
| MemphisCase2026_Mar7 | cuda_edge | `analyze_sec` | 15.283 | 21.563 | 0.709x | 6.280 |
| MemphisCase2026_Mar7 | cuda_edge | `elapsed_sec` | 16.445 | 24.287 | 0.677x | 7.842 |
| MemphisCase2026_Mar7 | cuda_edge | `solve_sec` | 1.162 | 2.724 | 0.427x | 1.562 |
| Texas7k_20220923 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 35.375 | 43.688 | 0.810x | 8.313 |
| Texas7k_20220923 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.618 | 1.842 | 0.879x | 0.223 |
| Texas7k_20220923 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.653 | 1.775 | 0.368x | 1.121 |
| Texas7k_20220923 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 1.103 | 3.106 | 0.355x | 2.002 |
| Texas7k_20220923 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.194 | 0.199 | 0.975x | 0.005 |
| Texas7k_20220923 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.726 | 2.708 | 0.268x | 1.982 |
| Texas7k_20220923 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 2.040 | 2.033 | 1.004x | -0.008 |
| Texas7k_20220923 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 36.997 | 45.533 | 0.813x | 8.536 |
| Texas7k_20220923 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.648 | 0.644 | 1.005x | -0.004 |
| Texas7k_20220923 | cuda_edge | `NR.analyze.total.total_sec` | 39.687 | 48.213 | 0.823x | 8.526 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.050 | 0.049 | 1.016x | -0.001 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 2.687 | 7.798 | 0.345x | 5.111 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.328 | 0.336 | 0.977x | 0.008 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.total.total_sec` | 3.137 | 8.253 | 0.380x | 5.116 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.067 | 0.065 | 1.026x | -0.002 |
| Texas7k_20220923 | cuda_edge | `NR.solve.download.total_sec` | 0.073 | 0.075 | 0.980x | 0.001 |
| Texas7k_20220923 | cuda_edge | `NR.solve.total.total_sec` | 3.726 | 8.794 | 0.424x | 5.068 |
| Texas7k_20220923 | cuda_edge | `NR.solve.upload.total_sec` | 0.513 | 0.463 | 1.107x | -0.050 |
| Texas7k_20220923 | cuda_edge | `analyze_sec` | 39.688 | 48.214 | 0.823x | 8.526 |
| Texas7k_20220923 | cuda_edge | `elapsed_sec` | 43.415 | 57.010 | 0.762x | 13.595 |
| Texas7k_20220923 | cuda_edge | `solve_sec` | 3.727 | 8.795 | 0.424x | 5.068 |
| case_ACTIVSg200 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 9.364 | 15.144 | 0.618x | 5.779 |
| case_ACTIVSg200 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.570 | 1.714 | 0.916x | 0.145 |
| case_ACTIVSg200 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.109 | 0.343 | 0.318x | 0.234 |
| case_ACTIVSg200 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.102 | 0.328 | 0.312x | 0.226 |
| case_ACTIVSg200 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.025 | 0.021 | 1.229x | -0.005 |
| case_ACTIVSg200 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.153 | 0.736 | 0.209x | 0.582 |
| case_ACTIVSg200 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.065 | 0.063 | 1.017x | -0.001 |
| case_ACTIVSg200 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 10.937 | 16.861 | 0.649x | 5.924 |
| case_ACTIVSg200 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.249 | 0.213 | 1.168x | -0.036 |
| case_ACTIVSg200 | cuda_edge | `NR.analyze.total.total_sec` | 11.252 | 17.140 | 0.657x | 5.887 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.026 | 0.022 | 1.151x | -0.003 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.393 | 1.430 | 0.275x | 1.037 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.079 | 0.072 | 1.093x | -0.007 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.total.total_sec` | 0.542 | 1.567 | 0.346x | 1.025 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.041 | 0.039 | 1.054x | -0.002 |
| case_ACTIVSg200 | cuda_edge | `NR.solve.download.total_sec` | 0.019 | 0.016 | 1.175x | -0.003 |
| case_ACTIVSg200 | cuda_edge | `NR.solve.total.total_sec` | 0.630 | 1.638 | 0.385x | 1.008 |
| case_ACTIVSg200 | cuda_edge | `NR.solve.upload.total_sec` | 0.067 | 0.053 | 1.257x | -0.014 |
| case_ACTIVSg200 | cuda_edge | `analyze_sec` | 11.253 | 17.141 | 0.657x | 5.888 |
| case_ACTIVSg200 | cuda_edge | `elapsed_sec` | 11.885 | 18.781 | 0.633x | 6.896 |
| case_ACTIVSg200 | cuda_edge | `solve_sec` | 0.631 | 1.639 | 0.385x | 1.008 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 17.879 | 24.142 | 0.741x | 6.263 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.593 | 1.819 | 0.876x | 0.226 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.467 | 1.109 | 0.421x | 0.642 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.715 | 1.839 | 0.389x | 1.124 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.081 | 0.082 | 0.995x | 0.000 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.537 | 2.169 | 0.248x | 1.632 |
| case_ACTIVSg2000 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.610 | 0.605 | 1.008x | -0.005 |
| case_ACTIVSg2000 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 19.475 | 25.963 | 0.750x | 6.489 |
| case_ACTIVSg2000 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.340 | 0.337 | 1.009x | -0.003 |
| case_ACTIVSg2000 | cuda_edge | `NR.analyze.total.total_sec` | 20.427 | 26.908 | 0.759x | 6.481 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.040 | 0.040 | 1.010x | -0.000 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 1.806 | 5.206 | 0.347x | 3.400 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.170 | 0.172 | 0.987x | 0.002 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.total.total_sec` | 2.086 | 5.488 | 0.380x | 3.402 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.064 | 0.065 | 0.991x | 0.001 |
| case_ACTIVSg2000 | cuda_edge | `NR.solve.download.total_sec` | 0.036 | 0.035 | 1.014x | -0.000 |
| case_ACTIVSg2000 | cuda_edge | `NR.solve.total.total_sec` | 2.308 | 5.709 | 0.404x | 3.402 |
| case_ACTIVSg2000 | cuda_edge | `NR.solve.upload.total_sec` | 0.183 | 0.183 | 0.999x | 0.000 |
| case_ACTIVSg2000 | cuda_edge | `analyze_sec` | 20.428 | 26.909 | 0.759x | 6.481 |
| case_ACTIVSg2000 | cuda_edge | `elapsed_sec` | 22.737 | 32.620 | 0.697x | 9.883 |
| case_ACTIVSg2000 | cuda_edge | `solve_sec` | 2.309 | 5.710 | 0.404x | 3.402 |
| case_ACTIVSg25k | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 100.901 | 113.178 | 0.892x | 12.278 |
| case_ACTIVSg25k | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.725 | 1.820 | 0.948x | 0.094 |
| case_ACTIVSg25k | cuda_edge | `CUDA.solve.factorization32.total_sec` | 1.280 | 3.576 | 0.358x | 2.296 |
| case_ACTIVSg25k | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 3.626 | 9.338 | 0.388x | 5.712 |
| case_ACTIVSg25k | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.864 | 0.827 | 1.045x | -0.037 |
| case_ACTIVSg25k | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.938 | 5.073 | 0.382x | 3.136 |
| case_ACTIVSg25k | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 7.474 | 7.451 | 1.003x | -0.022 |
| case_ACTIVSg25k | cuda_edge | `NR.analyze.linear_solve.total_sec` | 102.630 | 115.002 | 0.892x | 12.372 |
| case_ACTIVSg25k | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 1.769 | 1.761 | 1.004x | -0.008 |
| case_ACTIVSg25k | cuda_edge | `NR.analyze.total.total_sec` | 111.875 | 124.218 | 0.901x | 12.343 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.105 | 0.107 | 0.983x | 0.002 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.linear_solve.total_sec` | 7.746 | 18.851 | 0.411x | 11.105 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.mismatch.total_sec` | 1.267 | 1.251 | 1.013x | -0.016 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.total.total_sec` | 9.250 | 20.355 | 0.454x | 11.104 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.124 | 0.138 | 0.897x | 0.014 |
| case_ACTIVSg25k | cuda_edge | `NR.solve.download.total_sec` | 0.223 | 0.224 | 0.992x | 0.002 |
| case_ACTIVSg25k | cuda_edge | `NR.solve.total.total_sec` | 10.993 | 22.114 | 0.497x | 11.121 |
| case_ACTIVSg25k | cuda_edge | `NR.solve.upload.total_sec` | 1.516 | 1.531 | 0.990x | 0.015 |
| case_ACTIVSg25k | cuda_edge | `analyze_sec` | 111.877 | 124.220 | 0.901x | 12.343 |
| case_ACTIVSg25k | cuda_edge | `elapsed_sec` | 122.871 | 146.336 | 0.840x | 23.465 |
| case_ACTIVSg25k | cuda_edge | `solve_sec` | 10.995 | 22.116 | 0.497x | 11.121 |
| case_ACTIVSg500 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 10.702 | 16.783 | 0.638x | 6.082 |
| case_ACTIVSg500 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.580 | 1.760 | 0.898x | 0.180 |
| case_ACTIVSg500 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.228 | 0.636 | 0.359x | 0.407 |
| case_ACTIVSg500 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.243 | 0.854 | 0.285x | 0.611 |
| case_ACTIVSg500 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.047 | 0.048 | 0.981x | 0.001 |
| case_ACTIVSg500 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.257 | 1.336 | 0.192x | 1.080 |
| case_ACTIVSg500 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.143 | 0.146 | 0.985x | 0.002 |
| case_ACTIVSg500 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 12.284 | 18.547 | 0.662x | 6.262 |
| case_ACTIVSg500 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.257 | 0.271 | 0.950x | 0.014 |
| case_ACTIVSg500 | cuda_edge | `NR.analyze.total.total_sec` | 12.687 | 18.964 | 0.669x | 6.277 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.038 | 0.040 | 0.953x | 0.002 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.779 | 2.878 | 0.271x | 2.099 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.118 | 0.124 | 0.953x | 0.006 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.total.total_sec` | 1.004 | 3.112 | 0.323x | 2.109 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.063 | 0.065 | 0.972x | 0.002 |
| case_ACTIVSg500 | cuda_edge | `NR.solve.download.total_sec` | 0.022 | 0.022 | 0.964x | 0.001 |
| case_ACTIVSg500 | cuda_edge | `NR.solve.total.total_sec` | 1.114 | 3.228 | 0.345x | 2.114 |
| case_ACTIVSg500 | cuda_edge | `NR.solve.upload.total_sec` | 0.086 | 0.090 | 0.950x | 0.005 |
| case_ACTIVSg500 | cuda_edge | `analyze_sec` | 12.688 | 18.966 | 0.669x | 6.278 |
| case_ACTIVSg500 | cuda_edge | `elapsed_sec` | 13.804 | 22.195 | 0.622x | 8.392 |
| case_ACTIVSg500 | cuda_edge | `solve_sec` | 1.115 | 3.229 | 0.345x | 2.114 |
| case_ACTIVSg70k | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 269.121 | 286.256 | 0.940x | 17.135 |
| case_ACTIVSg70k | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.783 | 2.010 | 0.887x | 0.228 |
| case_ACTIVSg70k | cuda_edge | `CUDA.solve.factorization32.total_sec` | 3.033 | 8.365 | 0.363x | 5.332 |
| case_ACTIVSg70k | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 14.891 | 40.237 | 0.370x | 25.346 |
| case_ACTIVSg70k | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 3.305 | 3.252 | 1.016x | -0.053 |
| case_ACTIVSg70k | cuda_edge | `CUDA.solve.solve32.total_sec` | 5.971 | 11.802 | 0.506x | 5.830 |
| case_ACTIVSg70k | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 22.524 | 22.418 | 1.005x | -0.106 |
| case_ACTIVSg70k | cuda_edge | `NR.analyze.linear_solve.total_sec` | 270.909 | 288.272 | 0.940x | 17.363 |
| case_ACTIVSg70k | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 5.154 | 5.224 | 0.987x | 0.070 |
| case_ACTIVSg70k | cuda_edge | `NR.analyze.total.total_sec` | 298.593 | 315.920 | 0.945x | 17.327 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.391 | 0.394 | 0.991x | 0.003 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.linear_solve.total_sec` | 27.358 | 63.797 | 0.429x | 36.438 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.mismatch.total_sec` | 4.424 | 4.380 | 1.010x | -0.044 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.total.total_sec` | 32.476 | 68.874 | 0.472x | 36.398 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.292 | 0.292 | 0.998x | 0.000 |
| case_ACTIVSg70k | cuda_edge | `NR.solve.download.total_sec` | 0.608 | 0.644 | 0.945x | 0.036 |
| case_ACTIVSg70k | cuda_edge | `NR.solve.total.total_sec` | 37.460 | 73.721 | 0.508x | 36.261 |
| case_ACTIVSg70k | cuda_edge | `NR.solve.upload.total_sec` | 4.370 | 4.197 | 1.041x | -0.173 |
| case_ACTIVSg70k | cuda_edge | `analyze_sec` | 298.597 | 315.924 | 0.945x | 17.327 |
| case_ACTIVSg70k | cuda_edge | `elapsed_sec` | 336.060 | 389.648 | 0.862x | 53.588 |
| case_ACTIVSg70k | cuda_edge | `solve_sec` | 37.463 | 73.724 | 0.508x | 36.261 |

## Files

- `manifest.json`: configuration, commands, and environment
- `summary.csv`: one row per repeat
- `aggregates.csv`: grouped timing statistics
- `mg_comparison.csv`: top-level baseline vs MG timing
- `operator_comparison.csv`: all collected timers, baseline vs MG
- `raw/`: per-repeat parsed payload and stdout
