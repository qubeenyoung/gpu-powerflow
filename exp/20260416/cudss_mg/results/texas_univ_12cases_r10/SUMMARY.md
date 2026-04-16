# cuDSS MG Newton Solver Benchmark `texas_univ_12cases_r10`

## Setup

- Created UTC: 2026-04-16T04:12:15.331851+00:00
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
- CUDA_VISIBLE_DEVICES: ``

## end2end

| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Base_Eastern_Interconnect_515GW | cuda_edge | 384.959 | 415.718 | 0.926x | 31.280 | 41.781 | 0.749x | True/True | 6.5/6.7 |
| Base_Florida_42GW | cuda_edge | 41.080 | 51.743 | 0.794x | 4.159 | 7.669 | 0.542x | True/True | 5.0/5.0 |
| Base_MIOHIN_76GW | cuda_edge | 66.051 | 79.347 | 0.832x | 5.428 | 10.119 | 0.536x | True/True | 5.0/5.0 |
| Base_Texas_66GW | cuda_edge | 49.394 | 65.325 | 0.756x | 4.406 | 11.151 | 0.395x | True/True | 5.0/5.0 |
| Base_West_Interconnect_121GW | cuda_edge | 110.269 | 133.440 | 0.826x | 10.514 | 19.735 | 0.533x | True/True | 5.3/6.0 |
| MemphisCase2026_Mar7 | cuda_edge | 16.162 | 24.442 | 0.661x | 0.996 | 2.767 | 0.360x | True/True | 3.0/3.0 |
| Texas7k_20220923 | cuda_edge | 42.373 | 57.376 | 0.739x | 3.111 | 8.734 | 0.356x | True/True | 4.0/4.0 |
| case_ACTIVSg200 | cuda_edge | 11.658 | 18.786 | 0.621x | 0.523 | 1.608 | 0.325x | True/True | 3.0/3.0 |
| case_ACTIVSg2000 | cuda_edge | 22.340 | 31.489 | 0.709x | 2.031 | 5.024 | 0.404x | True/True | 4.0/4.0 |
| case_ACTIVSg25k | cuda_edge | 123.038 | 147.613 | 0.834x | 10.833 | 21.892 | 0.495x | True/True | 5.0/5.0 |
| case_ACTIVSg500 | cuda_edge | 13.540 | 22.279 | 0.608x | 0.956 | 3.209 | 0.298x | True/True | 4.0/4.0 |
| case_ACTIVSg70k | cuda_edge | 322.920 | 343.265 | 0.941x | 29.466 | 37.674 | 0.782x | True/True | 7.0/7.1 |

## operators

| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Base_Eastern_Interconnect_515GW | cuda_edge | 400.525 | 461.142 | 0.869x | 40.197 | 81.678 | 0.492x | True/True | 6.6/6.7 |
| Base_Florida_42GW | cuda_edge | 41.270 | 55.449 | 0.744x | 4.301 | 10.171 | 0.423x | True/True | 5.0/5.0 |
| Base_MIOHIN_76GW | cuda_edge | 67.983 | 85.879 | 0.792x | 6.540 | 14.485 | 0.452x | True/True | 5.0/5.0 |
| Base_Texas_66GW | cuda_edge | 51.047 | 65.634 | 0.778x | 5.220 | 11.215 | 0.465x | True/True | 5.0/5.0 |
| Base_West_Interconnect_121GW | cuda_edge | 111.834 | 135.690 | 0.824x | 11.985 | 23.844 | 0.503x | True/True | 5.9/5.6 |
| MemphisCase2026_Mar7 | cuda_edge | 16.574 | 24.534 | 0.676x | 1.184 | 2.865 | 0.413x | True/True | 3.0/3.0 |
| Texas7k_20220923 | cuda_edge | 43.749 | 57.218 | 0.765x | 3.773 | 8.746 | 0.431x | True/True | 4.0/4.0 |
| case_ACTIVSg200 | cuda_edge | 12.038 | 19.384 | 0.621x | 0.634 | 1.765 | 0.359x | True/True | 3.0/3.0 |
| case_ACTIVSg2000 | cuda_edge | 22.822 | 32.814 | 0.695x | 2.323 | 5.746 | 0.404x | True/True | 4.0/4.0 |
| case_ACTIVSg25k | cuda_edge | 123.464 | 146.975 | 0.840x | 11.095 | 22.251 | 0.499x | True/True | 5.0/5.0 |
| case_ACTIVSg500 | cuda_edge | 13.955 | 22.353 | 0.624x | 1.142 | 3.255 | 0.351x | True/True | 4.0/4.0 |
| case_ACTIVSg70k | cuda_edge | 336.643 | 389.510 | 0.864x | 37.553 | 72.912 | 0.515x | True/True | 7.0/7.1 |

## Operator Metrics

| case | profile | metric | baseline ms | MG ms | speedup | delta ms |
|---|---|---|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 322.140 | 341.039 | 0.945x | 18.899 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.846 | 2.129 | 0.867x | 0.283 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 3.636 | 9.918 | 0.367x | 6.282 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 16.147 | 44.930 | 0.359x | 28.783 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 3.276 | 3.272 | 1.001x | -0.004 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 5.739 | 12.312 | 0.466x | 6.574 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 29.414 | 29.454 | 0.999x | 0.040 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 323.991 | 343.174 | 0.944x | 19.183 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 6.912 | 6.826 | 1.013x | -0.086 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.analyze.total.total_sec` | 360.323 | 379.460 | 0.950x | 19.137 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.454 | 0.464 | 0.977x | 0.010 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 28.996 | 70.586 | 0.411x | 41.590 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 4.503 | 4.581 | 0.983x | 0.077 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.total.total_sec` | 34.278 | 75.973 | 0.451x | 41.694 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.314 | 0.332 | 0.946x | 0.018 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.solve.download.total_sec` | 0.677 | 0.662 | 1.022x | -0.014 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.solve.total.total_sec` | 40.194 | 81.675 | 0.492x | 41.481 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `NR.solve.upload.total_sec` | 5.235 | 5.034 | 1.040x | -0.201 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `analyze_sec` | 360.327 | 379.464 | 0.950x | 19.137 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `elapsed_sec` | 400.525 | 461.142 | 0.869x | 60.618 |
| Base_Eastern_Interconnect_515GW | cuda_edge | `solve_sec` | 40.197 | 81.678 | 0.492x | 41.481 |
| Base_Florida_42GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 32.733 | 40.857 | 0.801x | 8.124 |
| Base_Florida_42GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.673 | 1.830 | 0.914x | 0.157 |
| Base_Florida_42GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.607 | 1.594 | 0.381x | 0.987 |
| Base_Florida_42GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 1.436 | 4.054 | 0.354x | 2.617 |
| Base_Florida_42GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.233 | 0.241 | 0.967x | 0.008 |
| Base_Florida_42GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.942 | 3.226 | 0.292x | 2.284 |
| Base_Florida_42GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 1.896 | 1.894 | 1.001x | -0.002 |
| Base_Florida_42GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 34.410 | 42.690 | 0.806x | 8.280 |
| Base_Florida_42GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.659 | 0.690 | 0.956x | 0.031 |
| Base_Florida_42GW | cuda_edge | `NR.analyze.total.total_sec` | 36.967 | 45.276 | 0.816x | 8.309 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.063 | 0.066 | 0.955x | 0.003 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 3.231 | 9.127 | 0.354x | 5.896 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.369 | 0.401 | 0.921x | 0.032 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.total.total_sec` | 3.773 | 9.687 | 0.390x | 5.914 |
| Base_Florida_42GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.103 | 0.087 | 1.191x | -0.017 |
| Base_Florida_42GW | cuda_edge | `NR.solve.download.total_sec` | 0.064 | 0.065 | 0.986x | 0.001 |
| Base_Florida_42GW | cuda_edge | `NR.solve.total.total_sec` | 4.300 | 10.169 | 0.423x | 5.870 |
| Base_Florida_42GW | cuda_edge | `NR.solve.upload.total_sec` | 0.459 | 0.414 | 1.108x | -0.045 |
| Base_Florida_42GW | cuda_edge | `analyze_sec` | 36.968 | 45.278 | 0.816x | 8.309 |
| Base_Florida_42GW | cuda_edge | `elapsed_sec` | 41.270 | 55.449 | 0.744x | 14.179 |
| Base_Florida_42GW | cuda_edge | `solve_sec` | 4.301 | 10.171 | 0.423x | 5.870 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 55.298 | 65.027 | 0.850x | 9.730 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.692 | 1.901 | 0.890x | 0.209 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.898 | 2.370 | 0.379x | 1.472 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 2.373 | 6.280 | 0.378x | 3.907 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.385 | 0.380 | 1.013x | -0.005 |
| Base_MIOHIN_76GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.207 | 3.853 | 0.313x | 2.646 |
| Base_MIOHIN_76GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 3.504 | 3.464 | 1.012x | -0.040 |
| Base_MIOHIN_76GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 56.993 | 66.932 | 0.852x | 9.939 |
| Base_MIOHIN_76GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.942 | 0.994 | 0.947x | 0.052 |
| Base_MIOHIN_76GW | cuda_edge | `NR.analyze.total.total_sec` | 61.441 | 71.392 | 0.861x | 9.952 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.081 | 0.080 | 1.004x | -0.000 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 4.880 | 12.901 | 0.378x | 8.021 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.607 | 0.611 | 0.993x | 0.004 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.total.total_sec` | 5.666 | 13.689 | 0.414x | 8.023 |
| Base_MIOHIN_76GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.091 | 0.088 | 1.026x | -0.002 |
| Base_MIOHIN_76GW | cuda_edge | `NR.solve.download.total_sec` | 0.101 | 0.100 | 1.007x | -0.001 |
| Base_MIOHIN_76GW | cuda_edge | `NR.solve.total.total_sec` | 6.539 | 14.483 | 0.451x | 7.944 |
| Base_MIOHIN_76GW | cuda_edge | `NR.solve.upload.total_sec` | 0.769 | 0.691 | 1.114x | -0.079 |
| Base_MIOHIN_76GW | cuda_edge | `analyze_sec` | 61.442 | 71.394 | 0.861x | 9.952 |
| Base_MIOHIN_76GW | cuda_edge | `elapsed_sec` | 67.983 | 85.879 | 0.792x | 17.896 |
| Base_MIOHIN_76GW | cuda_edge | `solve_sec` | 6.540 | 14.485 | 0.452x | 7.944 |
| Base_Texas_66GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 40.850 | 49.366 | 0.827x | 8.516 |
| Base_Texas_66GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.697 | 1.860 | 0.912x | 0.163 |
| Base_Texas_66GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.734 | 1.770 | 0.415x | 1.036 |
| Base_Texas_66GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 1.839 | 4.583 | 0.401x | 2.745 |
| Base_Texas_66GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.284 | 0.288 | 0.988x | 0.003 |
| Base_Texas_66GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.069 | 3.361 | 0.318x | 2.292 |
| Base_Texas_66GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 2.548 | 2.425 | 1.051x | -0.123 |
| Base_Texas_66GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 42.550 | 51.229 | 0.831x | 8.679 |
| Base_Texas_66GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.725 | 0.761 | 0.953x | 0.036 |
| Base_Texas_66GW | cuda_edge | `NR.analyze.total.total_sec` | 45.825 | 54.418 | 0.842x | 8.592 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.068 | 0.067 | 1.012x | -0.001 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 3.940 | 10.017 | 0.393x | 6.076 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.435 | 0.445 | 0.978x | 0.010 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.total.total_sec` | 4.537 | 10.624 | 0.427x | 6.087 |
| Base_Texas_66GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.088 | 0.088 | 1.001x | -0.000 |
| Base_Texas_66GW | cuda_edge | `NR.solve.download.total_sec` | 0.077 | 0.077 | 1.001x | -0.000 |
| Base_Texas_66GW | cuda_edge | `NR.solve.total.total_sec` | 5.219 | 11.213 | 0.465x | 5.995 |
| Base_Texas_66GW | cuda_edge | `NR.solve.upload.total_sec` | 0.600 | 0.510 | 1.178x | -0.091 |
| Base_Texas_66GW | cuda_edge | `analyze_sec` | 45.827 | 54.419 | 0.842x | 8.592 |
| Base_Texas_66GW | cuda_edge | `elapsed_sec` | 51.047 | 65.634 | 0.778x | 14.587 |
| Base_Texas_66GW | cuda_edge | `solve_sec` | 5.220 | 11.215 | 0.465x | 5.995 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 89.191 | 101.226 | 0.881x | 12.034 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.759 | 1.916 | 0.918x | 0.157 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.solve.factorization32.total_sec` | 1.287 | 3.628 | 0.355x | 2.341 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 4.516 | 11.300 | 0.400x | 6.784 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.880 | 0.810 | 1.087x | -0.070 |
| Base_West_Interconnect_121GW | cuda_edge | `CUDA.solve.solve32.total_sec` | 2.155 | 5.045 | 0.427x | 2.890 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 7.161 | 7.041 | 1.017x | -0.120 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.analyze.linear_solve.total_sec` | 90.954 | 103.145 | 0.882x | 12.192 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 1.730 | 1.654 | 1.046x | -0.075 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.analyze.total.total_sec` | 99.847 | 111.843 | 0.893x | 11.996 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.123 | 0.117 | 1.056x | -0.006 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.linear_solve.total_sec` | 8.878 | 20.819 | 0.426x | 11.941 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.mismatch.total_sec` | 1.287 | 1.202 | 1.071x | -0.085 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.total.total_sec` | 10.445 | 22.287 | 0.469x | 11.841 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.148 | 0.140 | 1.064x | -0.009 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.solve.download.total_sec` | 0.190 | 0.182 | 1.043x | -0.008 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.solve.total.total_sec` | 11.983 | 23.842 | 0.503x | 11.860 |
| Base_West_Interconnect_121GW | cuda_edge | `NR.solve.upload.total_sec` | 1.342 | 1.370 | 0.980x | 0.028 |
| Base_West_Interconnect_121GW | cuda_edge | `analyze_sec` | 99.849 | 111.845 | 0.893x | 11.996 |
| Base_West_Interconnect_121GW | cuda_edge | `elapsed_sec` | 111.834 | 135.690 | 0.824x | 23.856 |
| Base_West_Interconnect_121GW | cuda_edge | `solve_sec` | 11.985 | 23.844 | 0.503x | 11.860 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 13.117 | 19.207 | 0.683x | 6.090 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.667 | 1.851 | 0.901x | 0.184 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.338 | 0.797 | 0.424x | 0.459 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.209 | 0.542 | 0.386x | 0.332 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.038 | 0.039 | 0.964x | 0.001 |
| MemphisCase2026_Mar7 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.264 | 1.145 | 0.231x | 0.880 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.298 | 0.298 | 1.001x | -0.000 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 14.787 | 21.061 | 0.702x | 6.274 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.302 | 0.307 | 0.984x | 0.005 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.analyze.total.total_sec` | 15.388 | 21.668 | 0.710x | 6.279 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.026 | 0.027 | 0.974x | 0.001 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.854 | 2.527 | 0.338x | 1.673 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.105 | 0.109 | 0.961x | 0.004 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.total.total_sec` | 1.030 | 2.709 | 0.380x | 1.679 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.042 | 0.043 | 0.979x | 0.001 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.solve.download.total_sec` | 0.026 | 0.028 | 0.931x | 0.002 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.solve.total.total_sec` | 1.183 | 2.864 | 0.413x | 1.681 |
| MemphisCase2026_Mar7 | cuda_edge | `NR.solve.upload.total_sec` | 0.125 | 0.124 | 1.002x | -0.000 |
| MemphisCase2026_Mar7 | cuda_edge | `analyze_sec` | 15.390 | 21.669 | 0.710x | 6.279 |
| MemphisCase2026_Mar7 | cuda_edge | `elapsed_sec` | 16.574 | 24.534 | 0.676x | 7.960 |
| MemphisCase2026_Mar7 | cuda_edge | `solve_sec` | 1.184 | 2.865 | 0.413x | 1.681 |
| Texas7k_20220923 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 35.494 | 43.805 | 0.810x | 8.311 |
| Texas7k_20220923 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.699 | 1.893 | 0.898x | 0.194 |
| Texas7k_20220923 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.672 | 1.782 | 0.377x | 1.110 |
| Texas7k_20220923 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 1.108 | 3.086 | 0.359x | 1.978 |
| Texas7k_20220923 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.199 | 0.197 | 1.010x | -0.002 |
| Texas7k_20220923 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.727 | 2.647 | 0.275x | 1.919 |
| Texas7k_20220923 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 2.081 | 2.049 | 1.016x | -0.032 |
| Texas7k_20220923 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 37.196 | 45.701 | 0.814x | 8.505 |
| Texas7k_20220923 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.695 | 0.718 | 0.968x | 0.023 |
| Texas7k_20220923 | cuda_edge | `NR.analyze.total.total_sec` | 39.974 | 48.470 | 0.825x | 8.496 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.048 | 0.049 | 0.990x | 0.000 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 2.716 | 7.722 | 0.352x | 5.006 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.338 | 0.334 | 1.014x | -0.005 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.total.total_sec` | 3.172 | 8.174 | 0.388x | 5.002 |
| Texas7k_20220923 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.065 | 0.064 | 1.011x | -0.001 |
| Texas7k_20220923 | cuda_edge | `NR.solve.download.total_sec` | 0.073 | 0.092 | 0.793x | 0.019 |
| Texas7k_20220923 | cuda_edge | `NR.solve.total.total_sec` | 3.772 | 8.745 | 0.431x | 4.973 |
| Texas7k_20220923 | cuda_edge | `NR.solve.upload.total_sec` | 0.524 | 0.477 | 1.099x | -0.047 |
| Texas7k_20220923 | cuda_edge | `analyze_sec` | 39.975 | 48.471 | 0.825x | 8.496 |
| Texas7k_20220923 | cuda_edge | `elapsed_sec` | 43.749 | 57.218 | 0.765x | 13.469 |
| Texas7k_20220923 | cuda_edge | `solve_sec` | 3.773 | 8.746 | 0.431x | 4.972 |
| case_ACTIVSg200 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 9.346 | 15.369 | 0.608x | 6.023 |
| case_ACTIVSg200 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.716 | 1.876 | 0.915x | 0.160 |
| case_ACTIVSg200 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.111 | 0.371 | 0.298x | 0.260 |
| case_ACTIVSg200 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.103 | 0.351 | 0.294x | 0.248 |
| case_ACTIVSg200 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.026 | 0.025 | 1.016x | -0.000 |
| case_ACTIVSg200 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.154 | 0.771 | 0.200x | 0.616 |
| case_ACTIVSg200 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.064 | 0.065 | 0.989x | 0.001 |
| case_ACTIVSg200 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 11.065 | 17.248 | 0.642x | 6.183 |
| case_ACTIVSg200 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.271 | 0.304 | 0.892x | 0.033 |
| case_ACTIVSg200 | cuda_edge | `NR.analyze.total.total_sec` | 11.402 | 17.618 | 0.647x | 6.216 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.026 | 0.026 | 0.992x | 0.000 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.396 | 1.521 | 0.261x | 1.125 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.078 | 0.082 | 0.952x | 0.004 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.total.total_sec` | 0.544 | 1.674 | 0.325x | 1.130 |
| case_ACTIVSg200 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.040 | 0.041 | 0.971x | 0.001 |
| case_ACTIVSg200 | cuda_edge | `NR.solve.download.total_sec` | 0.019 | 0.019 | 0.979x | 0.000 |
| case_ACTIVSg200 | cuda_edge | `NR.solve.total.total_sec` | 0.633 | 1.764 | 0.359x | 1.131 |
| case_ACTIVSg200 | cuda_edge | `NR.solve.upload.total_sec` | 0.068 | 0.068 | 0.991x | 0.001 |
| case_ACTIVSg200 | cuda_edge | `analyze_sec` | 11.403 | 17.619 | 0.647x | 6.216 |
| case_ACTIVSg200 | cuda_edge | `elapsed_sec` | 12.038 | 19.384 | 0.621x | 7.347 |
| case_ACTIVSg200 | cuda_edge | `solve_sec` | 0.634 | 1.765 | 0.359x | 1.131 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 17.898 | 24.174 | 0.740x | 6.276 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.622 | 1.885 | 0.860x | 0.264 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.486 | 1.143 | 0.425x | 0.658 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.714 | 1.836 | 0.389x | 1.121 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.080 | 0.081 | 0.984x | 0.001 |
| case_ACTIVSg2000 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.535 | 2.167 | 0.247x | 1.633 |
| case_ACTIVSg2000 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.611 | 0.636 | 0.959x | 0.026 |
| case_ACTIVSg2000 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 19.523 | 26.063 | 0.749x | 6.540 |
| case_ACTIVSg2000 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.361 | 0.365 | 0.990x | 0.003 |
| case_ACTIVSg2000 | cuda_edge | `NR.analyze.total.total_sec` | 20.497 | 27.067 | 0.757x | 6.569 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.039 | 0.040 | 0.995x | 0.000 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 1.820 | 5.234 | 0.348x | 3.414 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.165 | 0.171 | 0.965x | 0.006 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.total.total_sec` | 2.093 | 5.514 | 0.380x | 3.421 |
| case_ACTIVSg2000 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.063 | 0.065 | 0.980x | 0.001 |
| case_ACTIVSg2000 | cuda_edge | `NR.solve.download.total_sec` | 0.034 | 0.035 | 0.974x | 0.001 |
| case_ACTIVSg2000 | cuda_edge | `NR.solve.total.total_sec` | 2.322 | 5.744 | 0.404x | 3.423 |
| case_ACTIVSg2000 | cuda_edge | `NR.solve.upload.total_sec` | 0.192 | 0.193 | 0.996x | 0.001 |
| case_ACTIVSg2000 | cuda_edge | `analyze_sec` | 20.498 | 27.068 | 0.757x | 6.569 |
| case_ACTIVSg2000 | cuda_edge | `elapsed_sec` | 22.822 | 32.814 | 0.695x | 9.992 |
| case_ACTIVSg2000 | cuda_edge | `solve_sec` | 2.323 | 5.746 | 0.404x | 3.423 |
| case_ACTIVSg25k | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 101.136 | 113.539 | 0.891x | 12.403 |
| case_ACTIVSg25k | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.715 | 1.912 | 0.897x | 0.197 |
| case_ACTIVSg25k | cuda_edge | `CUDA.solve.factorization32.total_sec` | 1.343 | 3.710 | 0.362x | 2.367 |
| case_ACTIVSg25k | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 3.635 | 9.292 | 0.391x | 5.657 |
| case_ACTIVSg25k | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.833 | 0.820 | 1.015x | -0.012 |
| case_ACTIVSg25k | cuda_edge | `CUDA.solve.solve32.total_sec` | 1.942 | 5.102 | 0.381x | 3.161 |
| case_ACTIVSg25k | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 7.532 | 7.380 | 1.021x | -0.152 |
| case_ACTIVSg25k | cuda_edge | `NR.analyze.linear_solve.total_sec` | 102.855 | 115.455 | 0.891x | 12.599 |
| case_ACTIVSg25k | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 1.976 | 1.883 | 1.049x | -0.093 |
| case_ACTIVSg25k | cuda_edge | `NR.analyze.total.total_sec` | 112.366 | 124.722 | 0.901x | 12.355 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.105 | 0.128 | 0.817x | 0.024 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.linear_solve.total_sec` | 7.790 | 18.960 | 0.411x | 11.170 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.mismatch.total_sec` | 1.249 | 1.210 | 1.032x | -0.038 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.total.total_sec` | 9.274 | 20.436 | 0.454x | 11.162 |
| case_ACTIVSg25k | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.122 | 0.129 | 0.944x | 0.007 |
| case_ACTIVSg25k | cuda_edge | `NR.solve.download.total_sec` | 0.216 | 0.214 | 1.009x | -0.002 |
| case_ACTIVSg25k | cuda_edge | `NR.solve.total.total_sec` | 11.093 | 22.249 | 0.499x | 11.156 |
| case_ACTIVSg25k | cuda_edge | `NR.solve.upload.total_sec` | 1.599 | 1.595 | 1.002x | -0.003 |
| case_ACTIVSg25k | cuda_edge | `analyze_sec` | 112.369 | 124.724 | 0.901x | 12.355 |
| case_ACTIVSg25k | cuda_edge | `elapsed_sec` | 123.464 | 146.975 | 0.840x | 23.511 |
| case_ACTIVSg25k | cuda_edge | `solve_sec` | 11.095 | 22.251 | 0.499x | 11.156 |
| case_ACTIVSg500 | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 10.708 | 16.783 | 0.638x | 6.075 |
| case_ACTIVSg500 | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.673 | 1.860 | 0.900x | 0.186 |
| case_ACTIVSg500 | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.252 | 0.670 | 0.377x | 0.417 |
| case_ACTIVSg500 | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.243 | 0.846 | 0.287x | 0.604 |
| case_ACTIVSg500 | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.046 | 0.047 | 0.987x | 0.001 |
| case_ACTIVSg500 | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.257 | 1.341 | 0.191x | 1.084 |
| case_ACTIVSg500 | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.145 | 0.143 | 1.014x | -0.002 |
| case_ACTIVSg500 | cuda_edge | `NR.analyze.linear_solve.total_sec` | 12.385 | 18.646 | 0.664x | 6.261 |
| case_ACTIVSg500 | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.281 | 0.306 | 0.918x | 0.025 |
| case_ACTIVSg500 | cuda_edge | `NR.analyze.total.total_sec` | 12.812 | 19.096 | 0.671x | 6.284 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.040 | 0.039 | 1.013x | -0.000 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.803 | 2.909 | 0.276x | 2.106 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.118 | 0.122 | 0.965x | 0.004 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.total.total_sec` | 1.029 | 3.140 | 0.328x | 2.111 |
| case_ACTIVSg500 | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.064 | 0.064 | 0.994x | 0.000 |
| case_ACTIVSg500 | cuda_edge | `NR.solve.download.total_sec` | 0.022 | 0.021 | 1.023x | -0.001 |
| case_ACTIVSg500 | cuda_edge | `NR.solve.total.total_sec` | 1.141 | 3.254 | 0.351x | 2.113 |
| case_ACTIVSg500 | cuda_edge | `NR.solve.upload.total_sec` | 0.087 | 0.089 | 0.972x | 0.002 |
| case_ACTIVSg500 | cuda_edge | `analyze_sec` | 12.813 | 19.097 | 0.671x | 6.284 |
| case_ACTIVSg500 | cuda_edge | `elapsed_sec` | 13.955 | 22.353 | 0.624x | 8.397 |
| case_ACTIVSg500 | cuda_edge | `solve_sec` | 1.142 | 3.255 | 0.351x | 2.113 |
| case_ACTIVSg70k | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 269.261 | 286.680 | 0.939x | 17.419 |
| case_ACTIVSg70k | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.799 | 2.075 | 0.867x | 0.276 |
| case_ACTIVSg70k | cuda_edge | `CUDA.solve.factorization32.total_sec` | 3.023 | 8.429 | 0.359x | 5.406 |
| case_ACTIVSg70k | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 15.010 | 39.502 | 0.380x | 24.492 |
| case_ACTIVSg70k | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 3.235 | 3.182 | 1.017x | -0.053 |
| case_ACTIVSg70k | cuda_edge | `CUDA.solve.solve32.total_sec` | 5.960 | 11.717 | 0.509x | 5.758 |
| case_ACTIVSg70k | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 22.504 | 22.444 | 1.003x | -0.060 |
| case_ACTIVSg70k | cuda_edge | `NR.analyze.linear_solve.total_sec` | 271.066 | 288.762 | 0.939x | 17.696 |
| case_ACTIVSg70k | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 5.510 | 5.381 | 1.024x | -0.129 |
| case_ACTIVSg70k | cuda_edge | `NR.analyze.total.total_sec` | 299.086 | 316.593 | 0.945x | 17.507 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.392 | 0.400 | 0.980x | 0.008 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.linear_solve.total_sec` | 27.363 | 62.991 | 0.434x | 35.628 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.mismatch.total_sec` | 4.341 | 4.308 | 1.008x | -0.033 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.total.total_sec` | 32.398 | 68.007 | 0.476x | 35.609 |
| case_ACTIVSg70k | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.290 | 0.296 | 0.982x | 0.005 |
| case_ACTIVSg70k | cuda_edge | `NR.solve.download.total_sec` | 0.591 | 0.542 | 1.090x | -0.049 |
| case_ACTIVSg70k | cuda_edge | `NR.solve.total.total_sec` | 37.550 | 72.909 | 0.515x | 35.359 |
| case_ACTIVSg70k | cuda_edge | `NR.solve.upload.total_sec` | 4.557 | 4.355 | 1.046x | -0.202 |
| case_ACTIVSg70k | cuda_edge | `analyze_sec` | 299.090 | 316.597 | 0.945x | 17.507 |
| case_ACTIVSg70k | cuda_edge | `elapsed_sec` | 336.643 | 389.510 | 0.864x | 52.866 |
| case_ACTIVSg70k | cuda_edge | `solve_sec` | 37.553 | 72.912 | 0.515x | 35.359 |

## Files

- `manifest.json`: configuration, commands, and environment
- `summary.csv`: one row per repeat
- `aggregates.csv`: grouped timing statistics
- `mg_comparison.csv`: top-level baseline vs MG timing
- `operator_comparison.csv`: all collected timers, baseline vs MG
- `raw/`: per-repeat parsed payload and stdout
