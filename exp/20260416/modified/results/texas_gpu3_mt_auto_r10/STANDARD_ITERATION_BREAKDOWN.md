# Standard Newton Iteration Operator Breakdown

Filtered operators runs only. One standard full Newton iteration is defined as one iteration that reaches `linear_factorize`, i.e. `mismatch -> jacobian -> factorize -> solve -> voltage_update`. The final convergence-only mismatch is included in `mismatch_per_full_iteration_ms`, so `mismatches_per_full_iteration` can be greater than 1.

| case | full iters | mismatch/iter | mismatch ms | jacobian ms | factorize ms | solve ms | update ms | accounted ms | factorize % | solve % | mismatch % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 5.67 | 1.18 | 0.836 | 0.081 | 3.313 | 1.856 | 0.056 | 6.143 | 53.9 | 30.2 | 13.6 |
| Base_Florida_42GW | 4.00 | 1.25 | 0.092 | 0.016 | 0.494 | 0.296 | 0.021 | 0.920 | 53.8 | 32.2 | 10.0 |
| Base_MIOHIN_76GW | 4.00 | 1.25 | 0.158 | 0.020 | 0.785 | 0.423 | 0.023 | 1.409 | 55.7 | 30.1 | 11.2 |
| Base_Texas_66GW | 4.00 | 1.25 | 0.113 | 0.016 | 0.619 | 0.355 | 0.026 | 1.128 | 54.8 | 31.5 | 10.0 |
| Base_West_Interconnect_121GW | 4.80 | 1.21 | 0.121 | 0.023 | 1.112 | 0.512 | 0.029 | 1.798 | 61.9 | 28.5 | 6.8 |
| MemphisCase2026_Mar7 | 2.00 | 1.50 | 0.053 | 0.013 | 0.258 | 0.154 | 0.021 | 0.499 | 51.7 | 30.9 | 10.6 |
| Texas7k_20220923 | 3.00 | 1.33 | 0.121 | 0.017 | 0.596 | 0.354 | 0.022 | 1.109 | 53.7 | 31.9 | 10.9 |
| case_ACTIVSg200 | 2.00 | 1.50 | 0.040 | 0.013 | 0.106 | 0.092 | 0.020 | 0.272 | 39.1 | 34.0 | 14.7 |
| case_ACTIVSg2000 | 3.00 | 1.33 | 0.040 | 0.011 | 0.380 | 0.191 | 0.020 | 0.642 | 59.2 | 29.8 | 6.2 |
| case_ACTIVSg25k | 4.00 | 1.25 | 0.488 | 0.024 | 1.170 | 0.852 | 0.030 | 2.563 | 45.6 | 33.2 | 19.0 |
| case_ACTIVSg500 | 3.00 | 1.33 | 0.033 | 0.011 | 0.146 | 0.093 | 0.020 | 0.304 | 48.1 | 30.7 | 10.9 |
| case_ACTIVSg70k | 6.12 | 1.16 | 0.769 | 0.065 | 2.822 | 1.752 | 0.049 | 5.457 | 51.7 | 32.1 | 14.1 |
| __average_across_cases__ | 3.80 | 1.30 | 0.239 | 0.026 | 0.983 | 0.578 | 0.028 | 1.854 | 52.4 | 31.2 | 11.5 |

## Per-Call Times

| case | mismatch call ms | jacobian call ms | factorize call ms | solve call ms | update call ms |
|---|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 0.711 | 0.081 | 3.313 | 1.856 | 0.056 |
| Base_Florida_42GW | 0.073 | 0.016 | 0.494 | 0.296 | 0.021 |
| Base_MIOHIN_76GW | 0.126 | 0.020 | 0.785 | 0.423 | 0.023 |
| Base_Texas_66GW | 0.090 | 0.016 | 0.619 | 0.355 | 0.026 |
| Base_West_Interconnect_121GW | 0.100 | 0.023 | 1.112 | 0.512 | 0.029 |
| MemphisCase2026_Mar7 | 0.035 | 0.013 | 0.258 | 0.154 | 0.021 |
| Texas7k_20220923 | 0.090 | 0.017 | 0.596 | 0.354 | 0.022 |
| case_ACTIVSg200 | 0.027 | 0.013 | 0.106 | 0.092 | 0.020 |
| case_ACTIVSg2000 | 0.030 | 0.011 | 0.380 | 0.191 | 0.020 |
| case_ACTIVSg25k | 0.390 | 0.024 | 1.170 | 0.852 | 0.030 |
| case_ACTIVSg500 | 0.025 | 0.011 | 0.146 | 0.093 | 0.020 |
| case_ACTIVSg70k | 0.661 | 0.065 | 2.822 | 1.752 | 0.049 |
| __average_across_cases__ | 0.197 | 0.026 | 0.983 | 0.578 | 0.028 |

## Low-Level cuDSS Per Full Iteration

| case | CUDSS factorization ms | CUDSS refactorization ms | RHS prepare ms | CUDSS solve ms |
|---|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 0.609 | 2.704 | 0.590 | 1.234 |
| Base_Florida_42GW | 0.142 | 0.353 | 0.058 | 0.236 |
| Base_MIOHIN_76GW | 0.213 | 0.571 | 0.097 | 0.322 |
| Base_Texas_66GW | 0.172 | 0.446 | 0.072 | 0.280 |
| Base_West_Interconnect_121GW | 0.252 | 0.860 | 0.067 | 0.436 |
| MemphisCase2026_Mar7 | 0.154 | 0.104 | 0.021 | 0.131 |
| Texas7k_20220923 | 0.219 | 0.377 | 0.094 | 0.251 |
| case_ACTIVSg200 | 0.055 | 0.052 | 0.014 | 0.078 |
| case_ACTIVSg2000 | 0.146 | 0.234 | 0.017 | 0.173 |
| case_ACTIVSg25k | 0.320 | 0.850 | 0.353 | 0.485 |
| case_ACTIVSg500 | 0.068 | 0.077 | 0.012 | 0.079 |
| case_ACTIVSg70k | 0.476 | 2.346 | 0.552 | 1.172 |
| __average_across_cases__ | 0.236 | 0.748 | 0.162 | 0.406 |
