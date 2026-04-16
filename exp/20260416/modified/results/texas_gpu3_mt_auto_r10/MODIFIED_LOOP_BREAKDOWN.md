# Modified Loop Operator Breakdown

Filtered operators runs only. One modified loop is defined as one outer pass with one `linear_factorize` call. Times are divided by `NR.iteration.linear_factorize.count`.

| case | loops | solve/loop | jacobian ms | factorize ms | linear solve ms | update ms | mismatch ms | accounted ms | factorize % | solve % | mismatch % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 3.00 | 2.00 | 0.079 | 3.328 | 2.640 | 0.109 | 0.790 | 6.946 | 47.9 | 38.0 | 11.4 |
| Base_Florida_42GW | 3.00 | 1.67 | 0.016 | 0.503 | 0.575 | 0.035 | 0.226 | 1.355 | 37.1 | 42.4 | 16.7 |
| Base_MIOHIN_76GW | 3.00 | 1.67 | 0.020 | 0.786 | 0.719 | 0.037 | 0.249 | 1.810 | 43.4 | 39.7 | 13.7 |
| Base_Texas_66GW | 3.00 | 1.67 | 0.017 | 0.628 | 0.581 | 0.035 | 0.179 | 1.440 | 43.6 | 40.3 | 12.4 |
| Base_West_Interconnect_121GW | 2.40 | 1.83 | 0.023 | 1.138 | 0.947 | 0.054 | 0.226 | 2.389 | 47.6 | 39.7 | 9.5 |
| MemphisCase2026_Mar7 | 1.00 | 2.00 | 0.013 | 0.308 | 0.308 | 0.041 | 0.106 | 0.775 | 39.7 | 39.7 | 13.6 |
| Texas7k_20220923 | 2.00 | 1.90 | 0.016 | 0.589 | 0.597 | 0.040 | 0.199 | 1.441 | 40.9 | 41.4 | 13.8 |
| case_ACTIVSg200 | 1.00 | 2.00 | 0.013 | 0.110 | 0.181 | 0.040 | 0.079 | 0.424 | 26.0 | 42.8 | 18.7 |
| case_ACTIVSg2000 | 2.00 | 2.00 | 0.013 | 0.405 | 0.414 | 0.042 | 0.105 | 0.980 | 41.3 | 42.3 | 10.7 |
| case_ACTIVSg25k | 3.00 | 1.67 | 0.027 | 1.187 | 1.667 | 0.051 | 0.929 | 3.860 | 30.7 | 43.2 | 24.1 |
| case_ACTIVSg500 | 2.00 | 2.00 | 0.011 | 0.159 | 0.182 | 0.040 | 0.061 | 0.454 | 35.1 | 40.1 | 13.5 |
| case_ACTIVSg70k | 6.00 | 1.83 | 0.065 | 2.828 | 3.189 | 0.089 | 1.292 | 7.463 | 37.9 | 42.7 | 17.3 |
| __average_across_cases__ | 2.62 | 1.85 | 0.026 | 0.997 | 1.000 | 0.051 | 0.370 | 2.445 | 39.3 | 41.0 | 14.6 |

## Low-Level cuDSS Per Loop

| case | CUDSS factorization ms | CUDSS refactorization ms | RHS prepare ms | CUDSS solve ms |
|---|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 1.139 | 2.188 | 0.493 | 2.078 |
| Base_Florida_42GW | 0.190 | 0.313 | 0.175 | 0.395 |
| Base_MIOHIN_76GW | 0.283 | 0.503 | 0.168 | 0.544 |
| Base_Texas_66GW | 0.231 | 0.397 | 0.120 | 0.456 |
| Base_West_Interconnect_121GW | 0.503 | 0.635 | 0.124 | 0.810 |
| MemphisCase2026_Mar7 | 0.307 |  | 0.042 | 0.263 |
| Texas7k_20220923 | 0.319 | 0.270 | 0.123 | 0.468 |
| case_ACTIVSg200 | 0.110 |  | 0.027 | 0.152 |
| case_ACTIVSg2000 | 0.226 | 0.178 | 0.057 | 0.354 |
| case_ACTIVSg25k | 0.426 | 0.761 | 0.811 | 0.834 |
| case_ACTIVSg500 | 0.102 | 0.058 | 0.024 | 0.156 |
| case_ACTIVSg70k | 0.491 | 2.337 | 1.014 | 2.128 |
| __average_across_cases__ | 0.361 | 0.764 | 0.265 | 0.720 |
