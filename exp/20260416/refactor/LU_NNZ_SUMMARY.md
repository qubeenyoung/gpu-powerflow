# cuDSS LU NNZ by Reordering Algorithm

- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`
- Matrix: edge-based Newton Jacobian at `V0`
- cuDSS phases: `ANALYSIS` + one `FACTORIZATION` only
- No Newton iteration loop or solve loop was run.
- Output CSV: `lu_nnz_by_case_reorder.csv`

## LU NNZ

| case | J dim | J nnz | DEFAULT | ALG_1 | ALG_2 | ALG_3 |
| --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 361 | 2,489 | 4,285 | 5,839 | 5,839 | 3,985 |
| case_ACTIVSg500 | 943 | 6,275 | 10,547 | 13,763 | 13,763 | 8,835 |
| MemphisCase2026_Mar7 | 1,797 | 13,355 | 32,339 | 41,405 | 41,405 | 27,079 |
| case_ACTIVSg2000 | 3,607 | 26,345 | 87,037 | 131,256 | 131,256 | 77,389 |
| Base_Florida_42GW | 11,210 | 83,806 | 245,210 | 343,900 | 343,900 | 207,966 |
| Texas7k_20220923 | 12,843 | 91,901 | 293,397 | 476,227 | 476,227 | 254,561 |
| Base_Texas_66GW | 14,431 | 104,763 | 315,319 | 492,349 | 492,349 | 269,233 |
| Base_MIOHIN_76GW | 20,186 | 155,152 | 568,114 | 943,637 | 943,637 | 494,474 |
| Base_West_Interconnect_121GW | 40,748 | 302,486 | 933,954 | 1,485,990 | 1,485,990 | 823,060 |
| case_ACTIVSg25k | 47,246 | 318,672 | 850,278 | 1,293,441 | 1,293,441 | 699,220 |
| case_ACTIVSg70k | 134,104 | 900,558 | 2,671,844 | 4,466,768 | 4,466,768 | 2,306,786 |
| Base_Eastern_Interconnect_515GW | 154,916 | 1,143,804 | 4,047,100 | 6,978,879 | 6,978,879 | 3,581,364 |

## Ratio vs DEFAULT

| case | ALG_1/DEFAULT | ALG_2/DEFAULT | ALG_3/DEFAULT | best LU nnz |
| --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 1.36x | 1.36x | 0.93x | `ALG_3` (3,985) |
| case_ACTIVSg500 | 1.30x | 1.30x | 0.84x | `ALG_3` (8,835) |
| MemphisCase2026_Mar7 | 1.28x | 1.28x | 0.84x | `ALG_3` (27,079) |
| case_ACTIVSg2000 | 1.51x | 1.51x | 0.89x | `ALG_3` (77,389) |
| Base_Florida_42GW | 1.40x | 1.40x | 0.85x | `ALG_3` (207,966) |
| Texas7k_20220923 | 1.62x | 1.62x | 0.87x | `ALG_3` (254,561) |
| Base_Texas_66GW | 1.56x | 1.56x | 0.85x | `ALG_3` (269,233) |
| Base_MIOHIN_76GW | 1.66x | 1.66x | 0.87x | `ALG_3` (494,474) |
| Base_West_Interconnect_121GW | 1.59x | 1.59x | 0.88x | `ALG_3` (823,060) |
| case_ACTIVSg25k | 1.52x | 1.52x | 0.82x | `ALG_3` (699,220) |
| case_ACTIVSg70k | 1.67x | 1.67x | 0.86x | `ALG_3` (2,306,786) |
| Base_Eastern_Interconnect_515GW | 1.72x | 1.72x | 0.88x | `ALG_3` (3,581,364) |

## Geomean Ratios

- `ALG_1/DEFAULT`: 1.51x
- `ALG_2/DEFAULT`: 1.51x
- `ALG_3/DEFAULT`: 0.87x

## Notes

- `ALG_1` and `ALG_2` produced identical LU nnz for every Texas case in this probe.
- `ALG_3` produced lower LU nnz than DEFAULT for every case, but earlier timing still showed slower analyze/solve overall.
