# cuDSS Fill-in by Reordering Algorithm

- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`
- Matrix: edge-based Newton Jacobian at `V0`
- cuDSS phases: `ANALYSIS` + one `FACTORIZATION` only
- Definition: `fill_in_nnz = LU_NNZ - Jacobian_NNZ`
- Definition: `LU/J ratio = LU_NNZ / Jacobian_NNZ`
- Output CSV: `fill_in_by_case_reorder.csv`

## Fill-in NNZ

| case | J nnz | DEFAULT fill | ALG_1 fill | ALG_2 fill | ALG_3 fill |
| --- | --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 2,489 | 1,796 | 3,350 | 3,350 | 1,496 |
| case_ACTIVSg500 | 6,275 | 4,272 | 7,488 | 7,488 | 2,560 |
| MemphisCase2026_Mar7 | 13,355 | 18,984 | 28,050 | 28,050 | 13,724 |
| case_ACTIVSg2000 | 26,345 | 60,692 | 104,911 | 104,911 | 51,044 |
| Base_Florida_42GW | 83,806 | 161,404 | 260,094 | 260,094 | 124,160 |
| Texas7k_20220923 | 91,901 | 201,496 | 384,326 | 384,326 | 162,660 |
| Base_Texas_66GW | 104,763 | 210,556 | 387,586 | 387,586 | 164,470 |
| Base_MIOHIN_76GW | 155,152 | 412,962 | 788,485 | 788,485 | 339,322 |
| Base_West_Interconnect_121GW | 302,486 | 631,468 | 1,183,504 | 1,183,504 | 520,574 |
| case_ACTIVSg25k | 318,672 | 531,606 | 974,769 | 974,769 | 380,548 |
| case_ACTIVSg70k | 900,558 | 1,771,286 | 3,566,210 | 3,566,210 | 1,406,228 |
| Base_Eastern_Interconnect_515GW | 1,143,804 | 2,903,296 | 5,835,075 | 5,835,075 | 2,437,560 |

## LU/J Fill-in Ratio

| case | DEFAULT LU/J | ALG_1 LU/J | ALG_2 LU/J | ALG_3 LU/J | best |
| --- | --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 1.72x | 2.35x | 2.35x | 1.60x | `ALG_3` (1.60x) |
| case_ACTIVSg500 | 1.68x | 2.19x | 2.19x | 1.41x | `ALG_3` (1.41x) |
| MemphisCase2026_Mar7 | 2.42x | 3.10x | 3.10x | 2.03x | `ALG_3` (2.03x) |
| case_ACTIVSg2000 | 3.30x | 4.98x | 4.98x | 2.94x | `ALG_3` (2.94x) |
| Base_Florida_42GW | 2.93x | 4.10x | 4.10x | 2.48x | `ALG_3` (2.48x) |
| Texas7k_20220923 | 3.19x | 5.18x | 5.18x | 2.77x | `ALG_3` (2.77x) |
| Base_Texas_66GW | 3.01x | 4.70x | 4.70x | 2.57x | `ALG_3` (2.57x) |
| Base_MIOHIN_76GW | 3.66x | 6.08x | 6.08x | 3.19x | `ALG_3` (3.19x) |
| Base_West_Interconnect_121GW | 3.09x | 4.91x | 4.91x | 2.72x | `ALG_3` (2.72x) |
| case_ACTIVSg25k | 2.67x | 4.06x | 4.06x | 2.19x | `ALG_3` (2.19x) |
| case_ACTIVSg70k | 2.97x | 4.96x | 4.96x | 2.56x | `ALG_3` (2.56x) |
| Base_Eastern_Interconnect_515GW | 3.54x | 6.10x | 6.10x | 3.13x | `ALG_3` (3.13x) |

## Fill-in Ratio vs DEFAULT

| case | ALG_1 fill / DEFAULT | ALG_2 fill / DEFAULT | ALG_3 fill / DEFAULT |
| --- | --- | --- | --- |
| case_ACTIVSg200 | 1.87x | 1.87x | 0.83x |
| case_ACTIVSg500 | 1.75x | 1.75x | 0.60x |
| MemphisCase2026_Mar7 | 1.48x | 1.48x | 0.72x |
| case_ACTIVSg2000 | 1.73x | 1.73x | 0.84x |
| Base_Florida_42GW | 1.61x | 1.61x | 0.77x |
| Texas7k_20220923 | 1.91x | 1.91x | 0.81x |
| Base_Texas_66GW | 1.84x | 1.84x | 0.78x |
| Base_MIOHIN_76GW | 1.91x | 1.91x | 0.82x |
| Base_West_Interconnect_121GW | 1.87x | 1.87x | 0.82x |
| case_ACTIVSg25k | 1.83x | 1.83x | 0.72x |
| case_ACTIVSg70k | 2.01x | 2.01x | 0.79x |
| Base_Eastern_Interconnect_515GW | 2.01x | 2.01x | 0.84x |

## Geomean vs DEFAULT

- `ALG_1/DEFAULT` fill-in nnz: 1.81x; LU/J ratio: 1.51x
- `ALG_2/DEFAULT` fill-in nnz: 1.81x; LU/J ratio: 1.51x
- `ALG_3/DEFAULT` fill-in nnz: 0.78x; LU/J ratio: 0.87x

## Notes

- `ALG_1` and `ALG_2` have identical fill-in for every case.
- `ALG_3` has lower fill-in than DEFAULT for every case in this probe.
- This measures structural LU fill after one cuDSS factorization of the initial Jacobian; it does not run Newton iterations.
