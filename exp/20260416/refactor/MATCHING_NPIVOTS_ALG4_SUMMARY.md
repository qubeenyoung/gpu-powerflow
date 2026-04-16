# cuDSS Matching NPIVOTS Probe

- Reordering algorithm: `DEFAULT`
- Matrix: edge-based Newton Jacobian at `V0`
- cuDSS phases: `ANALYSIS` + one `FACTORIZATION` only
- No Newton iteration loop or solve loop was run.
- Output CSV: `matching_npivots_by_case.csv`

## NPIVOTS

| case | J dim | no matching | matching | delta | ratio |
| --- | --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 361 | 0 | 0 | 0 | - |
| case_ACTIVSg500 | 943 | 0 | 0 | 0 | - |
| MemphisCase2026_Mar7 | 1,797 | 0 | 0 | 0 | - |
| case_ACTIVSg2000 | 3,607 | 0 | 0 | 0 | - |
| Base_Florida_42GW | 11,210 | 0 | 0 | 0 | - |
| Texas7k_20220923 | 12,843 | 0 | 0 | 0 | - |
| Base_Texas_66GW | 14,431 | 0 | 0 | 0 | - |
| Base_MIOHIN_76GW | 20,186 | 0 | 0 | 0 | - |
| Base_West_Interconnect_121GW | 40,748 | 0 | 0 | 0 | - |
| case_ACTIVSg25k | 47,246 | 0 | 0 | 0 | - |
| case_ACTIVSg70k | 134,104 | 0 | 0 | 0 | - |
| Base_Eastern_Interconnect_515GW | 154,916 | 0 | 0 | 0 | - |

## LU NNZ Side Check

| case | LU no matching | LU matching | delta | ratio |
| --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 4,285 | 4,285 | 0 | 1.00x |
| case_ACTIVSg500 | 10,547 | 10,547 | 0 | 1.00x |
| MemphisCase2026_Mar7 | 32,339 | 32,339 | 0 | 1.00x |
| case_ACTIVSg2000 | 87,037 | 87,037 | 0 | 1.00x |
| Base_Florida_42GW | 245,210 | 245,210 | 0 | 1.00x |
| Texas7k_20220923 | 293,397 | 293,397 | 0 | 1.00x |
| Base_Texas_66GW | 315,319 | 312,791 | -2,528 | 0.99x |
| Base_MIOHIN_76GW | 568,114 | 568,114 | 0 | 1.00x |
| Base_West_Interconnect_121GW | 933,954 | 933,954 | 0 | 1.00x |
| case_ACTIVSg25k | 850,278 | 855,520 | 5,242 | 1.01x |
| case_ACTIVSg70k | 2,671,844 | 2,665,900 | -5,944 | 1.00x |
| Base_Eastern_Interconnect_515GW | 4,047,100 | 4,003,732 | -43,368 | 0.99x |

## Geomean Ratios

- NPIVOTS matching/on over off: not defined because every off value is 0.
- LU_NNZ matching/on over off: 1.00x

## Notes

- `CUDSS_DATA_NPIVOTS` returned 0 for every case, both with matching disabled and enabled.
- Matching was tested with `CUDSS_CONFIG_MATCHING_ALG=CUDSS_ALG_4`.
- Reordering was fixed to `CUDSS_ALG_DEFAULT`.
