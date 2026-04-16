# Standard vs Modified Comparison

## Average Speedup

| mode | elapsed speedup std/mod | solve speedup std/mod |
|---|---:|---:|
| end2end | 1.011x | 1.128x |
| operators | 0.913x | 1.100x |

Values above are standard divided by modified. Greater than 1 means modified is faster.

## End2End By Case

| case | std ms | mod ms | speedup | std solve ms | mod solve ms | solve speedup | std iter | mod iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 254.041 | 227.918 | 1.115x | 41.429 | 29.604 | 1.399x | 6.7 | 7.0 |
| Base_Florida_42GW | 38.468 | 31.617 | 1.217x | 4.614 | 3.969 | 1.162x | 5.0 | 6.0 |
| Base_MIOHIN_76GW | 43.292 | 51.728 | 0.837x | 6.335 | 6.157 | 1.029x | 5.0 | 6.0 |
| Base_Texas_66GW | 31.944 | 34.224 | 0.933x | 4.374 | 4.937 | 0.886x | 5.0 | 6.0 |
| Base_West_Interconnect_121GW | 67.207 | 73.826 | 0.910x | 9.258 | 7.856 | 1.178x | 5.8 | 5.1 |
| MemphisCase2026_Mar7 | 19.454 | 14.798 | 1.315x | 1.100 | 0.791 | 1.392x | 3.0 | 3.0 |
| Texas7k_20220923 | 29.982 | 36.895 | 0.813x | 3.594 | 2.802 | 1.283x | 4.0 | 4.7 |
| case_ACTIVSg200 | 13.507 | 11.671 | 1.157x | 0.589 | 0.477 | 1.234x | 3.0 | 3.0 |
| case_ACTIVSg2000 | 22.483 | 29.311 | 0.767x | 2.222 | 2.129 | 1.044x | 4.0 | 5.0 |
| case_ACTIVSg25k | 77.873 | 80.817 | 0.964x | 10.673 | 10.619 | 1.005x | 5.0 | 6.0 |
| case_ACTIVSg500 | 13.964 | 13.303 | 1.050x | 1.064 | 1.063 | 1.000x | 4.0 | 5.0 |
| case_ACTIVSg70k | 231.072 | 220.098 | 1.050x | 44.745 | 48.726 | 0.918x | 7.1 | 12.0 |

## Operator Solve By Case

| case | std solve ms | mod solve ms | solve speedup | std factorize count | mod factorize count | std solve count | mod solve count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 40.626 | 25.611 | 1.586x | 5.7 | 3.1 | 5.7 | 6.1 |
| Base_Florida_42GW | 4.203 | 4.651 | 0.904x | 4.0 | 3.0 | 4.0 | 5.0 |
| Base_MIOHIN_76GW | 6.496 | 6.657 | 0.976x | 4.0 | 3.0 | 4.0 | 5.0 |
| Base_Texas_66GW | 5.196 | 4.995 | 1.040x | 4.0 | 3.0 | 4.0 | 5.0 |
| Base_West_Interconnect_121GW | 9.444 | 6.531 | 1.446x | 4.8 | 2.4 | 4.8 | 4.4 |
| MemphisCase2026_Mar7 | 1.172 | 0.927 | 1.264x | 2.0 | 1.0 | 2.0 | 2.0 |
| Texas7k_20220923 | 4.016 | 3.467 | 1.158x | 3.0 | 2.0 | 3.0 | 3.8 |
| case_ACTIVSg200 | 0.641 | 0.518 | 1.238x | 2.0 | 1.0 | 2.0 | 2.0 |
| case_ACTIVSg2000 | 2.089 | 2.186 | 0.956x | 3.0 | 2.0 | 3.0 | 4.0 |
| case_ACTIVSg25k | 11.423 | 14.346 | 0.796x | 4.0 | 3.0 | 4.0 | 5.0 |
| case_ACTIVSg500 | 1.002 | 1.001 | 1.001x | 3.0 | 2.0 | 3.0 | 4.0 |
| case_ACTIVSg70k | 41.604 | 49.559 | 0.839x | 6.1 | 6.0 | 6.1 | 11.0 |
