# Filtered Standard vs Modified Comparison

Filtering rule: within each `(mode, case, profile)` group, a repeat is removed when any of `elapsed_sec`, `analyze_sec`, or `solve_sec` is above both `Q3 + 1.5 * IQR` and `1.20 * median`.

Removed metric hits: 70
Removed repeat rows: 36

## Average Speedup

| mode | elapsed speedup std/mod | solve speedup std/mod |
|---|---:|---:|
| end2end | 1.051x | 1.132x |
| operators | 0.999x | 1.108x |

Values are standard divided by modified. Greater than 1 means modified is faster.

## End2End By Case

| case | std ms | mod ms | speedup | std solve ms | mod solve ms | solve speedup | std runs | mod runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 246.292 | 227.918 | 1.081x | 41.402 | 29.604 | 1.399x | 9 | 10 |
| Base_Florida_42GW | 38.468 | 28.701 | 1.340x | 4.614 | 3.966 | 1.163x | 10 | 9 |
| Base_MIOHIN_76GW | 43.292 | 43.936 | 0.985x | 6.335 | 6.127 | 1.034x | 10 | 9 |
| Base_Texas_66GW | 31.944 | 33.412 | 0.956x | 4.374 | 4.946 | 0.884x | 10 | 9 |
| Base_West_Interconnect_121GW | 67.207 | 65.528 | 1.026x | 9.258 | 7.627 | 1.214x | 10 | 8 |
| MemphisCase2026_Mar7 | 16.126 | 14.800 | 1.090x | 1.100 | 0.769 | 1.430x | 9 | 9 |
| Texas7k_20220923 | 29.982 | 28.388 | 1.056x | 3.594 | 2.825 | 1.272x | 10 | 9 |
| case_ACTIVSg200 | 11.863 | 11.671 | 1.016x | 0.589 | 0.477 | 1.234x | 9 | 10 |
| case_ACTIVSg2000 | 18.410 | 18.598 | 0.990x | 2.219 | 2.107 | 1.053x | 8 | 8 |
| case_ACTIVSg25k | 77.873 | 75.138 | 1.036x | 10.673 | 10.617 | 1.005x | 10 | 9 |
| case_ACTIVSg500 | 13.499 | 13.303 | 1.015x | 1.031 | 1.063 | 0.970x | 9 | 10 |
| case_ACTIVSg70k | 224.071 | 220.098 | 1.018x | 45.304 | 48.726 | 0.930x | 9 | 10 |

## Operator Solve By Case

| case | std solve ms | mod solve ms | solve speedup | std factorize count | mod factorize count | std solve count | mod solve count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 40.366 | 24.006 | 1.681x | 5.666666666666667 | 3.0 | 5.666666666666667 | 6.0 |
| Base_Florida_42GW | 4.203 | 4.785 | 0.879x | 4.0 | 3.0 | 4.0 | 5.0 |
| Base_MIOHIN_76GW | 6.496 | 6.297 | 1.032x | 4.0 | 3.0 | 4.0 | 5.0 |
| Base_Texas_66GW | 5.196 | 4.995 | 1.040x | 4.0 | 3.0 | 4.0 | 5.0 |
| Base_West_Interconnect_121GW | 9.444 | 6.531 | 1.446x | 4.8 | 2.4 | 4.8 | 4.4 |
| MemphisCase2026_Mar7 | 1.172 | 0.927 | 1.264x | 2.0 | 1.0 | 2.0 | 2.0 |
| Texas7k_20220923 | 4.016 | 3.467 | 1.158x | 3.0 | 2.0 | 3.0 | 3.8 |
| case_ACTIVSg200 | 0.641 | 0.517 | 1.239x | 2.0 | 1.0 | 2.0 | 2.0 |
| case_ACTIVSg2000 | 2.089 | 2.188 | 0.955x | 3.0 | 2.0 | 3.0 | 4.0 |
| case_ACTIVSg25k | 11.529 | 13.895 | 0.830x | 4.0 | 3.0 | 4.0 | 5.0 |
| case_ACTIVSg500 | 1.002 | 1.001 | 1.001x | 3.0 | 2.0 | 3.0 | 4.0 |
| case_ACTIVSg70k | 38.166 | 49.480 | 0.771x | 6.125 | 6.0 | 6.125 | 11.0 |
