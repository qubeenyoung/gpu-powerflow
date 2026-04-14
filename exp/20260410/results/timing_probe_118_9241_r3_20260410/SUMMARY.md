# Result Summary `timing_probe_118_9241_r3_20260410`

## Setup

- Created (UTC): 2026-04-10T04:50:18.196286+00:00
- Cases: 118_ieee, 9241_pegase
- Warmup: 1
- CPU repeats: 3
- GPU repeats: 3
- Requested GPU index: 3

## Elapsed Time

| case | pypower (s) | cpu naive (s) | cpu optimized (s) | cuda edge (s) | cuda vertex (s) |
|---|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.011346 | 0.001457 | 0.000714 | 0.101552 | 0.013730 |
| pglib_opf_case9241_pegase | 0.319721 | 0.242275 | 0.075113 | 0.207370 | 0.133334 |

## Speedup

| case | cpu naive vs pypower | cpu optimized vs pypower | cuda edge vs pypower | cuda vertex vs pypower | cuda edge vs cpu optimized | cuda vertex vs cpu optimized |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 7.79x | 15.88x | 0.11x | 0.83x | 0.01x | 0.05x |
| pglib_opf_case9241_pegase | 1.32x | 4.26x | 1.54x | 2.40x | 0.36x | 0.56x |

## C++ Breakdown

| case | cpu naive analyze (s) | cpu naive solve (s) | cpu optimized analyze (s) | cpu optimized solve (s) | cuda edge analyze (s) | cuda edge solve (s) | cuda vertex analyze (s) | cuda vertex solve (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.000039 | 0.001417 | 0.000193 | 0.000520 | 0.055273 | 0.046279 | 0.012556 | 0.001174 |
| pglib_opf_case9241_pegase | 0.000944 | 0.241331 | 0.013097 | 0.062015 | 0.144596 | 0.062773 | 0.100591 | 0.032744 |

## Correctness Snapshot

| case | pypower success | cpu naive success | cpu optimized success | cuda edge success | cuda vertex success | pypower iter | cpu naive iter | cpu optimized iter | cuda edge iter | cuda vertex iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | True | True | True | True | True | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| pglib_opf_case9241_pegase | True | True | True | True | True | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |
