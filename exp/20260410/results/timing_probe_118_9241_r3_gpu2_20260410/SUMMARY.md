# Result Summary `timing_probe_118_9241_r3_gpu2_20260410`

## Setup

- Created (UTC): 2026-04-10T04:53:03.995869+00:00
- Cases: 118_ieee, 9241_pegase
- Warmup: 1
- CPU repeats: 3
- GPU repeats: 3
- Requested GPU index: 2

## Elapsed Time

| case | pypower (s) | cpu naive (s) | cpu optimized (s) | cuda edge (s) | cuda vertex (s) |
|---|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.011263 | 0.001486 | 0.000736 | 0.013865 | 0.013735 |
| pglib_opf_case9241_pegase | 0.317485 | 0.241614 | 0.075414 | 0.063174 | 0.063494 |

## Speedup

| case | cpu naive vs pypower | cpu optimized vs pypower | cuda edge vs pypower | cuda vertex vs pypower | cuda edge vs cpu optimized | cuda vertex vs cpu optimized |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 7.58x | 15.30x | 0.81x | 0.82x | 0.05x | 0.05x |
| pglib_opf_case9241_pegase | 1.31x | 4.21x | 5.03x | 5.00x | 1.19x | 1.19x |

## C++ Breakdown

| case | cpu naive analyze (s) | cpu naive solve (s) | cpu optimized analyze (s) | cpu optimized solve (s) | cuda edge analyze (s) | cuda edge solve (s) | cuda vertex analyze (s) | cuda vertex solve (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.000039 | 0.001445 | 0.000195 | 0.000540 | 0.012739 | 0.001125 | 0.012622 | 0.001113 |
| pglib_opf_case9241_pegase | 0.000900 | 0.240714 | 0.013007 | 0.062407 | 0.055971 | 0.007203 | 0.056265 | 0.007228 |

## Correctness Snapshot

| case | pypower success | cpu naive success | cpu optimized success | cuda edge success | cuda vertex success | pypower iter | cpu naive iter | cpu optimized iter | cuda edge iter | cuda vertex iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | True | True | True | True | True | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| pglib_opf_case9241_pegase | True | True | True | True | True | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |
