# Result Summary `optimized_fix_check_20260409`

## Setup

- Created (UTC): 2026-04-09T04:19:35.582017+00:00
- Cases: 118_ieee, 4601_goc, 9241_pegase
- Warmup: 1
- Repeats: 3
- CPU model: AMD EPYC 7313 16-Core Processor
- Logical CPUs: 32
- Pinned thread env: {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
- Python: 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0]
- NumPy: 2.2.6
- SciPy: 1.15.3

## Elapsed Time

| case | pypower (s) | cpp_pypowerlike (s) | cpp_optimized (s) | pypowerlike speedup vs pypower | optimized speedup vs pypower | optimized speedup vs pypowerlike |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.011094 | 0.000961 | 0.000237 | 11.54x | 46.81x | 4.06x |
| pglib_opf_case4601_goc | 0.134770 | 0.097625 | 0.040409 | 1.38x | 3.34x | 2.42x |
| pglib_opf_case9241_pegase | 0.308387 | 0.230506 | 0.072396 | 1.34x | 4.26x | 3.18x |

## C++ Breakdown

| case | cpp_pypowerlike analyze (s) | cpp_pypowerlike solve (s) | cpp_optimized analyze (s) | cpp_optimized solve (s) |
|---|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.000008 | 0.000953 | 0.000093 | 0.000144 |
| pglib_opf_case4601_goc | 0.000269 | 0.097356 | 0.004803 | 0.035606 |
| pglib_opf_case9241_pegase | 0.000634 | 0.229872 | 0.013545 | 0.058850 |

## Correctness Snapshot

| case | pypower success | cpp_pypowerlike success | cpp_optimized success | pypower iterations | cpp_pypowerlike iterations | cpp_optimized iterations |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | True | True | True | 4.0 | 4.0 | 4.0 |
| pglib_opf_case4601_goc | True | True | True | 5.0 | 5.0 | 5.0 |
| pglib_opf_case9241_pegase | True | True | True | 7.0 | 7.0 | 7.0 |

Raw data lives in `summary.csv`, `aggregates.csv`, and `raw/`.
