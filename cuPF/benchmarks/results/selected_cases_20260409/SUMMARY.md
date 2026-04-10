# Result Summary `selected_cases_20260409`

## Setup

- Created (UTC): 2026-04-09T03:37:12.285903+00:00
- Cases: 118_ieee, 793_goc, 1354_pegase, 2746wop_k, 4601_goc, 8387_pegase, 9241_pegase
- Warmup: 1
- Repeats: 3
- CPU model: AMD EPYC 7313 16-Core Processor
- Logical CPUs: 32
- Pinned thread env: {'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
- Python: 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0]
- NumPy: 2.2.6
- SciPy: 1.15.3

## Elapsed Time

| case | pypower (s) | cpp_pypowerlike (s) | cpp_optimized (s) | pypowerlike speedup vs pypower | optimized speedup vs pypower | optimized speedup vs pypowerlike |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.012553 | 0.000955 | 0.000527 | 13.14x | 23.84x | 1.81x |
| pglib_opf_case793_goc | 0.021807 | 0.008366 | 0.005779 | 2.61x | 3.77x | 1.45x |
| pglib_opf_case1354_pegase | 0.037496 | 0.018348 | 0.012262 | 2.04x | 3.06x | 1.50x |
| pglib_opf_case2746wop_k | 0.053444 | 0.034381 | 0.023155 | 1.55x | 2.31x | 1.48x |
| pglib_opf_case4601_goc | 0.134936 | 0.097436 | 0.076156 | 1.38x | 1.77x | 1.28x |
| pglib_opf_case8387_pegase | 0.235597 | 0.171670 | 0.113735 | 1.37x | 2.07x | 1.51x |
| pglib_opf_case9241_pegase | 0.311329 | 0.230211 | 0.148902 | 1.35x | 2.09x | 1.55x |

## C++ Breakdown

| case | cpp_pypowerlike analyze (s) | cpp_pypowerlike solve (s) | cpp_optimized analyze (s) | cpp_optimized solve (s) |
|---|---:|---:|---:|---:|
| pglib_opf_case118_ieee | 0.000008 | 0.000947 | 0.000126 | 0.000400 |
| pglib_opf_case793_goc | 0.000043 | 0.008323 | 0.000876 | 0.004902 |
| pglib_opf_case1354_pegase | 0.000085 | 0.018262 | 0.001536 | 0.010726 |
| pglib_opf_case2746wop_k | 0.000235 | 0.034145 | 0.003197 | 0.019958 |
| pglib_opf_case4601_goc | 0.000267 | 0.097169 | 0.006606 | 0.069550 |
| pglib_opf_case8387_pegase | 0.000576 | 0.171094 | 0.012113 | 0.101622 |
| pglib_opf_case9241_pegase | 0.000636 | 0.229575 | 0.013745 | 0.135157 |

## Correctness Snapshot

| case | pypower success | cpp_pypowerlike success | cpp_optimized success | pypower iterations | cpp_pypowerlike iterations | cpp_optimized iterations |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case118_ieee | True | True | True | 4.0 | 4.0 | 4.0 |
| pglib_opf_case793_goc | True | True | True | 4.0 | 4.0 | 4.0 |
| pglib_opf_case1354_pegase | True | True | True | 5.0 | 5.0 | 5.0 |
| pglib_opf_case2746wop_k | True | True | True | 4.0 | 4.0 | 4.0 |
| pglib_opf_case4601_goc | True | True | True | 5.0 | 5.0 | 5.0 |
| pglib_opf_case8387_pegase | True | True | True | 6.0 | 6.0 | 6.0 |
| pglib_opf_case9241_pegase | True | True | True | 7.0 | 7.0 | 7.0 |

Raw data lives in `summary.csv`, `aggregates.csv`, and `raw/`.
