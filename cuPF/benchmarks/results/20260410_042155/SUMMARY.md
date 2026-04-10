# Result Summary `20260410_042155`

## Setup

- Created (UTC): 2026-04-10T04:21:55.309273+00:00
- Cases: case30_ieee
- Warmup: 0
- Repeats: 1
- CPU model: AMD EPYC 7313 16-Core Processor
- Logical CPUs: 32
- Pinned thread env: not set
- Python: 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0]
- NumPy: 2.2.6
- SciPy: 1.15.3

## Elapsed Time

| case | pypower (s) | cpp_pypowerlike (s) | cpp_optimized (s) | pypowerlike speedup vs pypower | optimized speedup vs pypower | optimized speedup vs pypowerlike |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case30_ieee | 0.010554 | 0.001294 | 0.000802 | 8.16x | 13.16x | 1.61x |

## C++ Breakdown

| case | cpp_pypowerlike analyze (s) | cpp_pypowerlike solve (s) | cpp_optimized analyze (s) | cpp_optimized solve (s) |
|---|---:|---:|---:|---:|
| pglib_opf_case30_ieee | 0.000182 | 0.001111 | 0.000327 | 0.000474 |

## Correctness Snapshot

| case | pypower success | cpp_pypowerlike success | cpp_optimized success | pypower iterations | cpp_pypowerlike iterations | cpp_optimized iterations |
|---|---:|---:|---:|---:|---:|---:|
| pglib_opf_case30_ieee | True | True | True | 4.0 | 4.0 | 4.0 |

Raw data lives in `summary.csv`, `aggregates.csv`, and `raw/`.
