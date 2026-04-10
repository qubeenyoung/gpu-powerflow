# Benchmark Run `optimized_fix_check_20260409`

- Created (UTC): 2026-04-09T04:19:35.582017+00:00
- Cases: 118_ieee, 4601_goc, 9241_pegase
- Implementations: pypower, cpp_pypowerlike, cpp_optimized
- Warmup: 1
- Repeats: 3

## Environment

- OS: Linux-5.15.0-163-generic-x86_64-with-glibc2.35
- CPU model: AMD EPYC 7313 16-Core Processor
- Logical CPUs: 32
- Threads per core: 1
- Cores per socket: 16
- Sockets: 2
- Memory (KiB): 263976212
- Python: 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0]
- NumPy: 2.2.6
- SciPy: 1.15.3

## Files

- `manifest.json`: full environment and command log
- `SUMMARY.md`: benchmark result summary with speedup tables
- `summary.csv`: one row per run
- `aggregates.csv`: grouped statistics per implementation/case
- `raw/`: per-run raw payloads

## Aggregate Snapshot

| implementation | case | elapsed mean (s) | analyze mean (s) | solve mean (s) |
|---|---:|---:|---:|---:|
| cpp_optimized | pglib_opf_case118_ieee | 0.000237 | 9.266666666666666e-05 | 0.00014366666666666667 |
| cpp_optimized | pglib_opf_case4601_goc | 0.04040933333333333 | 0.004802666666666667 | 0.03560633333333333 |
| cpp_optimized | pglib_opf_case9241_pegase | 0.072396 | 0.013545333333333333 | 0.05885 |
| cpp_pypowerlike | pglib_opf_case118_ieee | 0.0009613333333333332 | 8e-06 | 0.0009526666666666667 |
| cpp_pypowerlike | pglib_opf_case4601_goc | 0.097625 | 0.00026933333333333334 | 0.09735566666666667 |
| cpp_pypowerlike | pglib_opf_case9241_pegase | 0.230506 | 0.0006336666666666667 | 0.229872 |
| pypower | pglib_opf_case118_ieee | 0.011094393208622932 | n/a | n/a |
| pypower | pglib_opf_case4601_goc | 0.13476958312094212 | n/a | n/a |
| pypower | pglib_opf_case9241_pegase | 0.3083868144700925 | n/a | n/a |
