# Benchmark Run `selected_cases_20260409`

- Created (UTC): 2026-04-09T03:37:12.285903+00:00
- Cases: 118_ieee, 793_goc, 1354_pegase, 2746wop_k, 4601_goc, 8387_pegase, 9241_pegase
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
| cpp_optimized | pglib_opf_case118_ieee | 0.0005266666666666666 | 0.000126 | 0.0004003333333333333 |
| cpp_optimized | pglib_opf_case1354_pegase | 0.012262 | 0.001536 | 0.010726 |
| cpp_optimized | pglib_opf_case2746wop_k | 0.023155333333333333 | 0.0031966666666666667 | 0.019957666666666665 |
| cpp_optimized | pglib_opf_case4601_goc | 0.07615633333333334 | 0.006606 | 0.06955 |
| cpp_optimized | pglib_opf_case793_goc | 0.005778666666666666 | 0.0008763333333333333 | 0.004902 |
| cpp_optimized | pglib_opf_case8387_pegase | 0.11373533333333333 | 0.012113333333333334 | 0.10162166666666667 |
| cpp_optimized | pglib_opf_case9241_pegase | 0.14890233333333333 | 0.013744666666666667 | 0.135157 |
| cpp_pypowerlike | pglib_opf_case118_ieee | 0.0009553333333333333 | 8e-06 | 0.000947 |
| cpp_pypowerlike | pglib_opf_case1354_pegase | 0.018347666666666665 | 8.533333333333334e-05 | 0.018262 |
| cpp_pypowerlike | pglib_opf_case2746wop_k | 0.034380666666666664 | 0.00023533333333333335 | 0.034145 |
| cpp_pypowerlike | pglib_opf_case4601_goc | 0.09743633333333333 | 0.000267 | 0.09716866666666667 |
| cpp_pypowerlike | pglib_opf_case793_goc | 0.008366333333333333 | 4.3333333333333334e-05 | 0.008322666666666667 |
| cpp_pypowerlike | pglib_opf_case8387_pegase | 0.17167033333333334 | 0.0005763333333333333 | 0.171094 |
| cpp_pypowerlike | pglib_opf_case9241_pegase | 0.230211 | 0.0006356666666666666 | 0.22957466666666668 |
| pypower | pglib_opf_case118_ieee | 0.012553138037522634 |  |  |
| pypower | pglib_opf_case1354_pegase | 0.037495734790960945 |  |  |
| pypower | pglib_opf_case2746wop_k | 0.053443690141042076 |  |  |
| pypower | pglib_opf_case4601_goc | 0.13493578073879084 |  |  |
| pypower | pglib_opf_case793_goc | 0.021807414169112842 |  |  |
| pypower | pglib_opf_case8387_pegase | 0.2355974124123653 |  |  |
| pypower | pglib_opf_case9241_pegase | 0.31132901397844154 |  |  |
