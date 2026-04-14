# Benchmark Run `selected_cases_20260410_gpu3_bg`

- Created (UTC): 2026-04-09T16:02:46.970498+00:00
- Cases: 118_ieee, 793_goc, 1354_pegase, 2746wop_k, 4601_goc, 8387_pegase, 9241_pegase
- Implementations: pypower, cpp_pypowerlike, cpp_optimized, cpp_cuda_edge, cpp_cuda_vertex
- Warmup: 1
- CPU repeats: 10
- GPU repeats: 10
- Requested GPU index: 3

## Environment

- OS: Linux-5.15.0-163-generic-x86_64-with-glibc2.35
- CPU model: AMD EPYC 7313 16-Core Processor
- Logical CPUs: 32
- Python: 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0]
- NumPy: 2.2.6
- SciPy: 1.15.3

## Files

- `manifest.json`: full environment and command log
- `SUMMARY.md`: compact comparison tables
- `summary.csv`: one row per run
- `aggregates.csv`: grouped statistics per implementation/case
- `raw/`: per-run raw payloads

## Aggregate Snapshot

| implementation | case | runs | elapsed mean (s) | analyze mean (s) | solve mean (s) |
|---|---|---:|---:|---:|---:|
| cpp_cuda_edge | pglib_opf_case118_ieee | 10 | 0.013105 | 0.012163 | 0.000941 |
| cpp_cuda_edge | pglib_opf_case1354_pegase | 10 | 0.020822 | 0.018710 | 0.002111 |
| cpp_cuda_edge | pglib_opf_case2746wop_k | 10 | 0.026778 | 0.024483 | 0.002294 |
| cpp_cuda_edge | pglib_opf_case4601_goc | 10 | 0.038674 | 0.034592 | 0.004082 |
| cpp_cuda_edge | pglib_opf_case793_goc | 10 | 0.017318 | 0.015712 | 0.001605 |
| cpp_cuda_edge | pglib_opf_case8387_pegase | 10 | 0.056551 | 0.050722 | 0.005829 |
| cpp_cuda_edge | pglib_opf_case9241_pegase | 10 | 0.061930 | 0.054927 | 0.007004 |
| cpp_cuda_vertex | pglib_opf_case118_ieee | 10 | 0.013123 | 0.012202 | 0.000920 |
| cpp_cuda_vertex | pglib_opf_case1354_pegase | 10 | 0.020816 | 0.018660 | 0.002155 |
| cpp_cuda_vertex | pglib_opf_case2746wop_k | 10 | 0.026775 | 0.024460 | 0.002314 |
| cpp_cuda_vertex | pglib_opf_case4601_goc | 10 | 0.038253 | 0.034277 | 0.003975 |
| cpp_cuda_vertex | pglib_opf_case793_goc | 10 | 0.017304 | 0.015749 | 0.001554 |
| cpp_cuda_vertex | pglib_opf_case8387_pegase | 10 | 0.056608 | 0.050718 | 0.005889 |
| cpp_cuda_vertex | pglib_opf_case9241_pegase | 10 | 0.062053 | 0.054965 | 0.007087 |
| cpp_optimized | pglib_opf_case118_ieee | 10 | 0.000547 | 0.000119 | 0.000427 |
| cpp_optimized | pglib_opf_case1354_pegase | 10 | 0.006493 | 0.001312 | 0.005181 |
| cpp_optimized | pglib_opf_case2746wop_k | 10 | 0.011655 | 0.003065 | 0.008590 |
| cpp_optimized | pglib_opf_case4601_goc | 10 | 0.042545 | 0.006321 | 0.036224 |
| cpp_optimized | pglib_opf_case793_goc | 10 | 0.003196 | 0.000753 | 0.002442 |
| cpp_optimized | pglib_opf_case8387_pegase | 10 | 0.058166 | 0.012452 | 0.045713 |
| cpp_optimized | pglib_opf_case9241_pegase | 10 | 0.075937 | 0.013817 | 0.062120 |
| cpp_pypowerlike | pglib_opf_case118_ieee | 10 | 0.001387 | 0.000038 | 0.001348 |
| cpp_pypowerlike | pglib_opf_case1354_pegase | 10 | 0.019972 | 0.000105 | 0.019866 |
| cpp_pypowerlike | pglib_opf_case2746wop_k | 10 | 0.033506 | 0.000155 | 0.033351 |
| cpp_pypowerlike | pglib_opf_case4601_goc | 10 | 0.103242 | 0.000282 | 0.102959 |
| cpp_pypowerlike | pglib_opf_case793_goc | 10 | 0.009346 | 0.000076 | 0.009270 |
| cpp_pypowerlike | pglib_opf_case8387_pegase | 10 | 0.176660 | 0.000596 | 0.176065 |
| cpp_pypowerlike | pglib_opf_case9241_pegase | 10 | 0.244154 | 0.000681 | 0.243473 |
| pypower | pglib_opf_case118_ieee | 10 | 0.011169 | n/a | n/a |
| pypower | pglib_opf_case1354_pegase | 10 | 0.038989 | n/a | n/a |
| pypower | pglib_opf_case2746wop_k | 10 | 0.054658 | n/a | n/a |
| pypower | pglib_opf_case4601_goc | 10 | 0.138448 | n/a | n/a |
| pypower | pglib_opf_case793_goc | 10 | 0.021945 | n/a | n/a |
| pypower | pglib_opf_case8387_pegase | 10 | 0.247531 | n/a | n/a |
| pypower | pglib_opf_case9241_pegase | 10 | 0.318198 | n/a | n/a |
