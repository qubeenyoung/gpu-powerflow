# Benchmark Run `timing_probe_118_9241_r3_20260410`

- Created (UTC): 2026-04-10T04:50:18.196286+00:00
- Cases: 118_ieee, 9241_pegase
- Implementations: pypower, cpp_pypowerlike, cpp_optimized, cpp_cuda_edge, cpp_cuda_vertex
- Warmup: 1
- CPU repeats: 3
- GPU repeats: 3
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
| cpp_cuda_edge | pglib_opf_case118_ieee | 3 | 0.101552 | 0.055273 | 0.046279 |
| cpp_cuda_edge | pglib_opf_case9241_pegase | 3 | 0.207370 | 0.144596 | 0.062773 |
| cpp_cuda_vertex | pglib_opf_case118_ieee | 3 | 0.013730 | 0.012556 | 0.001174 |
| cpp_cuda_vertex | pglib_opf_case9241_pegase | 3 | 0.133334 | 0.100591 | 0.032744 |
| cpp_optimized | pglib_opf_case118_ieee | 3 | 0.000714 | 0.000193 | 0.000520 |
| cpp_optimized | pglib_opf_case9241_pegase | 3 | 0.075113 | 0.013097 | 0.062015 |
| cpp_pypowerlike | pglib_opf_case118_ieee | 3 | 0.001457 | 0.000039 | 0.001417 |
| cpp_pypowerlike | pglib_opf_case9241_pegase | 3 | 0.242275 | 0.000944 | 0.241331 |
| pypower | pglib_opf_case118_ieee | 3 | 0.011346 | n/a | n/a |
| pypower | pglib_opf_case9241_pegase | 3 | 0.319721 | n/a | n/a |
