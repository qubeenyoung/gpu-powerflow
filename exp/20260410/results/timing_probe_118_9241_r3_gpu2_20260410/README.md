# Benchmark Run `timing_probe_118_9241_r3_gpu2_20260410`

- Created (UTC): 2026-04-10T04:53:03.995869+00:00
- Cases: 118_ieee, 9241_pegase
- Implementations: pypower, cpp_pypowerlike, cpp_optimized, cpp_cuda_edge, cpp_cuda_vertex
- Warmup: 1
- CPU repeats: 3
- GPU repeats: 3
- Requested GPU index: 2

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
| cpp_cuda_edge | pglib_opf_case118_ieee | 3 | 0.013865 | 0.012739 | 0.001125 |
| cpp_cuda_edge | pglib_opf_case9241_pegase | 3 | 0.063174 | 0.055971 | 0.007203 |
| cpp_cuda_vertex | pglib_opf_case118_ieee | 3 | 0.013735 | 0.012622 | 0.001113 |
| cpp_cuda_vertex | pglib_opf_case9241_pegase | 3 | 0.063494 | 0.056265 | 0.007228 |
| cpp_optimized | pglib_opf_case118_ieee | 3 | 0.000736 | 0.000195 | 0.000540 |
| cpp_optimized | pglib_opf_case9241_pegase | 3 | 0.075414 | 0.013007 | 0.062407 |
| cpp_pypowerlike | pglib_opf_case118_ieee | 3 | 0.001486 | 0.000039 | 0.001445 |
| cpp_pypowerlike | pglib_opf_case9241_pegase | 3 | 0.241614 | 0.000900 | 0.240714 |
| pypower | pglib_opf_case118_ieee | 3 | 0.011263 | n/a | n/a |
| pypower | pglib_opf_case9241_pegase | 3 | 0.317485 | n/a | n/a |
