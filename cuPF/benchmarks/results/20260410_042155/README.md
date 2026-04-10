# Benchmark Run `20260410_042155`

- Created (UTC): 2026-04-10T04:21:55.309273+00:00
- Cases: case30_ieee
- Implementations: pypower, cpp_pypowerlike, cpp_optimized
- Warmup: 0
- Repeats: 1

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
| cpp_optimized | pglib_opf_case30_ieee | 0.000802 | 0.000327 | 0.000474 |
| cpp_pypowerlike | pglib_opf_case30_ieee | 0.001294 | 0.000182 | 0.001111 |
| pypower | pglib_opf_case30_ieee | 0.010553734377026558 | n/a | n/a |
