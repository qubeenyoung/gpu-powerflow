# cuDSS MG Newton Solver Benchmark `smoke_modified_parse`

## Setup

- Created UTC: 2026-04-16T04:49:17.674128+00:00
- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`
- Cases: case_ACTIVSg200
- Profiles: cuda_edge_modified
- Modes: operators
- Warmup: 0
- Repeats: 1
- cuDSS MG indices: `0,1`
- cuDSS reordering: `DEFAULT`
- cuDSS MT: False
- cuDSS host threads: `AUTO`
- cuDSS ND_NLEVELS: `AUTO`
- CUDA_VISIBLE_DEVICES: `2,3`

## Files

- `manifest.json`: configuration, commands, and environment
- `summary.csv`: one row per repeat
- `aggregates.csv`: grouped timing statistics
- `mg_comparison.csv`: top-level baseline vs MG timing
- `operator_comparison.csv`: all collected timers, baseline vs MG
- `raw/`: per-repeat parsed payload and stdout
