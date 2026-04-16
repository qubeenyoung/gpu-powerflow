# cuDSS MG Newton Solver Benchmark `smoke_case30`

## Setup

- Created UTC: 2026-04-16T03:52:26.182389+00:00
- Dataset root: `/workspace/datasets/pglib-opf/cuPF_benchmark_dumps`
- Cases: case30_ieee
- Profiles: cuda_edge
- Modes: end2end, operators
- Warmup: 1
- Repeats: 1
- cuDSS MG indices: `0,1`
- cuDSS reordering: `DEFAULT`
- cuDSS MT: False
- cuDSS host threads: `AUTO`
- cuDSS ND_NLEVELS: `AUTO`
- CUDA_VISIBLE_DEVICES: ``

## end2end

| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| case30_ieee | cuda_edge | 11.115 | 19.254 | 0.577x | 0.695 | 2.298 | 0.302x | True/True | 5.0/5.0 |

## operators

| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| case30_ieee | cuda_edge | 11.148 | 18.626 | 0.599x | 0.758 | 2.231 | 0.340x | True/True | 5.0/5.0 |

## Operator Metrics

| case | profile | metric | baseline ms | MG ms | speedup | delta ms |
|---|---|---|---:|---:|---:|---:|
| case30_ieee | cuda_edge | `CUDA.analyze.cudss32.analysis.total_sec` | 8.524 | 14.347 | 0.594x | 5.823 |
| case30_ieee | cuda_edge | `CUDA.analyze.cudss32.setup.total_sec` | 1.604 | 1.793 | 0.895x | 0.189 |
| case30_ieee | cuda_edge | `CUDA.solve.factorization32.total_sec` | 0.067 | 0.255 | 0.263x | 0.188 |
| case30_ieee | cuda_edge | `CUDA.solve.refactorization32.total_sec` | 0.178 | 0.720 | 0.247x | 0.542 |
| case30_ieee | cuda_edge | `CUDA.solve.rhsPrepare.total_sec` | 0.038 | 0.036 | 1.056x | -0.002 |
| case30_ieee | cuda_edge | `CUDA.solve.solve32.total_sec` | 0.205 | 0.944 | 0.217x | 0.739 |
| case30_ieee | cuda_edge | `NR.analyze.jacobian_builder.total_sec` | 0.018 | 0.017 | 1.059x | -0.001 |
| case30_ieee | cuda_edge | `NR.analyze.linear_solve.total_sec` | 10.131 | 16.144 | 0.628x | 6.013 |
| case30_ieee | cuda_edge | `NR.analyze.storage_prepare.total_sec` | 0.238 | 0.230 | 1.035x | -0.008 |
| case30_ieee | cuda_edge | `NR.analyze.total.total_sec` | 10.389 | 16.393 | 0.634x | 6.004 |
| case30_ieee | cuda_edge | `NR.iteration.jacobian.total_sec` | 0.039 | 0.040 | 0.975x | 0.001 |
| case30_ieee | cuda_edge | `NR.iteration.linear_solve.total_sec` | 0.493 | 1.961 | 0.251x | 1.468 |
| case30_ieee | cuda_edge | `NR.iteration.mismatch.total_sec` | 0.093 | 0.096 | 0.969x | 0.003 |
| case30_ieee | cuda_edge | `NR.iteration.total.total_sec` | 0.692 | 2.165 | 0.320x | 1.473 |
| case30_ieee | cuda_edge | `NR.iteration.voltage_update.total_sec` | 0.061 | 0.061 | 1.000x | 0.000 |
| case30_ieee | cuda_edge | `NR.solve.download.total_sec` | 0.015 | 0.015 | 1.000x | 0.000 |
| case30_ieee | cuda_edge | `NR.solve.total.total_sec` | 0.757 | 2.230 | 0.339x | 1.473 |
| case30_ieee | cuda_edge | `NR.solve.upload.total_sec` | 0.047 | 0.046 | 1.022x | -0.001 |
| case30_ieee | cuda_edge | `analyze_sec` | 10.390 | 16.394 | 0.634x | 6.004 |
| case30_ieee | cuda_edge | `elapsed_sec` | 11.148 | 18.626 | 0.599x | 7.478 |
| case30_ieee | cuda_edge | `solve_sec` | 0.758 | 2.231 | 0.340x | 1.473 |

## Files

- `manifest.json`: configuration, commands, and environment
- `summary.csv`: one row per repeat
- `aggregates.csv`: grouped timing statistics
- `mg_comparison.csv`: top-level baseline vs MG timing
- `operator_comparison.csv`: all collected timers, baseline vs MG
- `raw/`: per-repeat parsed payload and stdout
