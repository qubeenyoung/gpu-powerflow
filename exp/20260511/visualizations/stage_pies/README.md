# cuPF Stage Pie Check: case13659pegase

## Verdict

- Feasible: yes, for measured CPU reference and full cuPF GPU buckets.
- Intermediate pies for `+ cuDSS linear solve` and `+ GPU Jacobian` are compositional estimates: they replace one measured CPU bucket at a time with the corresponding measured cuPF bucket.
- A true ablation pie would require dedicated `cuda_wo_cudss` / `cuda_wo_jacobian` benchmark paths in the current 20260511 flow.

## Inputs

- CPU reference: Python/SciPy Newton path from `exp/20260511/benchmarks/utils.py` with local per-stage timers.
- Full cuPF GPU: `/workspace/gpu-powerflow/cuPF/build/bench-operators/benchmarks/cupf_case_benchmark` profile `cuda_mixed_edge` with `ENABLE_TIMING=ON`.
- Linear bucket: `prepare_rhs + factorize + solve`.
- Expanded NR buckets: `ibus`, `mismatch`, `mismatch_norm`, and `voltage_update`; CPU reference has zero for `ibus` and `mismatch_norm` because those are not separately timed there.
- `outer / transfer` is retained in timing CSVs as a diagnostic but excluded from the pies and stage totals.

## Stage Totals

| stage | total ms | speedup vs CPU reference |
| --- | ---: | ---: |
| MATPOWER | 189.281 | 1.00x |
| 선형계 GPU 가속 | 53.323 | 3.55x |
| 선형계 + 자코비안 GPU 가속 | 9.295 | 20.36x |
| cuPF 전체 구조 | 6.571 | 28.80x |

## Component Means

| source | linear ms | jacobian ms | ibus ms | mismatch ms | mismatch norm ms | voltage update ms | excluded outer/transfer ms | measured solve ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU reference | 141.762 | 44.110 | 0.000 | 1.515 | 0.000 | 1.895 | 0.258 | 189.540 |
| Full cuPF GPU | 5.803 | 0.082 | 0.340 | 0.053 | 0.221 | 0.070 | 0.908 | 7.480 |

## Outputs

- `stage_pies_case13659pegase.png` / `.pdf`
- `stage_pie_data.csv`
- `cpu_reference_timing.csv`
- `gpu_timing.csv`
