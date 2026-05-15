# case9241pegase Jacobian Assembly Measurement

- Generated UTC: `2026-05-14 09:02:34`
- Case dir: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case9241pegase`
- cuPF operators binary: `/workspace/gpu-powerflow-master/cuPF/build/bench-operators/benchmarks/cupf_case_benchmark`
- Warmup: `3`
- Repeats: `30`
- Metric: average Jacobian assembly time per Newton Jacobian update.
- Python/PyPower uses the dump Ybus/Sbus/V/pv/pq and the repository's PyPower Newton wrapper, which calls PYPOWER `dSbus_dV`.
- C/CUDA use `METRIC name=NR.iteration.jacobian` from the cuPF operators benchmark.

## Summary

| Method | Profile | Mean ms | Median ms | Std ms | Speedup vs PyPower |
|---|---|---:|---:|---:|---:|
| PyPower bus | `pypower` | 11.423147 | 11.388767 | 0.211893 | 1.00x |
| C bus | `cpp_pypowerlike` | 4.336550 | 4.347417 | 0.121922 | 2.63x |
| C edge | `cpu_fp64_edge` | 0.479944 | 0.479500 | 0.002404 | 23.80x |
| CUDA edge | `cuda_mixed_edge` | 0.012072 | 0.012000 | 0.000113 | 946.23x |

## Files

- `raw.csv`: per-repeat rows.
- `summary.csv`: method-level aggregate rows.
- `stdout_*.txt` / `stderr_*.txt`: raw cuPF benchmark output.
- Output directory: `exp/20260423/jac_asm_9000/results/case9241pegase_w3_r30_20260514`
