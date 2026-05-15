# case9241pegase Jacobian Assembly Measurement

- Generated UTC: `2026-05-14 09:16:05`
- Case dir: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case9241pegase`
- cuPF operators binary: `/workspace/gpu-powerflow-master/cuPF/build/bench-operators/benchmarks/cupf_case_benchmark`
- Warmup: `3`
- Repeats: `30`
- Metric: average Jacobian assembly time per Newton Jacobian update.
- Python/PyPower uses the dump Ybus/Sbus/V/pv/pq and the repository's PyPower Newton wrapper, which calls PYPOWER `dSbus_dV`.
- C/CUDA use `METRIC name=NR.iteration.jacobian` from the cuPF operators benchmark.
- CUDA profiles are run with `CUDA_LAUNCH_BLOCKING=1`, so the scoped timer waits for kernel completion.
- `cuda_event_jac_asm_*.csv` is standalone CUDA-event timing for bus/edge fill kernels.
- CUDA bus is measured with the standalone `jac_asm_vertex_thread` kernel because the cuPF solver benchmark exposes CUDA solver profiles only for edge-based Jacobian assembly.

## Summary

| Method | Profile | Mean ms | Median ms | Std ms | Speedup vs PyPower |
|---|---|---:|---:|---:|---:|
| PyPower bus | `pypower` | 11.431966 | 11.404044 | 0.132380 | 1.00x |
| C bus | `cpp_pypowerlike` | 4.345850 | 4.366083 | 0.211852 | 2.63x |
| C edge | `cpu_fp64_edge` | 0.485844 | 0.482333 | 0.010993 | 23.53x |
| CUDA edge | `cuda_mixed_edge` | 0.016239 | 0.016167 | 0.000373 | 703.99x |

## CUDA Event Fill Summary

| Method | Kernel/Profile | Basis | Fill ms | Speedup vs CUDA bus |
|---|---|---|---:|---:|
| CUDA bus | `jac_asm_vertex_thread` | bus | 0.025242 | 1.00x |
| CUDA edge | `jac_asm_edge` | edge | 0.007581 | 3.33x |

## Files

- `raw.csv`: per-repeat rows.
- `summary.csv`: method-level aggregate rows.
- `cuda_event_summary.csv`: standalone CUDA event aggregate rows, when enabled.
- `cuda_event_jac_asm_*.csv`: raw standalone CUDA event timing, when enabled.
- `stdout_*.txt` / `stderr_*.txt`: raw cuPF benchmark output.
- Output directory: `exp/20260423/jac_asm_9000/results/case9241pegase_w3_r30_with_cuda_bus_20260514`
