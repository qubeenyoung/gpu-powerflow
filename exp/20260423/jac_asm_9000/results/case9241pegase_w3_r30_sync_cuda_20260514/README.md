# case9241pegase Jacobian Assembly Measurement

- Generated UTC: `2026-05-14 09:06:14`
- Case dir: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case9241pegase`
- cuPF operators binary: `/workspace/gpu-powerflow-master/cuPF/build/bench-operators/benchmarks/cupf_case_benchmark`
- Warmup: `3`
- Repeats: `30`
- Metric: average Jacobian assembly time per Newton Jacobian update.
- Python/PyPower uses the dump Ybus/Sbus/V/pv/pq and the repository's PyPower Newton wrapper, which calls PYPOWER `dSbus_dV`.
- C/CUDA use `METRIC name=NR.iteration.jacobian` from the cuPF operators benchmark.
- CUDA profiles are run with `CUDA_LAUNCH_BLOCKING=1`, so the scoped timer waits for kernel completion.
- `cuda_event_jac_asm_case9241pegase.csv` is a standalone CUDA-event timing of edge-based Jacobian fill only (`warmup=10`, `iters=1000`).

## Summary

| Method | Profile | Mean ms | Median ms | Std ms | Speedup vs PyPower |
|---|---|---:|---:|---:|---:|
| PyPower bus | `pypower` | 11.470393 | 11.448438 | 0.163207 | 1.00x |
| C bus | `cpp_pypowerlike` | 4.330800 | 4.365667 | 0.138381 | 2.65x |
| C edge | `cpu_fp64_edge` | 0.481644 | 0.481083 | 0.004672 | 23.82x |
| CUDA edge | `cuda_mixed_edge` | 0.016139 | 0.016167 | 0.000170 | 710.73x |

Standalone CUDA event fill time: `0.007556 ms`.

## Files

- `raw.csv`: per-repeat rows.
- `summary.csv`: method-level aggregate rows.
- `cuda_event_jac_asm_case9241pegase.csv`: CUDA event timing for the standalone edge-fill kernel path.
- `stdout_*.txt` / `stderr_*.txt`: raw cuPF benchmark output.
- Output directory: `exp/20260423/jac_asm_9000/results/case9241pegase_w3_r30_sync_cuda_20260514`
