# MATPOWER End-to-End/Solve Comparison

- Cases: 78 MATPOWER dump cases
- CPU runs: `cpp_naive` and `cpp`, batch 1, warmup 0, repeats 10
- GPU runs: `cuda_edge`, batch 1/4/16/64/256, warmup 3, repeats 10
- Timing columns are mean milliseconds from `aggregates_end2end.csv`.
- GPU per-case time is `gpu total time / batch_size`.
- Batch summary speedups use only cases where cpu reference, cpu cuPF, and that GPU batch all succeeded.

## Files

- `comparison_wide.csv`: one row per case with CPU baselines, GPU batch times, and speedups
- `comparison_long.csv`: tidy method/batch table
- `summary_by_batch.csv`: aggregate per-batch speedups
- `failures.csv`: unsuccessful case/profile combinations

## Batch Summary

| batch | eligible cases | ref elapsed gmean speedup | cpu elapsed gmean speedup | ref solve gmean speedup | cpu solve gmean speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 77 | 0.1577 | 0.04316 | 2.68 | 0.4561 |
| 4 | 77 | 0.339 | 0.09276 | 0.6825 | 0.1161 |
| 16 | 77 | 1.271 | 0.3479 | 2.434 | 0.4143 |
| 64 | 77 | 4.11 | 1.125 | 7.137 | 1.214 |
| 256 | 77 | 10.72 | 2.933 | 16.67 | 2.837 |

## Failures

| case | method | iterations mean | final mismatch max |
|---|---|---:|---:|
| case16am | cpu_ref | 50 | 2.00263e-08 |
| case16am | cpu_cupf | 50 | 2.00264e-08 |

