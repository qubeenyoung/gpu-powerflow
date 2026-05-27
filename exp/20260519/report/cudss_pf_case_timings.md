# cuDSS Power Flow Jacobian Benchmark

- precision: `fp64`
- rhs_mode: `synthetic`
- warmup/repeats: `1/5`
- cuDSS threading layer: `enabled`

| status | case | n_bus | linear_dim | nnz | analysis ms | factorize ms | solve ms | total ms | rel residual | rel error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ok | case118 | 118 | 181 | 1051 | 8.720 | 0.089 | 0.070 | 8.879 | 2.854e-16 | 4.912e-15 |
| ok | case1197 | 1197 | 2392 | 14344 | 11.138 | 0.211 | 0.109 | 11.458 | 1.414e-16 | 3.962e-14 |
| ok | case3012wp | 3012 | 5725 | 36263 | 19.793 | 0.458 | 0.218 | 20.469 | 1.045e-15 | 6.050e-13 |
| ok | case6468rte | 6468 | 12643 | 87845 | 30.542 | 0.692 | 0.288 | 31.523 | 1.062e-15 | 8.197e-14 |
| ok | case8387pegase | 8387 | 14908 | 110572 | 28.909 | 0.936 | 0.362 | 30.207 | 7.215e-16 | 5.716e-14 |
