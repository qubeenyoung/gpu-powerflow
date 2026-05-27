# cuDSS case6468rte Subphase Profile

Internal benchmark timer only; no Nsight Systems capture was used.

Command per run:

`/workspace/gpu-powerflow/exp/20260519/build/cudss_pf_benchmark --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case6468rte --rhs-mode synthetic --precision fp64 --warmup 1 --repeats 1 --csv --enable-mt --threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so --split-cudss-phases`

Method: 10 independent process runs. Each run uses `warmup=1` and `repeats=1`; the reported statistics aggregate the measured run from each process. The timer synchronizes the CUDA device before and after each timed range.

Metadata:

- case: `case6468rte`
- precision/rhs: `fp64` / `synthetic`
- cuDSS version: `0.7.1`
- linear_dim / nnz: `12643` / `87845`
- LU_NNZ after analysis/factor: `160245` / `160245`

## Phase Timings

| phase | mean_ms | median_ms | min_ms | max_ms | stddev_ms |
|---|---:|---:|---:|---:|---:|
| `analysis` | 54.392 | 61.048 | 20.865 | 78.574 | 23.011 |
| `factorization` | 0.702 | 0.689 | 0.679 | 0.836 | 0.045 |
| `solve` | 0.318 | 0.316 | 0.312 | 0.334 | 0.006 |
| `total` | 55.412 | 62.135 | 21.868 | 79.577 | 23.008 |

## Subphase Timings

| subphase | mean_ms | median_ms | min_ms | max_ms | stddev_ms | parent_share |
|---|---:|---:|---:|---:|---:|---:|
| `analysis.reordering` | 45.394 | 52.027 | 11.972 | 68.983 | 22.924 | 83.5% |
| `analysis.symbolic_factorization` | 8.996 | 8.919 | 8.851 | 9.589 | 0.205 | 16.5% |
| `solve.forward` | 0.127 | 0.125 | 0.124 | 0.136 | 0.004 | 39.8% |
| `solve.forward_to_rhs` | 0.008 | 0.007 | 0.007 | 0.011 | 0.001 | 2.4% |
| `solve.backward` | 0.182 | 0.182 | 0.180 | 0.187 | 0.002 | 57.3% |

## Accuracy

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| relative_residual | 1.010e-15 | 1.005e-15 | 9.007e-16 | 1.132e-15 |
| relative_error | 5.308e-14 | 4.258e-14 | 3.530e-14 | 9.756e-14 |

Files:

- raw per-run CSV: `exp/20260520/report/cudss_case6468rte_subphase_profile_runs.csv`
- summary CSV: `exp/20260520/report/cudss_case6468rte_subphase_profile_summary.csv`
