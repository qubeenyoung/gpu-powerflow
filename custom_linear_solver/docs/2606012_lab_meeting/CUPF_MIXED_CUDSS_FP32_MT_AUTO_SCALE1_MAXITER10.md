# cuPF mixed cuDSS FP32 MT-auto merged run

- B=1: previous native measurement, median initial/solve time; all cases converged within 10 iterations.
- B=16,64,256: identical batch replicas, `CUPF_BENCH_SCALE_STEP=0.0`, `CUPF_BENCH_MAX_ITER=10`, tolerance `1e-8`, repeats `5`.
- Backend: CUDA / `compute=mixed` / cuDSS / MT-auto (`CUDSS_THREADING_LIB=libcudss_mtlayer_gomp.so.0`).

## Init And Solve

| case | B | init ms | solve ms | ms/system | iterations | max mismatch |
|---|---:|---:|---:|---:|---:|---:|
| `case3012wp` | 1 | 30.019 | 1.672 | 1.672 | 4.0 | 3.188e-11 |
| `case3012wp` | 16 | 200.427 | 6.503 | 0.406 | 4.0 | 4.524e-11 |
| `case3012wp` | 64 | 30.578 | 21.728 | 0.339 | 4.0 | 6.502e-11 |
| `case3012wp` | 256 | 30.321 | 83.365 | 0.326 | 4.0 | 1.121e-10 |
| `case6468rte` | 1 | 43.639 | 2.078 | 2.078 | 4.0 | 1.732e-11 |
| `case6468rte` | 16 | 215.252 | 10.281 | 0.643 | 4.0 | 1.788e-11 |
| `case6468rte` | 64 | 42.801 | 36.112 | 0.564 | 4.0 | 1.714e-11 |
| `case6468rte` | 256 | 44.370 | 141.331 | 0.552 | 4.0 | 2.155e-11 |
| `case8387pegase` | 1 | 57.579 | 2.998 | 2.998 | 4.0 | 8.117e-10 |
| `case8387pegase` | 16 | 212.103 | 13.758 | 0.860 | 4.0 | 9.984e-10 |
| `case8387pegase` | 64 | 55.118 | 47.913 | 0.749 | 4.0 | 1.014e-09 |
| `case8387pegase` | 256 | 53.087 | 194.378 | 0.759 | 4.0 | 1.178e-09 |
| `case_ACTIVSg25k` | 1 | 107.061 | 6.600 | 6.600 | 5.0 | 8.401e-09 |
| `case_ACTIVSg25k` | 16 | 292.724 | 52.226 | 3.264 | 5.0 | 8.467e-09 |
| `case_ACTIVSg25k` | 64 | 105.195 | 215.921 | 3.374 | 6.0 | 2.906e-11 |
| `case_ACTIVSg25k` | 256 | 105.989 | 913.425 | 3.568 | 6.0 | 3.297e-11 |
| `case_SyntheticUSA` | 1 | 320.041 | 28.490 | 28.490 | 8.0 | 1.158e-09 |
| `case_SyntheticUSA` | 16 | 481.106 | 281.544 | 17.596 | 8.0 | 4.681e-11 |
| `case_SyntheticUSA` | 64 | 384.016 | 1116.640 | 17.448 | 8.0 | 1.315e-10 |
| `case_SyntheticUSA` | 256 | 308.906 | 4434.060 | 17.320 | 8.0 | 1.277e-10 |

## Dominant Solve Operators

| case | B | solve total | factorize | triangular solve | upload | download | jacobian | mismatch+norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `case3012wp` | 1 | 1.676 | 0.871 | 0.483 | 0.076 | 0.030 | 0.029 | 0.081 |
| `case3012wp` | 16 | 6.568 | 3.845 | 1.735 | 0.350 | 0.303 | 0.057 | 0.109 |
| `case3012wp` | 64 | 21.860 | 13.720 | 5.996 | 0.755 | 0.670 | 0.157 | 0.151 |
| `case3012wp` | 256 | 83.256 | 53.095 | 23.093 | 2.522 | 2.301 | 0.517 | 0.384 |
| `case6468rte` | 1 | 2.091 | 1.086 | 0.548 | 0.182 | 0.039 | 0.033 | 0.084 |
| `case6468rte` | 16 | 10.286 | 6.116 | 2.724 | 0.487 | 0.448 | 0.114 | 0.123 |
| `case6468rte` | 64 | 36.278 | 22.300 | 9.893 | 1.391 | 1.211 | 0.361 | 0.253 |
| `case6468rte` | 256 | 141.121 | 87.217 | 38.453 | 5.093 | 5.132 | 1.351 | 0.749 |
| `case8387pegase` | 1 | 3.022 | 1.736 | 0.723 | 0.242 | 0.047 | 0.038 | 0.090 |
| `case8387pegase` | 16 | 13.761 | 8.479 | 3.366 | 0.620 | 0.549 | 0.162 | 0.137 |
| `case8387pegase` | 64 | 47.778 | 30.307 | 11.966 | 1.748 | 1.494 | 0.553 | 0.284 |
| `case8387pegase` | 256 | 194.855 | 117.517 | 46.287 | 6.230 | 15.267 | 2.120 | 0.875 |
| `case_ACTIVSg25k` | 1 | 6.604 | 4.134 | 1.598 | 0.413 | 0.085 | 0.062 | 0.122 |
| `case_ACTIVSg25k` | 16 | 52.348 | 35.173 | 12.506 | 1.500 | 1.389 | 0.555 | 0.308 |
| `case_ACTIVSg25k` | 64 | 206.945 | 140.711 | 49.542 | 5.217 | 4.804 | 2.177 | 0.927 |
| `case_ACTIVSg25k` | 256 | 949.798 | 632.127 | 221.720 | 17.692 | 45.525 | 9.823 | 3.699 |
| `case_SyntheticUSA` | 1 | 27.777 | 18.520 | 6.537 | 1.400 | 0.365 | 0.259 | 0.227 |
| `case_SyntheticUSA` | 16 | 282.698 | 196.029 | 67.319 | 4.946 | 5.627 | 3.201 | 1.225 |
| `case_SyntheticUSA` | 64 | 1118.950 | 765.181 | 262.420 | 15.490 | 38.562 | 12.564 | 4.375 |
| `case_SyntheticUSA` | 256 | 4433.930 | 3038.590 | 1042.800 | 58.852 | 149.400 | 50.050 | 17.014 |

## Files

- Merged init/solve CSV: `custom_linear_solver/docs/2606012_lab_meeting/data/cupf_mixed_cudss_fp32_mt_auto_scale1_maxiter10_init_solve.csv`
- Merged operator CSV: `custom_linear_solver/docs/2606012_lab_meeting/data/cupf_mixed_cudss_fp32_mt_auto_scale1_maxiter10_ops_ms.csv`
