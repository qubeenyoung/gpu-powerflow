# cuDSS case6468rte Nsight Systems: split phase NVTX

Command target:

`exp/20260519/build/cudss_pf_benchmark --case-dir datasets/matpower8.1/cupf_all_dumps/case6468rte --rhs-mode synthetic --precision fp64 --warmup 1 --repeats 1 --csv --enable-mt --threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so --cuda-profiler-capture --split-cudss-phases`

Capture:

- `cudss_case6468rte_split_phases_nvtx.nsys-rep`
- `cudss_case6468rte_split_phases_nvtx.sqlite`

Summary CSV:

- `cudss_case6468rte_split_phases_nvtx_nvtx_sum.csv`
- `cudss_case6468rte_split_phases_nvtx_cuda_api_sum.csv`
- `cudss_case6468rte_split_phases_nvtx_cuda_gpu_kern_sum.csv`

NVTX ranges:

| range | ms |
|---|---:|
| `cudss.analysis` | 188.536 |
| `cudss.analysis.reordering` | 174.971 |
| `cudss.analysis.symbolic_factorization` | 13.533 |
| `cudss.factorize` | 0.698 |
| `cudss.solve` | 0.403 |
| `cudss.solve.backward` | 0.211 |
| `cudss.solve.forward` | 0.169 |
| `cudss.solve.forward_to_rhs` | 0.014 |

Benchmark row:

- `cudss_analysis_lu_nnz`: 160245
- `cudss_factor_lu_nnz`: 160245
- `relative_residual`: 1.06038806674e-15
- `relative_error`: 5.36443063121e-14

Note: analysis is split using public cuDSS phases
`CUDSS_PHASE_REORDERING` and `CUDSS_PHASE_SYMBOLIC_FACTORIZATION`. Solve is
split using `CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD`, an
intermediate `x -> rhs` device copy for the GENERAL/LU buffer convention, and
`CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM`.
