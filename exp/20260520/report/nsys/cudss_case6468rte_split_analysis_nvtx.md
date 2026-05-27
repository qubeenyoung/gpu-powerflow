# cuDSS case6468rte Nsight Systems: split analysis NVTX

Command target:

`exp/20260519/build/cudss_pf_benchmark --case-dir datasets/matpower8.1/cupf_all_dumps/case6468rte --rhs-mode synthetic --precision fp64 --warmup 1 --repeats 1 --csv --enable-mt --threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so --cuda-profiler-capture --split-cudss-analysis`

Capture:

- `cudss_case6468rte_split_analysis_nvtx.nsys-rep`
- `cudss_case6468rte_split_analysis_nvtx.sqlite`

NVTX ranges:

| range | ms |
|---|---:|
| `cudss.analysis` | 22.554 |
| `cudss.analysis.reordering` | 13.631 |
| `cudss.analysis.symbolic_factorization` | 8.917 |
| `cudss.factorize` | 0.707 |
| `cudss.solve` | 0.307 |

Benchmark row:

- `cudss_analysis_lu_nnz`: 160245
- `cudss_factor_lu_nnz`: 160245
- `relative_residual`: 9.95384217528e-16
- `relative_error`: 4.18052719737e-14

Note: analysis is split using public cuDSS phases
`CUDSS_PHASE_REORDERING` and `CUDSS_PHASE_SYMBOLIC_FACTORIZATION`. Solve remains
a single `CUDSS_PHASE_SOLVE` call because individually invoking solve subphases
produced an invalid solution in a smoke test.
