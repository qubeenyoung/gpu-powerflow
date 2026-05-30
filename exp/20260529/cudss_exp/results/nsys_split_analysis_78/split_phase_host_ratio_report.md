# cuDSS split analysis NVTX host ratio report

- Command pattern: `nsys profile --trace=cuda,nvtx,osrt ... cudss_run <case> --repeat 1 --matching --mt --split-analysis`
- Cases: 78 MATPOWER linear systems
- Primary CPU/host proxy: `(NVTX wall - union(kernel, memcpy, memset GPU activity)) / NVTX wall`. This is GPU-idle / host-dominated wall share, not sampled CPU core utilization.
- Secondary proxy: `(NVTX wall - union(CUDA runtime API calls)) / NVTX wall`, useful for estimating time spent outside visible CUDA API calls, such as CPU-side ordering work.

## Phase summary

| phase | cases | wall mean ms | wall median ms | GPU activity weighted % | host/GPU-idle mean % | host/GPU-idle median % | host/GPU-idle weighted % | outside CUDA API mean % | outside CUDA API weighted % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reordering | 78 | 18.915 | 10.420 | 16.3 | 76.7 | 73.1 | 83.7 | 39.0 | 59.9 |
| symbolic_factorization | 78 | 8.818 | 7.724 | 76.4 | 24.7 | 27.5 | 23.6 | 1.9 | 1.8 |

## Size bins

| phase | n bin | cases | wall mean ms | host/GPU-idle mean % | host/GPU-idle weighted % | outside CUDA API weighted % |
|---|---:|---:|---:|---:|---:|---:|
| reordering | <1k | 49 | 13.322 | 74.7 | 80.5 | 46.4 |
| reordering | 1k-10k | 18 | 15.414 | 76.5 | 80.7 | 51.7 |
| reordering | 10k-50k | 9 | 28.707 | 84.7 | 85.6 | 70.6 |
| reordering | >=50k | 2 | 143.361 | 92.1 | 92.1 | 89.1 |
| symbolic_factorization | <1k | 49 | 7.647 | 28.1 | 28.1 | 2.1 |
| symbolic_factorization | 1k-10k | 18 | 8.812 | 20.0 | 20.0 | 1.8 |
| symbolic_factorization | 10k-50k | 9 | 11.630 | 17.3 | 17.1 | 1.4 |
| symbolic_factorization | >=50k | 2 | 24.910 | 14.9 | 14.9 | 0.7 |

## Largest host-dominated reordering cases

| case | n | nnz | reordering wall ms | host/GPU-idle % | outside CUDA API % |
|---|---:|---:|---:|---:|---:|
| case_SyntheticUSA | 156255 | 1052085 | 157.325 | 92.2 | 89.6 |
| case_ACTIVSg70k | 134104 | 900558 | 129.397 | 91.9 | 88.5 |
| case2737sop | 5280 | 34242 | 70.328 | 95.7 | 89.2 |
| case_ACTIVSg25k | 47246 | 318672 | 51.247 | 88.4 | 80.2 |
| case136ma | 270 | 1556 | 44.660 | 94.1 | 83.5 |
| case69 | 136 | 808 | 37.974 | 93.1 | 81.2 |
| case_ACTIVSg10k | 18544 | 125174 | 38.542 | 89.4 | 77.9 |
| case51he | 100 | 592 | 35.768 | 92.7 | 79.9 |

## Top kernel families

### reordering

| kernel family | count | summed kernel ms |
|---|---:|---:|
| `offsets_par_ker` | 156 | 217.978 |
| `cudss::radix_sort_ker` | 156 | 7.239 |
| `cudss::trans_columns_ker` | 78 | 0.663 |
| `cudss::adjncy_ker` | 78 | 0.488 |
| `cudss::xadj_ker` | 78 | 0.450 |
| `cudss::trans_nnz_per_row_ker` | 78 | 0.333 |
| `cudss::copy_csr_columns_ker` | 78 | 0.190 |

### symbolic_factorization

| kernel family | count | summed kernel ms |
|---|---:|---:|
| `offsets_par_ker` | 312 | 446.233 |
| `cudss::define_superpanel_ker` | 78 | 20.124 |
| `cudss::csc_rows_ker` | 780 | 12.555 |
| `cudss::nnz_per_col_ker` | 156 | 10.113 |
| `cudss::dependency_map_ker` | 780 | 8.940 |
| `cudss::map_ker` | 78 | 6.909 |
| `cudss::radix_sort_ker` | 156 | 5.803 |
| `cudss::fwd_bwd_order_step_2_ker` | 969 | 2.066 |

## Top CUDA runtime APIs

### reordering

| runtime API | count | summed ms |
|---|---:|---:|
| `cudaLaunchKernel_v7000` | 702 | 290.620 |
| `cudaMemcpyAsync_v3020` | 1564 | 149.116 |
| `cudaFree_v3020` | 78 | 112.190 |
| `cudaMalloc_v3020` | 546 | 35.327 |
| `cudaStreamSynchronize_v3020` | 624 | 1.952 |
| `cudaMemsetAsync_v3020` | 468 | 1.748 |
| `cudaDeviceSynchronize_v3020` | 78 | 0.357 |

### symbolic_factorization

| runtime API | count | summed ms |
|---|---:|---:|
| `cudaMemcpyAsync_v3020` | 2260 | 489.033 |
| `cudaLaunchKernel_v7000` | 5280 | 171.125 |
| `cudaFree_v3020` | 234 | 5.742 |
| `cudaMemsetAsync_v3020` | 1431 | 5.207 |
| `cudaMalloc_v3020` | 234 | 3.036 |
| `cudaStreamSynchronize_v3020` | 1554 | 1.214 |
| `cudaDeviceSynchronize_v3020` | 78 | 0.146 |

## Inference

- Reordering is overwhelmingly host dominated in large cases. Visible GPU work is mostly matrix/graph preparation kernels (`xadj`, `adjncy`, transpose/copy, sort, offsets), while most wall time is outside CUDA runtime calls. With `--matching` enabled, matching/scaling-related CPU work may be included, but this profile cannot name cuDSS internal CPU functions.
- Symbolic factorization is much more CUDA-runtime/GPU driven. Its dominant kernels build and transform symbolic structures such as CSC rows, offsets, dependency maps, nnz-per-column, mapping, and superpanel metadata.
- Exact METIS/reordering internals are not directly visible because cuDSS is a closed library and this trace lacks CPU sampling call stacks. The split cuDSS phases let us bound/estimate where the host time occurs, not attribute it to a named internal function with certainty.
