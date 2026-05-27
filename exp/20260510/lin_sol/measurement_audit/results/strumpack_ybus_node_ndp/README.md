# STRUMPACK Ybus GPU Benchmark

Date: 2026-05-18

Binary:
`/workspace/gpu-powerflow/exp/20260510/lin_sol/solvers/strumpack/build/strumpack_ybus_benchmark`

Build:
- STRUMPACK 8.0.0 from `/workspace/gpu-powerflow/third_party/lin_sol/strumpack/install`
- CUDA enabled: yes
- GPU offload requested: yes
- SLATE ScaLAPACK backend: yes
- MAGMA: no

Runtime:
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB
- `LD_LIBRARY_PATH=/workspace/gpu-powerflow/third_party/lin_sol/slate/install_cuda/lib:/usr/local/cuda/lib64`
- `OMP_NUM_THREADS=1`
- dtype: complex fp64
- warmup: 1
- repeats: 3
- diagonal shift: 0
- METIS ordering option: `enable_METIS_NodeNDP()`

Method:
- Inputs are MatrixMarket `dump_Ybus.mtx` files from `datasets/matpower8.1/cupf_all_dumps`.
- RHS is generated as `rhs = A * x_ref` with deterministic complex `x_ref`.
- Timed phases are STRUMPACK `set_csr_matrix + reorder`, `factor`, and `solve`.
- GPU timings include `cudaDeviceSynchronize()` after each timed STRUMPACK phase.
- Nsight Compute confirmed kernels including `assemble_kernel`, `LU_block_kernel_batched`,
  `solve_block_kernel_batched`, and `Schur_block_kernel_batched`.

Results:

| case | n | nnz | analysis ms | factor ms | solve ms | total ms | rel residual | rel error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case118 | 118 | 476 | 0.232 | 9.033 | 0.061 | 9.326 | 1.69e-14 | 9.21e-15 |
| case2869pegase | 2869 | 10805 | 8.239 | 11.020 | 1.022 | 20.282 | 2.31e-14 | 4.85e-13 |
| case9241pegase | 9241 | 37655 | 30.875 | 19.051 | 3.134 | 53.060 | 2.11e-14 | 1.80e-13 |
| case_ACTIVSg25k | 25000 | 85220 | 87.917 | 52.200 | 8.779 | 148.895 | 2.37e-14 | 3.55e-13 |
| case_ACTIVSg70k | 70000 | 236636 | 273.139 | 88.287 | 26.333 | 387.759 | 2.31e-14 | 1.91e-13 |
| case_SyntheticUSA | 82000 | 278406 | 321.731 | 117.649 | 31.291 | 470.672 | 1.81e-14 | 3.08e-13 |

Notes:
- The old `dumped_systems` matrix dataset expected by the existing benchmark was absent,
  so this benchmark targets available Ybus MatrixMarket files directly.
- Initial runs with STRUMPACK's default METIS NodeND completed but produced deep
  elimination-tree warnings on large cases. The final reported run uses
  STRUMPACK's recommended `enable_METIS_NodeNDP()` option and completed without
  those warnings.
- No diagonal shift was used in the final results.
