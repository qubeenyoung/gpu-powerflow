# STRUMPACK GPU Path Check

Date: 2026-05-11

## Question

Verify whether the STRUMPACK benchmark uses a GPU path, not just whether the
library was compiled with CUDA.

## Static Build Evidence

- STRUMPACK build has CUDA enabled:
  - `third_party/strumpack/build_cuda_nolto/CMakeCache.txt`
  - `STRUMPACK_USE_CUDA:BOOL=ON`
- STRUMPACK build defines GPU support:
  - `third_party/strumpack/build_cuda_nolto/StrumpackConfig.h`
  - `#define STRUMPACK_USE_CUDA`
  - `#define STRUMPACK_USE_GPU`
- SLATE is not enabled:
  - `TPL_ENABLE_SLATE:BOOL=OFF`
  - `STRUMPACK_USE_SLATE_SCALAPACK` is undefined
- The benchmark wrapper explicitly requests STRUMPACK GPU offload:
  - `solvers/strumpack/strumpack_benchmark.cpp`
  - `solver.options().enable_gpu();`
- The executable links CUDA libraries:
  - `libcudart.so.12`
  - `libcublas.so.12`
  - `libcusolver.so.11`
  - `libcusparse.so.12`

## Runtime Profiling Evidence

Command:

```bash
/usr/local/cuda/bin/ncu \
    --target-processes all \
    --kernel-name-base demangled \
    --launch-count 40 \
    --section LaunchStats \
    --csv \
    --log-file exp/20260510/lin_sol/measurement_audit/logs/strumpack_gpu_path/ncu_direct_case2869.csv \
    exp/20260510/lin_sol/solvers/strumpack/build/strumpack_benchmark \
    --matrix exp/20260510/lin_sol/datasets/dumped_systems/case2869pegase/iter_000/J.mtx \
    --rhs exp/20260510/lin_sol/datasets/dumped_systems/case2869pegase/iter_000/rhs.txt \
    --xref exp/20260510/lin_sol/datasets/dumped_systems/case2869pegase/iter_000/x_ref.txt \
    --meta exp/20260510/lin_sol/datasets/dumped_systems/case2869pegase/iter_000/meta.json \
    --dtype fp64 \
    --repeats 1 \
    --warmup 0 \
    --out exp/20260510/lin_sol/measurement_audit/results/raw_json/strumpack_gpu_path_ncu_case2869_fp64.json
```

Nsight Compute connected to `strumpack_benchmark` and captured 40 CUDA kernel
launches. Distinct captured kernel groups:

| count | kernel |
| ---: | --- |
| 12 | `extend_add_kernel<double, 16>` |
| 7 | `assemble_kernel<double, 8>` |
| 7 | `LU_block_kernel_batched<double, 8, double>` |
| 7 | `solve_block_kernel_batched<double, 8>` |
| 7 | `Schur_block_kernel_batched<double, 8>` |

The profiled run also reported GPU memory use through the wrapper:

- `peak_gpu_memory_mb`: 373.375
- `gpu_resident_after_initial_load`: `cpu_gpu_hybrid_mpi_dist`
- `data_residency`: `host_input_output_with_internal_gpu_offload`

## Important Limitation

The same run emitted STRUMPACK's own warning:

```text
WARNING: SLATE is required for full GPU support.
Configure with -DTPL_ENABLE_SLATE=ON
and set the SLATE_DIR environment variable.
```

Therefore the current STRUMPACK result is a real CUDA-offload path, but not a
full GPU-resident distributed sparse direct path. Matrix input/output remain
host/MPI distributed in this wrapper, and full distributed-memory GPU support
would require a SLATE-enabled STRUMPACK build.

## Classification

- GPU kernel execution: yes
- CUDA build: yes
- Wrapper requests GPU: yes
- Full GPU support: no, because SLATE is disabled
- cuDSS-equivalent GPU residency: no
- Correct benchmark interpretation: STRUMPACK MPI/hybrid direct baseline with
  internal CUDA offload

