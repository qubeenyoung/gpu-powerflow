# GPU Block ILU(0) File Split

## Files

- `src/tools/gpu_block_ilu0_phase_bench.cu`
  - CLI parsing
  - J/F dump loading
  - benchmark execution
  - sanity-test launch
  - CSV and Markdown output

- `src/tools/gpu_block_ilu0_kernels.cuh`
  - dense scatter kernel
  - ILU update kernels
  - diagonal preparation/apply kernels
  - vector copy and off-diagonal GEMV-subtract kernels

- `src/tools/gpu_block_ilu0_preconditioner.cuh`
  - shared data types
  - `GpuBlockILU0` class declaration
  - cuBLAS handle ownership

- `src/tools/gpu_block_ilu0_setup.cuh`
  - symbolic block pattern setup
  - dense scatter metadata
  - device buffer allocation
  - `GpuBlockILU0::setup`

- `src/tools/gpu_block_ilu0_factorize.cuh`
  - numeric buffer reset
  - dense scatter
  - block ILU(0) row updates
  - diagonal LU/inverse
  - `GpuBlockILU0::factorize`

- `src/tools/gpu_block_ilu0_apply.cuh`
  - forward triangular apply
  - backward triangular apply
  - optional output norm diagnostic
  - `GpuBlockILU0::apply`

## Checks

- Build target: `gpu_block_ilu0_phase_bench`
- Profile smoke: `case2383wp`, block size 16, factorization succeeded
- Fast smoke: `case2383wp`, block size 16, factorization succeeded
- Fast 5-case run after setup/factorize/apply split: all selected cases completed with `factor_failed = 0`

## Note

The fast path remains the runtime path to read for performance. The profile path intentionally inserts many event synchronizations and is useful only for phase attribution.
