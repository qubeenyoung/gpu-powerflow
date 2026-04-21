# CUDA Batch Ops — Deprecated Separate Track

Do not add separate CUDA batch operator implementations in this directory.

The CUDA backend's default execution model is batch execution. A single case is
represented as `batch_size = 1` and must use the same storage, operator, and
execution-plan path as larger batches.

## Current Direction

- Extend the existing CUDA storage and ops to be batch-aware.
- Use batch-major device buffers for voltage, residual, Jacobian values, and
  solution vectors.
- Use cuDSS uniform batch descriptors for Jacobian factorization and solve.
- Keep sparse structure, maps, and bus index arrays shared across the batch.
- Treat single-case APIs as wrappers around the same batch path with `B=1`.

Current implementation state:

- CUDA Mixed uses the batch-aware path. `solve_batch(B>1)` is enabled there.
- CPU FP64 and CUDA FP64 remain `B=1` compatibility paths.
- Mixed voltage state is `Va/Vm` FP64 with `V_re/V_im` FP64 cache.
- Mixed mismatch uses custom batch CSR `Ibus`, not a separate cuSPARSE path.

## Do Not Implement

- `CudaBatch*` storage or operator classes.
- A separate `BatchExecutionPlan`.
- A separate `BatchPlanBuilder`.
- A separate `.cu` source tree for batch kernels.
