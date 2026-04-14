# CUDA Multi-Batch Ops — Future Track

This directory is reserved for CUDA multi-batch operator implementations.

Multi-batch is explicitly **out of scope** for the Phase A–D refactor.
Single-case experiments are the current priority.

## Planned contents (do not implement until single-case is stable)

- `CudaBatchMixedStorage`
- `CudaBatchFp32Storage`
- `CudaBatchMismatchOpMixed`
- `CudaBatchMismatchOpFp32`
- `CudaBatchJacobianOpEdgeF32`
- `CudaBatchLinearSolveCuDSS32`
- `CudaBatchVoltageUpdateMixed`
- `CudaBatchVoltageUpdateFp32`

## Principles

- Single-case and batch ops must **not** share `.cu` source files.
- Batch storage uses different buffer layouts and cuDSS UBATCH descriptors.
- A separate `BatchExecutionPlan` or `BatchPlanBuilder` may be needed.
- The `IStorage` / `IOp` interface concepts can be reused; concrete types cannot.
