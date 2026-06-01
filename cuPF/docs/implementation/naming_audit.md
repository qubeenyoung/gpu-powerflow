# Naming consistency audit (cycle 3 / s3)

- Date: 2026-05-31
- Scope: goal #1 (variable/function/type naming consistency)
- Verification: `WITH_CUDA=ON BUILD_EVALUATORS=ON` build green **and** full
  `WITH_CUDA=ON BUILD_PYTHON_BINDINGS=ON CUPF_WITH_TORCH=ON BUILD_EVALUATORS=ON`
  build green (`libcupf.a`, `cupf_cpp_evaluate`, `_cupf*.so`, `cupf_minimal_tests`).

This file records the canonical naming scheme the source code actually follows
(static-dispatch design, **not** the virtual `I*Op` interface design that the
older `docs/overview.md` / `docs/ops/README.md` still describe — that doc/src
gap is left for s5), and the renames applied this cycle to make the code
internally consistent.

## 1. Canonical scheme

| Category | Rule | Examples |
|---|---|---|
| Backend prefix | `Cpu` / `Cuda` | `CpuMismatchOp`, `CudaJacobianOp` |
| Profile / precision token | Pipelines & storage carry `Fp64` / `Fp32` / `Mixed`; ops are **templated** on `<double>`/`<float>` and carry no precision token | `CudaMixedPipeline`, `CudaFp64Storage`, `CudaJacobianOp<float>` |
| Storage type | `{Cpu,Cuda}{Fp64,Fp32,Mixed}Storage`, defined in the matching `*_storage.{hpp,cpp}` under `storage/` | `CudaFp64Storage` ↔ `cuda_fp64_storage.hpp` |
| Op struct | `{Cpu,Cuda}<Stage>Op`; stage ∈ {Ibus, Mismatch, MismatchNorm, Jacobian, VoltageUpdate} | `CpuVoltageUpdateOp`, `CudaIbusOp<S>` |
| Linear-solve op | library suffix instead of `Op` | `CpuLinearSolveKLU`, `CudaLinearSolveCuDSS<T,S>` |
| Pipeline | `{Cpu,Cuda}<Profile>Pipeline`, aggregated by the `SolverPipeline` variant | `CudaMixedPipeline` |
| Kernel pair | host launcher `launch_<verb>`, device kernel `<verb>_kernel` | `launch_compute_ibus` ↔ `compute_ibus_kernel` |
| Device buffer | `d_` prefix; an array belonging to a logical matrix keeps that matrix's prefix | `d_Ybus_re`, `d_Ybus_indptr`, `d_J_values` |
| State fields | `Va`/`Vm` (polar, authoritative) · `V_re`/`V_im` (rectangular cache) · `Ibus` · `F` (mismatch) · `normF` · `dx` | — |

### CSR member naming is provenance-based (intentional, kept)
External/input matrices (Ybus, and anything coming from the scipy CSR caller)
use the scipy spelling `indptr` / `indices`; the internally-built Jacobian uses
`row_ptr` / `col_idx`. This is a deliberate signal of where the array comes from
and is **not** an inconsistency, so it was left as-is.

## 2. Renames applied this cycle (pure token substitution, logic unchanged)

| Before | After | Why |
|---|---|---|
| `CpuFp64Buffers` | `CpuFp64Storage` | type name now matches the `storage/` dir, the `*_storage.{hpp,cpp}` filenames, and the docs' "Storage" role; 4 storage types were the lone `*Buffers` outliers |
| `CudaFp64Buffers` | `CudaFp64Storage` | same |
| `CudaFp32Buffers` | `CudaFp32Storage` | same |
| `CudaMixedBuffers` | `CudaMixedStorage` | same |
| `d_Y_row` | `d_Ybus_row` | every other Ybus array in the same struct uses the `d_Ybus_` prefix |
| `launch_voltage_update_state` | `launch_apply_voltage_update` | launcher now matches its kernel `apply_voltage_update_kernel`, like the sibling pair `launch_reconstruct_voltage` ↔ `reconstruct_voltage_kernel` |

Touched ~40 files across `core/`, `ops/`, `storage/`, `reference/`, plus the two
doc lines that referenced the old symbols (`docs/storage/README.md`,
`docs/implementation/deviations.md`). No CMake change needed (filenames are
unchanged; only symbols moved).

## 3. Known residual naming items (deferred)

- `CpuJacobianOpF64` / `CpuNaiveJacobianOpF64` keep the `F64` suffix even though
  the other CPU ops drop it. Left intentionally: the CPU path is FP64-only and
  `docs/ops/README.md` already documents these exact names, so renaming would
  *create* a doc/src gap. Revisit together with the doc reconciliation in s5.
- `cuda_custom_solver.{hpp,cpp}` was renamed for consistency but is gated by
  `CUPF_ENABLE_CUSTOM_SOLVER=OFF` and cannot be built (pre-existing `atomicAdd`
  bug recorded in s1), so the rename there is unverified by build.

## 4. Remaining doc↔src naming gaps for s5

- `docs/overview.md` / `docs/ops/README.md` describe a virtual interface design
  (`IStorage`, `IMismatchOp`, `IJacobianOp`, `ILinearSolveOp`, `IVoltageUpdateOp`,
  `IIbusOp`, `IMismatchNormOp`) and an `op_interfaces.hpp` file. None exist — the
  real design is stateless `Cpu*Op`/`Cuda*Op` structs dispatched through the
  `SolverPipeline` `std::variant`.
- Docs assume a `cupf::` namespace; the code uses the global namespace.
- `docs/overview.md` contains garbled identifiers from an earlier bulk
  substitution ("solver stage configuration::build", "NewtonSolver stage
  ownership").
- `benchmarks/` is referenced in `docs/overview.md` but absent from the tree.
