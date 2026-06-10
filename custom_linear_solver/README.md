# custom_linear_solver

`custom_linear_solver` is the cuDSS-like wrapper layer for the copied CUDA
multifrontal solver code.

The copied operation code is organized directly under `src`: `analyze`,
`factorize`, `solve`, and `internal`. It is a small subset of
`external/lin_solver`: only the GPU multifrontal kernels, symbolic analysis,
METIS nested-dissection ordering, and the minimal matrix/API state are kept
here. Benchmark drivers, matching experiments, GPU-ND experiments, and
third-party solver adapters are not part of this module.

## Public API

The wrapper follows the cuDSS phase model:

```text
set_data      upload/register matrix descriptors
set_rhs       register RHS vector
set_solution  register output vector
analyze       one-time symbolic analysis for a fixed sparsity pattern
factorize     numeric factorization for current values
solve         solve current RHS into the output vector
get_data      inspect currently registered descriptors
```

## Configurable knobs

Set on `SolverConfig` (passed to the `Solver` ctor). All values shown are
defaults.

| Field                            | Default      | Description                                                                                              |
|----------------------------------|--------------|----------------------------------------------------------------------------------------------------------|
| `precision`                      | `FP64`       | `FP64` / `FP32` / `TF32` (PTX, recommended)                                                              |
| `max_panel_width`                | `8`          | Max columns amalgamated into one supernode panel (1..64). Analyzer uses width 16 for n>=16k unless `CLS_RESPECT_PANEL_CAP`. |
| `use_parallel_nested_dissection` | `true`       | METIS-ND uses parallel host threads                                                                       |
| `use_multistream_subtrees`       | `true`       | Dispatch independent subtrees on separate CUDA streams (capped at 8). Disable for single-stream debugging. |
| `analyze_emit_info`              | `false`      | After `analyze()`, print front-size and subtree summary to stderr                                          |
| `analyze_dump_fronts_path`       | `""`         | If non-empty, after `analyze()` write per-front CSV `(q,p,fsz,nc,uc,level)` here                          |
| `use_matching`                   | `false`      | Reserved — row matching pre-permutation                                                                   |
| `enable_shift_retry`             | `true`       | Reserved — diagonal-shift fallback on singular pivot                                                      |
| `shift_retry_epsilon`            | `1e-8`       | Reserved — shift magnitude                                                                                |

CMake options that change build behavior:

| Option                  | Default | Effect                                                                                       |
|-------------------------|---------|----------------------------------------------------------------------------------------------|
| `CLS_BUILD_CUDA_OPS`    | `ON`    | Build the multifrontal CUDA kernels                                                          |
| `CLS_INTERNAL_GRAPH`    | `ON`    | Capture factor/solve into internal CUDA graphs (standalone). `OFF` = external capture mode.  |
| `CLS_BUILD_SCRIPTS`     | `ON`    | Build the `custom_linear_solver_run` CLI                                                     |
| `CLS_BUILD_CUDSS_SCRIPT`| `OFF`   | Build the cuDSS comparison driver                                                            |
| `CLS_CUDA_ARCHITECTURES`| `86`    | Must be ≥ 80 for the TF32 kernels                                                            |

## Precision matrix

| Mode        | Front storage | Phase-3 trailing GEMM            | Accumulate | Accuracy on power-grid Jacobians |
|-------------|---------------|----------------------------------|------------|----------------------------------|
| `FP64`      | FP64          | scalar FP64                      | FP64       | ~1e-13                            |
| `FP32`      | FP32          | staged-scalar FP32               | FP32       | ~1e-4                             |
| `TF32`      | FP32          | TF32 PTX mma.m16n8k8 / k4 hybrid | FP32       | ~1e-4 (TF32 rounding)             |

`TF32` is the **recommended TF32 path** for power-grid Jacobians: it bakes in
the V9h stack (`docs/03-optimization-notes/02-tf32-trailing-gemm.md`) and the
EXP-B `__launch_bounds__(512, 2)` for the big tier
(`docs/03-optimization-notes/01-kernel-engineering.md`).

`TF32` requires Ampere (sm_80+).

## Experimental combinations

Anything you can vary at runtime from the CLI runner without rebuilding:

| Axis                     | Values                                                | CLI flag           |
|--------------------------|-------------------------------------------------------|--------------------|
| Precision                | `fp64` / `fp32` / `tf32`                           | `--precision`      |
| Batch size B             | 1, 4, 16, 64, 256, …                                  | `--batch`          |
| Max panel width          | 1..64                                                 | `--max-panel-width` |
| Multi-stream subtrees    | on / off                                              | `--no-multistream` |
| Repeat (timing)          | 1..N (median reported)                                | `--repeat`         |
| Single-system input dtype| `fp64` / `fp32`                                       | `--single-precision` |
| Analyze diagnostics      | summary / per-front CSV                               | `--analyze-info` / `--dump-fronts <path>` |

The TF32 PTX path is now baked in (V9h + LB(512, 2)); selecting it is just
`--precision tf32`. Experiments that historically lived behind a build flag
(`CLS_TF32_BIG_PTX`, `CLS_TF32_MID_PTX`, `CLS_TF32_MMA_AREUSE`,
`CLS_TF32_SKIP_CONVERT`, `CLS_TF32_MID_K4_HYBRID`, `CLS_TF32_BIG_LB_512_2`)
are now the default behavior of `Precision::TF32`. Failed variants
(`CLS_TF32_PTX_VARIANT`, `CLS_TF32_BIG_T_512`, `CLS_TF32_BIG_LB_256_4`,
always-k4 `CLS_TF32_MID_K4`, `CLS_SMALL_WARPS_16`) have been removed; see
`docs/03-optimization-notes/02-tf32-trailing-gemm.md` §4 and
`docs/03-optimization-notes/01-kernel-engineering.md` for the reasoning.

## Build + run

```bash
cmake -S custom_linear_solver -B build/custom_linear_solver \
  -DCLS_BUILD_CUDA_OPS=ON \
  -DCLS_BUILD_SCRIPTS=ON
cmake --build build/custom_linear_solver -j

# Recommended TF32 path, 4 systems batched:
build/custom_linear_solver/custom_linear_solver_run \
  /datasets/matpower_linear_systems/case30 \
  --precision tf32 --batch 4 --repeat 5 \
  --solution-out /tmp/case30_cls_solution.mtx

# Compatibility alias for the TF32 PTX path:
build/custom_linear_solver/custom_linear_solver_run \
  /datasets/matpower_linear_systems/case30 \
  --precision tf32_wmma --batch 4 --repeat 5
```

## Docs

Start from `docs/README.md`. The two most relevant optimization notes are
`docs/03-optimization-notes/02-tf32-trailing-gemm.md`
(V9h TF32 PTX trailing stack) and
`docs/03-optimization-notes/01-kernel-engineering.md`
(EXP-B LB(512, 2) and other tier/kernel levers), both baked into `Precision::TF32`.
The honest TF32 speedup analysis is in
`docs/03-optimization-notes/03-tensor-core-investigation.md`.
