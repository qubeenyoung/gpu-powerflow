# Dead-code / over-implementation audit (cycle 4 / s4)

- Date: 2026-05-31
- Scope: goal #3 (comments/readability) + goal #4 (over-implementation), per
  subgoal s4 — "remove only what is genuinely unreachable/unused; preserve the
  evaluation and test paths".
- Verification: isolated `docker run --rm cupf:latest` —
  `cmake -S cuPF -DWITH_CUDA=ON -DBUILD_EVALUATORS=ON -DBUILD_TESTING=ON` configures
  and the full build links green (`libcupf.a`, `cupf_cpp_evaluate`,
  `cupf_minimal_tests`); `ctest` passes (1/1, `cupf_minimal_tests`). No compiled
  source file was modified, so there is no functional change to verify beyond a
  clean build + tests.

## Removed (truly unreachable — never compiled, never referenced)

These files were not listed in any `CMakeLists.txt` source list and were not
`#include`d or referenced anywhere in the tree (verified by grep). They defined
reference variants that no pipeline ever instantiates.

| Removed path | What it was | Why dead |
|---|---|---|
| `cpp/src/newton_solver/reference/cpu_naive_jacobian_f64.{cpp,hpp}` | `CpuNaiveJacobianOpF64` — PyPower/MATPOWER-style reference Jacobian | not in CMake; `CpuNaiveJacobianOpF64` instantiated nowhere |
| `cpp/src/newton_solver/reference/cpu_superlu_solve.{cpp,hpp}` | `CpuLinearSolveSuperLU` — one-shot SuperLU reference linear solve | not in CMake (not even under `CUPF_SUPERLU_ENABLED`); referenced nowhere |
| `cpp/src/newton_solver/core/solver_stages.cpp` | 1-line tombstone comment ("removed; moved to newton_solver.cpp") | vestigial empty file, not in CMake |

The `cpp/src/newton_solver/reference/` directory is now empty and was removed.

### CMake cleanup tied to the above

The SuperLU detection block in `CMakeLists.txt` was pure dead weight once
`cpu_superlu_solve.cpp` is gone — its only consumer was that file, and the
`CUPF_WITH_SUPERLU` compile define is used in **no** source. Removed:
`find_path(SUPERLU_INCLUDE_DIR ...)`, `find_library(LIB_SUPERLU ...)`, the
`CUPF_SUPERLU_ENABLED` gate, and the `$<...:SUPERLU_INCLUDE_DIR>`,
`$<...:LIB_SUPERLU>`, `$<...:CUPF_WITH_SUPERLU>` entries. The image never had
SuperLU, so this variant was already auto-disabled (see s1 baseline) — removal
changes no build output.

## Deliberately KEPT (reachable / live — removing would be a design change)

- **Custom CUDA FP64 linear solver** (`ops/linear_solve/cuda_custom_solver.*`,
  `#ifdef CUPF_ENABLE_CUSTOM_SOLVER` in `newton_solver.cpp`, `pipeline.hpp`,
  `newton_solver_torch_bridge.cpp`). This is a build-time-gated alternative
  backend to cuDSS (`option(CUPF_ENABLE_CUSTOM_SOLVER ... OFF)`), reachable when
  the flag is ON. It is an intentional optional feature, not unreachable code, so
  it is preserved. (Note: with the flag ON it currently fails to build because the
  sibling `../custom_linear_solver` has a pre-existing `atomicAdd` overload bug —
  a separate component, recorded in the s1 baseline, not a cuPF dead-code issue.)
- **Batch scaffolding** (`solve_batch()`, `NRBatchResult`, batch-major storage
  offsets, `--batch-size`). Per `deviations.md` Phase 06–07 this is *live* on the
  CUDA Mixed path (`B>1` runs, `B=2/B=4` smoke tests pass). Only the CPU-FP64 and
  CUDA-FP64 storages keep a `B=1` compatibility path. This is staged/in-progress
  functionality, not dead code, so it is preserved.

## Comments / readability (goal #3)

No stray `std::cout`/`printf`/`cerr` debug prints or `TODO/FIXME/HACK` markers
remain in compiled sources (the only `LOG_DEBUG`/`tmp` hits were the logger macro
definitions and the now-removed reference file). Removing the phantom
SuperLU / naive-Jacobian variants and the SuperLU CMake gate is itself a
readability win: the build no longer advertises capabilities the code does not
have.
