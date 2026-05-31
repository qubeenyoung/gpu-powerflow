# docs ↔ src reconciliation + refactoring summary (cycle 5 / s5)

- Date: 2026-05-31
- Scope: goal #8 — align the design docs with the refactored source tree, and
  summarize the whole refactoring (cycles 1–5) before any merge to master.
- Method: item-by-item diff of every doc under `docs/` against the final `cpp/`
  tree, fixing broken file paths, dead symbol references, and non-existent
  directories; recording structural/renaming changes in the docs.

This document is the s5 deliverable. It does **not** merge to master — that
boundary stays with the user (see "Next step" below).

---

## 1. Root cause of the doc/src gap

The user-facing design docs (`overview.md`, `core/README.md`, `ops/README.md`,
`storage/README.md`, `variants/README.md`) described an **earlier virtual-interface
design** (`IStorage`, `IMismatchOp`, `IJacobianOp`, `ILinearSolveOp`,
`IVoltageUpdateOp`, `IIbusOp`, `IMismatchNormOp` in an `op_interfaces.hpp`,
assembled by a `solver stage configuration::build()`), while the code had since
moved to a **static-dispatch design**: stateless `Cpu*Op`/`Cuda*Op` structs owned
by per-profile pipeline structs, unified by a `SolverPipeline` `std::variant`,
and assembled directly in the `NewtonSolver` constructor. The gap was already
flagged in `naming_audit.md` §4 and `structure_audit.md` §1 and deferred to s5.

---

## 2. docs ↔ src matching table (after this cycle)

| Doc | Claimed before | Actual src | Action this cycle |
|---|---|---|---|
| `overview.md` dir tree | `cpp/inc/.../ops/`, `src/.../reference/`, top-level `benchmarks/` | inc has only `core/` + `utils/`; no `reference/`; `benchmarks/` is external/optional | tree rewritten; benchmark sourcing (`BUILD_BENCHMARKS`/`CUPF_BENCHMARKS_DIR`) noted |
| `overview.md` design | "가상 함수만 호출", `cupf::` namespace, garbled "solver stage configuration::build" / "NewtonSolver stage ownership" | `SolverPipeline` `std::variant` static dispatch, global namespace | rewritten to variant model; namespace note added; garbled tokens removed |
| `overview.md` profiles | 3 (CPU FP64, CUDA FP64, CUDA Mixed) | 5 pipelines: + `CudaFp32Pipeline`, + gated `CudaFp64CustomPipeline` | profile table extended |
| `overview.md` loop | "4-stage Op" | ibus → mismatch → mismatch_norm → jacobian → prepare_rhs → factorize → solve → voltage_update | loop + substage relationship corrected |
| `core/README.md` components | `solver_stages.cpp`, `configure_*_stages()` | assembly in `newton_solver.cpp` ctor; `pipeline.hpp` defines variant | component table + "Pipeline 선택" section rewritten |
| `core/README.md` Stage Ownership | `unique_ptr<IStorage>`, `unique_ptr<I*Op>` | `std::unique_ptr<SolverPipeline>` (variant), `batch_supported` flag | replaced with real variant + pipeline struct |
| `core/README.md` IterationContext | has a `storage` (IStorage&) field | no storage field; storage passed per-call; fields are config/pv/pq/iter/normF/converged/jacobian_age | table corrected |
| `core/README.md` contexts | `IStorage::prepare/upload`, `ILinearSolveOp::initialize` | `buf.prepare/upload/download`, `linear_solve.initialize` | call sites corrected |
| `ops/README.md` interfaces | `op_interfaces.hpp`, virtual `I*Op` | concrete stateless `Cpu*Op`/`Cuda*Op`, pipeline stage methods | "stage 호출 규약" section rewritten; ibus/mismatch_norm ops added |
| `ops/README.md` linear solve | 3 cuDSS specializations | + `CudaLinearSolveCuDSS<float, CudaFp32Storage>`, + gated custom solver | table extended |
| `storage/README.md` interface | `class IStorage` virtual in `op_interfaces.hpp` (with `backend()`/`compute()`) | concrete structs, `prepare/upload/download/download_batch`, no virtuals | interface section rewritten; `CudaFp32Storage` noted |
| `variants/README.md` assembly | "`solver stage configuration::build()`에서 조립" | assembled in `NewtonSolver` ctor into the variant | intro + comparison table corrected; FP32 + custom profiles added |
| `RULE.md` | timer/NVTX labels | matches `utils/` + timer labels | no change needed (verified consistent) |
| `math.md` | NR math background | no file/symbol references | no change needed |
| `implementation/README.md` checklist | `cuda_batch/TODO.md`, "solver stage configuration", `Batch* stage ownership` | no such path/symbols | dangling path + garbled checklist items fixed (plan body kept) |

### Intentionally kept (not a gap)

- `CpuJacobianOpF64`/`CpuMismatchNormOp` `F64` suffix on the FP64-only CPU path
  (`naming_audit.md` §3) — docs now document the exact names.
- CSR member naming split `indptr`/`indices` (external Ybus) vs `row_ptr`/`col_idx`
  (internal Jacobian) — deliberate provenance signal (`naming_audit.md` §1).
- The historical audit docs (`structure_audit.md`, `naming_audit.md`,
  `dead_code_audit.md`) are dated cycle records and were left unedited; their
  "before" names are intentional.

---

## 3. Overall refactoring summary (cycles 1–5)

| Cycle / subgoal | Goal(s) | Outcome |
|---|---|---|
| s1 — build & baseline | #5 (broken deps) | restored a green build; recorded baseline (custom solver / SuperLU gaps) |
| s2 — structure & oversized files | #0, #2 | confirmed inc/src + Storage/Op + 4-stage structure is sound; split `csr_transpose.{hpp,cpp}` and `newton_solver_adjoint_math.{hpp,cpp}` out, removing duplicated CSR-transpose code (`newton_solver_adjoint.cpp` 579→350, `cuda_cudss.cpp` 620→586) |
| s3 — naming consistency | #1 | canonical scheme codified; renamed `*Buffers`→`*Storage`, `d_Y_row`→`d_Ybus_row`, `launch_voltage_update_state`→`launch_apply_voltage_update` (~40 files, logic unchanged) |
| s4 — dead code / over-impl | #3, #4 | removed unreachable `reference/` (naive Jacobian, SuperLU solve) + `solver_stages.cpp` tombstone + the dead SuperLU CMake block; kept gated custom solver and live batch scaffolding |
| s5 — docs ↔ src | #8 | rewrote the 5 design docs to the variant/static-dispatch reality; this report + deviations update |

Net effect: the public design docs now describe the code that actually ships —
a `std::variant`-based `SolverPipeline` over per-profile pipeline structs, with
Storage + stateless stage Ops, in the global namespace, with FP64 public I/O.

---

## 4. Verification status

- Doc-only change this cycle: no compiled source was touched, so the build is
  unaffected. The last green build (s4) was the isolated
  `docker run --rm cupf:latest` configure+build of `libcupf.a`,
  `cupf_cpp_evaluate`, `cupf_minimal_tests` with `WITH_CUDA=ON
  -DBUILD_EVALUATORS=ON -DBUILD_TESTING=ON`; `ctest` 1/1 pass.
- Every file path and symbol cited in the rewritten docs was grep-verified
  against the tree at the time of writing.

## 5. Next step (user-owned)

The `agent/03-cupf-workspace-gpu` branch now holds the cycles 1–5 refactor with
docs aligned to src. Merging to `master` is the standing human-owned boundary and
is **not** performed here; it is proposed to the user for review.
