# custom_linear_solver — Main Report

A GPU-resident sparse **direct** solver (LU, no pivoting) specialized for the
**Newton–Raphson power-flow Jacobians** that cuPF produces. It exposes a
cuDSS-style phase API (`analyze` → `factorize` → `solve`) and runs the whole
Newton-iteration linear solve on the device.

> Scope of this report: what the solver is, how it is structured, the measured
> performance, and the techniques that produced it. Per-cycle tuning logs and
> design deep-dives live in [`history/`](history/); external positioning lives in
> [`related-work-and-contribution.md`](related-work-and-contribution.md); the API
> and build are in [`api-and-build-design.md`](api-and-build-design.md).

---

## 1. Problem and why it is hard

Power-grid NR Jacobians are **near-planar** sparse matrices: nested-dissection
ordering produces an elimination tree of **many tiny dense fronts** (≈95 % of
fronts are `fsz ≤ 16`) and a thin "spine" of medium fronts near the root. The
total work is small (≈0.08 GFLOP for `case9241`) but the **critical path is
long** (deep etree, ~26 levels).

Consequence, confirmed by profiling: general GPU multifrontal/supernodal solvers
collapse to **latency-bound** on this matrix class (literature measures peak
multifrontal at ~0.004 % of GPU peak on near-planar systems — see
[related-work](related-work-and-contribution.md)). The whole design is therefore
about **fighting latency and launch/occupancy overhead**, not about FLOPs.

---

## 2. Pipeline

```
set_data (device CSR)
   │
analyze            ── once per sparsity pattern ───────────────────────────────
   │   GPU symmetric-graph build (A+Aᵀ)  →  METIS nested dissection
   │   etree → fill pattern → relaxed panels (amalgamation)
   │   multifrontal symbolic (front_ptr / extend-add maps)
   │   build MultifrontalPlan + device arena
   │   capture factor CUDA graph + solve CUDA graph
   │
factorize          ── every Newton iteration ──────────────────────────────────
   │   scatter CSR values into the front arena
   │   replay factor graph: per-level dense LU + extend-add into parent
   │   invert each front's pivot block (partitioned inverse, for a GEMV solve)
   │
solve              ── every Newton iteration ──────────────────────────────────
       gather permuted RHS → forward sweep (leaves→root)
       → backward sweep (root→leaves) → scatter permuted solution
```

The multifrontal method assembles each etree node into a small dense **front**,
does a dense LU on it, and **extend-adds** its Schur complement (the contribution
block) into the parent front. Levels are processed leaves→root; all fronts in one
level are independent and run concurrently, with one CUDA-graph node per level.

Source map:

| area | files |
|---|---|
| public API | `src/solver.{hpp,cpp}`, `src/matrix/view.hpp` |
| plan / arena / graphs | `src/plan/multifrontal_plan.{hpp,cu}` |
| analyze (ordering, symbolic) | `src/reordering/metis_nd.*`, `src/symbolic/*`, `src/matrix/pattern_kernels.*` |
| single-system factor / solve | `src/factorize/multifrontal.cu`, `src/solve/multifrontal.cu` |
| batched (B systems) factor / solve | `src/batched/*` |
| tensor-core trailing (batched) | `src/tc/factor_tc.cuh` |

Two execution modes share one symbolic/analyze:

- **Single-system (B=1)** — `src/{factorize,solve}/multifrontal.cu`. Used by cuPF
  per Newton iteration. Precisions: FP64, FP32, and Mixed (FP64-master assembly +
  FP32 working LU).
- **Batched (B>1)** — `src/batched/*`. B systems sharing the pattern, processed
  with one launch per level (`blockIdx.y = batch`). Precisions: FP32 / Mixed /
  FP64 / tensor-core.

---

## 3. Performance (RTX 3090, CUDA 12.8, kernel time, power-grid NR Jacobians)

### 3.1 Single-system vs cuDSS
FP64 single-system factor beats cuDSS on the measured cases (e.g. `case9241`
factor 0.90 vs 0.99 ms/iter, `25k` 1.10 vs 1.45). The warm-kernel F+S standing is
the NR regime (factor+solve replayed every iteration with the pattern fixed).

### 3.2 Single-system precision lever (B=1, factor+solve, kernel ms)
On GA102, FP64 runs at 1/64 the FP32 rate and moves 2× the bytes, so the FP32
single-system path is substantially faster than FP64 at FP32 accuracy:

| case | fp64 f+s | fp32 f+s | fp32 vs fp64 |
|---|---|---|---|
| case3120sp | 0.488 | 0.350 | −28 % |
| case6470rte | 0.718 | 0.502 | −30 % |
| case9241pegase | 1.082 | 0.667 | **−38 %** |
| case_ACTIVSg25k | 1.912 | 1.548 | −19 % |

Accuracy: fp64 relres 1e-13…1e-15, fp32 relres 1e-4…1e-6.

### 3.3 Batching (per-system factor / solve, B=128 vs single)
Batching amortizes the latency across systems and is the dominant throughput win:

| case | factor / system | solve / system |
|---|---|---|
| case3120sp | **−91 %** | **−90 %** |
| case9241pegase | **−85 %** | **−84 %** |
| case_ACTIVSg25k | −72 % | −74 % |
| case_ACTIVSg70k | −52 % | −61 % |

---

## 4. Key techniques

**Analyze**
- **GPU symmetric-graph build** (`thrust` sort/unique for A+Aᵀ) replaces the
  single-threaded CPU adjacency build → analyze −22…−34 % on small/medium.
- **Size-adaptive amalgamation cap** (panel merge): merges etree chains into wider
  fronts to cut solve levels, traded against fill; cap is 8/12/16/20 by `n`.

**Factorize**
- **Blocked rank-nc dense LU** per front (panel → U-panel → single rank-nc trailing
  update) instead of nc rank-1 passes: fewer trailing re-reads and `__syncthreads`.
- **Fused factor + extend-add** (one kernel/level instead of two): the parent is at
  a strictly higher level, so the extend-add atomics are race-free.
- **Multi-block big fronts**: large separator fronts split their trailing GEMM and
  extend-add across many blocks (tiled, shared-staged) so a few big fronts use all SMs.
- **Partitioned inverse**: invert each front's `nc×nc` pivot block at the end of
  factor so the solve becomes a parallel **GEMV** instead of a sequential triangular
  solve (the big solve win, −22…−28 % backward, −13…−18 % forward).
- **FP32 shared-front** (single-system FP32 spine): stage the front into dynamic
  shared so the latency-bound nc-step panel runs in ~30-cycle shared instead of
  ~500-cycle global; −5…−8 % factor on the mid/large pegase cases, accuracy bit-matched.

**Batched**
- **Warp-per-front** for the tens of thousands of tiny bottom-level fronts (one warp,
  many warps/block, `__syncwarp`, shared-staged) → bottom levels ~2.2× faster.
- **Shared-resident mid fronts** (CB write-back skipped) and **1024-thread big fronts**.
- **Tensor cores** help **only** for amalgamation-grown fronts (`fsz ≥ 64`); natural
  power-grid fronts have `nc ≤ 16` (thin K), where WMMA cannot beat FP32 FMA.

---

## 5. Honest limitations

- **Single-system (B=1) is at its critical-path floor.** Within a fixed precision,
  no kernel reorganization explored (cooperative level fusion, warp-per-front,
  full shared-front, thread/threshold/amalgam sweeps) yields a reliable ≥10 %
  factor/solve speedup — the GPU is >99 % idle but the work is serially dependent
  along the etree spine. The ≥30 % B=1 win comes from **precision** (§3.2); the
  ≥50 % throughput win comes from **batching** (§3.3). See
  [`history/b1-single-system-optimization.md`](history/b1-single-system-optimization.md).
- **Pure FP32 diverges on the largest cases** (`70k`); use Mixed/FP64 there.
- **Novelty is bounded**: batched single-shared-symbolic power-grid solving is prior
  art; the defensible contributions are the tiny-K tensor-core/FP64-master **negative
  result** and the FP32-native + tiny-front kernel engineering. No head-to-head vs
  KLU/STRUMPACK/batched-powerflow has been run. See
  [`related-work-and-contribution.md`](related-work-and-contribution.md).

---

## 6. Document index

See [`README.md`](README.md) for the full index. Quick links:

- [`api-and-build-design.md`](api-and-build-design.md) — API, build, cuPF integration.
- [`related-work-and-contribution.md`](related-work-and-contribution.md) — landscape + novelty.
- [`history/`](history/) — per-cycle optimization reports and design deep-dives.
