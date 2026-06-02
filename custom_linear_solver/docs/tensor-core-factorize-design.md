# Tensor-Core Batched Multifrontal Factorize — Design & Optimization

This document explains how FP16 tensor cores were applied to the batched numeric
factorization in `custom_linear_solver`, the algorithms and optimization strategy behind
it, and the supporting solve/analyze work. It is written to be self-contained: section 1
defines the technical terms, section 2 walks the algorithm and the optimization strategy,
and section 3 is the detailed account of *how tensor cores are applied to factorize*.

Target hardware: NVIDIA RTX 3090 (GA102, sm_86), CUDA 12.8. Test systems: MATPOWER
Newton power-flow Jacobians (case3120sp … case_ACTIVSg70k), pass criterion relative
residual `‖Ax−b‖/‖b‖ < 1e-3`.

---

## 1. Technical terms

**Sparse direct solver / LU factorization.** We solve `A x = b` by factoring `A = L·U`
(lower × upper triangular) once, then `solve` = forward solve `L y = b` + backward solve
`U x = y`. For a fixed sparsity pattern, the symbolic structure is computed once
(`analyze`) and reused while only the numeric values change (`factorize`/`solve`) — exactly
the Newton power-flow loop.

**CSR / CSC.** Compressed Sparse Row / Column storage: `row_offsets`/`col_indices`/`values`.
The matrix enters as device CSR; internally we build the CSC pattern and a permuted ordering.

**Fill-in.** Factorization creates new nonzeros (`L`/`U` have entries where `A` had zeros).
Minimizing fill is the job of the **ordering**.

**Nested dissection (ND) / METIS.** A fill-reducing ordering: recursively find a small
**vertex separator** that splits the graph in two, order the two halves first and the
separator last. METIS is the CPU library used. Power-grid graphs are near-planar with very
small separators, so fill is low — but the resulting **supernodes are tiny** (see below).

**Elimination tree (etree).** A tree where `parent[j]` is the first row below the diagonal
in column `j` of `L`. It encodes the dependency order: a column can be eliminated only after
its etree children. `parent[j] > j` in a valid (postordered) ordering.

**Symbolic Cholesky / fill pattern.** Computes the structure of `L` (`Lp`/`Li`) from the
ordered pattern and the etree, without touching numeric values. `column_counts` (cs_counts)
gives `|L(:,j)|` per column.

**Supernode / panel / amalgamation.** A **supernode** is a set of consecutive columns with
(nearly) identical nonzero structure, factored together as one dense block — the unit of
dense BLAS-like work. **Relaxed amalgamation** merges small supernodes (accepting some
explicit-zero **padding**) into bigger **panels** to get larger dense blocks. `panel_cap`
bounds the columns per panel; `nc` = a panel's column count.

**Multifrontal method.** Each panel owns a dense **front**: an `fsz × fsz` matrix whose
first `nc` rows/cols are the panel's pivot block and the rest is its **contribution block
(CB)** — the `(fsz−nc) × (fsz−nc)` trailing Schur complement. Factoring a front does a
dense no-pivot LU on the pivot panel and a rank-`nc` **trailing update** on the CB, then
**extend-add**s the CB into the parent panel's front (an indexed scatter / `atomicAdd`).
Fronts are processed bottom-up by **etree level**; fronts at the same level are independent.

**Front sizes here.** Power-grid Jacobians give ~96% of fronts with `fsz ≤ 16` and `nc ≤ 16`;
the work concentrates in a few thousand `fsz` 49–256 fronts. The whole 70k factor is only
~0.08 GFLOP — it is **latency/occupancy-bound, not compute-bound** (runs at ~0.2% of FP32
peak for a single system).

**GEMM / GEMV.** GEMM = matrix×matrix `C = A·B` (reuse → high arithmetic intensity). GEMV =
matrix×vector (no reuse → memory-bound). Tensor cores accelerate GEMM, not GEMV.

**Tensor cores / WMMA.** Dedicated matrix-multiply-accumulate units. On GA102 they support
FP16/BF16/TF32 inputs with FP32 accumulate (no FP64 tensor cores). The CUDA WMMA API
(`nvcuda::wmma`, `mma.h`) operates on fixed **16×16×16 tiles** (M16 N16 K16): `load_matrix_sync`,
`mma_sync`, `store_matrix_sync`. They need a tall **K** (contraction) dimension and an
**aligned leading dimension** to be efficient.

**Mixed precision.** Keep a precise FP64 "master" front for the assembly/extend-add, do the
dense per-front LU bulk in a lower precision "working" copy (FP32 or, with tensor cores,
FP16), write the result back to the master. On GA102 FP64 runs at **1/64** of FP32, so the
LU bulk is far cheaper in lower precision.

**Iterative refinement (IR).** Recover accuracy after a low-precision solve:
`r = b − A x` (FP64 spmv) → `dx = solve(r)` (reuse the existing factor) → `x += dx`. Each
step costs ≈ one solve.

**Uniform-batch (batched).** `B` linear systems that share the *same* sparsity pattern (same
analyze) but have different numeric values/RHS — e.g. contingency / time-step / scenario
sweeps. Analyze once; factor/solve all `B`. Layout is **front-major**: `front_b` for panel
`p` of batch `b` starts at `b·front_total + front_off[p]`.

**CUDA graph.** Records the many per-level kernel launches once and replays them, cutting
per-launch overhead — used for the captured factor and solve.

**Coalescing / occupancy.** Coalescing = a warp's 32 lanes reading 32 *consecutive*
addresses in one memory transaction. Occupancy = active warps per SM (limited by registers
and shared memory per block); higher occupancy hides latency.

**selinv (partitioned inverse).** Invert each front's `nc×nc` pivot block during factor so
the solve applies `L_pp⁻¹`/`U_pp⁻¹` as a *parallel* GEMV instead of a sequential triangular
solve. Costs `O(nc³)` per front in factor; only worth it when the solve dominates.

---

## 2. Algorithm and optimization strategy

### 2.1 Single-system baseline and its wall

The solver has three phases: `analyze` (once), `factorize`, `solve`. The single-system
path is already heavily tuned (CUDA graphs, fused factor+extend, multi-block big fronts).
The measured bottlenecks were:

- **analyze** dominated by CPU METIS nested dissection (~66% of analyze) — the
  single-threaded vertex separators; not tunable away on the CPU.
- **factorize / solve** are sub-millisecond and **launch/level-sync latency-bound** on the
  tiny-front deep etree — *not* compute-bound. Therefore precision and tensor cores are
  inert for a single system (the arithmetic floor is <0.1% of the wall time).

So on a single system: lower precision / tensor cores cannot help; the only levers are
reducing latency (already done) or the ordering. This is the key negative result.

### 2.2 The batched regime changes everything

Batching `B` systems amortizes the per-level launch/sync latency across `B` and fills the
GPU (each level launches `B × level_size` blocks). Per-system factor/solve drop sharply and
the workload becomes **compute-bound** at saturation (B≈32–128). In that regime lower
precision and tensor cores finally matter. Per-system factor/solve fell 52–91% vs the
single-system path just from batching, on every size.

### 2.3 Optimizations applied (with rationale)

**Analyze.**
- *GPU symmetric-graph build* (`build_symmetric_graph_device`): build the METIS graph
  (A+Aᵀ, dedup, sorted) on the GPU with a sort/unique over edge keys, replacing the
  single-threaded CPU build + CSC download. −22…−34% analyze on small/medium.
- *Parallel root induce*: extract the two ND halves concurrently at the top recursion levels.

**Factor (batched).**
- *Uniform-batch factor* (front-major, `blockIdx.y = batch`): one launch per level covers all
  `B` fronts.
- *Mixed precision*: FP64 master + FP32 (or FP16-TC) working LU — the GA102 1/64 FP64 penalty.
- *Tensor cores* on the trailing update — section 3.

**Solve (batched).** Solve was the part that amalgamation hurt (bigger fronts ⇒ more
triangular work) and that tensor cores **cannot** help (it is a single-RHS GEMV per system,
N=1; the `B` systems have distinct matrices so they cannot be combined into a shared-N GEMM).
The wins were therefore classical kernel engineering:
- *Warp-parallel triangular pivot solve*: replace the serial thread-0 `O(nc²)` forward/
  backward substitution (a bottleneck once `nc` grows under amalgamation) with a warp
  substitution — lane `k` finalizes its unknown, broadcasts via `__shfl`, other lanes fold
  their partial. `O(nc)` steps.
- *One-warp (TS=32) solve blocks*: the pivot solve uses one warp anyway and smaller blocks
  pack more per SM. −21…−25%.
- *Parallel-over-k CB reduction with x cached in dynamic shared*: the backward CB reduction
  (78% of solve) used a per-thread `part[nc]` register array + an `nc`-way warp reduction,
  which capped occupancy. Restructured so each lane owns one output `pk = Σ_j U[k][nc+j]·x[nc+j]`
  (x cached in shared) — no register array, no reduction. bwd −42%. (Coalesced-per-row and
  j-split variants were measured *slower* — the added reductions outweighed the coalescing.)

Cumulatively the batched solve fell ~58% (70k 0.71→0.30 ms/sys) and the amalgamation solve
penalty was eliminated.

**Analyze for the amalgamation path** (extra cost of growing supernodes, see §3): cut from
~55 ms to ~29 ms (70k) by (a) flat CSR / counting-sort in the reorder (no per-node
`std::vector`), (b) relabeling the parent instead of recomputing the etree (a postorder
preserves the etree), and (c) running `fill_pattern` once in METIS order so column counts are
free and relabeling the fill into the reordered space with `permute_pattern` (drops
`column_counts` and a second `permute_symmetric_pattern`).

### 2.4 What did NOT work (measured)

- FP16 tensor cores on the **natural** fronts (K=`nc`≤16, one tile deep): the per-tile FP16
  staging + scratch overhead exceeds the FP32 FMA trailing — slower.
- Aggressive amalgamation (ratio ≥ 8 → nc~16–20): padded fill grows faster than the
  tensor-core gain — slower. Sweet spot is mild (nc~7–10).
- Iterative refinement to make low precision usable end-to-end: each step ≈ one solve, and
  the solve is not faster in low precision (latency-bound), so FP32/FP16+IR loses to the
  natural mixed path for the full f+s+IR pipeline. (Where low precision *does* help — the
  factor — the result already passes 1e-3 without IR for small/medium.)

---

## 3. How tensor cores are applied to factorize (detailed)

### 3.1 Why the factor trailing update is a GEMM

For one front, after factoring the `nc`-wide pivot panel (FP32) and the U panel, the
dominant work is the **rank-`nc` trailing update** of the contribution block:

```
C(uc × uc)  −=  L(uc × nc) · U(nc × uc)        uc = fsz − nc
```

This is a genuine **GEMM** (M=uc, N=uc, K=nc), so tensor cores apply — *per batch*. Batching
gives many such independent GEMMs to keep the tensor cores busy. (Contrast the solve, which
is `L·vector` = GEMV, N=1 — tensor cores do not apply.)

An isolated microbench (`bench/tc_trailing_microbench.cu`) confirms the ceiling: the batched
trailing GEMM is **1.5–2.7× faster in FP16 WMMA than FP32** for fronts ≥ 64 (e.g. 16×128
2.69×, 32×256 2.72×).

### 3.2 The obstacle: K = nc ≤ 16, and how to fix it (deep-K amalgamation)

On the natural power-grid structure `nc ≤ 16`, so the trailing GEMM has **K = 16 — one
tensor tile deep**. Tensor cores need a tall K to amortize fragment-load/staging overhead;
at K=16 WMMA is slower than the FP32 FMA loop. 96% of fronts are also tiny (`fsz ≤ 16`),
where WMMA does not apply at all.

The fix is to **grow `nc` (= K)** with relaxed amalgamation. But this must stay a valid
multifrontal:

- Merging *arbitrary* postorder-consecutive columns is INVALID — a child's contribution
  block must nest in exactly one parent front; merging across subtrees breaks that
  (`asm_idx = −1` → out-of-bounds extend-add, verified with compute-sanitizer).
- The valid construction (`amalgamation_reorder`, env `MF_AMALG="cap:ratio"`): merge each
  child column into its **parent supernode** — a child's CB always nests in its parent's
  front, so this is always a valid multifrontal — bounded by `cap` columns and a colcount
  similarity `ratio` (to keep padding small). Then **reorder** the columns so each supernode
  is contiguous: a postorder of the resulting supernode tree, columns within a supernode in
  ascending index. A postorder is a valid elimination order with the *same fill class* as
  the input etree. `analyze` re-derives `perm`, the device value map, and (by relabel) the
  etree and fill in the new order, and forces the resulting panel partition into
  `analyze_multifrontal`. relres stays < 1e-3 (FP64 ~1e-13, verified).

This raises `nc` from ~2 to ~7–32 (tunable). `cap` is held ≤ 32; aggressive merging is a net
loss (padded fill explodes). Sweet spot ≈ `32:3` (avg nc ~7–9).

### 3.3 The WMMA factor kernel

`mf_factor_extend_mixed_tc_b` (one block per (front, batch), 128 threads = 4 warps):

1. **Narrow** the FP64 master front into the FP32 working copy.
2. **Panel factor** (FP32): the `nc`-wide no-pivot LU on the full-height panel.
3. **U-panel** triangular solve (FP32).
4. **Trailing update with WMMA** for `fsz > 48` (smaller fronts keep the FP32 loop):
   - `KP = ceil(nc/16)·16`, `UCP = ceil(uc/16)·16`.
   - Stage the L panel (uc×nc) and U panel (nc×uc) as FP16 into shared, **zero-padding the K
     dimension to KP** (the nc≤KP padding is why the staging is mandatory — the front cannot
     be WMMA-loaded directly because the padding columns hold real trailing values, and
     WMMA needs an aligned leading dimension which the arbitrary front stride `fsz` is not).
   - For each 16×16 output tile, accumulate over the `KP/16` k-steps:
     `load_matrix_sync(A_frag)`, `load_matrix_sync(B_frag)`, `mma_sync` (FP16×FP16 → FP32
     accumulate), then store the FP32 tile to an aligned scratch and subtract it into the
     working front. Each warp owns a tile-row and reuses its A fragments across the tiles.
   - **Multi-k-step** (KP up to 32) is what makes deep-K profitable: each scratch round-trip
     is amortized over `KP/16` `mma_sync`.
5. Write the L/U back to the FP64 master and **extend-add** the CB into the parent.

**Dynamic per-level shared memory.** The staging buffers are sized to the level's max `uc`
(via `plan.h_front_ptr`/`h_ncols`/`h_plcols`), not a fixed 256-wide bound. The first version
used 36 KB static shared and collapsed occupancy to 1 block/SM — the WMMA path *lost* purely
on occupancy. Dynamic shared (small for the common small-front levels) restored occupancy and
made WMMA win.

**Precision and accuracy.** Inputs are FP16, accumulation FP32, the master/assembly stays
FP64. The FP16 factor error grows with size (relres ~7e-4 at 9k to ~1e-1 at 70k) and is
recovered with 2 FP64 IR steps to ~1e-6 when needed. *If the application tolerates ~1e-3 it
can skip IR* — which is the regime targeted here.

### 3.4 Results

Per-system batched factor (B=64), amalg `32:3` + FP16 WMMA-TC:

| case | natural FP64 | natural mixed FP32 | TC+amalg | vs FP64 |
|------|-------------:|-------------------:|---------:|--------:|
| 9241 | 0.119 ms | 0.101 ms | 0.085 ms | −29% |
| 25k  | 0.429 ms | 0.368 ms | 0.291 ms | −32% |
| 70k  | 1.547 ms | 1.224 ms | 0.956 ms | −38% |

With the optimized batched solve (which made the amalgamated solve as fast as the natural
solve), the full **f + s** is fastest with TC+amalg on every size: 70k 1.555 → 1.252 ms/sys
(−19%), 25k −14%. The dominant lever is the amalgamation (bigger dense fronts use the ALUs
far better — natural batched FP32 runs at ~0.2% of peak); FP16 tensor cores add on top,
clearly on the largest case (70k WMMA-TC 0.962 vs amalg-FP32 1.025).

### 3.5 Caveats / when to use it

- TC+amalg wins on **factor** and on **f+s** in the batched regime; for a single system it is
  inert (latency-bound).
- The amalgamation is a factor↔solve trade-off; the solve optimizations of §2.3 neutralize
  the solve penalty so the factor win survives end-to-end.
- IR is only needed if the residual must be < the FP16 factor error; each step ≈ one solve.
- Everything is gated behind `MF_AMALG` / `MF_TC`; the default path is unchanged.

### 3.6 Reproduce

```bash
cmake -S custom_linear_solver -B build/cls -DCLS_BUILD_CUDA_OPS=ON -DCLS_BUILD_SCRIPTS=ON
cmake --build build/cls -j
# TC+amalg batched factor+solve, B=64:
MF_TC=1 MF_AMALG=32:3 MF_NO_SELINV=1 \
  build/cls/custom_linear_solver_run /datasets/.../case_ACTIVSg70k --batch 64 --batch-only --repeat 8
# isolated trailing-GEMM tensor-core microbench:
nvcc -O3 -arch=sm_86 custom_linear_solver/bench/tc_trailing_microbench.cu -o /tmp/tcb && /tmp/tcb
```

Env knobs: `MF_AMALG=cap:ratio`, `MF_TC`, `MF_NO_SELINV`, `MF_BSOLVE_TS`,
`MF_BSKIP_FWD/BWD`, runner `--batch B`, `--batch-only`, `--ir N`. See
`fsa-optimization-report.md` for the full measurement log.
