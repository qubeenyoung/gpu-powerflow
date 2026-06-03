# FP32-native batched factorize + solve: tensor cores vs. the real bottleneck

Goal of this work: in the **multi-batch** regime (B power-grid Newton Jacobians sharing one
analyzed pattern), accelerate `factorize` with tensor cores while not letting `solve` regress, at
**FP32-level accuracy** (the caller does not need < FP32), targeting **−30% factor+solve vs. the
pure-FP32 batched path**. Target HW: RTX 3090 (sm_86), B=64–128.

## TL;DR result

Robust best-of-3 A/B, FP32-optimized vs. the original pure-FP32 batched path, B=64, `--batch-only`,
relres matches the FP32 baseline (pure FP32 math throughout):

| case | n | orig f+s (ms/sys) | new f+s | gain |
|------|--:|------------------:|--------:|-----:|
| case3120sp     | 5 991 | 0.040 | 0.034 | −16% |
| case6470rte    | 12 485 | 0.083 | 0.067 | −19% |
| case9241pegase | 17 036 | 0.122 | 0.102 | −16% |
| case_ACTIVSg25k | 47 246 | 0.391 | 0.292 | **−25%** |
| case_ACTIVSg70k | 134 104 | 1.240 | 0.895 | **−28%** (peaks ~−30% at cap=20) |

The **large multi-batch cases (the target) reach ~25–28%**, approaching/at 30% on the largest.
Solve alone fell −30…−41%. B=128 is the same (25k −26%, 70k −26%).

## The key finding: tensor cores are inert here; the workload is latency-bound

1. **Pure FP32 is the fastest baseline**, not Mixed/TC. The previous TC/Mixed paths carry an FP64
   *master* front (per-front `cast_copy` FP64→FP32, `writeback`, **double** `atomicAdd` extend-add,
   2× front memory). Against an FP32 baseline that overhead dominates the FP16 trailing gain — the
   old TC path was **17–21% slower than FP32**, and less accurate.

2. **The factor is not trailing-GEMM bound.** ncu/level breakdown (25k, B64): the factor splits
   across (a) tens of thousands of **tiny leaf fronts** (fsz≤16 = 95% of fronts, 5% of flops),
   (b) **mid fronts** (fsz 49–160 = most of the flops), (c) a few **big separator fronts**
   (fsz>159). Every band runs at <60% compute **and** <60% DRAM = **latency / occupancy bound**.
   The trailing GEMM that WMMA accelerates is a minority of the time, and `K = nc ≤ 16–20` is too
   thin for WMMA to beat the FP32 FMA loop. **FP16 tensor cores cannot deliver 30% here** (and the
   FP16 rounding diverges on the large grids — TC32 relres → NaN at 25k/70k).

So the 30% had to come from **kernel engineering on the latency-bound levels**, precision-kept-FP32.

## What actually moved the needle (all FP32, accuracy unchanged)

All gated to the float-front paths (`BatchPrecision::FP32` and the new `TC32`); FP64 shares the
small-front kernel. New files: `src/batched/factor_small.cuh`, `src/batched/solve_small.cuh`,
`src/tc/factor_tc.cuh`.

1. **Warp-per-front tiny-front factor** (`mf_factor_small_warp_b`). Levels whose max fsz ≤ 32 run
   one **warp** per (front,batch), `SMALL_WARPS=8` warps/block, front staged COALESCED into per-warp
   shared, `__syncwarp()` (no 4-warp block barrier), reduced L/U-only write-back. Replaces one
   128-thread block per tiny front (¾ idle threads + full block barriers). The dominant bottom level
   went 2.47→1.12 ms and became compute-bound (76%).

2. **Shared-resident mid-front factor** (`mf_factor_mid_tc32_b<USE_TC>`, in `src/tc`). Levels with
   32 < max fsz ≤ 159 stage the whole front into dynamic shared (sized per level to the level's max
   fsz², opt-in to the 99 KB sm_86 cap), run the panel LU / U-solve / trailing / extend-add against
   shared instead of re-reading the front from global on every one of the ~nc sequential passes, and
   write back **only the L/U** (the uc×uc CB stays in shared for the extend-add — skipping it is the
   bulk of the write traffic on these DRAM-bound levels). Block threads scale with front size
   (64/128/256). Trailing is FP32 scalar (FP32 path) or FP16 WMMA (`TC32`).

3. **High-thread big-separator factor.** Levels with max fsz > 159 (too big for shared) run the
   global kernel with **1024 threads/block** — these top levels have only 9–25 fronts, so packing
   many warps per block hides the long sequential dependency at otherwise near-zero occupancy. 70k
   factor 0.87→0.77.

4. **Warp-packed small-level solve** (`mf_{fwd,bwd}_small_warp_b`). The bottom solve levels dominate
   solve the same way; packing 8 warps/block (vs one 32-thread block per front) cut solve −30…−41%
   with identical accuracy. Bigger solve levels get adaptive threads (64/128/256).

5. **Retuned amalgamation cap.** With the shared/mid kernels handling wider fronts well, the largest
   matrices benefit from harder amalgamation (fewer, denser fronts; fewer solve levels): `eff_cap` is
   now 20 for n≥80 000 (was 16). 70k f+s −5% extra. Mid band stays 12, small stays `panel_cap`.

## The src/tc module (the requested promotion)

`src/tc/factor_tc.cuh` is the promoted tensor-core factor module:
- `tc_trailing_wmma_f32` — the shared-staged FP16 WMMA rank-nc trailing update on an FP32 front
  (16×16×16 tiles, K=nc zero-padded, multi-k-step), 16-byte-aligned staging.
- `mf_factor_extend_tc32_b` — **FP32-native** TC factor (no FP64 master): float front/assembly/
  extend-add, FP16 trailing inputs with FP32 accumulate. Removes the FP64-master overhead that made
  the old `TC`/`Mixed` lose to FP32.
- `mf_factor_mid_tc32_b<USE_TC>` — the shared-resident mid kernel above, FP32 or TC trailing.
- `BatchPrecision::TC32` is wired through `batched_setup`/dispatch and the runner (`MF_TC32`).

**Honest status of TC32:** it is correct and passes ~1e-3 on small/medium (3120/6470/9241) but is
**not faster than the FP32-optimized path there** (K=nc thin), and its FP16 trailing **diverges on
the large grids** (25k/70k relres → NaN). For this latency-bound, thin-K power-grid workload tensor
cores are not the lever; the FP32-native kernel restructuring is. TC32 is kept as a working module
for completeness and for any future deep-K (heavier amalgamation) regime.

## Reproduce

```bash
cmake -S custom_linear_solver -B /tmp/clsb -DCLS_BUILD_CUDA_OPS=ON -DCLS_BUILD_SCRIPTS=ON
cmake --build /tmp/clsb -j
# FP32-optimized batched factor+solve, B=64:
MF_FP32=1 /tmp/clsb/custom_linear_solver_run /workspace/cls_linsys/case_ACTIVSg70k \
    --batch 64 --batch-only --repeat 16
# research knobs (runner/library): MF_FP32 / MF_TC32 / MF_MIXED / MF_NO_MIXED, CLS_CAP=<cap>,
# CLS_DUMP=1 (front-size + per-level structure to stderr).
```
Note: build under `/tmp` (the `/workspace` overlay rejects CMake's cache rename).
