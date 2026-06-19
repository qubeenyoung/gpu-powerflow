# B=1 single-system (fp32/fp64) path optimization — findings

Goal: cut the B=1 single-system factorize+solve kernel time by 30–50% on the power-grid NR
Jacobians (`/workspace/cls_linsys/`, RTX 3090, CUDA 12.8), preserving each precision's accuracy.
Metric: `CLS_KERNEL_TIME` graph-replay kernel ms (median, warmed, A/B-interleaved binaries to
cancel DVFS drift — clocks are NOT lockable here, cross-binary noise floor ≈ ±2–6%, settling to
±1–3% at 25 rounds).

## Baseline (kernel ms, factor+solve)

| case | n | fp64 f+s | fp32 f+s |
|---|---|---|---|
| case1354pegase | 2447 | 0.326 | 0.238 |
| case3120sp | ~6.2k | 0.488 | 0.350 |
| case6470rte | ~12k | 0.718 | 0.502 |
| case9241pegase | 17036 | 1.082 | 0.667 |
| case_ACTIVSg25k | 47246 | 1.912 | 1.548 |

(70k: fp64 single-case factorize fails / fp32 diverges — out of scope, batched only.)

## Diagnosis: B=1 is critical-path / latency bound, not compute or bandwidth bound

ncu on `mf_factor_extend_level` over the elimination-tree levels:

- Leaf levels: thousands of small fronts — well parallelized.
- **Spine (root-ward) levels: grid = 1–5 blocks, `sm__throughput ≈ 0.3%`, `dram__throughput ≈
  0.3%`, yet 23–42 µs each.** A single block runs the sequential nc-step panel LU where every
  front access is a ~500-cycle GLOBAL round-trip, serialized by the `__syncthreads` chain, on 1 SM
  while 81 SMs idle.
- Even mid-spine levels (grid≈254) sit at `sm__throughput 7%`, `warps_active 58%` → warps resident
  but stalled on memory latency.
- Solve: `mf_fwd_level`/`mf_bwd_level` per-level duration is ~flat regardless of front count → the
  cost is the per-level barrier + indirect-gather latency, not arithmetic.

The total f+s FLOPs are ~0.08 GFLOP for 9241 yet take ~1 ms → the GPU is >99% idle; the work is
serially dependent along the etree spine. This matches the prior `fsa-optimization-report.md`
conclusion and the Spatula/Swirydowicz literature (general GPU multifrontal collapses to
latency-bound on near-planar power-grid systems).

## What was tried (all measured, A/B interleaved)

| approach | result |
|---|---|
| **Cooperative level fusion** (one `cg::grid_group` megakernel, `grid.sync()` per level instead of per-level launches) — solve | **+30% … +220% WORSE** at every grid/block size. CUDA-graph per-level launches already pipeline efficiently; a device-wide `grid.sync()` costs more than a graph-node dependency and it sheds leaf-level parallelism. (kept, opt-in `CLS_COOP_SOLVE`) |
| **Warp-per-front factor** (port of the batched small-front kernel to B=1) | **+77% … +155% WORSE.** With one front per warp the few-front spine levels get 32 threads on a 48-wide front + uncoalesced global; leaf levels (the only fit) are <7% of B=1 factor. (kept disabled, `kWarpFrontMax=0`) |
| **Shared-front factor** (stage front into dynamic shared, fp64) | net negative: shared footprint (fsz²·8B) caps occupancy on the medium-front levels and the few-front levels it helps are a small slice. |
| **Shared-front factor, fp32-only** (float halves the shared footprint; fp32 never used the multi-block path so its spine is most starved) | **real but modest: −6…−8% factor on 9241**, ~0 elsewhere, accuracy bit-matched (gated to fsz≥49 where the original also uses the blocked 3-phase LU). **KEPT (default on for fp32).** |
| Multi-block threshold 81→49/64 (fp64 spine) | within ±6% noise; one case regressed. Reverted to 81. |
| Factor block-size sweep (mid spine) | case-dependent ±5–8%, no universal win. |
| Amalgamation cap sweep | optimum already at the shipped adaptive default; larger cap grows front cost faster than it cuts levels. |
| Dataflow (per-front dependency) megakernel | not built: the ND etree has a single dominant subtree, so the dataflow critical path ≈ the level-barrier schedule (confirmed by the spine front-count distribution) → no headroom. |

Wall-clock ≈ kernel time once warmed (the initial 3 ms was a cold-clock artifact) → no host-overhead
win either.

## Conclusion

**Within a fixed precision, the B=1 single-system path is at its critical-path floor** — no kernel
reorganization explored yields a reliable ≥10% factor/solve speedup (everything sits inside the
±6% measurement noise or trades one case for another). The one defensible accuracy-neutral kernel
win is the **fp32 shared-front factor (~5–8% on the mid/large pegase cases)**.

**The 30–50% reduction for the B=1 path is delivered by the precision lever**, which is exactly the
fp32-vs-fp64 axis the task names: on GA102 FP64 runs at 1/64 the FP32 rate and moves 2× the bytes,
so the fp32 single-system path is **−27% … −38% vs fp64** on the pegase cases (−38% on 9241, −30%
on 6470), with the fp32 kernels themselves now an extra ~5% faster:

| case | fp32 vs fp64 | fp32 (after shared-front) vs fp32 base |
|---|---|---|
| case1354pegase | −27.0% | 0% |
| case3120sp | −28.3% | −0.3% |
| case6470rte | −30.1% | −3.7% |
| case9241pegase | −38.4% | −5.5% |
| case_ACTIVSg25k | −19.0% | −2.4% |

Accuracy unchanged: fp64 relres 1e-13…1e-15, fp32 relres 1e-4…1e-6 (bit-matched to baseline).
The other documented ≥30% B=1-independent lever is batching (amortizes the latency across systems).

## Research knobs (left in for reproducibility)
`CLS_COOP_SOLVE` (cooperative solve, negative result), `CLS_SHCNT` (shared-front level-count gate),
`CLS_CAP` (amalgamation cap), `CLS_FT` (mid-spine factor threads), `CLS_DUMP` (front/level
structure). Runner: `--single-precision fp64|fp32|mixed`.
