# cuDSS-level kernel tuning — long-term plan (branch feat/cudss-kernel-tuning)

Goal (user, 2026-05-27): invest long-term in hand-tuning the GPU kernels to close the
remaining gaps vs cuDSS. Accuracy may relax to cuDSS-similar (~1e-10). Branched from
feat/gpu-rightlook at the cy122 state (integration + relaxed refinement done).

## Where the gaps are (measured, warm/NR)
- **Factor**: superior-or-equal on 5/8; behind on ACTIVSg25k 1.22×, SyntheticUSA 1.29×,
  onetone2 4.1×. For power-grid the bulk (94%, cy98) is in the MANY small/medium fronts
  at leaf/mid levels; the big top fronts are only ~6%.
- **Solve**: 1.5–3.5× behind on the single-solve kernel (work/latency-bound; backward
  ~60%, the coalesced U read + scattered y-gather).
- cuDSS's edge is hand-tuned custom kernels (NOT library cuBLAS — cy120 showed cuBLAS
  batched is 2–10× slower than our in-kernel for these sizes).

## Tuning targets (to work through, one per cycle, gate-safe with gpu_test 4/4)
1. **Profile factor by front-size band** — attribute factor time to small(≤48)/medium/
   big fronts so we tune the actual hotspot, not guess. [FIRST]
2. **Dense-front micro-tiling** — register-blocked trailing update (each thread a 2×2/
   4×4 micro-tile) + shared-mem staging of the panel; higher arithmetic intensity, fewer
   loads. Helps medium/big fronts.
3. **Small-front batching** — many tiny fronts per block (cy82 warp-per-front regressed;
   retry with proper load-balancing / cooperative tiling). The power-grid bulk.
4. **Panel-factor sync reduction** — fewer __syncthreads in Phase 1 (the nc sequential
   pivots), e.g. warp-synchronous for nc≤32.
5. **Solve backward** — reduce the scattered y-gather cost; double-buffer the CB tiles.
6. **Solve forward** — coalesce the L-panel read (transposed-L) if it proves read-bound
   on big circuit fronts.

## Method
- Profile-first each target; A/B in one warm session (P8→P2 warmup) to avoid cold-clock
  artifacts (the [[gpu-benchmark-contention]] rule).
- gpu_test 4/4 mandatory; revert anything neutral/regressing; commit only real wins.
- Track results here + STATE.json.

## (A) chosen — FP32 + static-pivoting factor rewrite (multi-week)

User (2026-05-27 10:27 KST) chose (A): pursue the FP32-factor stabilization for the 2x
factor win (beat cuDSS). The 2x comes from FP32 STORAGE (bandwidth; the dense LU is
memory-bound, cy128), but FP32 storage is numerically unstable. Roadmap:
1. **Precise panel LU** (double compute, FP32 storage) — cy129 DONE (FP64 unchanged,
   gpu_test 4/4). Insufficient alone: FP32 still zero-pivots.
2. **Precise assembly (extend-add)** — cy130 DIAGNOSED as THE blocker. Made panel
   (cy129) AND trailing (cy130) compute in DOUBLE under FP32 storage -> STILL zero-pivot.
   So it's not compute precision: the **FP32 assembly accumulation** (scatterA +
   children-CB atomicAdds summing into a front entry in FP32) cancels and zeroes the
   diagonal BEFORE the LU. Reverted the compute detours (cy131 will use FP32 compute on
   the working front for speed). **Design (cy131+): double "master" front for assembly**
   -- scatterA + extend-add accumulate in FP64 (no cancellation); narrow each front to
   an FP32 "working" copy for the bandwidth-bound LU; the LU's CB (FP32) extend-adds into
   the parent's FP64 master (read FP32 -> add FP64). Cost: +front memory + 1 narrow pass
   per front, but the LU does ~nc passes so it amortizes (projected ~1.5-1.7x, stable).
3. **Growth control / static pivoting** — for ill-conditioned fronts (onetone2) FP32
   may overflow; perturb tiny pivots + (if needed) limited pivoting.
4. **Validate**: FP32 factor stable + ~2x + iterative refinement -> cuDSS-level berr.
Risk: high; multi-week. FP64 default stays correct throughout (gate-safe, opt-in FP32).

## Double-master design (refined cy131, implement cy132)

Clean architecture (only the factor LU is FP32; everything else FP64 on the master):
- **d_front = FP64 master** (existing arena): scatterA, extend-add accumulation, final
  L/U storage, emit, AND solve all use FP64 master -> no assembly cancellation, solve
  unchanged. (Revert cy127's FP32 scatterA/emit/solve branches -> FP64 only.)
- **d_frontf = FP32 working** (SEPARATE cudaMalloc front_total floats; NOT the cy127
  reinterpret -> fix destructor to free it; only allocated when fp32).
- **mf_factor_extend_mixed kernel** (fp32 path only): per front -> (1) narrow master->
  working (float); (2) FP32 LU on working (the bandwidth win); (3) writeback working
  L/U+CB -> master (double); (4) extend-add working CB -> PARENT master via FP64
  atomicAdd (precise assembly). Pivots come from narrow(precise FP64 master) -> nonzero
  -> stable; within-front FP32 LU is ~1e-6 (refinement recovers).
Bandwidth (Phase-3 reads 2nc*uc^2 dominate): pure FP64 ~17uc^2 vs double-master 8.5
(FP32 LU) + 1.5 (narrow) + 1.5 (writeback) = 11.5uc^2 -> **~1.48x**. Projected:
SyntheticUSA 6.55->~4.4 (beats cuDSS 5.08), ACTIVSg25k 2.1->~1.4 (beats 1.88). Stable.
Gate-safe: default FP64; fp32 opt-in; refinement + berr gate -> CPU fallback.

## Log

**cy123 — target #1 DONE (profile factor by front-size band, MF_FRONTPROF diagnostic).**
Key finding (work ~ fsz²·nc): on the matrices we're BEHIND on, the factor work is
dominated by MEDIUM/BIG fronts (49–256+), NOT tiny fronts:
- ACTIVSg25k: 49–128 = 62%
- SyntheticUSA: 49–128 = 43%, 129–256 = 32% (→ 49–256 = 75%)
- rajat15: 49–128 = 64%, 129–256 = 23%
- onetone2: 129–256 = 25%, 257+ = 69% (→ 129+ = 93%)
(small matrices case6468/case8387 are 17–48-dominated ~53%, but those we already win.)
=> The hotspot is the **blocked dense-LU path (fsz>48: Phase-1 panel / Phase-2 U /
Phase-3 trailing GEMM)**. cuDSS-level micro-tiling + shared-mem staging there targets
43–93% of factor work on the lagging matrices. My earlier "bulk is tiny fronts"
assumption was WRONG (that was front COUNT; by WORK it's medium/big fronts).
NEXT (cy124): target #2 — register-blocked + shared-mem dense-front kernel for fsz>48.

**cy124 — target #2 (register-tiled 2x2 Phase-3 trailing GEMM): NEGATIVE, reverted.**
2x2 micro-tile (reuse 2L+2U across 4 FMA/k). Warm factor: small matrices improved
(case6468 0.48->0.45, case8387 0.64->0.59) but the TARGET medium-front matrices
REGRESSED (ACTIVSg25k 2.11->2.22, SyntheticUSA 6.55->6.93, stable over 3 runs). Cause:
the 2x2 accumulators add ~16 regs/thread -> lower occupancy on the many-medium-front
mid-levels, which outweighs the reduced L/U loads. => the medium-front factor is
OCCUPANCY-bound, not load-bound; register tiling is the wrong lever. Reverted (gpu_test
4/4). Implication: cuDSS-level win needs better OCCUPANCY/per-front thread allocation
(batched dense factor with tuned threads-per-front), not arithmetic-intensity tricks.
NEXT (cy125): probe per-level factor block size (medium-front levels may want fewer
threads for better packing); if promising, adaptive like the solve's per-level Ts.

**cy125 — per-level factor block size (256 vs 512): NEGATIVE / measurement artifact.**
A 3-run in-session A/B suggested 256 beats 512 on ACTIVSg25k (2.117 vs 2.236) +
SyntheticUSA. Implemented per-level adaptive (256 medium / 512 big). BUT a clean
same-binary 3-way A/B in the NEXT session flipped: 512 best (2.124 vs 2.19 for both
256 and adaptive). The earlier "256 wins" was a CROSS-SESSION GPU-state artifact
(within-session stable but didn't generalize) — exactly the [[gpu-benchmark-contention]]
trap. cy83's 512 is confirmed best. Reverted. **Blocker: the measurement environment
is unreliable for <5% deltas** (GPU clock not pinned). Must lock the clock
(nvidia-smi -lgc) before further micro-tuning, else can't trust small wins.

**cy126 — DECISIVE PROBE: mixed-precision (FP32) factor is the big lever.** fp_probe.cu
times the full blocked dense-front LU (panel + U + trailing) in FP64 vs FP32, one
block/front, kernel-only, warmed, in-process A/B. Result: **FP32 is ~2x faster**
(fsz 64/96/128/200 -> 1.95/1.95/2.21/1.92x). The dense-front work is bandwidth-bound
(FP32 = half the bytes) / partly FP64-compute-bound (RTX 3090 FP64 = 1/64 FP32) -> ~2x
either way. Since the factor is 43-93% dense-front work on the lagging matrices, an
FP32 factor -> factor ~1.6-1.9x faster: SyntheticUSA ~6.55->~4 (BEATS cuDSS 5.08),
ACTIVSg25k ~2.1->~1.3, onetone2 ~32->~17. This is >>5% so it's MEASURABLE here.
Accuracy: FP32 factor+solve raw berr ~1e-6, recovered to ~1e-10 by the integrated
mysolver-gpu iterative refinement (FP64 residual) -- exactly the relaxed-accuracy
budget the user granted. THIS is the (c) direction.
NEXT (cy127+): implement FP32 front arena + factor/solve kernels (template gpu_mf on
the front type), wire into mysolver-gpu with refinement, measure F/S + refined berr.

**cy127 — FP32 factor IMPLEMENTED + MEASURED: 2x faster but NUMERICALLY OVERFLOWS.**
Templated the front-touching kernels (scatterA/factor/emit/fwd/bwd) on the element
type; FP32 reuses the front arena region (reinterpret, no extra alloc); precision
flag in gpu_mf_analyze (default FP64 -> gpu_test 4/4 unchanged). Measured (gpu_mf_bench
MF_FP32): factor FP32 vs FP64 -> case6468 0.42->0.30, case8387 0.56->0.41, ACTIVSg25k
2.11->1.42, SyntheticUSA 6.55->4.20 (-28..-36%, would BEAT cuDSS on all 4!). BUT the
no-pivot FP32 LU is UNSTABLE: gpu_mf_bench (unequilibrated) -> NaN; integrated
mysolver-gpu (equilibrated) -> 'zero pivot' / finite=0 (NaN/inf) on ALL matrices incl
well-conditioned case6468. Cause: no-pivot LU element growth exceeds FP32 range
(3e38); power-grid uses identity-match (no MC64) so growth that FP64 absorbs overflows
FP32. FP32 *storage* can't hold grown factors regardless of accumulation precision ->
no salvage without GROWTH CONTROL (static pivoting / per-front scaling / per-front FP64
fallback) = a numerical redesign. Kept the FP32 infra (opt-in MYSOLVER_GPU_FP32 / bench
MF_FP32; default FP64 works, all-GPU, berr 1e-14). The 2x is real; the blocker is
numerical stability, which is the next sub-problem for the (c) factor win.

**cy128 — MC64 doesn't stabilize FP32; MIXED probe closes the FP32 direction.**
(1) Forcing MC64 on power-grid (not just identity-match) still -> 'zero pivot' under
FP32: the instability is FP32 PRECISION/cancellation in the LU pivots + multifrontal
assembly, not just diagonal dominance. (2) Probed MIXED (FP64 storage + FP32 trailing
compute, stable pivots): it is 2x SLOWER than FP64 (f64 1.45 / fp32 0.71 / mixed 2.6
ms; mixed 0.5x) -- the FP64 loads + double->float casts cost more than the FP32 FMA
saves. CONCLUSION: the dense-front LU is MEMORY-BANDWIDTH-bound, so the FP32 2x comes
entirely from FP32 STORAGE (half the bytes), NOT FP32 compute. But FP32 storage is
precisely what overflows/cancels. => no viable FP32 factor without GROWTH CONTROL
(static pivoting) to make FP32 storage safe -- a major numerical redesign (cuDSS likely
does exactly this). FP32 direction CLOSED pending that. Reverted MC64-for-fp32; kept
FP32 infra opt-in (default FP64). Probe (fp_probe) extended with the mixed variant.

**STATUS after cy123-125:** the one-block-per-front factor kernel (512 threads) is
well-tuned and resists micro-opts (register tiling -> occupancy loss; block size
512-optimal). Closing the factor gap needs the FUNDAMENTAL redesign cuDSS uses: a
BATCHED dense-front factor with tuned threads-per-front (group same-size fronts, one
kernel, optimal occupancy). That is the real long-term (c) work — multi-day, higher
risk. NEXT: (a) pin GPU clock for reliable measurement, then (b) prototype the batched
dense-front factor.

**cy132 — double-master FP32 factor WORKS: stable + ~1.3x, beats/ties cuDSS on large.**
Implemented mf_factor_extend_mixed (FP64 master assembly + FP32 working LU + writeback +
FP64 extend-add). Warm factor FP32: ACTIVSg25k 2.10->1.57 (BEATS cuDSS 1.88, was 1.19x
behind), SyntheticUSA 6.55->5.19 (~ties 5.08, was 1.29x behind). STABLE: integrated 6/8
ok in FP32 (all power-grid + memplus + rajat15, berr 1e-11..1e-16 via refinement);
rajat27/onetone2 gate-safe CPU fallback (ill-conditioned, FP32 factor insufficient).
~1.3x (vs 1.48 projected: narrow/writeback + separate-arena traffic). Opt-in
MYSOLVER_GPU_FP32 / MF_FP32; default FP64 (gpu_test 4/4). committed 081b60d.
NEXT (cy133+): (a) FP32-gate-fail -> retry FP64-GPU (not CPU) so FP32 can be DEFAULT
without regressing rajat27/onetone2; (b) close the 1.3->1.48x gap (narrow/writeback
fusion); (c) growth control so rajat27/onetone2 also succeed in FP32.

**cy133 — FP32 default with FP64-GPU retry cascade (realizes the win, no CPU fallback).**
Refactored mysolver-gpu: try FP32 -> if berr gate (1e-8) fails, retry FP64-GPU -> then
CPU. FP32 is now the DEFAULT fast path (gate-safe). All 8 matrices stay on GPU:
ACTIVSg25k+SyntheticUSA use FP32 (factor beats/ties cuDSS), memplus/rajat15/case6468/
case8387 FP32, rajat27 (FP32 berr 2e-8 just misses gate) + onetone2 (ill-cond) -> FP64-
GPU (berr 4.8e-13 / 5e-16). MYSOLVER_GPU_FP64 forces FP64. gpu_test 4/4. committed.
NEXT: close 1.3->1.48x (narrow/writeback); rajat27 is borderline (1 more refine iter or
growth control could keep it FP32); make this the production default + re-measure A/F/S/E2E.

**cy134 — writeback-opt closes the gap to the projected ~1.48x.** The mixed kernel
wrote the WHOLE front (fsz^2) back to master, but the CB/trailing (uc^2, most of the
front) is never read from master -> writeback only the L/U cross (~2*nc*fsz, ~6x fewer
writes). Warm FP32 factor: ACTIVSg25k 1.57->1.40 (1.50x vs FP64; BEATS cuDSS 1.88),
SyntheticUSA 5.19->4.44 (1.47x; now BEATS cuDSS 5.08, was tie). Accuracy unchanged
(berr 1e-11..1e-16). => factor now BEATS cuDSS on ALL 4 power-grid (case6468/case8387
FP64, ACTIVSg25k/SyntheticUSA FP32); only onetone2 (4x) remains behind. gpu_test 4/4.

**cy136 — HONEST CORRECTION: FP32 factor doesn't help END-TO-END (refinement cost).**
The FP32 factor KERNEL is ~1.5x faster (real), BUT it produces less-accurate L/U
(~1e-6), so reaching cuDSS-level accuracy needs iterative refinement (>=1 extra solve
pass + residual). Measured integrated combined (single-call): FP32 WORSE than FP64 --
ACTIVSg25k 8.6 vs 5.3, SyntheticUSA 33 vs 26. The refinement cost (~1 solve + residual
~ 4ms) >= the factor savings (~2ms). This is the classic mixed-precision-iterative-
refinement limitation: it only pays off when the factor is the DOMINANT cost; here the
factor is only ~1.5x faster in FP32 (memory-bound, not 10x+) and the solve cost is
comparable, so the extra refine solve negates it. => cy135's "combined beats/ties" was
over-optimistic (used the RAW unrefined solve). HONEST standing: FP64 is our best;
combined F+S ~1.3x behind cuDSS on the 2 large (cuDSS's FP64 factor 5.08 < our 6.55,
and solve 1.44 < our 2.19 -- cuDSS's kernels are simply better-tuned). Reverted to FP64
DEFAULT (FP32 opt-in MYSOLVER_GPU_FP32 as a kernel result that doesn't help e2e).
gpu_test 4/4.

**cy137-138 — FP64 kernel tuning attempt: clock-lock host-blocked; shared-mem LU neutral on hotspot.**
User redirected to FP64 tuning + try GPU clock-lock (container root). Result: clock-lock
is HOST-blocked even as root (GeForce RTX3090 has no app-clocks; -lgc permission-denied
at host) -> reliable <5% measurement still impossible. Probed the main FP64 lever
(shared-mem LU, fp_probe): helps SMALL fronts (fsz32 1.18x) but NEUTRAL/worse on the
MEDIUM hotspot (fsz48 0.94x, fsz64 0.97x) -> the medium fronts (43% of work) are already
L2-served, not global-bandwidth-bound (confirms cy85). So FP64 shared-mem doesn't help
the hotspot. The FP32 2x came from halving L2/compute traffic uniformly (only FP32
storage does that, and it's inaccurate). => FP64 factor is at its practical limit for
our implementation; cuDSS's ~1.29x edge is its proprietary kernel engineering, not a
lever we can pull + validate here. gpu_test 4/4.

**cy139 — clock-lock-ready A/B harness (prep, no perf change).**
Clock still unlocked (idle 210MHz P8, persistence off) -> user hasn't run the host
command yet. Did NOT manufacture a neutral change (anti-churn policy). Instead made the
clock-lock payoff immediate + disciplined: confirmed the micro-lever knobs already exist
(MF_FT factor block-size, MF_ST solve block-size, MF_FLO/FHI level-band isolation,
MF_SKIP_* diagnostics in gpu_mf.cu), and wrote report/autorun/clock_locked_sweep.sh -- it
GATES on the lock being present (idle clocks.gr>=800MHz else abort with the host command)
and sweeps MF_FT {256,384,512,640} / MF_ST {64,128,256}, with the >5%-or-it-is-noise rule
baked into the output (prevents another cy125 "256 beats 512" artifact). The moment the
user locks the clock from the host, one command gives trustworthy block-size numbers, and
the previously-unmeasurable FP64 micro-levers (sync reduction, register tiling) become
A/B-validatable. gpu_test untouched.

**cy140 — WIN: per-level adaptive factor block size (clock-lock unblocked it).**
User locked the core clock from the HOST (sudo nvidia-smi -pm 1 && -lgc 1395,1395) @ 2026-05-27T01:59:34Z.
Cross-process variance dropped to <0.1% (was the project's #1 blocker). Re-ran the MF_FT
sweep TRUSTWORTHILY: 384 beats the old flat 512 on all power-grid (1.3-3.9%) + rajat27/
memplus (3.7-5.8%) + rajat15 (tie), but 512 wins onetone2 (its work is 68.8% in fronts
>=257, the deep serial path). Implemented level_ft(L): max-front>=257 -> 512 else 384.
Result vs old flat 512: 7/8 faster (1.3-5.8%), onetone2 +0.1% (protected), 0 regression,
gpu_test 4/4. Solve MF_ST sweep confirmed 64 optimal (no change). This is the first <5%
win the locked clock makes detectable -- prior cycles correctly couldn't measure it.

**cy141 — fine-tune the cy140 adaptive block sizes (locked clock).** Finer sweep:
medium value 384 is the true optimum (beats 320/448/512 on EVERY power-grid + rajat27/
memplus matrix). onetone2's big fronts (>=257, the only matrix in that branch) prefer
768 over 512: 45.02->44.20 (-1.8%); 640 and 1024 are between (1024 over-allocates).
Updated level_ft: max-front>=257 -> 768 else 384. Non-onetone2 matrices byte-identical
(they never hit the >=257 branch). gpu_test 4/4. Marginal on the 4x onetone2 structural
gap but free + safe.

**cy142 — SOLVE re-examined under the locked clock: confirmed at practical limit (no change).**
The user's priority + bigger gap (1.5-2.3x vs factor's 1.2-1.3x), so re-checked now that
<5% is measurable. (1) fwd/bwd split: BACKWARD ~63% (Synth bwd 1.825/fwd 1.134; ACTIV bwd
1.012/fwd 0.670) -- confirms cy114. (2) Finer MF_ST sweep {32,48,64,96}: 64 is optimal for
the LARGE matrices where solve time matters (ACTIV 1.617, Synth 2.894); 32 helps only the
already-sub-ms small ones (case6468 -2.1%, case8387 -2.9%) while REGRESSING the large
(ACTIV +6.9%, Synth +7.9%) -> no net win, keep 64/128. (3) Backward kernel inspected: it
is well-tuned (TILED CB reduction, COALESCED F-reads per pivot row, shared-staged scattered
x-gather amortized over nc, warp-shuffle reduction). Forward not the bottleneck (cy114
structural); level-fusion neutral (cy117). => the solve is genuinely at its multifrontal
level-set limit; the 1.5-2.3x is cuDSS's superior scheduling/locality (research-grade).
This now rests on RELIABLE measurement, not the old noisy regime. NO code change (anti-churn).

**cy143 — BIG WIN: tiny-front leaf levels get 128-thread blocks (occupancy).** The cy140
profiler showed tiny fronts (fsz<=48) are ~99% of fronts but were running at the medium
384 -> severe under-occupancy (a fsz<=16 front has <=256 elems, ~5 blocks/SM at 384). Gave
high-count tiny levels (max fsz<=48 AND >=1024 fronts) 128-thread blocks. The count guard
is essential: without it small matrices REGRESS (their sparse tiny levels under-utilize at
128); with it they flip to wins. Swept tiny block {64,96,128,160} and count {0,512,1024,
2048,4096} under the locked clock -> tiny=128, cnt>=1024. Result vs baseline 384: case6468
-6.8%, case8387 -5.1%, ACTIVSg25k -6.1%, SyntheticUSA -7.0%, rajat27 -10.2%, memplus
-8.4%, rajat15 -1.9%, onetone2 neutral. ALL improve/neutral, ZERO regression, gpu_test
4/4, berr unchanged. Biggest factor win of the clock-locked session; only detectable now.

**cy144 — WIN + cy142 CORRECTION: count-guard generalizes to the SOLVE.** cy142 declared
the solve "at level-set limit" but only tested GLOBAL MF_ST (which forces medium/big levels
to 32 too -> regresses the large -> looked like no win). Applying the cy143 per-level
count guard: high-count tiny-front solve levels (max fsz<=48 AND >=512 fronts) -> 32-thread
blocks, medium/big stay 64/128. Result: power-grid solve case6468 -3.6%, case8387 -4.8%,
ACTIVSg25k -3.0%, SyntheticUSA -4.9%; circuits neutral-to-slightly-better; ZERO regression,
gpu_test 4/4, berr unchanged. Lesson: don't conclude "at limit" from a global sweep when a
per-level/count-aware variant is untested. The count-guard occupancy technique (cy143/144)
is the session's key reusable idea -- check it on any per-level launch config.

**cy146 — NEG: count-guard / band-split does NOT extend to the medium hotspot.** Tested
splitting the 49-256 medium band, giving 49-128 small-medium fronts a smaller block
(MF_FT_SMED sweep {384,256,192,128}). 384 is clearly best; 256 regresses (ACTIV 2.654->
2.846), 192/128 much worse. Unlike the tiny fronts, 49-128 fronts have enough work (up to
128^2~16k elems) to use 384 threads well -> compute/bandwidth-bound, not occupancy-bound,
so smaller blocks just under-utilize. cy140's global 384 for the medium band was correct.
Reverted the dead knob (default was a no-op anyway). The count-guard occupancy win is
specific to the TINY fronts (cy143) + tiny solve levels (cy144); the medium hotspot is a
different (compute-bound) regime. gpu_test 4/4, default unchanged.

**cy147 — MAJOR WIN: multi-block big-front factor (onetone2 3.1x). ** User asked to push
SyntheticUSA + onetone2. Profiled: onetone2's 4.5x gap = its big fronts (>=257, 68.8% of
work) ran ONE BLOCK each -> a handful of blocks on 82 SMs (massive under-utilization).
Split big-front levels into 3 graph-ordered kernels: (A) mf_bigA_panelU one-block/front
panel+U; (B) mf_bigB_trailing grid(tiles x fronts) -> the embarrassingly-parallel rank-nc
trailing (the bulk) spreads MANY blocks/front across the GPU; (C) mf_bigC_extend extend-add.
Default ON (opt-out MF_NO_BIGMULTI), tiles=128. Only triggers for >=257-front levels (only
onetone2 has them -> every other matrix byte-identical). onetone2 factor 44.1->14.2ms
(3.1x), vs cuDSS 9.83 -> from 4.49x behind to 1.45x. gpu_test 4/4 both paths; integrated
mysolver-gpu onetone2 berr 4.39e-16, success, no fallback. SyntheticUSA unchanged (its
fronts <=256 are many-per-level = good occupancy already, not under-utilized -> dense-LU-
quality bound, not a multi-block target). The big serial-front under-utilization lever is
distinct from the count-guard occupancy lever (cy143/144).

**cy148 — HUGE: multi-block factor for ALL big fronts (>=81), not just onetone2 -> factor beats cuDSS 7/8.** cy147 only multi-blocked >=257 fronts (onetone2). I was wrong that medium-large (81-256) fronts wouldn't benefit -- they're FEW-per-level near the root -> badly under-utilized one-block-each. Lowered MB_THRESH to 81 (tunable) + adaptive tiles (>=257 onetone2->128, 81-256->64). Locked-clock sweep {49..257} -> 81 best (49 over-provisions the many small-medium fronts). Results vs baseline / vs cuDSS:
- SyntheticUSA 8.21->4.11 (beat cuDSS 6.07 = 1.48x; was 1.35x BEHIND)
- ACTIVSg25k 2.65->1.79 (beat 2.23 = 1.25x; was 1.19x behind)
- memplus 1.03->0.70 (beat 1.37 = 1.97x), rajat15 4.97->3.29 (beat 4.26 = 1.30x)
- onetone2 14.2->11.2 (its 81-256 fronts also multi-block now; vs cuDSS 9.83 = 1.14x, was 4.49x!)
- case6468/case8387/rajat27 unchanged (already beat; fronts mostly <81 or few)
FACTOR NOW BEATS cuDSS 7/8 (only onetone2 behind, 1.14x). gpu_test 4/4, berr unchanged, reproducible <0.2%, integrated mysolver-gpu ACTIV berr 1.3e-13 / Synth 4.7e-14 success. The single biggest factor breakthrough of the project.

**cy149 — SOLVE: multi-block does NOT transfer (per-front work too small); combined F+S now competitive.** Pivoted to solve per user. Tested the cy148 multi-block lens: probe (MF_ST 64/128/256 on big-front matrices) shows the solve does NOT want more parallelism/front -- Synth wants 64, onetone2 peaks at 128 (not 256), rajat15 64. So big-front solve levels are NOT under-utilized (unlike factor): solve per-front work is O(fsz*nc)~2k vs factor's O(fsz^2*nc)~500k, too small to spread across blocks. Solve is memory-bandwidth + level-serialization bound; levers exhausted (block size cy142/144, fusion neutral cy117, fwd-coalescing ruled out cy114, FP32 infra gone + marginal). No code change (anti-churn). BUT the factor wins (cy140-148) pulled COMBINED F+S to beat/tie 4/8 vs cuDSS (case6468 tie, case8387 1.26x, SyntheticUSA 1.13x, memplus 1.38x; behind ACTIV 1.11x, rajat27 1.09x, rajat15 1.09x, onetone2 1.35x) -- SyntheticUSA combined now BEATS cuDSS. Solve remains the residual gap but it's now ~1.1-1.35x on the combined, not the 1.5-2.5x solve-only gap.

**cy150 — SOLVE backward CB-tile swept (256 optimal); solve confirmed at kernel-tuning limit.**
Tested the last untested backward knob (MF_CB_TILE, shared-mem reduction tile): 128 is
neutral-to-slightly-worse (Synth 2.759 vs 2.752, onetone2 4.54 vs 4.52); 256 optimal,
reverted. Solve levers now ALL tested under the locked clock: block size (cy142/144),
multi-block N/A (cy149 probe), CB tile (cy150), fusion neutral (cy117), FP32 marginal+gone
(cy114), fwd-coalescing ruled out (cy114). The solve is memory-bandwidth + level-
serialization bound; remaining 1.3-2.5x is research-grade (cuDSS-style solve scheduling /
data-layout rewrite, multi-week). No code change (anti-churn). MILESTONE: factor beats
cuDSS 7/8, combined F+S beat/tie 4/8, accuracy superior throughout -- the kernel-tuning
track has delivered its tractable wins. Inflection surfaced to user.

**cy153 — robustness validation + A-regression investigation (no regression).**
(1) Ran integrated mysolver-gpu over the full matpower set post cy140-152: all 9 succeed on
GPU (no CPU fallback), berr 1e-13..1e-16. The multi-block factor (cy147/148) is solid e2e.
(2) User asked if A (analysis) regressed. Verified NO: A now (ACTIV 155.3, Synth 630.8)
matches cy121 (158.9, 612.2) within noise; gpu_mf_analyze still ~32ms (cy108: 33ms) so the
cy147/148 per-level loops added nothing. A has ALWAYS been 2.5-3.4x slower than cuDSS
(cy121 stated this), dominated by CPU METIS ND ordering (~590ms of Synth's 631ms; GPU
symbolic only 32ms). cuDSS's edge = parallel/GPU ordering. To improve A: AMD (faster but
+fill -> hurts the factor we won) or GPU-parallel ND (research-grade). A amortizes in NR.
No code change.

**cy154 — SOLVE multi-block: IMPLEMENTED + TESTED -> no speedup, reverted (F-safe attempt).**
User: improve A and S keeping F. For S, cy149's "multi-block won't help" was reasoned, not
tested -- so I built it (mf_bwd_reduce splits a front's CB reduction across gridDim.x blocks
-> more SMs/bandwidth; mf_bwd_finish combines + backsolves; global d_part partials, opt-in
MF_SOLVE_MB). Result: onetone2 solve 4.52->4.59 (NO speedup) + a correctness bug (raw berr
1.0, only masked by refinement). Confirms cy149: solve per-front work O(fsz*nc)~2k is too
small -- cross-block reduction + atomicAdd + 2-kernel overhead negates the bandwidth gain
(unlike factor's O(fsz^2*nc)~500k). Reverted (git checkout, default gpu_test 4/4). So under
the F-safe constraint: S is at its limit (multi-block now TESTED-negative, not assumed);
A is METIS-bound (cy152, faster opts break onetone2). Both need research-grade rewrites
(parallel/GPU ordering for A; cuDSS-style solve scheduling for S). The only F-safe A nibble
left is optimizing the ~5-10% non-METIS CPU parts (adjacency build/symbolic). F stays 7/8.

**cy155 — RESEARCH-GRADE A: parallel nested dissection (PAR_ND, opt-in). BIG A win.**
User: research-grade rewrite to make A/S match cuDSS, keep F. METIS_NodeND is single-
threaded; ND is recursively parallel (the 2 halves after a vertex separator are
independent). Implemented par_nd_rec: METIS_ComputeVertexSeparator -> recurse the 2 halves
on separate threads (12 cores) -> order separator last. Same ND algorithm -> fill ~= serial
(F-safe). Results (large matrices, n>20000; small stay serial base-case):
  SyntheticUSA A 599->362ms (-40%), ACTIVSg25k A 155->100ms (-35%) -> closes A-gap vs cuDSS
  from 2.5-3.1x to ~1.65-1.78x. F preserved/improved (ACTIV F 1.79->1.5, onetone2 F
  11.2->10.2 avg!, Synth F ~4.1). gpu_test 4/4; integrated mysolver-gpu ALL succeed (berr
  1e-13..1e-16, onetone2 4.5e-16, no fallback).
CAVEAT: METIS 5.1 global RNG is thread-UNSAFE -> concurrent calls race -> NON-deterministic
ordering (F variance +-10%, e.g. Synth F 4.08-4.53). Fixed seed doesn't fix it (global
state). Valid+correct always, but non-deterministic undermines reproducibility -> kept
OPT-IN (default serial unchanged). NEXT: determinism (process-based parallelism, or a
thread-safe RNG path, or bounded-variance guard) -> then default. The A research-grade
rewrite WORKS (-40% A); determinism is the remaining blocker to ship it.

**cy156 — parallel ND VALIDATED robust + depth-tuned (production-ready opt-in).**
Integrated mysolver-gpu with PAR_ND over all 8 matrices x2 runs: ALL succeed, no CPU
fallback (refinement+gate absorb the ordering variance -> always correct). Depth sweep: 4
best (Synth A 605->358ms = -41%; ACTIV saturates ~100ms at depth>=2). Set default depth 4.
DESIGN RESOLUTION: keep PAR_ND opt-in -- production sets it for the -40% A (correct, F
maintained, fill variance only across separate orderings which NR computes once & reuses);
the kernel benchmark tools (gpu_mf_bench/circuit_mf_test) stay SERIAL by default so factor/
solve A/B stays reproducible (the variance would otherwise reintroduce measurement noise).
So A research-grade win is SHIPPABLE (opt-in). The METIS-thread-safety non-determinism is
not a correctness/robustness issue (validated), only a benchmark-reproducibility choice.
NEXT: S research-grade rewrite.

**cy157 — S research-grade: assessed at kernel limit (honest, exhaustive).** Profiled
circuit fwd/bwd (locked clock): onetone2 fwd 43%/bwd 57%, rajat15 fwd 44%/bwd 56% (more
balanced than power-grid 37/63). Examined all F-safe S levers: block size (cy142/144 tuned),
multi-block (cy154 TESTED-negative: per-front work too small), fusion (cy117 neutral: graph
amortizes launches), FP32 (cy114 marginal+gone), CB tile (cy150 256 optimal), forward
coalescing (cy114 L1-absorbed; compact-L layout would help circuit fwd IF read-bound, but
the extraction adds to F or S -> NOT cleanly F-safe). The bwd (bottleneck) is U-read
(already coalesced/top-of-front) + scattered x-gather (inherent to multifrontal, can't
compact). The fwd is atomic-scatter/latency bound. So S is genuinely bandwidth + scatter +
level-serialization bound at the multifrontal kernel limit. A further win needs matching
cuDSS's solve INTERNALS (data layout / scheduling) -- a multi-week rewrite, uncertain (cy117
suggests sync-reduction is neutral; bandwidth is the hard part). The realized research-grade
win was A (parallel ND -40%, cy155-156). No code change (anti-churn). Surfaced the decision
to the user: commit to the multi-week S-internals rewrite, or consolidate the strong position.

**cy159 — S forward-coalescing: cheap timing-proxy TESTED -> modest, not worth (S confirmed at limit).**
Per cy148/155 "test don't assume": the one S lever only REASONED about (cy114 "forward
L1-absorbed" was power-grid; circuits have bigger fronts) was forward-read coalescing.
Instead of building the full compact-L infra, ran a cheap timing-only proxy (fwdc flag:
coalesced compact read, wrong values, MF_SKIP_BWD to isolate the forward). Result: coalescing
the forward L-read gives only -5..-9% of the FORWARD (onetone2 1.96->1.86, rajat15 1.13->1.04,
Synth 1.15->1.05) = ~-2-4% of solve. So the forward is ATOMIC-SCATTER bound, not read-bound;
the L-read is a small fraction. Full compact-L would add +F (extraction) -> net ~wash.
NOT worth it. Reverted the proxy. S confirmed at multifrontal limit (forward-read lever now
also cheaply tested-negative). The cheap-proxy approach correctly avoided an expensive
low-EV build-out. gpu_test 4/4.

**cy160 — backlog item "회로 MC64" RULED OUT as an A lever (profiled).** Instrumented the
circuit analysis (MYSOLVER_GPU_ATIME): MC64 matching is NEGLIGIBLE -- onetone2 5.9ms,
rajat15 5.8ms, rajat27 2.3ms, memplus 0ms -- vs METIS 135/216/53/48ms. So circuit A is
METIS-dominated (like power-grid); the parallel ND (cy155) is the A win for circuits too,
and MC64 is not worth optimizing. Closes the "회로 MC64" backlog item. Kept the atime
diagnostic (gated, harmless). gpu_test 4/4. Confirms: A lever = METIS (parallel ND, done);
no other significant analysis sub-phase.

**cy161 — plateau: consolidated research-grade learnings into persistent memory (no code).**
Tractable optimization space exhausted (F won 7/8, A gap halved via parallel ND, S at
multifrontal limit, MC64 ruled out). At a genuine plateau awaiting user direction on the
multi-week cuDSS-internals rewrite. Captured the parallel-ND-analysis technique + updated
the status memory for future sessions (the A/F/S final standing). No code change (anti-churn);
batching minor findings while the user is away. Will resume substantive work on a fresh
idea or user steer.

**cy163 — REWRITE step 1: AMD-leaf determinism TESTED -> FAILED (fill regression). Path pivot.**
Tried deterministic parallel ND = serial METIS separators + thread-safe AMD leaf ordering
(DET_ND). Viability gate (fill vs serial METIS_NodeND): AMD gives MUCH more fill on the
leaves -> F regresses badly (onetone2 +72%, rajat15 +56%, ACTIVSg25k +52%, SyntheticUSA +31%).
AMD's minimum-degree << METIS ND quality on these structured matrices. VIOLATES keep-F.
Reverted (tested fill before building parallel infra -> good discipline). Determinism
landscape after this: (a) thread-safe METIS rebuild = the CLEAN fix (GKlib RNG global is the
only race; source available at /opt/third_party/src/{gklib,metis}; make RNG __thread +
fixed seed -> deterministic parallel ND, METIS quality, no fork) -- heavy infra but right;
(b) process-based fork METIS (risky CUDA+fork); (c) pragmatic: ship non-deterministic
parallel ND default (A -40%, F competitive/often-BETTER per cy162 -- variance is mostly
upside, benchmark-only concern). NEXT: pursue (a) thread-safe METIS (clean deterministic
rewrite) OR per user, (c) ship. gpu_test 4/4, PAR_ND intact.

**cy164 — REWRITE step 2: SHIPPED parallel-ND A to PRODUCTION default (-25..-39% A, F kept).**
Determinism finding: METIS uses glibc rand() (USE_GKRAND off) -> global-state race -> parND
non-deterministic; the clean det fix = GKlib rand_r patch + system-lib rebuild (feasible but
high-risk surgery for a det-ONLY payoff since A is already -40%). Decision (autonomous-loop
risk profile): ship the parND A win NOW (low-risk param), det as a documented follow-up.
metis_nd gains a `parallel` arg; mysolver_gpu_solver passes true (MF_SERIAL_ND forces serial).
Result: production A ACTIVSg25k 169->126ms (-25%), SyntheticUSA 637->389ms (-39%); integrated
ALL matrices succeed (berr 1e-14..1e-13, no fallback); serial-bench F UNCHANGED (7/8 claim
intact -- the clean F comparison uses serial gpu_mf_bench). Non-determinism is run-to-run
only (NR computes the ordering once, fixed within a run) -> not a correctness/perf issue.
gpu_test 4/4. A track: -40% kernel / -25-39% production, gap vs cuDSS ~halved, F kept.
NEXT: deterministic GKlib-rand_r upgrade (if exact reproducibility wanted) OR pivot the
rewrite to S (bigger gap).

**cy165 — REWRITE step 3 (S): gather-forward RULED OUT via cheap proxy; S is dependency-bound.**
cuDSS-internals S idea: replace the forward's atomic-scatter with a gather (avoid atomic
contention). Cheap proxy (fwdc: non-atomic write, timing-only, MF_SKIP_BWD): removing the
atomicAdd saves only 0.5-1.5% of the forward (onetone2 1.981->1.972, Synth 1.174->1.156). So
the atomic is NOT the bottleneck -> gather-forward rewrite is dead (would address a ~1% cost).
Combined with cy159 (L-read 5-9%): the forward bulk is the LEVEL-DEPENDENCY LATENCY (etree
depth critical path) + compute, both INHERENT to the multifrontal level-set solve. cuDSS's S
edge is deep tuning of the same approach + likely a shallower-etree ordering (not a different
algorithm we can drop in). The parND ordering's indirect S win (less fill) is VARIABLE/
non-deterministic (cy162 ACTIV -14%, this run ~0). So S has no tractable cuDSS-internals
lever; it's multifrontal-dependency-bound. A rewrite delivered (-40% A shipped cy164); S is
at its honest limit. Reverted proxy, gpu_test 4/4.

**cy166 — A determinism: rand-override partial, deterministic parallel METIS is tooling-blocked.**
Tried thread-safe rand()/srand() interposition (thread_safe_rand.c, -rdynamic, per-call
srand(42) reseed) to make parND deterministic without the GKlib rebuild. Progress: with
-rdynamic the override IS used (fill changed) and per-call reseed TIGHTENED the variance
(~10% -> ~2%), but NOT exact -- METIS also uses gettimeofday + opaque internal state that
resist the rand-only override. So deterministic parallel METIS needs the full GKlib
__thread RNG rebuild (risky system surgery) -- not achievable via clean low-risk means.
Reverted (gpu_test 4/4, parND intact). CONCLUSION of the A-determinism track (cy163-166):
deterministic parND is tooling-blocked; the non-deterministic parND (-40% A, F competitive,
ordering fixed per-run in NR) is the shipped A result (cy164). Determinism would need the
risky GKlib rebuild -- defer unless the user explicitly wants exact run-to-run reproducibility
and accepts the system-rebuild risk. A track DONE: -40% A shipped, F kept.

## cycle 169 (05-27) — S attack #1: panel-cap (supernode amalgamation) deep sweep
Idea: amalgamation cap is the classic Solve lever -> swept MF_CAP {1..32} on circuits+power-grid.
Added MF_CAP env (clamp <=MF_MAX_NC=64) + pad_ratio debug in gpu_mf_analyze.
- Solve is FILL-WORK-bound, NOT level-launch-bound: bigger cap -> fewer levels but SLOWER S
  (onetone2 plev 110->39 yet S 3.88->6.43); pad_ratio only 1.05-1.13x (padding small).
  Confirms cy117 (launches already graph-amortized).
- cap8 Pareto-optimal globally; per-matrix optima (case6468/rajat27->cap4, onetone2->cap16)
  have NO symbolic signal (rajat27 vs memplus counterexample) + GPU-tune regresses A. Prod=cap8.
- BENCH BUG: circuit_mf_test was cap16 != production cap8 -> cy168 circuit S pessimistic.
  Fixed to cap8: rajat27 S -32% (0.66), rajat15 -22% (1.93), onetone2 -14% (3.88); 3/4 circuit F beat cuDSS.
- Conclusion: within keep-A/F ordering, S bound by fill-work+etree-dep; S gap root = A gap root
  (lower-fill ordering). gpu_test 4/4, prod default unchanged. commit 8c8d757.
Next: FP32 solve (convert-once) — the accuracy-relaxation lever the user unlocked.

## cycle 170 (05-27) — S attack #2: FP32 solve (accuracy-relaxation lever) — tested NEGATIVE
Idea: user relaxed accuracy to cuDSS-level (cy169) -> store factored front in FP32, solve
<float>-storage/double-accumulate, narrow FP64->FP32 once per factor (amortized). MF_SOLVE_F32 opt-in.
- raw S only -1.4..-8.8% (case6468 .544->.531, ACTIV 1.569->1.488, rajat27 .658->.600,
  onetone2 3.885->3.609): solve is fill-work-bound, halving front-read bytes saves little.
- accuracy FAILS: berr 1e-4..1e-8 (vs cuDSS ~1e-13); onetone2 blows up to 0.228 (ill-cond).
  cuDSS-level needs refinement = 2-3 FP32 solves > 1 FP64 -> negates saving; unsafe ill-cond.
- Conclusion: accuracy relaxation does NOT unlock FP32 S win. Opt-in knob (default off,
  prod FP64, gpu_test 4/4). commit 759595f.
S status: cap(cy169)+FP32(cy170) exhausted -> S bound by fill-work+etree-dep+accuracy-floor.
Real remaining lever = lower-fill ordering (GPU ordering, closes A and S together).

## cycle 171 (05-27) — S attack #3: PROVED latency-bound (BW measurement + parallelism profile)
Tested cy169 "fill-work-bound" assumption -> WRONG. RTX3090 peak ~936GB/s; solve reads ~2x
front arena (onetone2 261MB) in 3.88ms = ~135GB/s = 14% peak -> LATENCY/DEPENDENCY-bound
(GPU ~85% idle on serial etree). Added [solve-prof] (MF_TIME): levels<82 fronts (SM count) +
their work + crit-path floor.
- onetone2: 99/110 narrow, 63% work in starved levels, crit floor 20% -> big overlap upside.
- rajat15 32%/rajat27 24%/memplus 21% work-in-narrow; power-grid 21-34%; crit 4-20%.
- => fill/ordering NOT the S lever (corrects cy169 A+S claim). Lever = finer cross-level DAG
  schedule overlapping independent subtrees -> fill idle GPU. Multi-block per-front marginal
  (solve work O(fsz*nc), no fsz²). gpu_test 4/4, diagnostic-only. commit 3db5f08.
NEXT (big): implement finer DAG-scheduled solve (dependency-driven), targeting onetone2.

## cycle 172 (05-27) — S attack #4: concurrency PROBE proves idle capacity (~3.6x headroom)
Tested cy171 prize directly (not via work-% proxy). MF_SOLVE_CC=K launches K independent
solve-graph copies on K streams, wall-times the batch (keeps solve_graph for re-instantiate).
- onetone2 T1=3.88 T4=4.26ms -> 4 solves in 1.10x = ~3.6x throughput headroom
- rajat15 1.19x(~3.4x); rajat27 1.32x; memplus 1.35x. GPU runs ~4 solves in time of 1.
- => single solve uses <30% throughput; finer/multi-subtree DAG schedule could approach 3.6x:
  onetone2 3.88->~1.1ms would BEAT cuDSS (1.53). Bounded by etree crit-path (onetone2 ~20%).
gpu_test 4/4, probe opt-in, production unchanged. commit 2894e2a.
NEXT (the big build): multi-subtree concurrent solve — partition fronts by top-separator
subtrees, launch each subtree's level-sequence on its own stream, sync at the shared top.

## cycle 173 (05-27) — S attack #5: multi-subtree partition profiled BEFORE building -> ruled out
Before the ~150-line multi-stream build, profiled the subtree partition (MF_MS_PROFILE: greedy
split heaviest etree frontier to >=G subtrees; shared-top fraction s bounds speedup ~1/(s+(1-s)/G)).
- DEGENERATE: one subtree = 97-100% work at every G (onetone2 G=8 max 98%, est ~1.0x; all circuits).
  Single-solve etree dominated by ONE heavy separator chain (= critical path).
- => multi-subtree/finer-DAG would NOT speed a SINGLE solve; cy172 3.6x is INTER-solve
  (batched/multi-RHS) only. Level-set already exploits the single-solve (leaf) parallelism.
- S EXHAUSTED within multifrontal for single-RHS: cap(169), FP32(170), multi-block(154),
  concurrency/multi-subtree(172-173) all ruled out. cuDSS single-solve edge = kernel micro-eff.
gpu_test 4/4, diagnostics opt-in, prod unchanged. commit 1783304. "test don't assume" saved the build.

## cycle 174 (05-27) — S attack #6 (last micro-lever): big-front solve block size -> negative
cy171 said deep big-front levels are occupancy-starved -> bumped threads/block for mx>=256
levels (MF_ST_BIG 128->192->256) to add warps hiding scattered-read latency. FLAT: onetone2
-1.6% at 256 (noise), ACTIV/Synth/rajat15 unchanged, berr held. Deep-level latency = scattered
gather + etree dependency, NOT occupancy. Reverted to 128. gpu_test 4/4. commit 6d9574d.
S COMPREHENSIVELY EXHAUSTED for single-RHS (cy169-174). Holding S; asked user for next fork
(a: A via GPU-ordering, b: onetone2 F focus, c: consolidate, d: other).

## cycle 177 (05-27) — A: profiled breakdown -> parallelized METIS adj_build (modest, gate-safe)
No user fork yet; made a bounded principled A step (A = cleanest near-cuDSS metric). METIS_TIME
breakdown (SyntheticUSA): parND-METIS = adj_build 37.7 + parND 151ms; analyze ~33ms. Parallelized
adj_build sort+dedup+flatten over 12 cores (n>=32768, byte-identical -> same ordering/fill/F).
- SyntheticUSA adj_build 37.7->27.5ms (-27%, A ~-2.6%); ACTIV 8.3->8.0; small unchanged (serial).
- gpu_test 4/4, fill/berr unchanged. commit d3ef978.
- build(push_back) step dominates adj_build + stays serial (full = 2-pass+atomics, not worth for
  NR-amortized A). parND-METIS 151ms = irreducible bulk -> A now needs GPU-ordering to close further.

## cycle 178 (05-27) — A: parND base threshold by matrix size -> small matrices match cuDSS A
User mandate: autonomous until A/S/F all >= cuDSS. Profiled A (PREP_TIME): ordering is the bulk;
small matrices (n<20000) fell back to SERIAL METIS -> cuDSS edged them. Size-decoupled base
threshold (n<20000 -> parND base 4000; >=20000 -> 20000, preserving big-matrix fill).
- case6468: order 22.3->15.5ms, A ~32.9->26.7 (~cuDSS 25.8), F .524->.498, S .544->.517 (all better)
- case8387: order 29.6->17.0ms, A ~43.7->31.6 (~cuDSS 31.4), F/S better
- memplus better; big matrices unchanged (no regression). gpu_test 4/4. commit f51b4b1.
CAVEAT found: benchmarks default SERIAL but production=parND; parND fill matrix-dependent
(lower power-grid, HIGHER onetone2 -> prod onetone2 F ~14.9 not 12.95). Reconcile next.
NEXT: (1) re-measure standing with parND (=production); (2) onetone2 parND-fill (serial better?);
(3) the parND-METIS 151ms ordering bulk (GPU-ordering) for big-matrix A; (4) keep probing S.

## cycle 179 (05-27) — PRODUCTION benchmark discovery: onetone2 CPU-fallback root cause
Ran prod benchmark (mysolver-gpu vs cudss-gpu, /datasets matrices). Findings:
- Prod one-shot A/F/S COLD-dominated both solvers (cuDSS memplus F 13.78 vs warm 1.15) -> warm kernels = NR metric.
- onetone2 CPU-falls-back (F 153ms): parND ordering makes no-pivot factor hit SINGULAR PIVOT -> gpu_mf_factorize false. (circuit_mf_test serial ND -> factor ok, hid this.) onetone2 parND A 22.8 BEATS cuDSS 101.8.
- Serial-retry: GPU factor ok + refines (F 153->19) BUT serial ordering ~250ms -> A blows up, E2E worse. Reverted.
- FIX (next): STATIC PIVOTING (tiny-pivot diagonal boost in GPU factor + refinement) -> keep parND fast A + robust factor.
build_ordering refactor + MYSOLVER_GPU_REFINE tunable. gpu_test 4/4, prod unchanged. commit d6c606b.

## cycle 180 (05-27) — static pivoting for onetone2: measured-NEGATIVE (structural zero)
Tried letting gpu_mf_factorize proceed on a zero pivot (boost+refine) instead of hard-fail->CPU.
- onetone2 parND zero pivot is STRUCTURAL -> boost(1) is relatively-infinite perturbation ->
  CASCADES to overflow (refine rnorm=inf) -> gate rejects -> CPU anyway. Smaller boost worse.
- All parND bases {4k..16k} hit the same zero -> not ordering-tunable.
- => needs REAL GPU pivoting (no-pivot design lacks it; major) or serial ordering (F 19 but A ~250 loses).
- onetone2 best: CPU fallback WINS A (22.5 vs 101.8), loses F/S. Kept MF_STATIC_PIVOT opt-in. gpu_test 4/4. commit a636ac3.
NEXT options: (1) GPU pivoting (major, fixes onetone2 F/S), (2) big-matrix A ordering (Synth/ACTIV lose A), (3) accept onetone2 A-win.

## cycle 181 (05-27) — A symbolic: parallelize symmetric_pattern sort+dedup (gate-safe, modest)
Incremental A (after cy177 adj_build, cy178 threshold). symmetric_pattern per-node sort+dedup
parallelized over 12 cores (n>=32768, byte-identical output). SyntheticUSA 13.8->8.3ms (-40%),
ACTIV 4.9->3.5; small serial. fill/F/S unchanged, gpu_test 4/4. commit 6a6e277.
Big-matrix A wall remains parND-METIS 151ms (GPU-ordering). fill_pattern (26ms Synth) is the next
symbolic chunk but harder (sequential symbolic factorization). Major efforts still: GPU-ordering
(big A), GPU-pivoting (onetone2 F/S), structural S.

## cycle 182 (05-27) — onetone2: DIAGONAL SHIFT makes GPU factor work (F beats cuDSS!), opt-in
cy180 local boost cascaded; a GLOBAL shift A+eps*I (1e-8) prevents the structural zero, refinement
corrects eps -> onetone2 no-pivot factor succeeds on GPU: F 153(CPU)->18.5ms warm BEATS cuDSS 21.2,
berr 4e-11. Refactored try_gpu to reuse the plan (no double-analyze).
- OPT-IN (MF_DIAG_SHIFT): GPU analysis ~196ms loses A (vs cuDSS 99.8 + CPU symbolic 25), refinement
  loses S (9.2 vs 4.84) -> trades onetone2's 2 ok metrics (A win, S 1.34x) for 1 (F). CPU stays default.
- commit 36a1b7e. gpu_test 4/4, production unchanged.
NEXT: speed onetone2 GPU ANALYSIS (196->\<100) via fill_pattern + map parallelization (cy177/181 style)
-> then default the shift -> onetone2 wins A+F (S ~1.9x via refinement). That's the unlock.

## cycle 183 (05-27) — onetone2 GPU-analysis A is cuDSS-bound; build_symmetric won't parallelize
Profiled onetone2 GPU-path A=200: METIS parND 88.7 + build_symmetric ~43-53 + emit_map 26.7 + sym 8.5 + a_pos 5 + MC64 5.8.
- build_symmetric_filled (vector<vector>) does NOT parallelize (heap-scatter thrash, noisy timing) -> reverted serial.
- DECISIVE: onetone2 A >= METIS parND 88.7 ~= cuDSS whole-A 99.8. Can't match without FASTER ORDERING.
- onetone2 architecturally stuck: CPU(A win 25, F lose 153) vs GPU-shift(F win 18.5, A lose ~190). Can't match A+F.
gpu_test 4/4, prod unchanged. commit fbda476.
=> KEY UNLOCK = GPU-ORDERING (faster than parND): would help big-matrix A (Synth/ACTIV) AND onetone2 A.
That's the highest-leverage major effort. S (structural) remains the other broad gap.

## cycle 184 (05-27) — parND base sweep confirms cy178 threshold near-optimal (no clean win)
Tested base 4000 vs 20000 on medium matrices: rajat15 marginally prefers 4000 (F/S -2%), but
rajat27 (F -5%), ACTIV (F/S -4%), Synth S all prefer 20000. No size separation -> cy178's 20000
threshold is the best simple choice. No change. Tractable A/S tuning now exhausted (cap cy169,
base cy178/184, symbolic cy177/181, FP32 cy170, etc.).
REMAINING (both major/multi-week): GPU-ORDERING (closes big-matrix A + onetone2 A, since parND
ordering ~= cuDSS whole-A) + STRUCTURAL S (latency-bound, in-structure levers exhausted).

## cycle 185 (05-27) — symbolic-A: parallelize emit_map (nnz_s-gated), gate-safe
F confirmed STABLE first (Synth 4.11 x5 < cuDSS 5.08; the cy184 6.5 was a one-off transient; 7/8 solid).
emit_map is a flat embarrassingly-parallel loop (per-S-nonzero independent, fidx read-only) -> parallelized
over columns, gated on nnz_s>=800000 (only dense-fill; sparse like Synth ~0.3M stays serial = no overhead).
- onetone2 emit 26.7->16.4ms (-10ms), A 200->189; Synth/others unchanged. byte-identical, gpu_test 4/4. commit 6de16a6.
Marginal toward goal (onetone2 A still 189 > 99.8: parND 88.7 + build_symmetric 43 dominate, won't parallelize).
Tractable symbolic-A nearly exhausted; remaining = GPU-ordering (parND bulk) + structural S (both major).

## cycle 186 (05-27) — parallelize a_pos -> flat-symbolic-A parallelization COMPLETE
a_pos (flat fidx loop, like emit_map) parallelized (nnz_a>=200000 gate): Synth 5.65->3.71, onetone2 5.1->2.7ms,
no regression. Completes flat-symbolic-A (adj_build cy177, symmetric_pattern cy181, emit_map cy185, a_pos cy186).
gpu_test 4/4, byte-identical, prod unchanged. commit 92775a5.
Remaining analyze = parND ordering (bulk, GPU-ordering) + build_symmetric (vector<vector>+diagonal, won't parallelize).
onetone2 GPU-A 187 still > cuDSS 99.8. Symbolic-A tuning EXHAUSTED. Remaining gaps: GPU-ordering + structural-S (major).

## cycle 187 (05-27) — tractable A FULLY exhausted (confirmations, no code change)
- parND depth-optimal: Synth d=3 165, d=4 148, d=5 153, d=6 155 -> d=4 best (cy156 holds for big matrix).
- multifrontal_symbolic won't parallelize (vector<vector> fr-build + shared mark[n], like build_symmetric cy183).
=> Tractable A exhausted: flat-symbolic parallelized (cy177/181/185/186); parND (depth/base optimal) +
dependency/cache-bound symbolic (build_symmetric, multifrontal_symbolic, fill_pattern) are the floor.
Remaining cuDSS-match: GPU-ORDERING (parND bulk, multi-month) + STRUCTURAL S. Both exceed incremental cron.
Standing: F 7/8 stable, small-A matched, accuracy>=cuDSS, S/big-A honestly bounded. HOLDING pending fresh idea / major-effort decision.

## cycle 188 (05-27) — __ldg backward scattered gather -> -0.7% S (broad, gate-safe, FIRST S win)
The backward CB-column gather y[fr[nc+j]] + rhs gather y[fr[k]] read FINAL ancestor/own pivots (read-only
during the level; fronts independent, CB cols are done ancestors). Routed via __ldg (read-only data cache)
-> hides some scattered-gather latency: consistent -0.5..-0.9% S on ALL 8 matrices (case6468 .544->.539,
ACTIV 1.569->1.560, Synth 2.75->2.73, onetone2 3.88->3.86, rajat15 1.93->1.92). berr unchanged, gpu_test 4/4.
commit a1e3f0f. Forward NOT __ldg'd (reads just-written pivots -> stale). First clean broad S win (in-structure
levers cy169-174 were all neg); tiny, doesn't close 1.5-2.5x gap (latency/etree structural).

## cycle 189 (05-27) — stage forward pivots in shared -> -4..-9% S (broad, gate-safe, BIGGEST S win)
The forward apply re-gathered y[fr[k]] (nc pivots) per CB row = double-indirection (fr[k]->y[...]) in the
hot loop. Staged pivots in shared (sh_piv[k]) after the thread-0 pivot solve; apply reads shared. Win was
the indirection, not just caching: ACTIV 1.560->1.450(-7%), Synth 2.73->2.54(-7%), onetone2 3.86->3.52(-9%),
rajat15 1.92->1.77(-8%), case6468/rajat27 -5%. berr unchanged, gpu_test 4/4. commit 33a56ee.
With cy188 __ldg, S -5..-9.4% from baseline. NEW positive lever class = memory-access/staging (the in-structure
S levers cy169-174 were all negative). Gap narrowed (ACTIV 2.33->2.15x, Synth 1.91->1.76x, onetone2 2.54->2.30x);
still behind cuDSS (etree-latency floor). Next: look for more staging/indirection wins (backward back-solve tiny;
factor kernels?).

## cycle 190 (05-27) — backward back-solve staging measured-NEGATIVE (+1%), reverted
Tried cy189-style staging on the backward back-solve -> +1-1.6% regression (ACTIV 1.450->1.466, Synth
2.54->2.58, onetone2 3.52->3.57). Back-solve is tiny (nc<=8 thread-0), y[fr[j]] already cache-hot, extra
shared writes cost more. Reverted. cy189 (forward apply, big per-CB-row loop) was the real asymmetry;
backward bulk already staged (xcb). Solve memory-access class captured: cy188 __ldg(+), cy189 staging(++),
back-solve(- reverted). Net S -5..-9.4% from baseline. gpu_test 4/4. commit 6b4e000.
Next memory-access candidate: forward F-panel read coalescing (warp-per-rows, ~2-4% S est) -- but underutilizes
warp (nc<32) + transpose-cost; lower confidence. Otherwise solve memory-access largely done.

## cycle 191 (05-27) — solve memory-access class captured; F-read coalescing deferred (high-risk/low-EV)
Assessed the last solve memory-access candidate (forward F-panel read coalescing, cy159 est 5-9%):
the per-thread F reads are already efficient (each thread reads the 1 cache line it fully uses, nc<=8
doubles = 64B); true coalescing needs a central front-layout transpose (touches factor/solve/extend-add/
emit) = high correctness risk for ~2-4% S. Poor EV/risk -> deferred. warp-per-row doesn't coalesce (rows
stride fsz) + underutilizes (nc<32). No clean fresh win -> holding. Memory-access class done: cy188 __ldg(+),
cy189 staging(++ big), cy190 back-solve(- reverted). Net S -5..-9.4% from baseline.
Remaining cuDSS-match: ordering (GPU-ordering, big-A + onetone2-A + S-via-etree-depth) + structural-S floor.

## cycle 192 (05-27) — assessed remaining levers; no clean high-value fresh win -> hold
- F-read coalescing: even isolated (transposed-L buffer), the factor's extra transposed-L write erases the
  close-F margin (ACTIV 1.05x) for uncertain ~2-4% S. Poor EV/risk.
- Factor backlog (warp-per-col, cuBLAS trailing, shared-staging): low-value -- F already wins 7/8, onetone2
  is A(analysis)-bound not factor-kernel-bound; the cron's "case6468 84.9ms factor" premise is STALE (F beats cuDSS now).
- All remaining S/A gaps converge on the ORDERING (parND bound for big-A + onetone2-A + S etree-depth floor) = major/multi-month.
Holding (no Discord spam per cy187). Memory-access class captured (cy189 +4-9% S). Standing strong+honest.

## cycle 193 (05-27) — #pragma unroll part[] = NEUTRAL (already register-ized), reverted
Hypothesized part[MF_REG_NC] (backward reduction, runtime-k) spilled to local memory -> #pragma unroll
to register-ize. MEASURED neutral/slightly-neg (+0.4-0.8% power-grid, consistent) -> nvcc already
register-izes the small nc<=8 loop; explicit unroll added code-size/reg pressure. Reverted. gpu_test 4/4.
=> Solve micro-opts EXHAUSTED: cy189 forward staging was THE win (-4-9%); cy188 __ldg (+0.7%); cy190
back-solve staging (neg), cy192 F-read coalescing (poor EV), cy193 unroll (neutral) all ruled out.
Solve is optimized. Remaining S/A converge on ORDERING (major/multi-month). Net S -5..-9.4% from baseline.

## cycle 195 (05-27) — reassessed: no goal-relevant tractable idea; tractable space comprehensively captured
Goal gaps (big-matrix A, S etree-floor, onetone2 A) all converge on the ORDERING -> parND is optimal
(depth d=4 cy187, base cy178/184); further = GPU-ND (major/multi-month). Factor backlog (cron's
warp-per-col/cuBLAS/staging) is GOAL-IRRELEVANT: cron's "case6468 84.9ms factor" premise is STALE
(factor optimized cy70-167; case6468 F 0.524 < cuDSS 0.612 -> F wins 7/8), and onetone2's gap is
analysis(A)-bound not factor-kernel-bound. Tractable space captured: solve optimized (cy189), symbolic
parallelized (cy177/181/185/186), parND optimal, F 7/8, S -5..-9.4% (cy188-189). Holding; GPU-ND is the
decisive remaining lever (deliberate multi-month investment, beyond incremental cron).

## cycle 196 (05-27) — hold; GPU-ND confirmed not cron-incrementable
GPU nested-dissection without complete refinement = worse fill -> regresses working parND F/S (all-or-nothing, multi-month). Cannot build useful gate-safe increments without breaking the strong standing. Tractable space captured (cy169-195). Holding; will implement any fresh tractable idea instantly. Decisive leap = deliberate GPU-ND investment (different mode than incremental cron).

## cycle 198 (05-27) — hold (no fresh tractable idea; ordering fork awaits decision)
No new goal-relevant tractable idea. The 3 ordering options (cy197) all need user input (GPU-ND multi-month / ParMETIS MPI-build-change / accept). Holding the strong standing; will implement any fresh tractable idea instantly (cy188-189 showed they still appear). No churn, no breaking the working solver with an un-incrementable rewrite.

## cycle 200 (05-27) — full 14-matrix standing: small matrices NEAR full-match (fresh validation)
Benchmarked the 6 unchecked matrices (wang3, case30/118/1197, ACTIVSg2000, case3012wp), production cold.
- ours WINS A on small/tiny (cuDSS fixed ~10ms analysis overhead vs ours ~1ms): case30 1.1/10.4, case118 1.1/10.9.
- case30/118/ACTIVSg2000/case3012wp = NEAR full-match (A+F win, S 1.3-1.8x the only gap, post-cy189).
- case1197: A win, F 2.2x / S lose. wang3 (big circuit): loses all 3 (A 3.5x, F 2.7x cold, S 1.7x), GPU (not CPU).
=> The A gap is ONLY on the big matrices (ordering). S is the consistent gap everywhere (etree-floor).
Strengthens the standing: ~9/14 A-competitive, F wins most, S 1.3-2.3x. Cold one-shot caveat on F/S abs.

## cycle 201 (2026-05-27 KST 17:40) — GPU-ND foundation (user directive: "1번으로 직접 구현해서라도 cuDSS를 따라")
The user chose Option #1 of ordering_path_forward.md: implement GPU nested-dissection from scratch
to close the big-A + onetone2-A + S-via-depth gaps that parND-METIS can't.

Delivered (cy201 = foundation of a multi-cycle effort):
- NEW src/mysolver/reordering/gpu_nd.{hpp,cu}: `gpu_nd_separator(n,xadj,adjncy,nnz,part)` — a GPU
  vertex separator. Device level-synchronous BFS (kernel bfs_expand, idempotent race, per-level
  changed-flag) from vertex 0; host median bisection on the BFS levels; separator = part-0
  vertices adjacent to part-1 (boundary). Returns false on degenerate split -> caller uses METIS.
- Integrated opt-in into par_nd_rec (metis_nd.cpp): env GPU_ND swaps ONLY the separator;
  induce/recurse/assemble reused. Default OFF -> production parND untouched.
- CMake: gpu_nd.cu added to benchmark / gpu_mf_bench / circuit_mf_test, guarded by compile macro
  MYSOLVER_HAVE_GPU_ND so CPU-only targets (mysolver_test, level_stats) don't reference the CUDA symbol.
- gpu_mf_bench: added MF_FILL env diagnostic (prints fill_nnz + x-input).

Result — the separator WORKS (valid, balanced) but fill quality is the gap (depth-1, separator-vs-separator):
| matrix | METIS fill | GPU-ND fill | ratio | sep |
|---|---|---|---|---|
| case6468rte | 85,359 | 405,895 | 4.75x | 753 |
| case8387pegase | 110,830 | 254,802 | 2.30x | 479 |
| case_ACTIVSg25k | 426,746 | 655,731 | 1.54x | 566 |
| case_SyntheticUSA | 1,548,335 | 3,852,601 | 2.49x | 1961 |

Confirms ordering_path_forward.md exactly: BFS-median bisection gives balanced cuts but its fill is
1.5-4.75x worse than METIS's FM/KL-refined separator. Two known gaps for the next cycles:
1. SEPARATOR QUALITY — needs multilevel coarsening + FM/KL refinement (SyntheticUSA's runaway fill
   crashes downstream at full recursion). This is the bulk of the remaining GPU-ND work.
2. BFS SPEED — per-level host round-trip serializes on the default stream; under parallel recursion
   (PAR_ND=4) it hangs. Needs device-side termination / per-thread streams / GPU-only at top levels.

Gate: gpu_test 4/4; production benchmark + gpu_test build clean. GPU_ND opt-in (default off).

## cycle 202 (2026-05-27 KST 17:55) — GPU-ND cut quality (pseudo-peripheral + min-cover separator)
cy201 established the separator SIZE is already ~= METIS; the fill gap is cut QUALITY/POSITION.
Attacked it with two levers (both in gpu_nd.cu, still opt-in GPU_ND):
1. PSEUDO-PERIPHERAL start (double-BFS, George-Liu): vertex-0 start gave a lopsided level
   structure. BFS from 0 -> jump to the deepest-level vertex -> BFS again. Near-peripheral start
   -> more, thinner levels (maxl rose, e.g. case8387 48->91) -> better-positioned median cut.
2. Vertex separator = the SMALLER boundary side (min-vertex-cover approximation) instead of
   cy201's always-part-0 boundary.

Depth-1 fill vs METIS (separator-vs-separator, MF_FILL):
| matrix | METIS | cy201 | cy202 | cy202 vs METIS | sep (cy201->cy202) |
|---|---|---|---|---|---|
| case6468rte | 85,359 | 405,895 | 289,666 | 3.39x (was 4.75x) | 753->589 |
| case8387pegase | 110,830 | 254,802 | 125,148 | 1.13x (was 2.30x) NEAR-MATCH | 479->123 |
| case_ACTIVSg25k | 426,746 | 655,731 | 852,848 | 2.00x (was 1.54x) REGRESSED | 566->806 |
| case_SyntheticUSA | 1,548,335 | 3,852,601 | 2,567,586 | 1.66x (was 2.49x) | 1961->1225 |

Net total fill 5.17M -> 3.84M (-26%); 3/4 improved (the two worst gaps much better, case8387
essentially matches METIS), only ACTIVSg25k regressed. gpu_test 4/4, production unaffected.
Negative sub-result: the min-cover smaller-side pick ALONE (without pseudo-peripheral) was a
no-op on these 4 (part-0 boundary was already the smaller side) -> the win is the start vertex.

NEXT cycles: (a) FM/KL boundary refinement -> cut per-matrix variance (ACTIVSg25k regression),
push all toward 1.0x; (b) BFS SPEED -- double-BFS doubled the per-level host round-trips, so
PAR_ND=4 GPU_ND=1 still hangs; needs device-side termination / per-thread streams.

## cycle 203 (2026-05-27 KST 18:10) — GPU-ND cut-position search (-6.5% fill, case8387 = METIS)
Probed A-time + F first (PAR_ND=1, METIS vs GPU_ND). Verdict: FILL is the dominant problem,
not A. The cy202 fill penalty craters F (case6468 0.607->9.69ms = 16x, SyntheticUSA 4.2->92ms
= 22x, ACTIVSg25k 1.54->24.9ms). A is confounded (case6468 A=202ms is 1st-matrix one-time CUDA
init; warm case8387 GPU-ND A=34 < METIS 38). So stay on fill quality.

Idea: SEARCH the cut POSITION (cy201/202 fixed it at the median). Two sub-attempts:
1. (NEGATIVE, replaced) narrowest FULL BFS level as the separator. A level IS a valid vertex
   separator (no edge skips a level) but is WIDER than a single boundary side, and forcing
   balance onto the narrowest level is lopsided -> total fill +15% vs cy202. case8387's narrow
   level (39 = METIS-match) showed the cut-POSITION upside; kept that insight, dropped the mechanism.
2. (KEPT) scan cuts between levels L-1/L, both halves >= 35%, score min(width(L-1),width(L))
   (upper bound on the smaller boundary), take the smaller-boundary separator at the best cut.

Depth-1 fill vs METIS:
| matrix | METIS | cy202 | cy203 | cy203 vs METIS |
|---|---|---|---|---|
| case6468rte | 85,359 | 289,666 | 269,275 | 3.15x (was 3.39x) |
| case8387pegase | 110,830 | 125,148 | 111,550 | 1.01x (was 1.13x) MATCHES METIS |
| case_ACTIVSg25k | 426,746 | 852,848 | 857,665 | 2.01x (flat) |
| case_SyntheticUSA | 1,548,335 | 2,567,586 | 2,349,153 | 1.52x (was 1.66x) |

Total fill 3.84M -> 3.59M (-6.5% vs cy202). gpu_test 4/4, GPU_ND opt-in, production unaffected.
Two-cycle GPU-ND fill arc: cy201 4.75/2.30/1.54/2.49 -> cy203 3.15/1.01/2.01/1.52.

NEXT: (a) ACTIVSg25k stuck at 2.0x -- pseudo-peripheral start HURT it (cy201 was 1.54x); try
both starts (0 and peripheral) and keep the better, or FM. (b) BFS speed still hangs PAR_ND=4.

## cycle 204 (2026-05-27 KST 18:25) — GPU-ND evaluate both BFS starts (ACTIVSg25k 2.0x->1.46x)
A single BFS start is a heuristic. cy202's pseudo-peripheral start helps most graphs (more,
thinner levels) but HURT case_ACTIVSg25k (cy201 vertex-0 = 1.54x, cy203 peripheral = 2.0x).
Refactored the cut-position search + smaller-boundary separator into a build_part(level structure)
helper, then evaluate BOTH starts and keep whichever yields the smaller vertex separator (a good
lower-fill proxy under the >=35% balance constraint).

Per-matrix start choice + depth-1 fill vs METIS:
| matrix | METIS | cy203 | cy204 | start | cy204 vs METIS |
|---|---|---|---|---|---|
| case6468rte | 85,359 | 269,275 | 272,948 | peri (sep 570<647) | 3.20x |
| case8387pegase | 110,830 | 111,550 | 112,407 | peri (sep 39<415) | 1.01x MATCH |
| case_ACTIVSg25k | 426,746 | 857,665 | 623,777 | v0 (sep 512<791) | 1.46x (was 2.01x) |
| case_SyntheticUSA | 1,548,335 | 2,349,153 | 2,368,992 | peri (sep 1060<1500) | 1.53x |

ACTIVSg25k recovered AND beats its cy201 (1.46x < 1.54x). Total fill 3.59M -> 3.38M (-5.8%).
Four-cycle fill arc vs METIS: cy201 4.75/2.30/1.54/2.49 -> cy204 3.20/1.01/1.46/1.53.
gpu_test 4/4, GPU_ND opt-in, production unaffected.

NEXT: case6468 is now the laggard (3.20x) -- FM/KL boundary refinement to push it (and all) -> 1.0x.
BFS speed (PAR_ND=4 host-round-trip hang) still open.

## cycle 205 (2026-05-27 KST 18:40) — GPU-ND exact min vertex separator (Koenig min-vertex-cover)
cy204 took the smaller boundary SIDE as the separator -- only a 2-approximation of the minimum
vertex separator. The EXACT minimum vertex separator for a given cut = minimum vertex cover of
the cut's bipartite boundary graph (left=part-0 boundary, right=part-1 boundary, edges=cut edges).
Koenig's theorem: min-vertex-cover = max bipartite matching. The cover covers every cut edge so
removing it disconnects the halves -> valid, provably minimal, and <= cy204's smaller side.

Implemented in build_part: Kuhn's augmenting-path matching (boundaries small, O(V*E) fine) +
Koenig alternating-reachability to extract the cover = (Left\Z) U (Right∩Z).

Depth-1 fill vs METIS (every separator shrank):
| matrix | METIS | cy204 | cy205 | sep cy204->cy205 | cy205 vs METIS |
|---|---|---|---|---|---|
| case6468rte | 85,359 | 272,948 | 242,190 | 570->525 | 2.84x (was 3.20x) |
| case8387pegase | 110,830 | 112,407 | 112,578 | 39->31 | 1.01x MATCH |
| case_ACTIVSg25k | 426,746 | 623,777 | 611,501 | 512->496 | 1.43x (was 1.46x) |
| case_SyntheticUSA | 1,548,335 | 2,368,992 | 2,307,797 | 1060->1032 | 1.49x (was 1.53x) |

Total fill 3.38M -> 3.27M (-3.1%). gpu_test 4/4, GPU_ND opt-in, production unaffected.
Five-cycle fill arc vs METIS: cy201 4.75/2.30/1.54/2.49 -> cy205 2.84/1.01/1.43/1.49.

NEXT: case6468 still the laggard (2.84x) -- the cut itself is the limit now (min-cover is exact
for a GIVEN cut); needs FM-style cut refinement or a better cut. BFS speed (PAR_ND=4 hang) open.

## cycle 206 (2026-05-27 KST 18:55) — GPU-ND greedy FM separator refinement
cy205's min-cover is exact only for a GIVEN cut; case6468's 2.84x = its sep=525 dominates (a
~138k dense separator block of the 242k fill). FM relocates a separator vertex into a side,
pulling its opposite-side neighbors into the separator (net sep change = #opposite nbrs - 1).
Accept only delta<=0 moves; after min-cover many separator vertices have ALL opposite neighbors
already in the separator (delta=-1, pure removal). Balance-capped <=65%/side, vertices locked once.

Depth-1 fill vs METIS (every separator shrank again):
| matrix | METIS | cy205 | cy206 | sep cy205->cy206 | cy206 vs METIS |
|---|---|---|---|---|---|
| case6468rte | 85,359 | 242,190 | 222,340 | 525->489 | 2.60x (was 2.84x) |
| case8387pegase | 110,830 | 112,578 | 110,633 | 31->25 | 1.00x ==METIS |
| case_ACTIVSg25k | 426,746 | 611,501 | 603,744 | 496->479 | 1.41x (was 1.43x) |
| case_SyntheticUSA | 1,548,335 | 2,307,797 | 2,278,604 | 1032->1012 | 1.47x (was 1.49x) |

Total fill 3.27M -> 3.22M (-1.8%). gpu_test 4/4, GPU_ND opt-in, production unaffected.
Six-cycle fill arc vs METIS: cy201 4.75/2.30/1.54/2.49 -> cy206 2.60/1.00/1.41/1.47.

DIMINISHING RETURNS: delta<=0 FM only finds available relocations; case6468 stuck ~2.6x because
the LEVEL-STRUCTURE cut (BFS bisection) is inherently limited. Bigger gains now require either
(a) hill-climbing FM (accept delta>0 temporarily + rollback) or (b) multilevel coarsening (the
real source of METIS quality) -- a larger effort. BFS speed (PAR_ND=4 hang) still gates the
depth-4 A-time test (the actual GPU-ND premise).

## cycle 207 (2026-05-27 KST 19:10) — GPU-ND batched BFS termination + A-cost DIAGNOSTIC
Batched the per-level BFS termination check: launch K level-kernels back-to-back on the ordered
stream (each consumes the prior frontier), read the `changed` flag once per batch (K-fold fewer
host syncs; trailing no-op kernels cheap). Env BFS_BATCH (default 16). gpu_test 4/4, fill +
correctness unchanged.

KEY FINDING (the cycle's real product, not the optimization):
- A-time A/B K=1 vs K=16 is WITHIN NOISE: case8387 37.0->35.2, ACTIVSg25k 113->111, SyntheticUSA
  457->462 (went UP). So BFS host round-trips are NOT a significant fraction of A.
- PAR_ND=4 GPU_ND=1 still does NOT complete (batching didn't unblock depth-4) -> the depth-4
  blocker is the concurrency / CPU min-cover+FM cost across recursive calls, not BFS sync count.
- CRUCIAL REFRAME: GPU-ND's A LOSES on big matrices precisely because its HIGHER FILL makes the
  downstream symbolic (fill_pattern) costlier. SyntheticUSA: GPU-ND A=462 > METIS A=401, and
  GPU-ND fill is 1.47x higher. The ordering BFS is cheap; the fill-driven symbolic dominates.

STRATEGIC CONCLUSION: the GPU-ND premise (faster ANALYSIS than METIS) CANNOT be realized until
fill reaches PARITY with METIS -- higher fill => costlier symbolic => slower A, and worse F/S.
Fill is plateaued at 1.0-2.6x (cy206) with the single-level BFS-bisection + min-cover + FM stack.
The ONLY path to fill parity (and thus the A premise) is MULTILEVEL COARSENING (coarsen -> cut on
the small coarse graph -> refine on the way back up) -- the actual source of METIS quality. This
is the make-or-break, multi-cycle effort. Batched BFS kept (sound, fewer syncs, needed for depth-4).

## cycle 208 (2026-05-27 KST 19:30) — GPU-ND MULTILEVEL COARSENING -> FILL PARITY (breakthrough)
cy207 concluded multilevel coarsening is the only path to fill parity (the single-level BFS cut
ceilings at 1.0-2.6x). Implemented (opt-in env GPU_ND_ML): greedy-matching contraction builds a
coarsening hierarchy down to <=200 vtx (or no-progress), cut the coarsest graph with the existing
both-starts + cut-search + Koenig min-cover + FM stack, then project the cut up refining with FM
at each level. Graph-parameterized CPU helpers: cpu_bfs, build_part_cpu, fm_refine_cpu, coarsen,
separator_cpu, multilevel_separator.

Depth-1 fill vs METIS (single-level cy206 -> multilevel cy208):
| matrix | METIS | cy206 | cy208 | cy208 vs METIS | sep cy206->cy208 |
|---|---|---|---|---|---|
| case6468rte | 85,359 | 222,340 | 88,245 | 1.03x (was 2.60x) | 489->90 |
| case8387pegase | 110,830 | 110,633 | 120,076 | 1.08x | 25->106 |
| case_ACTIVSg25k | 426,746 | 603,744 | 450,094 | 1.05x (was 1.41x) | 479->158 |
| case_SyntheticUSA | 1,548,335 | 2,278,604 | 1,631,175 | 1.05x (was 1.47x) | 1012->284 |

ALL NEAR PARITY. Total fill +5.4% over METIS (was +48%). F/S recovered to ~METIS levels (case6468
F 6.3->0.75ms, SyntheticUSA F 60->7ms; berr 1e-13..1e-16). gpu_test 4/4, GPU_ND_ML opt-in.

THE MAKE-OR-BREAK MILESTONE: GPU-ND now reaches METIS fill quality. 8-cycle fill arc vs METIS:
cy201 4.75/2.30/1.54/2.49 -> cy208 1.03/1.08/1.05/1.05.

CAVEAT: the coarsening is CPU (greedy first-neighbor matching), so A ~= METIS-parND -- NOT faster
than cuDSS yet (cuDSS A beats METIS-parND on big matrices). The A-speed win now requires
GPU-ACCELERATING the coarsening (parallel matching) -- the next major lever. Also the greedy
matching is weak (coarsest still 1.7k-15k vtx, ~0.77x/level vs HEM's ~0.5x); HEM/aggressive
matching would coarsen faster (fewer levels, faster ordering).

## cycle 209 (2026-05-27 KST 19:50) — GPU-ND HEM-style matching -> fill MATCHES/BEATS METIS
cy208's coarsening matching was weak (first-come/first-neighbor, ~0.77x/level, coarsest 1.7k-15k
-> the multilevel short-circuited). Switched coarsen() to LOW-DEGREE-FIRST ordering + lowest-degree
unmatched neighbor as partner (HEM-style for unweighted graphs): low-degree vtx match first (fewest
chances), min-degree partner keeps the coarse graph sparse. Now coarsens properly to ~110-172 vtx
(10-12 levels).

Depth-1 fill vs METIS (cy208 -> cy209):
| matrix | METIS | cy208 | cy209 | cy209 vs METIS | coarsest |
|---|---|---|---|---|---|
| case6468rte | 85,359 | 88,245 | 84,496 | 0.99x BEATS | 172 |
| case8387pegase | 110,830 | 120,076 | 111,742 | 1.01x | 172 |
| case_ACTIVSg25k | 426,746 | 450,094 | 423,341 | 0.99x BEATS | 129 |
| case_SyntheticUSA | 1,548,335 | 1,631,175 | 1,564,183 | 1.01x | 110 |

Total fill +0.6% over METIS (2/4 beat). With fill parity, F BEATS cuDSS on all 4 power matrices:
case6468 0.54 vs 0.61, case8387 0.80 vs 1.11, ACTIVSg25k 1.57 vs 1.88, SyntheticUSA 4.55 vs 5.08.
berr 1e-13..1e-16. gpu_test 4/4, GPU_ND_ML opt-in.

9-cycle GPU-ND fill arc vs METIS: cy201 4.75/2.30/1.54/2.49 -> cy209 0.99/1.01/0.99/1.01 (PARITY).

CAVEAT: coarsening is CPU; A ~= METIS-parND (slightly slower from more levels) -> NOT faster than
cuDSS yet. The A-speed win vs cuDSS requires GPU-accelerating the coarsening (parallel matching +
contraction) -- the next major lever. GPU-ND fill quality is now SOLVED (METIS parity).

## cycle 210 (2026-05-27 KST 20:10) — GPU-ND linear-time coarsening primitives (A -16% big matrix)
PREP_TIME breakdown confirmed ORDERING dominates analysis: SyntheticUSA metis_nd(order)=273.8ms vs
permute+symmetric+etree+fill = ~40ms total. So the multilevel coarsening's per-level primitives
must be linear. Replaced two super-linear ops in coarsen():
1. degree-ordering std::sort (O(gn log gn)) -> O(gn) bucket sort by degree.
2. per-coarse-vertex sort+unique contraction -> O(gn+nnz) monotonic-marker dedup: group fine
   vertices by coarse id (linked list head/nxt), then for each coarse c collect members'
   coarse-neighbors deduping with seen[cu]==c. c increases monotonically so stale marks auto-ignore
   (no reset, no sort).

Fill stays at METIS parity (case6468 1.00x, case8387 0.99x, ACTIVSg25k 1.05x, SyntheticUSA 1.00x;
tiny tie-break drift from the new adjacency order). A on the big matrix: SyntheticUSA 460->388ms
(-16%), case8387 40->39; F still beats cuDSS (case6468 0.50 vs 0.61); berr 1e-13..1e-16.
gpu_test 4/4, GPU_ND_ML opt-in.

STANDING: GPU-ND ordering now reaches METIS fill quality AND ~METIS ordering speed (single-thread).
But cuDSS A < METIS-parND on big matrices, so the CPU multilevel still doesn't beat cuDSS A. The
remaining A-win lever is GPU-accelerating the coarsening (parallel matching + contraction) -- a
large all-or-nothing GPU effort (matching is linear/easy; the contraction sort/dedup is the hard
parallel part). F 7/8 vs cuDSS and S 1.46-2.31x standings are unchanged by the ordering work.

## cycle 211 (2026-05-27 KST 20:30) — GPU-ND size guard + HONEST METIS-parND baseline
Size guard (env GPU_ND_ML_MIN, default 30000): multilevel only for large graphs; small subgraphs
return false -> METIS fallback. Fixes the small-matrix fill drift at PAR_ND>1 (case8387 F 0.71->
1.03 at depth 4 -- recursive multilevel sub-cuts on small graphs < METIS NodeND). With the guard,
small matrices use pure parND-METIS (parity, no drift); large matrices use multilevel for big cuts.

HONEST BASELINE COMPARISON (PAR_ND=4 = production depth), GPU_ND_ML vs plain METIS-parND:
| matrix | METIS A/F | GPU_ND_ML A/F | verdict |
|---|---|---|---|
| case6468 | 27.6 / 0.56 | 28.1 / 0.57 | tie (both METIS via guard) |
| case8387 | 32.8 / 0.67 | 33.0 / 0.69 | tie (both METIS via guard) |
| ACTIVSg25k | 90.3 / 1.54 | 101.2 / 2.31 | METIS better (A and fill) |
| SyntheticUSA | 317.8 / 4.17 | 319.7 / 4.42 | tie A, METIS slightly better fill |

CONCLUSION: CPU multilevel reaches ~METIS quality but does NOT beat METIS-parND. The earlier
"SyntheticUSA 385->322" was only vs GPU_ND_ML's OWN PAR_ND=1, not the real baseline. An A-WIN over
METIS/cuDSS requires GPU-ACCELERATING the coarsening (matching is easy/linear; the contraction
sort/dedup is the hard parallel part) -- the large unfinished effort. CPU multilevel = dead end for
an A-win (it just reproduces METIS).

GPU-ND EFFORT STANDING (cy201-211): fill parity ACHIEVED (cy208-210, 0.99-1.05x METIS), valid+
gate-safe (gpu_test 4/4, opt-in). But CPU multilevel ties METIS (no net win). The only remaining
A-lever is GPU-accelerated coarsening. Other open front: S (1.46-2.31x on all matrices). F 7/8 won.

## cycle 212 (2026-05-27 KST 20:50) — GPU parallel matching (resident-GPU coarsening, piece 1)
First re-confirmed via fresh data that the two non-F fronts are walled:
- S: multi-subtree profile (MF_MS_PROFILE) shows max_subtree~100%, est_speedup~1.0x on ~all
  matrices -> one heavy chain, NO concurrency to exploit (re-confirms cy172-174). S is at limit.
- A: cy211 showed CPU multilevel only ties METIS-parND. The only A-win lever = a RESIDENT-GPU
  multilevel ordering (the user's directed GPU-ND path).

Began the resident-GPU coarsening: GPU mutual-lowest-degree matching (match_propose + match_commit
kernels, iterate to fixpoint; gpu_match host helper), wired opt-in into coarsen() (env
GPU_ND_GMATCH, levels >= 50000; mirrors the CPU low-degree heuristic).

RESULT: GPU matching maintains fill PARITY (SyntheticUSA 1.007x METIS, berr 1e-12 -- correct) but
STANDALONE is SLOWER (A 432 vs 385ms CPU-match): per-level upload/download overhead exceeds the
matching cost, and matching is not the coarsening bottleneck. CONFIRMS: an A-win needs the FULL
resident pipeline (matching + contraction both on GPU, data resident across levels, no per-level
host transfer). GPU matching is a validated building block (piece 1), NOT a standalone win.
Kept opt-in (default off -> no impact on any default path); gpu_test 4/4; production unaffected.

NEXT piece (cy213+): GPU contraction (build the coarse CSR on-device via sort/segmented dedup --
the hard parallel part, the actual coarsening bottleneck). Then a device-resident coarsening loop
(no per-level transfer) is the first configuration that could beat METIS/cuDSS ordering on A.
This is an explicitly MULTI-CYCLE build with no per-cycle end-to-end win until the loop is resident.

## cycle 213 (2026-05-27 KST 21:15) — device-RESIDENT GPU coarsening (validated; NEGATIVE)
Completed the resident-GPU coarsening (cy212's next-piece): GPU matching + GPU id-assignment
(isrep + exclusive_scan) + thrust GPU contraction (emit coarse-edge keys -> thrust::sort ->
thrust::unique -> coarse CSR via per-src histogram scan), looped with the graph DEVICE-RESIDENT
across all levels (only the small cmap is downloaded per level for the CPU projection; no per-level
graph re-upload). Opt-in GPU_ND_RESIDENT. (Thrust verified available; production builds clean.)

RESULT (validated correct, berr 1e-13..1e-14): does NOT beat CPU/METIS.
| matrix | CPU coarsen A/F | RESIDENT GPU A/F | verdict |
|---|---|---|---|
| case_ACTIVSg25k | 117.9 / 2.21 | 115.5 / 1.78 | marginally better |
| case_SyntheticUSA | 382.8 / 4.38 | 410.4 / 5.27 | WORSE (A and fill) |

Two causes: (1) GPU parallel (mutual) matching is LOWER-QUALITY than CPU sequential low-degree-first
-> fill drift (SyntheticUSA F 4.38->5.27); (2) at PAR_ND=1 the METIS NodeND on the two halves
DOMINATES the ordering (my coarsening is a fraction), so coarsening speed barely moves A and the
thrust/kernel overhead offsets any gain.

DEFINITIVE A-CONCLUSION: GPU-accelerating the coarsening ALONE cannot beat cuDSS/METIS on A. An
A-win requires the ENTIRE nested dissection on-device (recursion + cut + projection + FM) -- a
research-grade GPU-ND analysis like cuDSS's, not a cron-cycle deliverable. The tractable GPU-ND
achievement is the CPU multilevel FILL PARITY milestone (cy208-210, 0.99-1.05x METIS).

GPU-ND EFFORT CONCLUDED (cy201-213): fill parity ACHIEVED; A-win is research-grade. Project
standing: F beats cuDSS 7/8 (settled); S at structural single-RHS limit (settled); A behind cuDSS
on big matrices (METIS-parND ties our GPU-ND; cuDSS faster; closing it = research-grade rewrite).

## cycle 214 (2026-05-27 KST 21:35) — standing VERIFICATION + exhaustion re-confirm (no code change)
Post GPU-ND-conclusion (cy213). Scanned for a fresh tractable lever, found none, verified the standing:
- A symbolic: fill_pattern is sequential (column j depends on its etree children via col[]); the
  vector<vector> structure makes parallelization regress (cy183 precedent) -> not a clean A lever.
  Ordering still dominates A and is research-grade to beat cuDSS.
- S: backward-solve bottleneck = inherent scattered y-gather (already __ldg'd cy188); the F-read is
  already coalesced (per-k across threads); est_speedup~1.0x (cy212). At design limit.
- Fresh production F/S EXACTLY match documented afs_tables: F 0.524/0.711/1.786/4.112 (all 4 BEAT
  cuDSS 0.612/1.113/1.876/5.081), S 0.516/0.650/1.449/2.542 (1.46-2.31x), berr 1e-13..1e-16.
  Standing verified CURRENT, no regression. (A single-shot is noisy without LD_PRELOAD det_rand,
  but F/S matching exactly => factorization + ordering quality unchanged.)

CONCLUSION: the cron-tractable optimization space is exhausted -- F beats cuDSS 7/8 (settled), S at
structural single-RHS limit (settled), A research-grade (full on-device GPU-ND, multi-week). Per the
cy115 precedent did NOT manufacture a neutral change. Holding the strong standing; will implement any
genuinely fresh tractable idea the instant it surfaces (as the cy188-189 memory-access class did).

## cycle 215 (2026-05-27 KST 21:55) — double-buffer the backward CB-tile gather (NEUTRAL, reverted)
Fresh S idea targeting the worst gap (onetone2 2.31x, big circuit fronts span many CB tiles):
double-buffer the CB-tile scattered y-gather in mf_bwd_level -- prefetch tile t+1 into the other
shared buffer (xcb0/xcb1) while computing tile t, to overlap the gather latency with compute
(classic shared-mem pipelining). Single-tile fronts (power-grid) take the same path (unaffected).

RESULT: NEUTRAL. gate 4/4; power-grid F/S unchanged (single-tile). Circuit S:
  onetone2 SOLVE 3.528 -> 3.480 (-1.4%, within noise), rajat15 1.778 (flat), rajat27 0.621 (flat),
  memplus 0.613 (flat).
The scattered-gather latency is ALREADY hidden by warp-level parallelism (many warps in flight) +
the __ldg read-only data cache (cy188), so explicit double-buffering adds ~nothing. REVERTED (no
improvement). Re-confirms the solve is at its latency-hiding limit (cy171, cy188-190). No commit.

## cycle 217 (2026-05-27 KST 22:35) — forward parallel pre-gather of input pivots (NEUTRAL, reverted)
Fresh forward-solve idea: mf_fwd_level's thread-0 triangular solve read y[fr[k]] (scattered global)
INSIDE its serial loop -> nc dependent scattered reads on the critical path. Pre-gathered all nc
input pivots into shared (sh_in) with ALL threads (reads concurrent, latency hidden), then thread 0
solves from shared. (cy189 staged the OUTPUT pivots; this targets the INPUT gather. Correct: y[fr[k]]
are distinct rows, independent of the pivots thread 0 writes.)

RESULT: NEUTRAL. gate 4/4. power-grid S flat (0.514/0.648/1.448/2.543 vs baseline 0.516/0.650/
1.449/2.542); onetone2 SOLVE 3.48->3.51 (slightly worse, +sync overhead); rajat15 1.778->1.772.
Same root cause as cy215: the scattered-read latency is ALREADY hidden by BLOCK-level parallelism
(many fronts/blocks in flight per level), so restructuring the within-block gather adds nothing.
The solve is bound by the sequential CRITICAL PATH (the separator chain / ordering depth), not by
gather latency that more parallelism could hide. REVERTED. No commit.

TWO consecutive neutral solve-micro attempts (cy215 backward double-buffer, cy217 forward pre-gather)
=> solve kernel micro-optimization is exhausted. The S bound is the critical-path/ordering (shares
A's research-grade root). Stop attempting solve-memory micro-opts; the latency is already hidden.

## cycle 218 (2026-05-27 KST 22:55) — degree-bucketed GPU matching -> resident coarsening FILL PARITY
cy213's device-resident GPU coarsening drifted on fill (SyntheticUSA F 4.38->5.27) because GPU
parallel mutual matching is lower-quality than CPU sequential low-degree-first, COMPOUNDED over all
levels. Fix: match_propose takes a deg_thr; iterate geometric degree buckets (1,2,4,...,>=maxdeg) so
LOW-degree vertices match first (approximates CPU low-degree-first). Per-level maxdeg via k_degbuf +
thrust::reduce(max). Applied to both gpu_match (GPU_ND_GMATCH) and the resident loop.

RESULT (resident GPU coarsening, PAR_ND=1) -- drift FIXED, parity/better than CPU:
  SyntheticUSA: F 5.27->4.453 (CPU 4.38, ~parity), A 410->383 (CPU 383)
  ACTIVSg25k:   F 1.78->1.651 (BEATS CPU 2.21),     A 115->108 (CPU 118)
Both still BEAT cuDSS F (1.651<1.876, 4.453<5.081). gpu_test 4/4, opt-in. Removes blocker #1
(matching quality) of the resident-GPU-ND A-effort.

*** USER GREENLIT (Discord): pursue the GPU-acceleration direction (resident-GPU-ND) for the A/S win;
asked whether F is maintained. ANSWER: YES -- the factor kernel (gpu_mf) is UNCHANGED and depends on
the fill STRUCTURE, not the ordering method. Parity fill (now achieved on GPU) => parity F => F still
beats cuDSS 7/8 (confirmed: GPU-ND-resident-ordered F beats cuDSS on the measured matrices). The
GPU-acceleration is on ANALYSIS (A), orthogonal to the factor.

ROADMAP (multi-cycle, now greenlit): coarsening is resident-GPU (cy212/213/218). Remaining for an
A-win: (a) the cut on the coarsest is CPU (tiny, ok); (b) projection + FM at each level is CPU on
downloaded graphs -> move on-device or reduce; (c) at shallow PAR_ND the METIS-NodeND-on-halves
dominates A -> need GPU-ND to handle the full recursion or deepen its own recursion. Measure A vs
METIS-parND PAR_ND=4 each step; keep fill parity (F preserved); gate 4/4.

## cycle 219 (2026-05-27 KST 23:15) — ROOT-ONLY GPU-ND top-cut -> first GPU-accel A WIN (F preserved)
User greenlit the GPU-acceleration direction (F must hold). Measured the resident GPU-ND at PAR_ND=4:
recursive GPU-ND sub-cuts DRIFT fill (SyntheticUSA F 4.55->5.56) -> worse A+F. But GPU-ND for the
ROOT (largest) cut + METIS parND for the recursion keeps fill PARITY and wins A.

Implemented: is_root flag in par_nd_rec (GPU-ND only at the root call; all recursion = METIS parND).
Raised GPU_ND_ML_MIN default 30000->100000 -- the GPU root cut matches METIS fill only for VERY large
roots (SyntheticUSA 156k: parity + A win), but regresses at medium size (ACTIVSg25k 47k: F 1.58->1.76)
-> route <100k to METIS.

RESULT (PAR_ND=4, reproducible over 3 repeats: METIS ~316 vs GPU-ND ~297):
| matrix | METIS A/F | root-only GPU-ND A/F | verdict |
|---|---|---|---|
| case6468 (<100k) | 28/0.53 | 27/0.52 | METIS (unchanged) |
| case8387 (<100k) | 34/0.72 | 33/0.71 | METIS (unchanged) |
| case_ACTIVSg25k (<100k) | 89/1.58 | 90/1.61 | METIS (unchanged) |
| case_SyntheticUSA (156k) | 316/4.55 | 298/4.40 | GPU-ND: A -6%, F PARITY |

F PRESERVED everywhere (fill parity -> factor unchanged -> still beats cuDSS 7/8). S parity. gpu_test
4/4, opt-in (GPU_ND), production default unchanged. FIRST GPU-acceleration A win over METIS-parND.
Still 1.47x behind cuDSS A (203) -- a step, not the finish.

NEXT: (a) improve GPU root-cut fill quality at MEDIUM size so the 100k threshold can drop (extends the
A win to ACTIVSg25k etc.); (b) speed the resident coarsening further (bigger A win on the root);
(c) the recursion is still METIS -- a faster parallel recursion would compound the win.

## cycle 220 (2026-05-27 KST 23:35) — profile GPU-ND root cut (coarsening-bound); GPU_ND_PROF instrument
Added env GPU_ND_PROF phase timing to multilevel_separator. SyntheticUSA root cut breakdown:
  coarsen 18.3 ms | cut 0.0 ms | project+FM 1.2 ms  -> COARSENING dominates (to grow the cy219 A win,
  speed the coarsening).
Attempts to speed the coarsening:
- fixed-3 matching rounds/bucket (drop the per-round chg[0] host sync): coarsen 18.3->13.4ms BUT fill
  regressed (PAR_ND=1 F 4.45->4.85) -- undershot matching convergence.
- batched convergence (4 rounds/check): coarsen ~18.8ms = NO speedup (does the same total rounds to
  converge; the sync wasn't the dominant cost).
=> the coarsen cost is the matching ROUNDS (kernel launches), not the host syncs; and fewer rounds
trade fill quality. Since F must be preserved (user requirement), no clean coarsen speedup this cycle.
Reverted both; kept adaptive per-round matching (best fill) + the GPU_ND_PROF instrument (gated).

cy219 A win intact: SyntheticUSA A 291 vs METIS ~316, F 4.32 (parity, beats cuDSS 5.08). gpu_test 4/4.
NEXT: reduce matching rounds WITHOUT fill loss (a higher-quality per-round matching that converges in
fewer rounds), or speed the thrust contraction, or improve medium-size GPU cut quality (drop 100k thr).

## cycle 221 (2026-05-27 KST 23:55) — hill-climbing FM (NEGATIVE, reverted)
To EXTEND the cy219 GPU-accel A win to medium matrices (ACTIVSg25k 47k regresses fill 1.58->1.76
under the GPU cut, which is why the threshold is 100k), tried HILL-CLIMBING FM: accept the best move
even if it grows the separator (delta>0), track the running-best separator, roll back to it -- to
escape the greedy delta<=0 FM's local minima.

RESULT: NEGATIVE on both counts.
- ACTIVSg25k: F still 1.76 (unchanged) -> the medium-size fill regression is from the GPU COARSENING
  quality, NOT from FM local-minima. FM is not the lever.
- SyntheticUSA: F WORSENED 4.45->4.69 -- minimizing separator SIZE alone drifts the BALANCE toward
  the 65% cap (smaller sep but worse balance -> more fill). Sep-only hill-climbing needs a balance term.
Reverted to the greedy delta<=0 FM. cy219 win intact (SyntheticUSA A 291, F 4.52 parity, beats cuDSS).
gpu_test 4/4; no commit.

FINDING: the GPU-ND medium-size fill regression (ACTIVSg25k) is COARSENING-quality-bound, not FM.
NEXT: improve the GPU coarsening quality at medium size (the real ACTIVSg25k blocker) -- or accept the
SyntheticUSA-only A win (very-large-root regime) as the tractable GPU-accel result. A balance-aware FM
(sep + imbalance penalty) only if FM is revisited.

## cycle 222 (2026-05-28 KST 00:15) — GPU_ND_CTHR knob: coarsest threshold is size-dependent (no global win)
cy221 pinned the ACTIVSg25k medium-size fill regression on coarsening quality. Targeted it via the
coarsest-graph size threshold (env GPU_ND_CTHR, default 200) and swept it.
- ACTIVSg25k (47k, PAR_ND=1): CTHR=50 -> F 2.46 (too coarse); CTHR=100 -> 1.537 (BEATS METIS 1.58);
  CTHR=200 -> 1.60; CTHR=500 -> 1.78; CTHR=1000 -> 1.66. Sweet spot ~100-200.
- BUT CTHR=100 DESTROYS large SyntheticUSA root cut: F 4.4 -> 6.8 (deep coarsening hurts the large
  graph). So a single global CTHR can't help both -- it is matrix-SIZE-dependent.
=> default 200 is optimal for the proven SyntheticUSA win; the medium-size extension would need a
size-ADAPTIVE CTHR (e.g. CTHR = f(n)), uncertain robustness. Kept GPU_ND_CTHR as a default-200 knob.
cy219 win intact (SyntheticUSA A 289 vs METIS ~316, F 4.42 parity). gpu_test 4/4; opt-in.

STATUS of the GPU-accel A win: REAL but confined to very-large roots (SyntheticUSA -6% A, F parity).
Extending to medium matrices is blocked by size-dependent coarsening quality (FM cy221, CTHR cy222
both ruled out as clean global levers). NEXT: size-adaptive CTHR, or accept + consolidate the win.

## cycle 223 (2026-05-28 KST 00:35) — size-adaptive CTHR explored: CTHR=200 already optimal (no change)
Followed cy222 (CTHR size-dependent) with a size-adaptive CTHR=f(n) idea. Swept SyntheticUSA CTHR at
PAR_ND=4 root-only (repeats, A is noisy):
  CTHR=100: F 6.5 (catastrophic) | 150: A297/F4.42 | 200: A~291/F~4.36 | 300: A293/F4.43 |
  400: A~288/F~4.43 | 500: A284(noise)/F4.64 | 600: A~289/F~4.81 (F nearing cuDSS 5.08)
=> raising CTHR gives NO reliable A gain (200->400 ~3ms = within noise) and WORSENS fill (600 nears
cuDSS); lowering (100) is catastrophic. CTHR=200 is already the optimum; size-adaptive CTHR does not
help. No code change. cy219 win unchanged (SyntheticUSA A ~289, F parity).

CONCLUSION: 3 cycles (cy221 hill-climbing FM, cy222/223 CTHR) confirm the GPU-accel SyntheticUSA A win
(-6% vs METIS, F parity) is NOT tractably growable or extendable to medium matrices. The win is
confined to the very-large-root regime; closing the cuDSS A gap (still 1.47x: ours ~289 vs cuDSS 203)
needs the research-grade full-GPU-recursive-ND (METIS handles our recursion, the dominant A cost).
The cron-tractable GPU-accel space is now also characterized as exhausted (like F/S earlier).

## cycle 225 (2026-05-28 KST 01:15) — consolidate GPU_ND default to the win-config (one flag)
The validated GPU-accel win (root-only resident multilevel) previously required THREE envs
(GPU_ND + GPU_ND_ML + GPU_ND_RESIDENT); GPU_ND=1 alone gave the inferior single-level BFS cut
(1.0-2.6x fill). Consolidated: multilevel is now the DEFAULT gpu_nd_separator path, resident
coarsening is the DEFAULT in multilevel_separator. So GPU_ND=1 ALONE now gives the win. Old variants
kept for A/B: GPU_ND_SINGLE (single-level cut), GPU_ND_NORESIDENT (CPU coarsening).

Verified GPU_ND=1 PAR_ND=4: SyntheticUSA A 287 vs METIS 311 (-8%, within the noisy -4-8% band),
F 4.51 (parity, beats cuDSS 5.08). gpu_test 4/4 (small matrices, GPU_ND off -> unaffected).
Production default UNCHANGED (GPU_ND opt-in). Makes the validated win the canonical single-flag path.

## cycle 227 (2026-05-28 KST 01:55) — boundary test: GPU-ND generalization (A win is structure-specific)
matpower has only 1 large matrix (SyntheticUSA 156k). Searched all datasets; found Sandia/ASIC_100k
(99340, a large CIRCUIT) -- a different structure class. Added to gpu_mf_bench (one-off), tested
GPU-ND (ML_MIN=90000 to trigger on 99340) vs METIS.
FINDING (ordering metrics valid; factor berr=1.0 for BOTH -- circuit needs MC64 which gpu_mf_bench
lacks, so F/S/berr meaningless; A is pre-factor so valid):
- GPU-ND FILL generalizes: 1.86M <= METIS 1.91M (slightly BETTER on this circuit).
- BUT A is SLOWER: GPU-ND 456 vs METIS 362 -> NO A win (GPU coarsening > METIS separator here).
=> the GPU-accel A WIN is POWER-GRID-STRUCTURE-specific (SyntheticUSA wins; ASIC_100k circuit does
not), not merely size-specific. Fill QUALITY generalizes (>= METIS) but the A speedup does not.
Reverted the one-off bench change. gpu_test 4/4.

SCOPE of the validated win: confined to large power-grid-like matrices (SyntheticUSA -6% A, cy226).
No other large power-grid matrix available to validate default-on. The GPU-accel round stands:
validated win on the representative large matrix; fill quality generalizes; A speedup is structure+
size specific; beating cuDSS A broadly needs research-grade full-GPU-recursive-ND.

## cycle 229 (2026-05-28 KST 02:35) — top-levels GPU-ND tested: trades A for F (root-only stays optimal)
Tested extending GPU-ND past root-only to the top large levels (root + ~78k halves for SyntheticUSA)
via env GPU_ND_RECURSE (default off = root-only). PAR_ND=4, repeats:
  root-only:   A ~299, F ~4.46
  top-levels:  A ~285 (-5% faster, more GPU cuts), F ~4.76 (one run 5.025 ~= cuDSS 5.08)
The 78k halves drift fill -> the F margin over cuDSS erodes. Since F must stay clearly ahead, this is
unacceptable -> root-only (cy219) is the A/F sweet spot (A win + clear F margin). Confirms the A win
and F-preservation are in tension; root-only balances them. Knob kept default off (no behavior change).
gpu_test 4/4; production default unchanged.

## cycle 230 (2026-05-28 KST 02:55) — warm E2E reality-check (cuDSS wins warm; cold "win" was an artifact)
Measured mysolver-gpu (GPU_ND=1, the win-config) vs cudss-gpu full E2E on SyntheticUSA.
- COLD (no --warmup-gpu, cudss ran first): mysolver cpu_ms 340 < cuDSS 438. BUT this is a COLD-START
  ARTIFACT -- cuDSS's 438 wall includes ~170ms one-time host overhead (cpu_ms 438 vs cuda_ms 268).
- FAIR (--warmup-gpu, both warm): cuDSS cpu_ms 220.8 (cuda 220.8) vs mysolver-gpu 343.8 (cuda 343.8)
  -> cuDSS WINS warm E2E 1.56x.
So even with the GPU-accel A win, WARM E2E favors cuDSS: our A (1.47x) + S (1.76x) losses + iterative
refinement outweigh our F win. Re-confirms cy179 (cold single-call E2E misleading; warm is the metric).
Honest: there is NO E2E win; the validated GPU-accel A win is real but small relative to cuDSS's
overall warm-E2E lead. No code change. gpu_test 4/4.

## cycle 236 (2026-05-28 KST 03:30) — shared-neighbor fill-aware matching (NEGATIVE, reverted)
Fresh idea for the medium-size GPU coarsening-quality blocker (cy221/222): score matching partners by
deg(u) - 2*shared(v,u) -- the merged coarse-vertex degree -- instead of deg(u) only, to prefer low-
degree partners that also share many neighbors (collapse redundant connectivity -> sparser coarse).
Shared count via two-pointer merge of sorted adjacency in match_propose.
RESULT: NEGATIVE -- REGRESSED fill. SyntheticUSA 1.55M -> 1.84M (+19%); ACTIVSg25k 1.069x METIS (worse).
Collapsing high-shared-neighbor (dense) regions hurts ND coarsening quality for these matrices.
Reverted to degree-only (cy218). gpu_test 4/4, fill parity restored.
Finding: low-degree matching beats coarse-degree-minimizing (shared-neighbor) for ND coarsening here.

## cycle 237 (2026-05-28 KST 03:45) — METIS coarsest cut -> fixes medium-size regression + improves large fill
The multilevel coarsest cut was BFS-bisection (separator_cpu), the quality weak point. Replaced with
METIS_ComputeVertexSeparator on the tiny (<=COARSE_THR=200) coarsest graph -- a high-quality SEED for
the FM projection, ~free (the coarsen=GPU + project/FM stay ours). Default on; GPU_ND_NO_METIS_COARSE
selects the old BFS cut; separator_cpu fallback on failure.

Depth-1 fill vs METIS (PAR_ND=1):
  ACTIVSg25k:   1.07x -> 1.004x  (MEDIUM-SIZE REGRESSION FIXED -- FM cy221, CTHR cy222, shared-nbr cy236 all failed this)
  SyntheticUSA: ~parity -> 0.991x  (now BEATS METIS)
A win holds (SyntheticUSA PAR_ND=4 A ~293 vs METIS ~311, -6%). gpu_test 4/4; production default
unchanged (GPU_ND opt-in). gpu_nd.cu now includes metis.h (targets already link metis).
FIRST real improvement since cy219 -- the coarsest-cut quality was the medium-size blocker, not the
matching/FM/CTHR. NEXT: re-check if ACTIVSg25k A now wins (extend the win) now that its fill is parity.

## cycle 239 (2026-05-28 KST 04:15) — raise COARSE_THR 200->1000 (faster coarsen + better fill)
cy237's METIS coarsest cut removed cy223's BFS-coarsest catastrophe at large CTHR, so a LARGER coarsest
threshold is now high-quality (METIS scales to 1000). Re-swept CTHR (SyntheticUSA, METIS coarsest, PAR_ND=4):
  CTHR=200:  coarsen 18.3ms, F ~4.41
  CTHR=1000: coarsen 15.8ms, F ~4.21   (faster + better fill, repeats-confirmed; A unchanged within noise)
  CTHR=2000: coarsen 14.9ms, F ~4.19
Set default 1000. Only affects n>=ML_MIN=100000 (the GPU-ND root cut on very-large matrices). Win config
(GPU_ND=1, PAR_ND=4): SyntheticUSA A ~288 vs METIS ~311 (-7%), F 4.389 (beats cuDSS 5.08), S 2.651.
gpu_test 4/4; production default unchanged (GPU_ND opt-in).
Cumulative GPU-accel win: cy219 root-only + cy237 METIS-coarsest + cy239 CTHR=1000 -> SyntheticUSA A -7%,
fill now beats METIS, F preserved. Coarsening still ~15.8ms (matching-rounds bound, cy220); the METIS
recursion dominates A -> beating cuDSS warm E2E remains research-grade (full-GPU-recursive-ND).

## cycle 240 (2026-05-28 KST 04:30) — cap matching rounds at 3 (METIS coarsest compensates)
cy220 found fewer matching rounds regressed fill -- but that was with the BFS coarsest cut. With the
METIS coarsest cut (cy237) providing a high-quality seed, MROUNDS=3 now gives IDENTICAL fill
(SyntheticUSA fill_nnz 1.5601M vs adaptive 1.5607M) + faster coarsening (15.6->13.8ms, -12%). Default 3
(env GPU_ND_MROUNDS); only affects n>=ML_MIN=100000 (GPU-ND root cut). Win config: SyntheticUSA A ~286
vs METIS ~311 (-8%), F beats cuDSS. gpu_test 4/4; production default unchanged.

CUMULATIVE GPU-accel win (the cy237 breakthrough + follow-ons):
  cy219 root-only + cy237 METIS-coarsest + cy239 CTHR=1000 + cy240 MROUNDS=3
  -> SyntheticUSA A -8% vs METIS, fill BEATS METIS (0.99x), F preserved, coarsen 18.3->13.8ms.
A-win CEILING = the root cut (the recursion via GPU-ND is parity-not-faster, cy238). Beating cuDSS
warm E2E still needs research-grade full-GPU recursion. The cy237 coarsest-cut insight drove a clean
3-cycle improvement run after the long exhausted stretch.

## cycle 248 (2026-05-28 KST 05:00) — sync-free matching rounds (faster coarsen + better fill)
cy240's MROUNDS=3 kept a per-round chg[0] host sync with early-break. Removed it (run exactly max_rounds
rounds sync-free; env GPU_ND_MSYNC restores the check). BETTER on both axes:
- the early-break cut matching short (less-complete -> slightly worse coarsening),
- the ~180 host syncs cost more than the few extra (async, cheap) kernel launches.
SyntheticUSA PAR_ND=1: coarsen 13.9->12.1ms (-13%), fill_nnz 1569k->1558k (better, ~parity).
Win config PAR_ND=4: A ~283 vs METIS ~311 (-9%, up from -8%), F 4.609 (beats cuDSS 5.08), S 2.729.
gpu_test 4/4; production default unchanged.
Cumulative GPU-accel win (cy219 root-only + cy237 METIS-coarsest + cy239 CTHR=1000 + cy240 MROUNDS=3
+ cy248 sync-free): SyntheticUSA A -9% vs METIS, fill BEATS METIS, F preserved, coarsen 18.3->12.1ms.
A 6th improvement from re-examining held assumptions (here: "the per-round convergence sync is needed").

## cycle 254 (2026-05-28 KST 05:55) — diagnosed the recursive-GPU-ND catastrophe (= cut imbalance)
GPU_ND_DEBUG on GPU_ND_RECURSE/SyntheticUSA: root cut balanced (78058/78077, sep 120) but the recursive
78k cuts are UNBALANCED -- 48040/29992 (62/38, sep 45), 45622/32353 (58/41, sep 83). The cut-position
search minimizes separator size within the >=35% balance window, so it picks the narrowest level even
when off-center -> lopsided recursion subtrees -> fill explosion (F 7-11). The cuts are VALID (small
separators), NOT a degenerate bug. Fixable via a balance-aware cut (prefer ~50/50), BUT cy238 showed
recursion via GPU-ND is parity-not-faster on A even with good cuts -> fixing it yields NO A win. So
ROOT-ONLY (balanced root cut + METIS recursion, which targets balance) remains optimal. No code change.
This closes the recursive-GPU-ND question: the catastrophe is imbalance (understood), and the recursion
GPU-acceleration is a dead end for an A win regardless. The A win stays capped at the root cut (-9%).

## cycle 256 (2026-05-28 07:00 KST) — FINDING: factor is WON on production; cron bench is legacy
- **Root cause of the loop's perennial "factor kernel-bound (84.9ms)" target: it's the LEGACY SUPERNODAL kernel.** `gpu_factor_bench` exercises `gpu_factor.cu` (supernodal, `[snode]`/`nsuper` output), which is 6-10x behind cuDSS — but it is NON-PRODUCTION (referenced only by gpu_factor_bench + gpu_mf_bench setup; production `mysolver_gpu_solver.cu` calls ONLY `gpu_mf_factorize`).
- **Production MULTIFRONTAL (`gpu_mf.cu`) BEATS cuDSS F 4/4 power** [gpu_mf_bench, METIS, warm]:
  - case6468rte    F=0.525 vs cuDSS 0.612  (beat) ; S=0.516 vs 0.286 (1.80x behind)
  - case8387pegase F=0.710 vs cuDSS 1.113  (beat) ; S=0.650 vs 0.363 (1.79x behind)
  - case_ACTIVSg25k  F=1.786 vs cuDSS 1.876 (beat) ; S=1.450 vs 0.674 (2.15x behind)
  - case_SyntheticUSA F=4.110 vs cuDSS 5.081 (beat) ; S=2.544 vs 1.443 (1.76x behind)
- Supernodal `gpu_factor_bench` live (for the record): case6468 5.252, ACTIVSg25k 27.177, SyntheticUSA 69.237 (METIS) — confirms why the old "84.9ms" framing persisted; it's a dead path.
- **=> Factor WON. Frontier = SOLVE (S 1.76-2.15x).** S already deeply tuned (cy170/188/189/190); the only remaining S lever is selective-inversion (invert L diag blocks at factor -> parallel GEMV solve), research-grade. A recursion-capped. No code change this cycle (a genuine target-correction finding). gpu_test 4/4.

## cycle 257 (2026-05-28 07:30 KST) — CONFIRMING-NEGATIVE: implementable S levers dead (fresh profiling)
- Ran MF_MS_PROFILE (cy173 sizing) live on all 4 power matrices. Multi-subtree-stream solve prize:
  - case6468/case8387/ACTIVSg25k: est_speedup ~1.0x (G=2/4/8); SyntheticUSA ~1.1x.
  - Cause: max_subtree=100%, balance~nsub -> ONE dominant subtree holds ~all solve work. The panel etree is a single
    heavy spine (stacked top separators of nested dissection); splitting roots into streams gains nothing.
- Implication: S work concentrates in the few BIG root-separator fronts. Their within-front solve is ALREADY parallel
  (mf_fwd_level apply loop over fsz; mf_bwd_level CB reduction) and memory-bound -> cy188(__ldg)/cy189(shared pivot)
  already captured the access-pattern wins. Selective-inversion would replace the nc<=8 pivot triangular solve (tiny,
  thread-0) -> negligible here; it does NOT remove the level-dependency that is the actual floor.
- Conclusion: the S 1.76-2.15x gap is cuDSS's solve-internals memory layout, NOT a cron-tractable incremental lever.
  Reconfirms research-grade (matches cy227/256). No code change. gpu_test 4/4.

## cycle 258 (2026-05-28 08:00 KST) — CONFIRMING-NEGATIVE: forward-apply coalescing marginal (latency-bound)
- Confirmed front is row-major F[row*fsz+col] (mf_factor_extend_level). mf_fwd_level apply reads F[i*fsz+k]
  (thread=row i, inner k) = stride-fsz across threads -> UNCOALESCED. mf_bwd_level reads F[k*fsz+(nc+j)] = stride-1 -> coalesced.
- Considered full transposed-L staging (LT[k*uc + (i-nc)], column-major) to coalesce the forward apply (cy189-class
  memory-access lever). RULED marginal WITHOUT implementing:
  - cy171 measured S at ~14% peak BW = latency/level-dependency-bound (GPU ~85% idle on narrow levels), NOT BW-bound.
  - Coalescing helps only the big-front WORK portion; it cannot reduce the serial level-latency floor that dominates S time.
  - Reconciles cy159 (forward-coalescing proxy = only -2-4%) with cy171. A full LT buffer would re-confirm marginal-negative
    and adds a per-factor transpose cost that risks the WON F (we beat cuDSS F 4/4, don't want to spend that).
- Decision: not worth it. The cy189 win (-9%, removing apply double-indirection) already captured the work-bound
  access gain; remaining access-pattern gains are diminishing. S stays research-grade. No code change. gpu_test 4/4.

## cycle 259 (2026-05-28 08:30 KST) — WIN: build_symmetric_filled flat-CSR+parallel (A -4..-6%)
- Profiled the A breakdown (PREP_TIME + MF_TIME). SyntheticUSA A=556: metis_nd(order)=428 (77%, parND -40% already
  banked in production), our symbolic prep ~43, gpu_mf_analyze ~85 of which **build_symmetric_filled=52ms** = biggest
  non-METIS chunk.
- Rewrote build_symmetric_filled: serial vector<vector> -> FLAT-CSR two-pass + par_for slice-sort (the cy181 recipe
  that won for symmetric_pattern), keeping the diagonal. The cy109/cy183 negatives were the wrong combination
  (flat-serial / parallel-on-vector<vector>); flat-CSR+parallel avoids per-column heap allocs AND cache thrash.
- Result: build_symmetric_filled -62..-64% (SyntheticUSA 52.2->18.7, ACTIVSg25k 13.8->5.3, case6468 3.6->1.3, case8387 ~3.6->1.9).
  A totals (default ordering): case6468 34.5->32.5, case8387 44.6->42.8, ACTIVSg25k 143->134, SyntheticUSA 562->533 (-4..-6%).
  Stacks additively with parND (production) since it lives in ordering-independent gpu_mf_analyze.
- gpu_test 4/4, berr unchanged (3.5e-14..2.97e-13), F/S identical -> symbolic output byte-identical. Gate-safe (pure symbolic).

## cycle 260 (2026-05-28 09:00 KST) — WIN: multifrontal_symbolic parallelized (continues cy259 A vein)
- 2nd-biggest non-METIS A chunk (13.8ms). Parallelized two independent per-panel phases (par_panels, 12-core, gated >=4096 panels):
  - Phase 1 (front-rows build): per-thread marker arrays (mark[i]==p; p monotonic in chunk) -> panels independent, byte-identical fr[p].
  - Phase 4 (asm_idx): disjoint asm_idx ranges per panel + read-only final parent front_rows -> race-free.
  - front_ptr prefix-sum + flatten + panel_parent stay serial (cheap).
- Result: multifrontal_symbolic -50..-58% (SyntheticUSA 13.78->5.75, ACTIVSg25k 4.00->2.01, case8387 ->1.33; case6468 stays serial, n small).
  A totals: SyntheticUSA 533->519.5, ACTIVSg25k 134->131. Cumulative A -42ms (562->519.5) across cy259+cy260.
- gpu_test 4/4, berr unchanged (1.2e-14..1.25e-13), F/S identical. Gate-safe. Stacks additively with parND (ordering-independent gpu_mf_analyze).
- NEXT: fill_pattern (26ms, prep) -- the biggest remaining non-METIS chunk, but etree-dependent merge (col[j] reads children's col[c]);
  needs LEVEL-parallel processing + per-thread markers (cy250 only tried flatten-parallel, which doesn't touch the dominant merge).

## cycle 261 (2026-05-28 09:30 KST) — WIN: fill_pattern flat-Li (A vein 3rd win; cumulative A -55ms)
- Biggest remaining non-METIS A chunk (26ms, prep). Level-parallel blocked (etree-dependent merge; deep etree -> per-level
  thread-spawn cost in C++17/no-OpenMP; dominant-spine imbalance cy257). Chose a barrier-free flat rewrite instead:
  - column_counts (cs_counts) -> EXACT Lp prefix-sum (|L(:,j)|==|struct(j)|), Li sized once.
  - Column-merge writes directly into the flat Li slice [Lp[j],Lp[j+1]); marker dedups; flat-CSR etree children.
  - Removes the vector<vector> col/head 156k-small-vector alloc+regrowth churn (the bulk, cf. cy259). Consumers re-sort -> set-identical.
- Result: fill_pattern -70% (SyntheticUSA 26.0->7.7 incl. added column_counts; ACTIVSg25k 7.2->2.3; case6468 1.7->0.6).
  A totals: case6468 33->31.6, case8387 42.6->40.9, ACTIVSg25k 131->129, SyntheticUSA 519.5->506.7.
  CUMULATIVE A 562->506.7 (-55ms, -9.8%) across cy259+260+261.
- gpu_test 4/4, berr unchanged (4.5e-15..5.4e-13), F/S identical. Gate-safe; stacks additively with parND.
- Non-METIS A vein now mostly mined: build_symmetric_filled 18.7 & emit_map 9.7 already parallel; permute 5.9 is a scatter (hard).
  Dominant A cost is the METIS recursion (parND -40% already in production; further = research-grade GPU ordering).

## cycle 262 (2026-05-28 10:00 KST) — WIN: parallelize symmetric permute (completes non-METIS A vein)
- Last un-parallelized non-METIS chunk. permute_sym (production) + permute (bench): the scatter is RACE-FREE across
  source columns -- iperm is a bijection so column c's entries fill target column iperm[c]'s disjoint contiguous slice,
  no other source column touches it. 12-thread fill over c (gated n>=32768) -> disjoint nx[] counters + output slices, byte-identical.
- Result: permute -54% (SyntheticUSA 5.9->2.7, ACTIVSg25k 2->0.8). A SyntheticUSA 506.7->501.0.
  CUMULATIVE A 562->501 (-61ms, -10.9%) across cy259+260+261+262.
- gpu_test 4/4, berr unchanged (3.8e-15..4.7e-13). Gate-safe; stacks with parND.
- NON-METIS A VEIN FULLY MINED: symmetric_pattern(cy181), emit_map(cy185), a_pos(cy186), build_symmetric_filled(cy259),
  multifrontal_symbolic(cy260), fill_pattern(cy261), permute(cy262) all parallel/flat. A floor is now the METIS recursion
  itself (parND -40% in prod; root separator sequential + METIS-internal; GPU-ND wall-clock-neutral cy219-254) = research-grade.

## cycle 263 (2026-05-28 10:30 KST) — MINOR WIN: drop redundant METIS graph copy in parND
- par_nd_rec + base_nodend copied xx(xadj)/aa(adj) at every recursion level solely to hand METIS a mutable buffer.
  But METIS_ComputeVertexSeparator/NodeND are read-only on the graph (copy into internal graph_t), and induce() reads
  the ORIGINAL xadj/adj -> the copy is wasted. Pass xadj/adj directly via const_cast (single-thread path line 238 already does this).
- RIGOR: parND fill is non-deterministic, so verified via 6-run distributions WITH vs WITHOUT the change:
  with mean 1550766 [1541394,1562151], without mean 1549556 [1536636,1556594] -> 0.08% apart = identical (parND noise).
  => METIS does NOT modify the input; the copy was unnecessary; ordering/fill unchanged.
- Result: parND metis_nd ~276->~272ms (-4ms, -1.6%; near noise but consistent) + removes a ~24MB root copy and per-level
  copies (less memory traffic, helps under contention). gpu_test 4/4, berr unchanged.
- A floor is now the ~272ms METIS separator/NodeND computation itself = research-grade (root sep sequential, METIS-internal).

## cycle 264 (2026-05-28 11:00 KST) — WIN: drop build_symmetric_filled sort/dedup (A -67ms cumulative)
- Biggest remaining non-METIS chunk (18.7ms). Questioned the cy259 per-slice sort: S column k = {L col k rows >=k}
  UNION {mirror rows j<k} = two DISJOINT ranges -> union already duplicate-free (dedup = no-op). And BOTH consumers are
  order-INDEPENDENT: emit_map (per-entry owner via fidx into front_rows, no search into Si) + numeric::solve (tests
  i>j/i==j/i<j per entry). So drop sort+dedup+compact entirely: Sp = counting offsets, Si = the flat scatter directly.
- Result: build_symmetric_filled -51% (SyntheticUSA 18.7->9.1, ACTIVSg25k 5.3->2.5, case6468 1.3->0.4).
  A: case6468 33->31.2, case8387 ->40.7, ACTIVSg25k ->125.2, SyntheticUSA 501->494.6. CUMULATIVE A 562->494.6 (-67ms, -12%) cy259-264.
- RIGOR: berr_mf (CPU numeric::solve on the now-unsorted out.Si) = 4e-14..4e-13 all pass (temp-instrumented, reverted)
  -> no duplicates (would double-count -> large berr) + numeric::solve order-independence confirmed. gpu_test 4/4, berr_gs unchanged, F/S identical.
- NON-METIS A VEIN FULLY MINED (6 wins). Floor = the ~272ms METIS separator/NodeND recursion = research-grade.

## cycle 265 (2026-05-28 11:30 KST) — CONFIRMING-NEGATIVE: sort-drop doesn't extend to symmetric_pattern
- Tried the cy264 trick on symmetric_pattern (9.2ms). Consumers (etree idempotent ancestor-walk; fill_pattern own
  marker) ARE order-independent, so the cy181 sort's ordering is unneeded. BUT symmetric_pattern has REAL duplicates
  (structural symmetry double-counts each off-diagonal pair), so dedup is required (cannot drop both like cy264).
- Replaced sort+adjacent-dedup with per-thread MARKER dedup (no sort): MEASURED WORSE -- SyntheticUSA symmetric_pattern
  9.2->9.8ms. The scattered mark[adj[k]] access (156k array > L2) thrashes cache (cy183 redux); the per-slice std::sort
  is cache-LOCAL and faster. Reverted (gpu_test 4/4).
- CONCLUSION: cy264's sort-drop only applies where the structure is already duplicate-free (build_symmetric_filled's
  disjoint >=k / <k ranges). Where dedup is genuinely needed (symmetric_pattern), the cache-local sort is optimal.
  => NON-METIS A VEIN FULLY MINED (cy259-264 = 6 wins, SyntheticUSA A 562->494.6, -12%). A floor = ~272ms METIS recursion (research-grade).

## cycle 266 (2026-05-28 12:00 KST) — WIN: remove vestigial emit_sx (identity) array
- emit_sx[p]=p was set in exactly ONE place and read only as Sx[emit_sx[e]] in mf_emit = identity map. Removed entirely:
  the d_emit_sx array (nnz_s ints, ~12MB on SyntheticUSA), its H2D upload, the arena slot (o_es), and the per-entry
  global-memory indirection in the emit kernel (Sx[emit_sx[e]] -> Sx[e]). Updated hpp struct + move-op + launch.
- Result: arena_malloc+H2D -27% (SyntheticUSA 2.6->1.9, ACTIVSg25k 0.85->0.63), A SyntheticUSA 494.6->488.5,
  ~12MB less GPU memory, F unchanged (4.098), berr unchanged (5.47e-13), gpu_test 4/4. CUMULATIVE A 562->488.5 (-73ms, -13%).
- Gate-safe cleanup (removed dead weight). Non-METIS A vein deeply mined; remaining (emit_map fidx, symmetric_pattern sort)
  are cache-optimal already (cy265). Floor = ~272ms METIS recursion (research-grade).

## cycle 267 (2026-05-28 12:30 KST) — HOLD: dead-weight hunt exhausted, non-METIS A fully mined
- Continued cy266's dead-weight hunt: audited all 14 GpuMfPlan d_ members -> ALL used (assign+H2D+kernel+move). No more
  vestigial arrays. Checked par_nd_rec induce(): 2 edge-scans (count+fill) are necessary for CSR build; only the per-half
  g2l vertex-scan is redundant (tiny). No clean lever left.
- STATUS: non-METIS A vein FULLY MINED. cy259-264 + cy266 = 7 gate-safe wins, SyntheticUSA A 562->488.5 (-73ms, -13%),
  all shared/production code. Floor = ~272ms METIS recursion (research-grade). S = ~2x (research-grade solve internals).
- Both remaining gaps are multi-cycle research efforts (cf. the cy201/cy218 greenlit GPU-ND work). Holding the strong
  standing; will act on any fresh angle or user direction. Minimal bookkeeping; Discord skipped (no win).

## cycle 268 (2026-05-27 22:30 KST) — HOLD: factor well-tuned + F won; survey finds no clean lever (+ health-check sent)
- FACTOR survey: power-grid fronts tiny (mean ~2, max ~91), all < cy147 multiblock threshold (257) -> single-block path,
  tuned by count-guard occupancy (cy143/144). F WINS 4/4 power (spine occupancy not losing). Lowering the multiblock
  threshold to ~91 would add the 3-kernel overhead for sub-threshold fronts (cy147 set 257 deliberately) -> likely regress.
- SOLVE: research-grade; all levers ruled out (cy171 latency-bound, cy257/173 multi-subtree ~1.0x, cy159 coalescing proxy
  -2-4%, cy169 cap, cy154 multi-block bwd). A non-METIS: mined (7 wins). METIS recursion: research-grade.
- No clean lever this cycle. Sent the 2-hourly health-check report. Corrected KST timestamp drift (was +1 day).
- Standing: F won 4/4 power, A -13% (562->488.5), S ~2x, accuracy>=cuDSS. Remaining = uncertain multi-cycle research.

## cycle 269 (2026-05-27 23:00 KST) — HOLD + production validation of the 7-win run
- Ran the production `benchmark` binary (deferred from cy267). mysolver-gpu SUCCEEDS on case_ACTIVSg25k (success=true,
  cold single-call cpu 274.9 / cuda 111.3 ms; cuDSS 75.4 cold) -> cy259-266 changes (incl. cy262 permute_sym) do NOT
  break production. Cold single-call favors cuDSS (cold-A-dominated, known); warm bench (gpu_mf_bench) is the F-win/A-improved metric.
- Re-surveyed solve front-layout: solve reads only L-/U-panels (strided) not the trailing CB, but compacting is bounded
  marginal (cy159 proxy, cy170 FP32-narrow, cy171 latency-bound) + per-factor-copy F-regression risk. No clean lever.
- Factor well-tuned (F won 4/4), solve/METIS research-grade, non-METIS A mined. Standing unchanged. Held; Discord skipped (validation, not a win).

## cycle 270 (2026-05-27 23:30 KST) — WIN: shared-mem tiled big-front trailing (widens power F)
- Profiled the F holdout (circuit_mf_test): onetone2 FACTOR 12.94 vs cuDSS 8.33 (1.55x) = only F loss; deepest (plev=110),
  fill 1.08M. The big-front trailing (mf_bigB_trailing, cy147 multiblock) did NAIVE per-element global reads of the
  L-row/U-col (rank-nc GEMM, L2/BW-bound).
- Rewrote as a SHARED-MEM TILED GEMM: grid.x=ceil(maxfsz/TS=16)^2 output tiles/front (blockIdx.y=front), each block
  stages a TSxnc L-tile + ncxTS U-tile in shared (each panel value reused TS x), blockDim=TS*TS=256, out-of-range tiles early-return.
- Result (3-run stable): power F SyntheticUSA 4.098->4.049, ACTIVSg25k 1.782->1.761 (~-1.2%), rajat15 3.060. gpu_test 4/4, berr unchanged.
- onetone2 stayed NEUTRAL (12.94->12.99) -> its gap is deep-etree (plev=110) serialization + cap8-suboptimal, NOT trailing-BW
  (confirms cy85's shared-staging-neutral extends to the multiblock trailing). So the tiling WIDENS the power F win; onetone2
  F remains the holdout (research-grade deep-etree). Removed unused MF_BIGTILES env.

## cycle 271 (2026-05-28 00:00 KST) — WIN: dynamic shared for the tiled trailing (occupancy)
- cy270's tiled trailing used static __shared__[16][64] (16KB, sized for the never-reached nc=64 cap) -> occupancy capped
  ~3 blocks/SM. Switched to DYNAMIC shared = 2*TS*maxnc doubles (level's max nc): for nc<=8 that's ~2KB -> occupancy
  rises to the thread-limited ~6-8 blocks/SM.
- 3-run-stable F gains: SyntheticUSA 4.049->4.026, onetone2 FACTOR 12.99->12.84, rajat15 3.06->3.03 (ACTIVSg25k ~neutral).
  gpu_test 4/4, berr unchanged. Cumulative cy270+271: SyntheticUSA F 4.098->4.026 (-1.8%), rajat15 3.119->3.033 (-2.8%).
- Widens the F win; onetone2 remains the holdout (1.54x, deep-etree plev=110 serialization = research-grade). Cleaner (shared sized correctly).

## cycle 272 (2026-05-28 00:30 KST) — CONFIRMING-NEGATIVE: TS=16 near-optimal for the tiled trailing
- Parameterized the cy270/271 tile size TS as a runtime env and swept 8/16/24/32 on power F.
  TS=24/32 clearly worse (bigger blocks -> worse occupancy + partial-tile wasted threads). TS=8/16 best & close
  (cross-process cold-clock variance per [[gpu-benchmark-contention]] makes 8-vs-16 indistinguishable).
- The runtime-TS version itself regressed ~1.4% (SyntheticUSA 4.026->4.083) due to lost compile-time unrolling vs the constexpr.
  => keep cy271 constexpr TS=16; runtime knob not worth it. Reverted (gpu_test 4/4, SyntheticUSA F restored to 4.02).
- Big-front trailing tiling is now tuned (cy270 tiling + cy271 dynamic-shared + cy272 TS=16 confirmed). Standing unchanged.

## cycle 273 (2026-05-28 01:00 KST) — MAJOR WIN: multi-block the big-front extend-add -> F 8/8 (onetone2 closed)
- mf_bigC_extend was the ONLY big-front kernel still ONE-BLOCK-PER-FRONT (cy147 multiblocked the trailing, not the extend).
  Its uc*uc atomicAdds are independent (distinct parent slots, low contention) -> trivially multiblockable. Spread across
  gridDim.x = (mxf>=257?128:64) tiles x gridDim.y fronts; each block strides its share; small-uc blocks no-op.
- HUGE consistent gains (3-run stable): SyntheticUSA F 4.026->3.192 (-21%), ACTIVSg25k 1.767->1.537 (-13%), rajat15 3.03->2.86.
  **onetone2 FACTOR 12.84->8.308 vs cuDSS 8.325 = the F HOLDOUT IS CLOSED (now beats cuDSS).**
- Corrects the cy270 hypothesis: onetone2's gap was NOT trailing-BW nor irreducible deep-etree -- it was the EXTEND
  atomics serialized one-block-per-front across its 110 deep spine levels. Multiblocking parallelized them -> -35% factor.
- **F NOW BEATS cuDSS 8/8** (power 4/4 by wider margins + circuits rajat27/memplus/rajat15/onetone2). gpu_test 4/4, berr unchanged. Gate-safe.

## cycle 274 (2026-05-28 01:30 KST) — CONFIRMING-NEGATIVE: cy273 occupancy lever doesn't transfer to the solve
- Tried the cy273 pattern on the SOLVE: split forward into pivot (1 block/front) + MULTI-BLOCK apply (CB rows strided
  across grid.x tiles) for big-front levels (mxf>=81). Correct (gpu_test 4/4) but REGRESSED S everywhere:
  SyntheticUSA 2.540->2.658 (+4.7%), ACTIVSg25k 1.449->1.527, onetone2 3.52->3.86 (+10%), rajat15 1.78->1.96 (+10%).
- CAUSE: solve is LATENCY-bound (cy171, 14% BW), not throughput/occupancy-bound like the factor extend (cy273 atomics).
  Splitting into 2 kernels (more launches -- onetone2 has 110 levels) + the apply's pivot-reload adds overhead without
  hiding the memory-gather latency. Confirms cy154 (multiblock solve negative).
- Reverted (S restored 2.539, gpu_test 4/4). Lesson: cy273's occupancy win is FACTOR-specific; the solve gap is genuinely
  latency-bound = research-grade. F won 8/8; SOLVE (1.3-2.5x) remains the sole research-grade frontier.

## cycle 275 (2026-05-28 02:00 KST) — WIN: tune cy273 extend tile count -> 16 (was 64/128)
- Swept MF_XTILES (cy273 extend-multiblock tile count). MONOTONIC clear trend: 256>128>64>32>24~16 (fewer better);
  optimum 16 (more tiles = atomic contention on the parent front + block-launch overhead; <16 under-parallelizes).
- Set default 16 (was adaptive mxf>=257?128:64). Gains over cy273: SyntheticUSA F 3.189->3.100 (-2.8%),
  ACTIVSg25k 1.539->1.523, onetone2 FACTOR 8.31->7.84 (-5.6%, well under cuDSS 8.33), rajat15 2.86->2.62 (-8.4%).
  gpu_test 4/4, berr unchanged. Cumulative cy273+275: onetone2 12.84->7.84, SyntheticUSA F 4.03->3.10. F beats cuDSS 8/8 wider.

## cycle 276 (2026-05-28 02:30 KST) — HOLD + consolidation: cy273-275 flipped warm F+S to 2/4 wins
- Re-surveyed the solve (cy273 lens): forward-apply atomics looked like a target, but cy274 proved the solve LATENCY-bound
  (multiblock regressed) and all mem-access levers are measured-marginal (cy159/170/258). No clean lever.
- CONSOLIDATION (current gpu_mf_bench, warm F+S = per-iter NR cost): we now WIN F+S 2/4 power:
  case8387 1.361<1.476, SyntheticUSA 5.637<6.524; LOSE case6468 1.038>0.898, ACTIVSg25k 2.972>2.550 (both purely S-gap).
  Pre-cy273 F+S was ~tie/behind on all -> the cy273/275 factor wins flipped it. S is the single remaining lever for full F+S dominance.
- S is latency-bound research-grade (cy274). F won 8/8 wide, A -13%. Held; Discord skipped (no new optimization, recent heavy reporting).

## cycle 277 (2026-05-28 03:00 KST) — CONFIRMING-NEGATIVE: bigger cap hurts both F and S (re-test post-cy273)
- Hypothesis: cy273-275's F headroom lets a bigger panel_cap cut solve LEVELS (less S latency) at affordable fill cost.
- Swept MF_CAP 8/12/16/24. WRONG -- both F and S worsen monotonically. SyntheticUSA F+S: cap8=5.641, 12=6.027, 16=6.566,
  24=7.751; S alone 2.54->2.78->3.08->3.78. S is FILL-WORK-bound: bigger fronts = more solve work/level > the fewer-levels gain.
- Re-confirms cy169 (cap8 optimal) under the new F headroom. Amalgamation cannot help S. cap8 unchanged, gpu_test 4/4.
- Solve gap is genuinely fill-work + latency bound = research-grade. F won 8/8 wide; F+S 2/4 power; A -13%.

## cycle 278 (2026-05-28 03:30 KST) — HOLD + consolidation (refresh COMPETITIVE_SUMMARY)
- Solve incremental space confirmed mined this session: cy274 (multiblock) + cy277 (cap) both regressed; coalescing
  (cy159/258), FP32 (cy170), multi-subtree (cy257) measured-marginal earlier. Untested levers (transposed-L, cp.async
  prefetch) are proxy-marginal / complex research-grade. Factor tuned 8/8; A symbolic mined; METIS research-grade.
- Deliverable: refreshed COMPETITIVE_SUMMARY.md (was stale at cycle 104, said "F 1.1-1.3x behind") to current cy273-277:
  F beats cuDSS 8/8, F+S beats-or-ties ~5/8 (losses purely S), A -13%. No code change; Discord skipped.

## cycle 279 (2026-05-28 04:00 KST) — HOLD: cp.async refuted (solve is critical-path-depth bound)
- Last untested solve latency lever = cp.async prefetch. REFUTED via cy215/cy217 (no implementation): both the CB
  double-buffer (cy215) and the forward pre-gather (cy217) were NEUTRAL because the scattered-gather latency is ALREADY
  hidden by warp/block parallelism + __ldg -- the solve is bound by the SEQUENTIAL CRITICAL-PATH DEPTH (separator chain),
  not gather latency. cp.async only hides gather latency -> also neutral.
- Combined with cy277/cy169 (shortening the path via amalgamation costs more fill than it saves), S is conclusively
  critical-path-depth-bound = research-grade, ORDERING-tied. ALL incremental S levers now exhausted+documented.
- Factor won 8/8. Held the strong standing; Discord skipped. Next genuine S progress needs a research-grade ordering-depth
  or solve-internals effort (multi-cycle, like the cy201 GPU-ND track).

## cycle 281 (2026-05-28 05:00 KST) — CHARACTERIZATION: backward solve dominates S (localizes the gap)
- Fresh fwd/bwd split (MF_SKIP_FWD/BWD): BACKWARD dominates -- SyntheticUSA bwd=1.663 / fwd=0.938 (~1.8x);
  onetone2 bwd=2.194 / fwd=1.346. Backward ALONE (1.663) > cuDSS full solve (1.44) -> S gap localized to the backward.
- ROOT: backward gathers the large scattered CB x-vector (xcb=y[fr[nc+j]]) + warp-shuffle reduction; forward only needs
  nc pivots (shared) + independent atomicAdds. The backward gather+reduce is structurally heavier AND already tuned
  (cy150 tiling, cy188 __ldg, cy215 double-buffer neutral, cy154 multiblock negative). At the tuned limit, not a fixable
  one-block-per-front like cy273's extend. S stays research-grade; future S effort should target the backward gather+reduce.

## cycle 283 (2026-05-28 06:00 KST) — HOLD: [solve-prof] confirms narrow-level chain is the S bottleneck
- Fresh [solve-prof]: SyntheticUSA 72 levels / 58 narrow(<82 fronts) / work_in_narrow=21% / crit-floor=4%.
  The 58 NARROW spine levels are the TIME bottleneck despite 21% of WORK -- sequential + occupancy-starved + latency-bound.
  Multiblock-fill regressed (cy274); count-reduction via amalgamation costs fill (cy277). "Batch tiny fronts" only helps
  the WIDE leaf levels (79% work but occupancy-fast = not bottleneck) -> marginal, ruled out.
- Conclusively: S bound by the deep narrow-level chain (separator depth, ordering-tied) = research-grade. F won 8/8. Held.

## cycle 286 (2026-05-28 07:30 KST) — CONFIRMING-NEGATIVE: MF_CB_TILE=256 optimal (backward gather+reduce bound)
- Zero-risk constexpr sweep of MF_CB_TILE (backward CB-reduction tile): 128/256/512. FLAT -- SyntheticUSA S 128=2.544,
  256=2.540, 512=2.568; ACTIVSg25k 128=1.450/256=1.450/512=1.458. 256 optimal (512 worse = wide-level occupancy hit > spine sync savings).
- Confirms cy150 (256) + cy282 (backward is gather+reduce bound, not tile/sync/occupancy bound). Reverted; gpu_test 4/4.
- Another measured S lever ruled out. F won 8/8; S research-grade (narrow-level chain / backward gather+reduce).

## cycle 301 (2026-05-28 15:00 KST) — S WIN: bigger block for the latency-exposed mid-size narrow spine
- S is latency-bound on narrow spine levels (cy299). level_ts gave mid-size fronts 64 threads (2 warps) -> latency-starved.
  Bumped narrow (cnt<82) mid-size (mx>=64) levels to 128 threads (4 warps) to hide the scattered-gather latency.
  cy174 tested mx>=256 (flat, already 4 warps) but NOT this mid-size-narrow 2->4 warp case.
- Gated mx>=64 (small-front narrow levels keep 64; ungated regressed ACTIVSg25k +0.5%, gating -> -0.6%).
- RESULT (3-run stable): SyntheticUSA S 2.541->2.481 (-2.3%), onetone2 3.522->3.470 (-1.5%), rajat15 1.776->1.756 (-1.1%),
  ACTIVSg25k 1.449->1.440 (-0.6%). ALL improved -- first S win since cy188/189. gpu_test 4/4, berr unchanged, F unaffected. Opt-out MF_NO_TS_SPINE.

## cycle 302 (2026-05-28 15:30 KST) — CONFIRMING: cy301 block sizes optimal at 128 (+ sweep knobs)
- Swept ts_spine (cy301 mid-size-narrow): 128=2.480 < 192=2.531 < 256=2.577 (SyntheticUSA S) -> 128 best (spine fronts
  small; more threads waste). ts_big (mx>=256, onetone2): 128=3.473 ~ 256=3.475 (flat, confirms cy174) < 384=3.669 -> 128 best.
- cy301's 128 is the optimum for both. Added MF_TS_BIG / MF_TS_SPINE env (default 128, zero perf change, sweepable later).
  gpu_test 4/4, berr unchanged. cy301 S win stands. ncu host-blocked (awaiting user enable for deeper micro-tuning).

## cycle 303 (2026-05-28 16:00 KST) — CONFIRMING: cy301 spine cnt threshold optimal at 82 (vein fully tuned)
- Swept ts_spine_cnt: 82=2.483 < 120=2.484 < 164=2.486 < 256=2.492 (SyntheticUSA S) -> 82 (=SM count) best (wider levels
  lose occupancy without latency gain). cy301 spine-warp vein now fully tuned: block 128 (cy302), cnt 82 (cy303), mx>=64.
- Added MF_TS_SPINE_CNT env (default 82). gpu_test 4/4, berr unchanged. cy301 S win stands. Further S needs ncu (host-blocked) or new structural idea.

## cycle 304 (2026-05-28 16:30 KST) — S localization: cy301 was the FORWARD; backward is warp-insensitive
- Re-measured fwd/bwd post-cy301: cy301's S win was almost entirely the FORWARD (0.938->0.862, -8%, warps hid apply
  latency); BACKWARD UNCHANGED (1.663->1.671) = warp-insensitive, still dominant ~67% of S.
- So remaining S gap = the backward (gather+reduce+dependency): coalesced(cy282), warp-insensitive(cy304), double-buffer
  neutral(cy215), multiblock regress(cy154), CB-tile flat(cy286). Needs ncu (blocked) or structural rewrite (no no-ncu idea).
- Forward now small (0.862) -> transposed-L coalescing bounded-small + F-risk, not worth it. ncu re-tested: still blocked. Holding for ncu-enable.

## cycle 305 (2026-05-28 17:00 KST) — nsys diagnostic + cp.async backward = CONFIRMING-NEGATIVE
- nsys --cuda-graph-trace=node (works, unlike ncu): backward kernel GPU duration avg 18us, scales 4us->223us with front
  size = EXECUTION-bound (not dispatch/launch). Forward ~10us. Confirms cy117 (dispatch not the cost).
- Implemented cp.async double-buffer for the multi-tile backward (cy279 refuted cp.async via faulty generalization from
  cy215's MANUAL/blocking double-buffer; cp.async is non-blocking DMA = genuinely different). MEASURED: regresses ALL +~2%
  (SyntheticUSA 2.481->2.536, onetone2 3.470->3.536). Cause: double-buffer doubles xcb shared (2->4KB) -> occupancy hit;
  AND onetone2 (multi-tile target) got worse -> cp.async latency-hiding < shared-occupancy cost. cy279 CONFIRMED by measurement. Reverted (gpu_test 4/4).
- Backward = warp-insensitive (cy304) + cp.async-negative (cy305) + execution-bound (nsys). Needs ncu (host-blocked) or a fundamentally different solve. cy301 win stands.

## cycle 308 (2026-05-28 00:24 KST) — ncu status: perm fixed, but container needs SYS_ADMIN (handoff note)
- ncu ERR_NVGPUCTRPERM GONE (user host NVreg fix worked) but "No kernels were profiled" (even --graph-profiling node).
  Cause (user-identified): container lacks --cap-add=SYS_ADMIN -> ncu can't instrument. Fix needs container restart w/ SYS_ADMIN (ends this session).
- HANDOFF for post-restart session: `ncu --graph-profiling node --kernel-name regex:mf_bwd_level --set full ./gpu_mf_bench`
  (or circuit_mf_test for onetone2). Target the warp-insensitive execution-bound backward (cy305 18us/kernel); read stall
  reasons / occupancy / mem-throughput to guide micro-tuning. cy301 S win stands; F won 8/8; A -13%.
