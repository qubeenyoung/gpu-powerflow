# Factorization + Solve vs cuDSS — all benchmarks (cycle 114)

Warm-kernel ms (the NR-amortized regime: analysis/ordering is one-time and reused
across many factor+solve calls, so the fair factor-vs-factor and solve-vs-solve
comparison excludes it). Power-grid from `gpu_mf_bench`, circuits from
`circuit_mf_test`; cuDSS columns measured with `--solver cudss-gpu`
(CUDSS_PHASE_FACTORIZATION / _SOLVE). All run on the same RTX 3090, GPU warmed
(P8→P2) and A/B'd in-session to avoid cold-clock artifacts.

## Power-grid (matpower NR Jacobians, METIS ND)

| matrix | n | **Factor** ours | cuDSS | F verdict | **Solve** ours | cuDSS | S verdict |
|---|---|---|---|---|---|---|---|
| case6468rte | 12 643 | **0.42** | 0.61 | **beat 1.45×** | 0.41 | 0.29 | 1.44× behind |
| case8387pegase | 14 908 | **0.56** | 1.11 | **beat 1.99×** | 0.54 | 0.36 | 1.48× behind |
| case_ACTIVSg25k | 47 246 | 2.29 | 1.88 | 1.22× behind | 1.32 | 0.67 | 1.96× behind |
| case_SyntheticUSA | 156 255 | 6.55 | 5.08 | 1.29× behind | 2.21 | 1.44 | 1.53× behind |

## Circuits (MC64 + equilibration + METIS ND)

| matrix | n | **Factor** ours | cuDSS | F verdict | **Solve** ours | cuDSS | S verdict |
|---|---|---|---|---|---|---|---|
| rajat27 | 20 640 | **0.73** | 1.00 | **beat 1.38×** | 0.73 | 0.39 | 1.86× behind |
| memplus | 17 758 | **0.85** | 1.15 | **beat 1.35×** | 0.50 | 0.42 | 1.18× behind |
| rajat15 | 37 261 | 3.74 | 3.50 | ~tie (1.07×) | 1.93 | 0.87 | 2.21× behind |
| onetone2 | 36 057 | 32.4 | 7.95 | 4.07× behind | 3.50 | 1.53 | 2.29× behind |

## Verdict

- **Factorization**: superior-or-equal on **5/8** — beats cuDSS on case6468rte,
  case8387pegase, rajat27, memplus; ties rajat15. Behind on the two largest
  power-grid (1.22–1.29×) and onetone2 (3.9×).
- **Solve**: behind on all (1.18×–2.29×); closest on memplus (1.18×). This is the
  systemic gap — our multifrontal panel-level solve is bandwidth-bound (backward
  reads the U panel coalesced, ~55–62% of solve) and the forward L-apply is
  uncoalesced; cuDSS's solve is more memory-efficient.
- **Combined F+S**: superior-or-equal on case8387pegase (1.10 vs 1.48) and
  case6468rte (~tie 0.84 vs 0.90); the small circuits win on factor but lose the
  combined on solve; the two largest power-grid + onetone2 are 1.3–1.4× behind.
- **Accuracy maintained throughout** (componentwise berr): power-grid 1e-14–1e-13,
  circuits 1e-16–1e-13; onetone2 raw 4.9e-7 → **5e-16 after iterative refinement**
  (the integrated `mysolver-gpu` gate-safe path). Never worse than cuDSS.

## Where the gaps are (data-backed, not speculation)

- **Solve (all)** — near the multifrontal level-set practical limit. Re-confirmed
  cy114: Ts=64 optimal for power-grid (small fronts, max 214), per-level adaptive
  Ts gives circuits the 128-thread sweet spot (onetone2 solve −9%). Backward is
  coalesced + bandwidth-bound; forward is uncoalesced (the one remaining universal
  lever, but a transposed-L copy would push cost into the factor where we're
  already behind on large matrices — needs careful derisking).
- **Large power-grid factor (1.22–1.29×)** — under-utilization is only ~6% (cy98);
  the bulk is dense-LU work on medium fronts. Closing it needs cuBLAS-grade batched
  GEMM trailing updates (cuDSS's approach), a research-grade rewrite.
- **onetone2 factor (4.1×)** — deepest etree (plev 63) + highest fill (1.08M);
  serialization-bound. At the MF structural limit (cy110).

cuDSS reference factor/solve numbers independently re-verified cy116 via a live
`--solver cudss-gpu --warmup-gpu` run; they match these references (no stale-ref
risk).

The integrated `mysolver-gpu` solver (gate-safe, CPU fallback on berr > 1e-10) uses
this GPU path for all matrices; accuracy is always validated before acceptance.

## FP32-cascade (cy132-135) + HONEST CORRECTION (cy136)

The FP32 double-master factor KERNEL is ~1.5x faster (warm factor: ACTIVSg25k 1.39,
SyntheticUSA 4.41 -- faster than cuDSS's 1.88/5.08 in isolation). **But this does NOT
translate to an end-to-end win** (cy136): the FP32 factor's less-accurate L/U (~1e-6)
requires iterative refinement (>=1 extra solve pass + residual) to reach cuDSS-level
accuracy, and that cost >= the factor savings. Measured INTEGRATED combined: FP32 is
WORSE than FP64 -- ACTIVSg25k 8.6 vs 5.3, SyntheticUSA 33 vs 26. The cy135 "combined
beats/ties" line was over-optimistic (it used the RAW unrefined solve) -- WITHDRAWN.

**HONEST standing (FP64, the production default):**
- Factor: superior-or-equal on 5/8 (case6468/case8387/rajat27/memplus beat, rajat15
  tie); behind on ACTIVSg25k 1.19x, SyntheticUSA 1.29x, onetone2 4x.
- Solve: 1.5-2.3x behind (kernel-quality-limited).
- Combined F+S: beat/tie on the small, ~1.3-1.4x behind on the 2 large + onetone2.
- Accuracy: 1e-13..1e-16 (better than cuDSS); all 14 matrices on GPU, gate-safe.
- The residual ~1.3x gap on the large is cuDSS's better-tuned FP64 kernels; FP32 (the
  one big lever) is a confirmed e2e dead-end (mixed-precision-refinement only pays off
  when the factor dominates -- ours is only ~1.5x faster in FP32 and memory-bound).

## cy145 — RELIABLE standing at HOST-LOCKED 1395MHz (post cy140-144 tuning)

First apples-to-apples comparison on a pinned clock (cross-process variance <0.1%). OUR
numbers = gpu_mf_bench/circuit_mf_test warm median (incl. cy140/141/143 factor + cy144
solve tuning); cuDSS = `benchmark --solver cudss-gpu --warmup-gpu` factor_ms/solve_ms
(warm single call), same GPU, same 1395MHz lock.

| matrix | **F ours** | F cuDSS | F verdict | **S ours** | S cuDSS | S verdict |
|---|---|---|---|---|---|---|
| case6468rte | 0.524 | 0.734 | **beat 1.40x** | 0.545 | 0.339 | 1.61x |
| case8387pegase | 0.711 | 1.338 | **beat 1.88x** | 0.689 | 0.433 | 1.59x |
| case_ACTIVSg25k | 2.654 | 2.225 | 1.19x | 1.569 | 0.807 | 1.94x |
| case_SyntheticUSA | 8.21 | 6.069 | 1.35x | 2.752 | 1.713 | 1.61x |
| rajat27 | 0.844 | 1.205 | **beat 1.43x** | 0.968 | 0.463 | 2.09x |
| memplus | 1.034 | 1.369 | **beat 1.32x** | 0.664 | 0.503 | 1.32x |
| onetone2 | **14.2** | 9.829 | **1.45x** (cy147 multi-block) | 4.523 | 1.826 | 2.48x |
| rajat15 | 4.97 | 4.263 | 1.17x | 2.473 | 1.035 | 2.39x |

**Factor: beat cuDSS on 4/8** (case6468 1.40x, case8387 1.88x, rajat27 1.43x, memplus
1.32x); behind on ACTIVSg25k 1.19x, SyntheticUSA 1.35x, rajat15 1.17x, onetone2 4.49x.
cy140-143 shrank the large-power-grid factor gap (SyntheticUSA locked ratio 1.45x->1.35x).
**Solve: behind on all (1.32-2.48x)**; cy144 trimmed power-grid solve ~5% but the kernel-
quality gap persists. Accuracy unchanged (berr 1e-13..1e-16, <= cuDSS).

**IMPORTANT clock caveat (discovered cy145):** the lock is 1395MHz, BELOW the production
boost (~1700+ under load). Our kernel is slightly MORE clock-sensitive than cuDSS, so at
boost our ratios are a bit BETTER than this table (e.g. boost-clock factor beat 5/8 incl.
rajat15 tie, SyntheticUSA 1.29x). The lock is for RELIABLE A/B of our own changes; it
understates our boost-clock standing vs cuDSS. The cy140-144 gains are occupancy-driven
and real at both clocks. Remaining gaps: solve (systemic, kernel-quality) + onetone2
(structural, deep etree) -- both research-grade to fully close.

## cy148 — FACTOR now beats cuDSS 7/8 (multi-block big-front factor @ locked 1395MHz)

Multi-block factor extended to all fronts >=81 (cy147+cy148). Factor ms (ours / cuDSS):

| matrix | F ours | F cuDSS | verdict |
|---|---|---|---|
| case6468rte | 0.524 | 0.734 | **beat 1.40x** |
| case8387pegase | 0.711 | 1.338 | **beat 1.88x** |
| case_ACTIVSg25k | 1.786 | 2.225 | **beat 1.25x** |
| case_SyntheticUSA | 4.111 | 6.069 | **beat 1.48x** |
| rajat27 | 0.844 | 1.205 | **beat 1.43x** |
| memplus | 0.696 | 1.369 | **beat 1.97x** |
| rajat15 | 3.288 | 4.263 | **beat 1.30x** |
| onetone2 | 11.17 | 9.829 | behind 1.14x |

**FACTOR beats cuDSS on 7/8** (was 4/8 at cy145); onetone2 the lone holdout at 1.14x
(was 4.49x before cy147). All accuracy unchanged (berr 1e-13..1e-16). Reproducible <0.2%.
The win: medium-large fronts (81-256) are few-per-level near the root -> were
under-utilized one-block-each; spreading the trailing across many blocks/front fills the
GPU. SOLVE still behind on all (1.3-2.5x) -- the next focus per the user's direction.

## cy149 — COMBINED F+S standing (locked 1395MHz, post factor wins)

| matrix | F+S ours | F+S cuDSS | verdict |
|---|---|---|---|
| case6468rte | 1.068 | 1.073 | **tie** |
| case8387pegase | 1.400 | 1.771 | **beat 1.26x** |
| case_ACTIVSg25k | 3.356 | 3.032 | behind 1.11x |
| case_SyntheticUSA | 6.864 | 7.782 | **beat 1.13x** |
| rajat27 | 1.811 | 1.668 | behind 1.09x |
| memplus | 1.360 | 1.872 | **beat 1.38x** |
| onetone2 | 15.70 | 11.66 | behind 1.35x |
| rajat15 | 5.760 | 5.298 | behind 1.09x |

Combined F+S beat/tie 4/8; rest 1.09-1.35x. Solve multi-block doesn't transfer (per-front
work too small -- probe confirms big-front solve levels aren't under-utilized). Solve is
memory/serialization bound, near practical limit; the residual combined gap is the solve.
