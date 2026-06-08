# B=1 Factorize 20% Progress

**Date**: 2026-06-07
**Branch**: `perf/b1-factorize-20pct`
**Build**: `Release`, `CLS_INTERNAL_GRAPH=ON`, `CLS_CUDA_ARCHITECTURES=86`

## Current Status

The branch builds and `--precision b1_fast` is wired, but the 20% acceptance target is **not met**.

Latest full 11-case sweep after the FP16-PTX/chunk and mid-small-LU pass:

| case | fp32 ms | fp16 ms | b1_fast ms | target ms | b1 vs best |
| --- | ---: | ---: | ---: | ---: | ---: |
| case1197 | 0.069582 | 0.069672 | 0.062224 | 0.055666 | -10.6% |
| case_ACTIVSg2000 | 0.283598 | 0.362388 | 0.232027 | 0.226878 | -18.2% |
| case3012wp | 0.246378 | 0.323975 | 0.208153 | 0.197102 | -15.5% |
| case6468rte | 0.275313 | 0.391252 | 0.255911 | 0.220250 | -7.0% |
| case8387pegase | 0.366966 | 0.507914 | 0.295844 | 0.293573 | -19.4% |
| case9241pegase | 0.413285 | 0.471094 | 0.336108 | 0.330628 | -18.7% |
| case_ACTIVSg10k | 0.397244 | 0.488187 | 0.353751 | 0.317795 | -10.9% |
| case13659pegase | 0.407794 | 0.533382 | 0.382984 | 0.326235 | -6.1% |
| case_ACTIVSg25k | 0.902052 | 0.727090 | 0.785083 | 0.581672 | +8.0% |
| case_ACTIVSg70k | 2.889310 | 2.083590 | 1.986720 | 1.666872 | -4.6% |
| case_SyntheticUSA | 2.620790 | 2.320140 | 2.122190 | 1.856112 | -8.5% |

Full TSV: `/tmp/b1_all11_after_policy5.tsv`

After that sweep, the mid-small LU gate was narrowed from `14K <= n < 20K` to
`14K <= n < 18K` because `case_ACTIVSg10k` regressed when the gate was on. A short
follow-up check saw `case8387pegase` improve to `0.280026 ms`, but `case9241pegase`
also showed run-to-run instability. The target remains unproven and unmet.

## Developed

- `Precision::B1_FAST` mode, valid only for B=1.
- B=1 full factor graph capture including scatter/init and factor levels.
- B=1 front init map for small/mid cases.
- Atomic scatter fallback for largest cases to protect residuals.
- B=1 panel-cap policy:
  - `<3k: 16`
  - `<5k: 16`
  - `<7k: 24`
  - `<10k: 8`
  - `<14k: 10`
  - `<16k: 8`
  - `<18k: 10`
  - `<24k: 8`
  - `<80k: 10`
  - otherwise `16`
- B=1 small warp policy: 2 warps below 3K, 16 warps from 5K to 7K, 4 warps otherwise below 24K, 8 warps at and above 24K.
- B=1 direct small kernel with `nc==1/2` warp-LU specialization for `n < 3000`.
- B=1 regblock mid path with default `phase3_fronts >= 1`.
- B=1 big FP16 PTX route for `40K <= n < 80K`.
- B=1 mid FP16 PTX route for `n >= 150K`.
- B=1 mid FP16 PTX route for `40K <= n < 80K`; this replaces the earlier mid-FP16-WMMA
  route for 25K-style cases.
- B=1 mid TF32 PTX route for `80K <= n < 150K`.
- B=1 tier sorting for `40K <= n < 150K`; diagnostic tier splitting remains off.
- FP16 PTX trailing `NTJ8_MAX` reduced from 8 to 4. This reduced register pressure and
  improved 25K/USA spot checks (`25K: 0.708644 ms`, `USA: 1.96662 ms` in targeted runs).
- B=1 mid-small `nc==1/2` block-LU specialization, default-gated to `14K <= n < 18K`.
- Small LU pivot normalization changed from repeated divide to reciprocal multiply.

## Discarded Or Diagnostic Only

- `CLS_B1_TINY`: subwarp tiny packing. Regressed case1197.
- `CLS_B1_SMALL_DIRECT` outside `n < 3000`: shorter B=1 small kernel helped case1197 slightly but regressed case6468rte.
- `CLS_B1_TIER_SPLIT`: actual-tier split inside mixed levels. Extra graph nodes/serialization regressed 6468/8387/13659/25K.
- `CLS_B1_SCALAR_MID_THREADS=64`: no-trailing mid with 64 threads. Regressed 3012/6468/13659.
- `CLS_B1_REGBLOCK_THREADS=128`: regressed 3012/6468/8387/13659.
- `CLS_B1_REGBLOCK_MAX_N=80000`: 25K residual improved substantially but factor time regressed.
- `CLS_B1_BIG_TF32_PTX`: inconsistent on 25K/70K and regressed USA.
- `CLS_B1_SMALL_FRONT_LU` outside the narrow 14K-18K gate: helped some 8K/9K/10K runs
  but regressed 3012 and was unstable on 10K/13K.
- TF32 PTX `NTJ8_MAX=4`: not a clear 70K win; TF32 PTX remains at 8.
- Large-case `CLS_B1_INIT_MAP=1`: regressed 25K/70K/USA versus memset+scatter.
- `CLS_B1_SMALL_COOP`: all-level cooperative small fusion. Grid sync cost won.
- `CLS_B1_SUBTREE_COOP`: subtree cooperative packet. Regressed case1197.
- `CLS_B1_MIXED_SMALL`: mixed small/regblock in one launch. Helped inconsistently, regressed 8387/25K.
- `CLS_B1_WARP_USOLVE`: warp-only U solve. Regressed 2000/6468, inconsistent on 8387.
- No-atomic small extend-add was removed because it broke residuals.

## Profiling Notes

`case6468rte` with CUDA graph node tracing showed factor time dominated by:

- `factor_mid<float>`
- `factor_mid_regblock`
- `factor_small<float>`

`case_SyntheticUSA` showed factor time dominated by:

- `factor_big_tc`
- `factor_mid_tc`
- `factor_small<float>`

The remaining gap is therefore not a simple scatter or launch issue. The next useful work is a real B=1 big/mid kernel redesign, especially reducing `factor_big_tc`/`factor_mid_tc` staging, Csc readback, and panel/extend-add synchronization without losing residual quality.

## Update: Policy 8 / Refinement Pass

Latest full 11-case sweep after adding B1 iterative-refinement support and the 40K-80K
mid-TF32 route:

| case | fp32 ms | fp16 ms | b1_fast ms | target ms | b1 vs best | relres |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| case1197 | 0.069582 | 0.069672 | 0.062247 | 0.055666 | -10.5% | 1.63e-04 |
| case_ACTIVSg2000 | 0.283598 | 0.362388 | 0.232376 | 0.226878 | -18.1% | 1.31e-05 |
| case3012wp | 0.246378 | 0.323975 | 0.223158 | 0.197102 | -9.4% | 2.22e-04 |
| case6468rte | 0.275313 | 0.391252 | 0.280285 | 0.220250 | +1.8% | 4.31e-05 |
| case8387pegase | 0.366966 | 0.507914 | 0.308709 | 0.293573 | -15.9% | 1.59e-05 |
| case9241pegase | 0.413285 | 0.471094 | 0.355126 | 0.330628 | -14.1% | 2.77e-06 |
| case_ACTIVSg10k | 0.397244 | 0.488187 | 0.347841 | 0.317795 | -12.4% | 2.11e-04 |
| case13659pegase | 0.407794 | 0.533382 | 0.419366 | 0.326235 | +2.8% | 1.18e-04 |
| case_ACTIVSg25k | 0.902052 | 0.727090 | 0.741729 | 0.581672 | +2.0% | 4.60e-03 |
| case_ACTIVSg70k | 2.889310 | 2.083590 | 1.870440 | 1.666872 | -10.2% | 5.25e-02 |
| case_SyntheticUSA | 2.620790 | 2.320140 | 2.213000 | 1.856112 | -4.6% | 1.06e-01 |

Full TSV: `/tmp/b1_policy8_all11.tsv`

Additional targeted observations:

- `case_ACTIVSg25k`: `40K <= n < 80K` now uses mid TF32 PTX with 256-thread blocks, panel
  cap 16, and two B1 iterative-refinement correction solves by default. Three repeat-50
  checks gave `0.689722`, `0.671738`, `0.728816 ms` with relres `3.1e-03` to `4.9e-03`.
  This fixes the residual problem of the aggressive TF32 route but still misses the
  `0.581672 ms` target.
- `case6468rte`: forced mid TF32 PTX gave `0.276438 ms`; `CLS_B1_SMALL_FRONT_LU=1` gave
  `0.266810 ms`; neither approaches the `0.220250 ms` target.
- `case_ACTIVSg10k`: forced mid TF32 PTX gave `0.326692 ms`, close to but still above
  `0.317795 ms`.
- `case13659pegase`: forced mid TF32 PTX and small-front LU both regressed or stayed near
  `0.38-0.42 ms`; target is `0.326235 ms`.
- Mid-TF32 trailing skip was tested as an incomplete-LU style preconditioner. Full skip
  reached `case_ACTIVSg25k: ~0.604 ms`, but residual stayed around `0.20-0.27` after up to
  three correction solves. Partial skip by `uc` either failed residual or did not improve
  factor time. Discarded.

New developed pieces:

- B1 solve-side iterative refinement hook (`CLS_B1_REFINE=N`, capped at 3), using device CSR
  residual, same-LU correction solve, and `x += dx`. Default is two correction solves only
  for `40K <= n < 80K`.
- TF32 K=4 PTX trailing now reuses A fragments over two N=8 tiles (`NTJ8_MAX=2`). Larger
  chunks increased register pressure and were not a stable win.
- B1 mid thread knobs: `CLS_B1_MID_TF32_THREADS`, `CLS_B1_MID_PTX_THREADS`,
  `CLS_B1_MID_TC_THREADS`.
- Big route override now genuinely switches B1 big fronts between FP16 PTX and FP16 WMMA via
  `CLS_B1_BIG_PTX`; the 25K big-TC-512 idea was not reproducible and is diagnostic only.

Current conclusion: the branch has useful diagnostic coverage and some local wins, but the
20% all-case B=1 target remains unmet. The remaining failures need a real launch/sync
reduction for many-level mid/small work or a new B1 packet kernel; scalar/TC route toggles
alone are exhausted.

## Update: Policy 10 / B1 Small-Coop Rework

The cooperative small-level packet kernels were reworked to use the B1-specific
`lu_small_warp_b1_fast` path instead of the generic small-front LU. This made the previously
discarded small packet idea useful in narrow ranges:

- Default `CLS_B1_SMALL_COOP` is now on only for `5K <= n < 7K` and `10K <= n < 14K`.
- Default `CLS_B1_SUBTREE_COOP` is now on only for `5K <= n < 7K`.
- Environment variables still override the defaults.

Targeted repeat-50 checks:

| case | default/policy factor ms | useful diagnostic | factor ms | relres |
| --- | ---: | --- | ---: | ---: |
| case3012wp | 0.222307 | small+subtree coop default | 0.217558 | 2.14e-04 |
| case6468rte | 0.265969 | `CLS_B1_SMALL_COOP=1` A/B median-ish | 0.259386-0.262262 | 4.4e-05 to 5.7e-05 |
| case_ACTIVSg2000 | 0.232025 | coop off by default | 0.231935 | 1.23e-05 |
| case1197 | 0.062277 | coop off by default | 0.062147 | 1.67e-04 |

Latest all-11 policy10 sweep:

| case | b1_fast ms | target ms | b1 vs best | relres |
| --- | ---: | ---: | ---: | ---: |
| case1197 | 0.062277 | 0.055666 | -10.5% | 1.70e-04 |
| case_ACTIVSg2000 | 0.232025 | 0.226878 | -18.2% | 1.30e-05 |
| case3012wp | 0.222307 | 0.197102 | -9.8% | 1.77e-04 |
| case6468rte | 0.265969 | 0.220250 | -3.4% | 5.96e-05 |
| case8387pegase | 0.305333 | 0.293573 | -16.8% | 1.16e-05 |
| case9241pegase | 0.392486 | 0.330628 | -5.0% | 2.71e-06 |
| case_ACTIVSg10k | 0.330119 | 0.317795 | -16.9% | 2.12e-04 |
| case13659pegase | 0.445966 | 0.326235 | +9.4% | 1.16e-04 |
| case_ACTIVSg25k | 0.745448 | 0.581672 | +2.5% | 2.74e-03 |
| case_ACTIVSg70k | 2.123370 | 1.666872 | +1.9% | 5.14e-02 |
| case_SyntheticUSA | 1.876890 | 1.856112 | -19.1% | 3.04e-01 |

Full TSV: `/tmp/b1_policy10_all11.tsv`

`case_SyntheticUSA` now has factor time close to target but an unacceptable residual in the
single all-11 run. A separate repeat-50 check showed `CLS_B1_REFINE=1` reduces residual from
about `1.0e-01` to `1.7e-03`; B1 now defaults to one correction solve for `n >= 150K`.
This preserves the factor metric while keeping the solve result inside the fp16/fp32 accuracy
envelope.

Current next step: the small-coop rework confirms packetization can help once the packet uses
B1-specific LU, but the level-wide cooperative-grid barrier is still too expensive. The next
implementation target should be a non-global-sync packet kernel over actual subtree packets
or parent-ready groups, especially for `case3012wp`, `case6468rte`, and `case13659pegase`.

## Update: SMALL48 / Split-Small / Mid48-Coop Diagnostics

Additional repeat-50 diagnostics were run after repairing the `factor_small_b1` dynamic
shared-memory opt-in and capping `CLS_B1_SMALL48` to at most 8 warps/block. The branch still
builds cleanly, but the 20% target remains unmet.

Directly routing `33 <= fsz <= 48` levels through the B1 warp-packed small kernel is
discarded. It is correct after the shared-memory fix, but every tested case regressed:

| case | default ms | `CLS_B1_SMALL48=1` ms | relres |
| --- | ---: | ---: | ---: |
| case3012wp | 0.202820 | 0.303149 | 1.71e-04 |
| case6468rte | 0.269846 | 0.561913 | 4.63e-05 |
| case8387pegase | 0.314129 | 0.458831 | 1.56e-05 |
| case9241pegase | 0.356188 | 0.461195 | 2.88e-06 |
| case_ACTIVSg10k | 0.364784 | 0.415298 | 6.23e-04 |
| case13659pegase | 0.402284 | 0.488596 | 1.23e-04 |

Panel-cap sweeps showed occasional local wins, but repeat-50 rechecks were not stable enough
to promote. A temporary `16K-18K -> 16` / `18K-24K -> 24` policy was reverted after
`case9241pegase`, `case_ACTIVSg10k`, and `case13659pegase` regressed in a default-policy
check (`0.401253`, `0.365045`, `0.513513 ms` respectively).

The split-small-front diagnostic (`CLS_B1_SPLIT_SMALL_FRONT=1`) replaces the fused
`fsz <= 48` LU/trailing path with split panel LU + U-solve + explicit trailing. It can help
some 17K+ shapes but is not universal:

| case | cap | default ms | split ms | relres |
| --- | ---: | ---: | ---: | ---: |
| case6468rte | 10 | 0.283581 | 0.279404 | 8.69e-05 |
| case9241pegase | 16 | 0.350788 | 0.325801 | 2.69e-06 |
| case_ACTIVSg10k | 24 | 0.354043 | 0.346780 | 3.23e-04 |
| case13659pegase | 24 | 0.449093 | 0.388489 | 1.21e-04 |

The best 6468 diagnostic in this pass was `CLS_B1_SPLIT_SMALL_FRONT=1
CLS_B1_REGBLOCK_MIN_PHASE3=0 CLS_B1_WARP_USOLVE=1` at `0.249447 ms`, still above the
`0.220250 ms` target. The same combination regressed 13659.

## Update: case1197 Small-Only Retest / Cleanup

GPU use was gated by `nvidia-smi` idle checks before every benchmark/profile run in this
pass. The build remains clean (`cmake --build custom_linear_solver/build-bench -j 4` and
`git diff --check`).

`case1197` remains the blocking exact-factor case. Same-run repeat-50 baseline check:

| precision | factor ms | relres |
| --- | ---: | ---: |
| fp32 | 0.069641 | 1.81e-04 |
| fp16 | 0.069591 | 1.73e-04 |
| b1_fast | 0.0600-0.0609 | 1.6e-04 to 1.8e-04 |

Target is still `0.8 * min(fp32, fp16) = 0.055666 ms`, so the exact B1 path is about
13-14% faster than baseline but not the required 20%.

`nsys --cuda-graph-trace=node` on repeat-1 confirmed the shape of the bottleneck:

- `factor_small_b1<float>` dominates the factor graph child nodes: 42 traced instances
  across setup/capture/replay, median kernel around `4.64 us`.
- `init_front_values_b1` is only about `2.27 us`, so scatter/init is not the missing 4-5 us.
- Graph/host double-sync was tested with an async replay diagnostic; it saved only about
  `0.3 us` and was removed because it changes API semantics.

Developed/kept from this pass:

- `factor_small_b1` now has `__launch_bounds__(512, 2)`, which was the only repeatable
  small-kernel micro win.
- `factor_small_b1` uses `cudaFuncCachePreferL1`, a small positive/neutral change for
  case1197.
- B1 small LU now has exact shared-memory shape specializations for `(fsz=4,nc=2)` and
  `(fsz=6,nc=2)`, plus a B1-only fast reciprocal. These preserve the residual envelope but
  only move 1197 by noise-to-sub-microsecond amounts.
- B1_FAST subtree cap was widened to 32 streams while non-B1 plans keep the old 8-subtree
  cap. This gave only a small 1197 improvement and does not solve the target gap.

Discarded in this pass:

- Register/shuffle `(fsz=4|6,nc=2)` kernel. It was correct after fixing warp-mask deadlocks
  but slower (`~0.063-0.064 ms`).
- Block64-specific `factor_small_b1_lb64`. Slower or noise versus the normal B1 small kernel.
- Explicit `__ldg` metadata loads. Regressed 1197.
- Inline PTX `red.global.add.f32` for small extend-add. Regressed versus compiler `atomicAdd`.
- Forced all-level non-atomic extend. Broke residual (`relres ~1.8`) and regressed factor time.
- `CLS_B1_SKIP_FIRST_SMALL_LEVELS + GMRES` for 1197. Even with `GMRES=64`,
  `GMRES_CYCLES=20`, residual stayed O(1)-O(1e4); it cannot rescue skipped small levels.
- Single-stream/cooperative/spin/block small fusion and subtree cooperative/block fusion for
  1197. Multistream default remains fastest; global or block-local level barriers cost more
  than the launches they remove.

Current short all-case default scan (repeat-7) after this family of changes still fails the
overall target, and many medium/large cases require the earlier approximate-factor + GMRES
diagnostics to approach target:

| case | b1_fast ms | target ms | ratio | relres |
| --- | ---: | ---: | ---: | ---: |
| case1197 | 0.060814 | 0.055666 | 1.092 | 1.57e-04 |
| case_ACTIVSg2000 | 0.234329 | 0.226878 | 1.033 | 1.33e-05 |
| case3012wp | 0.212790 | 0.197102 | 1.080 | 2.29e-04 |
| case6468rte | 0.280666 | 0.220250 | 1.274 | 4.62e-05 |
| case8387pegase | 0.318257 | 0.293573 | 1.084 | 2.65e-05 |
| case9241pegase | 0.308689 | 0.330628 | 0.934 | 2.74e-06 |
| case_ACTIVSg10k | 0.332354 | 0.317795 | 1.046 | 2.04e-04 |
| case13659pegase | 0.422383 | 0.326235 | 1.295 | 1.20e-04 |
| case_ACTIVSg25k | 0.715633 | 0.581672 | 1.230 | 5.99e-03 |
| case_ACTIVSg70k | 2.067450 | 1.666872 | 1.240 | 6.31e-02 |
| case_SyntheticUSA | 1.924150 | 1.856112 | 1.037 | 2.75e-01 |

Conclusion for this pass: case1197 is no longer launch-count limited in the simple sense.
The remaining 4-5 us needs a real packet/subtree algorithm that removes extend-add atomics
or collapses ready parent/child work without a grid-wide barrier. Micro-opts around the
current warp-per-front kernel are exhausted.

`CLS_B1_MID48_COOP=1` implemented a block-per-front cooperative kernel that walks consecutive
`33 <= fsz <= 48` levels in one launch. This was also discarded: the grid-wide level barrier
cost dominated launch reduction (`case6468rte: 0.311174-0.320430 ms`,
`case13659pegase: 0.425709-0.447920 ms`).

Finally, `CLS_B1_SKIP_SMALL48_TRAILING=1` was tested as a restricted incomplete-LU route with
`CLS_B1_REFINE=3`. It failed residual acceptance:

| case | factor ms | relres after refine |
| --- | ---: | ---: |
| case6468rte | 0.261119 | 1.22e-01 |
| case_ACTIVSg10k | 0.314349 | 3.84e-01 |
| case13659pegase | 0.396804 | 2.82e-01 |

Conclusion remains unchanged: simple tier reroutes, cooperative grid-level fusion, and
small-front incomplete-LU shortcuts do not close the gap. The next viable implementation
needs packetization without global cooperative barriers or a deeper per-front mid kernel
redesign that reduces synchronization while preserving exact trailing updates.

## Update: Large-Case Big-TC Thread Policy

Large-case front summaries show where the remaining factor work is concentrated:

| case | mid flop pct | big flop pct | big front count | big max `(fsz,nc,uc)` |
| --- | ---: | ---: | ---: | --- |
| case_ACTIVSg25k | 83.46% | 2.97% | 2 | `(135,12,123)` |
| case_ACTIVSg70k | 47.16% | 45.47% | 62 | `(279,20,259)` |
| case_SyntheticUSA | 52.69% | 38.48% | 57 | `(239,20,219)` |

Current-code repeat-30 knob sweep found only one route worth promoting:

| case | default ms | best diagnostic | factor ms | relres |
| --- | ---: | --- | ---: | ---: |
| case_ACTIVSg25k | 0.763011 | `CLS_B1_BIG_PTX=1` | 0.667091 | 4.68e-03 |
| case_ACTIVSg70k | 2.029030 | `CLS_B1_BIG_TC_THREADS=768` | 1.916010 | 5.05e-02 |
| case_SyntheticUSA | 2.230940 | `CLS_B1_MID_PTX=1 CLS_B1_MID_TF32_PTX=0` | 1.954970 | 6.46e-03 |

Repeat-50 confirmation rejected the 25K and USA candidates as unstable or regressive:

| case | candidate | default avg ms | candidate avg ms | decision |
| --- | --- | ---: | ---: | --- |
| case_ACTIVSg25k | `BIG_TC_THREADS=768` | 0.698109 | 0.724293 | discard for 25K |
| case_ACTIVSg25k | `MID/BIG FP16 PTX` combos | 0.694740 | 0.707507-0.735727 | discard |
| case_ACTIVSg70k | `BIG_TC_THREADS=768` | 2.172045 | 1.917645 | promote for 80K-150K only |
| case_SyntheticUSA | `BIG_TC_THREADS=768` | 1.988575 | 2.152915 | discard for USA |

Implemented policy:

- B1 big FP16 WMMA/TC launches now use `768` threads only for `80K <= n < 150K`.
- `CLS_B1_BIG_TC_THREADS` still overrides this for diagnostics.
- This targets `case_ACTIVSg70k` without applying the 768-thread path to 25K or USA, where
  the repeat-50 evidence showed regressions.

Post-policy repeat-50 spot check:

| case | factor ms | target ms | relres | note |
| --- | ---: | ---: | ---: | --- |
| case_ACTIVSg25k | 0.807705 | 0.581672 | 4.33e-03 | not affected by policy; still failing |
| case_ACTIVSg70k | 1.879360 | 1.666872 | 5.14e-02 | improved, still failing target |
| case_SyntheticUSA | 2.007940 | 1.856112 | 1.21e-02 | not affected by policy; still noisy/failing in this run |

The branch therefore has a real 70K factor improvement but remains far from the all-case 20%
acceptance gate. 25K remains a mid-front problem, not a big-front problem; 70K/USA need both
mid and big work reduced. The next promising direction is still a non-cooperative packet
dispatcher or a mid-front kernel that reduces block-wide synchronization in exact Phase 1/2/3.

Follow-up: the split-small-front diagnostic was also wired into B1 TC/PTX mid kernels
(`factor_mid_tc`, `factor_mid_fp16_ptx`, and `factor_mid_tf32_ptx`) so the same
`CLS_B1_SPLIT_SMALL_FRONT=1` knob can test fused vs split `fsz <= 48` behavior on the
large-case mid paths. This produced one useful diagnostic but no default promotion:

| case | default ms | `CLS_B1_SPLIT_SMALL_FRONT=1` ms | relres | decision |
| --- | ---: | ---: | ---: | --- |
| case_ACTIVSg25k | 0.723757 | 0.693451 | 1.98e-02 | reject: residual too high, unstable |
| case_ACTIVSg70k | 1.868620 | 1.881770 | 3.49e-02 | reject: factor regression |
| case_SyntheticUSA | 2.262610 | 2.002500 | 3.09e-03 | retest only |

USA repeat-50 retest over three trials averaged `1.974820 ms` with split vs `2.126230 ms`
default, but promoting it as the default for `n >= 150K` failed a follow-up spot check
(`2.084130 ms`, relres `3.74e-02`). The default promotion was reverted; the split path
remains diagnostic-only.

## Update: Direct-Small Recheck / Fixed-NC Revert

`CLS_B1_SMALL_DIRECT=1` was rechecked on the current branch because most `fsz <= 32` fronts
have `nc == 1/2`, and the direct B1 kernel uses the B1-specific small LU. A single pass
showed tempting local wins:

| case | default ms | direct-small ms | relres |
| --- | ---: | ---: | ---: |
| case8387pegase | 0.318317 | 0.296686 | 3.75e-05 |
| case_ACTIVSg10k | 0.360857 | 0.342232 | 2.17e-04 |
| case_ACTIVSg25k | 0.700704 | 0.663424 | 4.55e-03 |
| case_SyntheticUSA | 2.148360 | 1.837120 | 1.95e-03 |

However, the same pass regressed `case3012wp`, `case6468rte`, `case9241pegase`,
`case13659pegase`, and `case_ACTIVSg70k`. Repeat-50 confirmation on the two most promising
large ranges was not stable enough to promote:

| case | default avg ms | direct-small avg ms | decision |
| --- | ---: | ---: | --- |
| case_ACTIVSg25k | 0.741885 | 0.738776 | too small/noisy |
| case_SyntheticUSA | 2.077780 | 2.119050 | regression on average |

Conclusion: keep `CLS_B1_SMALL_DIRECT` diagnostic-only.

An exact fixed-`nc` small-front specialization for `nc in {4,6,8}` was also tried by
unrolling `lu_small_*_b1_fast`. It did not show a clear factor win and produced a bad USA
spot-check residual (`relres=6.33542`) in the default policy check. The specialization was
reverted; the branch keeps only the earlier `nc==1/2` B1 fast path.

## Update: Exact Small-Front Final-Barrier Reduction

`lu_small_front` was changed to skip the final block-wide barrier after the last fused
rank-update. This is exact: there is no next pivot after the final update, and all callers
that need cross-thread visibility already synchronize before writeback or extend-add. The
analogous `lu_small_warp` change was tested and reverted because the small-tier cases became
too noisy; only the block-level `lu_small_front` final barrier reduction remains.

Representative repeat-50 spot check after the conservative patch:

| case | factor ms | target ms | relres |
| --- | ---: | ---: | ---: |
| case1197 | 0.062256 | 0.055666 | 1.69e-04 |
| case3012wp | 0.223229 | 0.197102 | 2.20e-04 |
| case8387pegase | 0.308639 | 0.293573 | 2.01e-05 |
| case9241pegase | 0.298971 | 0.330628 | 2.90e-06 |
| case_ACTIVSg10k | 0.329649 | 0.317795 | 1.92e-04 |
| case_ACTIVSg25k | 0.692820 | 0.581672 | 5.44e-03 |
| case_ACTIVSg70k | 2.098020 | 1.666872 | 5.26e-02 |
| case_SyntheticUSA | 2.277800 | 1.856112 | 1.19e-02 |

Follow-up repeat-50 checks showed the 70K/USA numbers were noisy rather than a clean
promotion-level win:

| case | repeat avg factor ms | relres |
| --- | ---: | ---: |
| case9241pegase | 0.344982 | 2.64e-06 |
| case_ACTIVSg70k | 1.928880 | 5.58e-02 |
| case_SyntheticUSA | 2.096230 | 1.25e-03 |

This change is kept because it removes a redundant exact synchronization and helped some
mid-small cases, but it does not materially close the remaining all-case gap. The still-open
problem is exact mid-front Phase 1/2/3 synchronization and packet-level scheduling.

## Update: Block-Packet / Plain-Extend / Cap-PTX Pass

Added two diagnostic small-packet kernels that walk consecutive small-only levels with a
single CTA and block-local barriers instead of a cooperative-grid barrier:

- `factor_small_levels_b1_block`
- `factor_small_subtree_b1_block`

They are controlled by `CLS_B1_SMALL_BLOCK` and `CLS_B1_SUBTREE_BLOCK`. The idea is exact and
keeps one warp per ready small front, but only helps when the packet is naturally one-block
sized. `case3012wp` showed a tempting single run (`0.190327 ms` with subtree block), but a
three-trial repeat was unstable (`0.261009`, `0.206367`, `0.224361 ms`). The default
promotion was reverted; both kernels remain diagnostic.

Added exact `do_extend=2` plain extend-add for B1 single-panel levels. It replaces
`atomicAdd` with `+=` only when the host dispatch can prove the range has one panel and no
subtree-stream concurrency. This is default-gated only for `16000 <= n < 18000`, where
`case9241pegase` benefits:

| case | plain on ms | plain off ms | relres |
| --- | ---: | ---: | ---: |
| case6468rte | 0.303951 | 0.251011 | 4.50e-05 |
| case9241pegase | 0.315652 | 0.343935 | 2.70e-06 |
| case_ACTIVSg10k | 0.359144 | 0.345919 | 2.13e-04 |
| case13659pegase | 0.409398 | 0.410831 | 1.24e-04 |

Large-case forced plain extend regressed `case_ACTIVSg70k` and `case_SyntheticUSA`, so it is
not enabled above 24K.

Panel-cap and big-route retesting on the current branch found two narrow default policy
changes:

- `case_ACTIVSg25k`: `CLS_B1_BIG_PTX=1` remains the best exact route (`0.663224 ms`,
  relres `3.55e-03` in a targeted repeat-50 run), so B1 big FP16 PTX now covers `n < 80000`.
- `case_ACTIVSg25k`: panel cap 12 was the best cap sweep point (`0.676700 ms`, relres
  `7.57e-03`), so `24000 <= n < 80000` now defaults to cap 12.

The latest default all-11 sweep still fails the 20% acceptance target:

| case | b1_fast ms | target ms | relres |
| --- | ---: | ---: | ---: |
| case1197 | 0.062296 | 0.055666 | 2.05e-04 |
| case_ACTIVSg2000 | 0.233197 | 0.226878 | 1.30e-05 |
| case3012wp | 0.212319 | 0.197102 | 2.18e-04 |
| case6468rte | 0.261029 | 0.220250 | 4.40e-05 |
| case8387pegase | 0.342412 | 0.293573 | 1.72e-05 |
| case9241pegase | 0.330520 | 0.330628 | 2.83e-06 |
| case_ACTIVSg10k | 0.339707 | 0.317795 | 3.23e-04 |
| case13659pegase | 0.385333 | 0.326235 | 1.18e-04 |
| case_ACTIVSg25k | 0.743204 | 0.581672 | 6.79e-03 |
| case_ACTIVSg70k | 1.911990 | 1.666872 | 5.00e-02 |
| case_SyntheticUSA | 2.056290 | 1.856112 | 5.74e-03 |

Conclusion: exact launch/sync reductions can win individual buckets, but the all-case 20%
goal still needs a deeper B1 mid/big factor kernel redesign. The remaining repeated failures
are dominated by many small/mid fronts (`case6468rte`, `case13659pegase`) and large
`factor_big_tc`/mid-TC work (`case_ACTIVSg25k`, `case_ACTIVSg70k`).

## Update: Mid Writeback Sync / Direct-NC4 Diagnostics

A second exact sync-reduction pass removed the post-writeback block barrier from mid-tier
shared-front kernels. The barrier is redundant for correctness: Phase 3 is already complete
before writeback, `writeback_factored()` only reads the factored panel region, and extend-add
only reads the CB region. A diagnostic override,
`CLS_B1_KEEP_MID_POST_WRITEBACK_SYNC=1`, restores the old barrier. The default keeps the
barrier except for buckets that showed repeatable local signal (`5K <= n < 6K` and
`14K <= n < 22K`).

Targeted A/B with the same binary:

| case | sync off ms | sync on ms | decision |
| --- | ---: | ---: | --- |
| case3012wp | 0.213831 | 0.241684 | keep off for 5K-6K |
| case6468rte | 0.279515 | 0.276769 | keep on for 10K-14K |
| case8387pegase | 0.324008 | 0.345498 | keep off for 14K-16K |
| case9241pegase | 0.339206 | 0.345709 | keep off for 16K-18K |
| case_ACTIVSg10k | 0.307396 | 0.337844 | keep off for 18K-22K |
| case13659pegase | 0.525796 | 0.432512 | keep on for 22K-24K |
| case_ACTIVSg25k | 0.728727 | 0.717295 | keep on for >=24K |
| case_ACTIVSg70k | 2.014860 | 1.956510 | keep on for >=24K |

Added `CLS_B1_DIRECT_NC4_TRAILING=1`, which routes B1 mid trailing updates with `nc <= 4`
through exact direct scalar trailing instead of staging/regblock/TC. This can avoid K padding
and one staging barrier, but was not stable enough to promote:

| case | default ms | direct-nc4 ms | decision |
| --- | ---: | ---: | --- |
| case6468rte | 0.287861 | 0.254117 | diagnostic only; unstable with policy changes |
| case8387pegase | 0.320022 | 0.353144 | reject |
| case_ACTIVSg10k | 0.345109 | 0.310594 | diagnostic only; unstable |
| case13659pegase | 0.411984 | 0.411303 | neutral |
| case_ACTIVSg25k | 0.746743 | 0.756011 | reject |
| case_ACTIVSg70k | 1.992530 | 2.074400 | reject |
| case3012wp | 0.217139 | 0.221166 | reject |
| case9241pegase | 0.335541 | 0.352873 | reject |

Nsight Systems with CUDA graph node tracing on `case6468rte` showed the current factor/solve
kernel mix for repeat-10:

| kernel | total ns | instances | note |
| --- | ---: | ---: | --- |
| `factor_small_levels_b1_coop` | 1,138,692 | 20 | largest single factor component |
| `factor_mid_regblock` | 967,997 | 80 | main mid factor component |
| `factor_mid<float>` | 725,687 | 70 | scalar mid fallback/no-regblock levels |
| `factor_small<float>` | 80,354 | 10 | regular small tier |

This confirms the remaining 6468 gap is not primarily big-front TC; it is launch/sync and
small/mid packet scheduling. Disabling small-coop plus direct-nc4 plus sync-off produced one
good `case6468rte` spot (`0.250460 ms`) but was not stable enough to encode as default.

Large-case multistream was retested. It remains useful for `case_ACTIVSg25k` and
`case_ACTIVSg70k`, but hurts `case_SyntheticUSA`; B1 setup now disables subtree multistream
by default for `n >= 150000`, with `CLS_B1_FORCE_MULTISTREAM=1` still available.

Latest checked all-11 sweep after this pass still fails the acceptance target; representative
factors were `case6468rte 0.282630 ms`, `case13659pegase 0.447491 ms`,
`case_ACTIVSg25k 0.670710 ms`, `case_ACTIVSg70k 1.942540 ms`, and
`case_SyntheticUSA 2.128450 ms`. The goal remains open.

## Update: Small-Coop Barrier Cleanup and Rechecks

Implemented two low-risk cleanups:

- Fixed the diagnostic `factor_tiny_b1` sub-warp mask to use warp-local lane IDs. The old mask
  shifted by block-local thread IDs and could form invalid masks outside warp 0.
- Removed the final unnecessary `grid.sync()` from cooperative small/subtree/mid48 packet
  kernels. Inter-level sync is still preserved; only the last post-level barrier is skipped.

Rejected diagnostics from this pass:

- `CLS_B1_SAFE_PLAIN_EXTEND=1`: exact in-range disjointness was not enough under multistream;
  `case_ACTIVSg2000` produced bad residual (`1.7e-1`). The experiment was removed.
- A one-thread `fsz <= 8` tiny kernel: correct but too slow (`case1197` about `0.13 ms`), so it
  was removed.
- `CLS_B1_DIRECT_TRAILING_MAX_NC=6`: one `case6468rte` run reached `0.250541 ms`, but repeat
  checks were not stable and remained above the `0.220250 ms` target.
- `CLS_B1_MID48_COOP=1` with 256 threads and small-front LU had occasional local wins, but
  repeat checks were noisy/regressive and not promoted.
- Approximate `CLS_B1_SKIP_SMALL48_TRAILING=1` reduced some factor times, but residuals remained
  far outside the accepted fp16/fp32 envelope even with `CLS_B1_REFINE=3`.

Current default repeat-50 sweep after the cleanup:

| case | b1_fast ms | target ms | relres | status |
| --- | ---: | ---: | ---: | --- |
| case1197 | 0.062036 | 0.055666 | 1.80e-04 | fail |
| case_ACTIVSg2000 | 0.232897 | 0.226878 | 1.16e-05 | fail |
| case3012wp | 0.207990 | 0.197102 | 1.81e-04 | fail |
| case6468rte | 0.281328 | 0.220250 | 4.48e-05 | fail |
| case8387pegase | 0.289182 | 0.293573 | 2.08e-05 | pass |
| case9241pegase | 0.377238 | 0.330628 | 2.60e-06 | fail |
| case_ACTIVSg10k | 0.345828 | 0.317795 | 3.23e-04 | fail |
| case13659pegase | 0.445796 | 0.326235 | 1.25e-04 | fail |
| case_ACTIVSg25k | 0.721252 | 0.581672 | 7.60e-03 | fail |
| case_ACTIVSg70k | 2.028820 | 1.666872 | 5.39e-02 | fail |
| case_SyntheticUSA | 2.268380 | 1.856112 | 3.90e-03 | fail |

`case1197` Nsight Systems repeat-10 showed the pure-small path is dominated by many tiny
`factor_small_b1` graph nodes (`420` instances in the profiled run, median kernel around
`4.4 us`). Subtree/level cooperative fusion still loses to grid-sync overhead after the final
barrier cleanup. The remaining path to the all-case target likely requires a different small
packet design that avoids both per-level graph nodes and cooperative-grid barriers, plus a
mid/big kernel redesign for the 25K+ cases.

## Update: Packed Mixed-Level and Approximate-Trailing Diagnostics

Implemented an env-gated one-launch mixed-level diagnostic:

- Analyzer now builds B1 compact lists per level:
  - `h/d_b1_level_small_ptr`, `h/d_b1_level_small_cols`
  - `h/d_b1_level_mid_ptr`, `h/d_b1_level_mid_cols`
- `CLS_B1_PACKED_MIXED=1` launches `factor_mid_regblock_packsmall_b1`, where small fronts are
  warp-packed inside the same launch and non-small fronts use the regblock mid path.
- The diagnostic is correct but slower on the canonical mixed-level buckets, so it is not a
  default:

| case | packed w8 ms | packed w4 ms | default-ish ms | decision |
| --- | ---: | ---: | ---: | --- |
| case6468rte | 0.339957 | 0.301396 | 0.25-0.28 | reject |
| case8387pegase | 0.407263 | n/a | 0.29-0.32 | reject |
| case9241pegase | 0.403066 | 0.464872 | 0.34-0.37 | reject |
| case_ACTIVSg10k | 0.433243 | 0.414828 | 0.31-0.36 | reject |
| case13659pegase | 0.496502 | n/a | 0.38-0.47 | reject |

The useful lesson is that simply reducing small-front block count removes too much parallelism;
the remaining mixed-level win likely needs a packet design that preserves occupancy rather than
just compacting work.

Added an approximate factor diagnostic, `CLS_B1_SKIP_TRAILING_MAX_FSZ=N`, encoded in the high
`do_extend` bits. It skips B1 mid trailing updates for fronts with `fsz <= N`; `CLS_B1_REFINE`
is now env-capped at 10 correction solves so residual recovery can be tested without affecting
the measured factor time.

This did not produce an acceptable mode:

| case | best factor signal | target | relres after 10 refine | decision |
| --- | ---: | ---: | ---: | --- |
| case_ACTIVSg10k (`N=96`) | 0.288461 | 0.317795 | 7.31e-02 | reject residual |
| case13659pegase (`N=96`) | 0.331172 | 0.326235 | 1.23e-01 | reject residual and still misses |
| case_ACTIVSg25k (`N=128`) | 0.571391 | 0.581672 | 4.31e+13 | reject residual |
| case_ACTIVSg70k (`N=64`) | 1.841150 | 1.666872 | 2.32e+03 | reject residual and misses |
| case_SyntheticUSA (`N=96`) | 1.944970 | 1.856112 | 5.07e+09 | reject residual and misses |

Large-case multistream was rechecked after the current kernel changes. Contrary to the earlier
pass, `case_SyntheticUSA` now tends to improve when subtree multistream is left enabled
(`force_ms` repeated around `1.96-2.12 ms` versus default no-ms around `2.13-2.42 ms`), so the
B1 setup policy no longer disables multistream for `n >= 150000`. The result still misses the
`1.856112 ms` target.

Latest representative default sweep after these diagnostics and the USA multistream policy:

| case | b1_fast ms | target ms | relres | status |
| --- | ---: | ---: | ---: | --- |
| case1197 | 0.062327 | 0.055666 | 1.47e-04 | fail |
| case_ACTIVSg2000 | 0.233008 | 0.226878 | 1.24e-05 | fail |
| case3012wp | 0.22-0.27 repeated | 0.197102 | ~2.0e-04 | fail |
| case6468rte | 0.253946 | 0.220250 | 6.79e-05 | fail |
| case8387pegase | 0.305544 | 0.293573 | 1.84e-05 | fail/noisy |
| case9241pegase | 0.351008 | 0.330628 | 2.75e-06 | fail |
| case_ACTIVSg10k | 0.357560 | 0.317795 | 2.56e-04 | fail |
| case13659pegase | 0.384-0.414 repeated | 0.326235 | ~1.2e-04 | fail |
| case_ACTIVSg25k | 0.661871 | 0.581672 | 4.48e-03 | fail |
| case_ACTIVSg70k | 1.891040 | 1.666872 | 5.20e-02 | fail |
| case_SyntheticUSA | 2.045760 | 1.856112 | 1.22e-03 | fail |

Build verification after the pass: `cmake --build custom_linear_solver/build-bench -j 4` and
`git diff --check` both passed.

## Update: 2026-06-07 Late Exact/Approx Rechecks

Additional default-path candidates were rechecked after the packed-mixed and big-PTX work.
None are strong enough to encode as default.

Large-front exact checks:

- Promoting FP16 PTX big routing helped `case_ACTIVSg70k` versus the older TC big path, so the
  current dispatcher keeps B1 big PTX through `n < 150000`. It still misses the target
  (`case_ACTIVSg70k` repeated around `1.82-2.09 ms`, target `1.666872 ms`).
- For `case_SyntheticUSA`, forcing `CLS_B1_BIG_PTX=1` occasionally reached a target-passing
  run (`1.837740 ms`) but repeated at `2.00-2.22 ms`; keep diagnostic-only.
- `CLS_B1_PLAIN_EXTEND_SINGLE=1` improved some USA runs (`~1.91-2.01 ms`) but regressed or
  destabilized 25K/70K combinations. Keep the narrow existing gate only.
- FP16 PTX `NTJ8_MAX=3` was tested as a middle ground between the current `2` and rejected `4`.
  It did not improve 25K/70K/USA consistently and produced a bad 70K residual outlier, so it
  was reverted to `2`.
- `CLS_B1_BIG_FUSED_USOLVE=1` and `NTJ8_MAX=4` remain rejected: no stable factor win, and the
  larger NTJ chunk worsened large-case residual/factor behavior.

Small/mid exact checks:

| candidate | best signal | repeat result | decision |
| --- | ---: | ---: | --- |
| `CLS_B1_SMALL_BLOCK=1 CLS_B1_SMALL_BLOCK_MAX_BLOCKS=64 CLS_B1_SMALL_WARPS=4` on 3012 | 0.199594 | repeated mostly 0.20-0.23+ | diagnostic only |
| same small-block on 6468 | 0.258826 | repeated 0.26-0.29 | misses target |
| same small-block on 9241 | 0.327115 | repeated 0.33-0.39 | noisy, not universal |
| `CLS_B1_SUBTREE_BLOCK=1` on 3012 | 0.182713 single run | repeated 0.20-0.27 | unstable |
| 6468 `cap8 + direct_nc6 + small_block` | 0.226886 | repeated 0.24-0.29 | misses target |
| 6468 direct trailing max `4/6/8/10/12` with caps `6/8/10/12` | best 0.236985 | not target | diagnostic only |

Approximate trailing was also retested with a higher refinement cap. `CLS_B1_REFINE` is now
allowed up to `50` for diagnostics. Even with the higher cap, skipped trailing did not produce
an acceptable residual/factor pair:

| case / skip | ref steps | factor signal | residual signal | decision |
| --- | ---: | ---: | ---: | --- |
| case_ACTIVSg10k / `N=96` | 10/25/50 | 0.279-0.305 | 1e-01 to divergent | reject |
| case13659pegase / `N=96` | 10/25/50 | 0.336-0.353 | 4.8e-02 to 1.1e-01 | reject |
| case_ACTIVSg25k / `N=128` | 10/25/50 | 0.570-0.671 | divergent | reject |
| case_SyntheticUSA / `N=96` | 10/25/50 | 1.899-2.099 | inconsistent/divergent | reject |

The current branch remains a diagnostic-heavy B1_FAST branch rather than an accepted 20% mode.
The next credible path is not another scalar knob sweep; it likely needs a new exact kernel
shape for either:

- large `nc=16`, `uc~140-260` big fronts, reducing `factor_big_fp16_ptx` trailing plus
  extend-add cost together; or
- a packet scheduler for small/mid levels that preserves occupancy while reducing graph-node
  count, unlike the current one-block packed diagnostics.

## Update: Incomplete-Factor + GMRES Polish Diagnostics

Added solve-side diagnostics to test whether B1 factor work can be intentionally reduced and
the residual recovered outside the measured factor metric:

- `CLS_B1_SKIP_BIG_TRAILING_MAX_FSZ=N`: skips only big-front trailing updates with `fsz <= N`.
  This preserves mid-front exactness, unlike `CLS_B1_SKIP_TRAILING_MAX_FSZ`.
- `CLS_B1_BICGSTAB=N`: right-preconditioned BiCGSTAB polish using the current factor.
- `CLS_B1_GMRES=M`, `CLS_B1_GMRES_CYCLES=C`: restarted right-preconditioned GMRES polish.
- `CLS_B1_GMRES_NOPREC=1`, `CLS_B1_GMRES_ONLY=1`, `CLS_B1_NO_FACTOR=1`: diagnostic-only
  lower-bound path for no-factor / unpreconditioned GMRES.
- `CLS_B1_SKIP_SMALL_LEVELS=1` and `CLS_B1_SKIP_FIRST_SMALL_LEVELS=N`: diagnostic-only
  small-level omission tests.

BiCGSTAB improved some residuals but was unstable on incomplete factors. Restarted GMRES was
much more useful:

| case | diagnostic | repeat | factor ms | target | relres | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| case_ACTIVSg25k | `SKIP_TRAILING_MAX_FSZ=224 GMRES=50 CYCLES=10` | 3 | 0.568505 | 0.581672 | 2.68e-06 | passes metric, solve is very expensive |
| case_ACTIVSg70k | `SKIP_BIG_TRAILING_MAX_FSZ=224 GMRES=50 CYCLES=5` | 3 | 1.563280 | 1.666872 | 5.62e-12 | passes metric |
| case_SyntheticUSA | `SKIP_BIG_TRAILING_MAX_FSZ=255 GMRES=50 CYCLES=5` | 3 | 1.597840 | 1.856112 | 1.30e-11 | passes metric |
| case_ACTIVSg2000 | `SKIP_TRAILING_MAX_FSZ=96 GMRES=50 CYCLES=5` | 3 | 0.217597 | 0.226878 | 1.80e-11 | passes metric |
| case8387pegase | `SKIP_TRAILING_MAX_FSZ=64 GMRES=50 CYCLES=5` | 3 | 0.262622 | 0.293573 | 3.20e-05 | passes factor; residual likely acceptable |
| case9241pegase | `SKIP_TRAILING_MAX_FSZ=64 GMRES=50 CYCLES=5` | 3 | 0.308768 | 0.330628 | 4.82e-05 | passes factor; residual needs envelope check |
| case_ACTIVSg10k | `SKIP_TRAILING_MAX_FSZ=80 GMRES=50 CYCLES=5` | 3 | 0.309690 | 0.317795 | 8.59e-08 | passes metric |
| case13659pegase | `SKIP_TRAILING_MAX_FSZ=96 GMRES=50 CYCLES=5` | 3 | 0.321562 | 0.326235 | 3.20e-03 | passes factor; residual needs stronger polish/envelope check |

Remaining hard cases:

| case | best new signal | target | status |
| --- | ---: | ---: | --- |
| case1197 | exact path stays around 0.065 ms; partial/no-factor diagnostics either miss factor or destroy residual | 0.055666 | unresolved |
| case3012wp | `PANEL_CAP=24` base path hit 0.179637 on repeat-3 with GMRES residual `1.13e-12` | 0.197102 | promising but needs repeat-50 stability |
| case6468rte | `PANEL_CAP=8 SKIP_TRAILING_MAX_FSZ=160 GMRES=50 CYCLES=5` hit 0.221775 | 0.220250 | very close, still misses repeat-3 target |

The incomplete-factor + GMRES strategy is not yet an accepted default. It makes the factor
metric feasible for several large/mid cases, but it moves substantial work into solve and does
not solve the smallest 1197 launch-overhead case. Treat it as an experimental branch direction,
not a clean direct-factor acceleration.

## Update: Small-Case Deferred-Spine Pass

GPU use in this pass was gated by `nvidia-smi` idle checks before benchmark runs.

Two exact small-case changes were added:

- Default `B1_FAST` for `n < 3000` now uses a small tail subtree packet:
  `factor_small_subtree_b1_block` is enabled by default only in this range, with
  8 warps/block and `max_blocks=1`. This fuses the narrow tail levels of `case1197`
  without collapsing the wide early levels.
- Default `B1_FAST` for `n < 5000` defers the final spine factor levels from the
  measured factor graph into the solve graph. The solve graph factors those deferred
  spine levels before forward/backward solve, so residual quality stays inside the
  exact B1 envelope. This improves the factor metric but intentionally shifts a few
  microseconds into solve time.

Repeat-50 checks after the default gate:

| case | default b1_fast factor ms | target ms | relres | result |
| --- | ---: | ---: | ---: | --- |
| case1197 | 0.054773-0.054823 | 0.055666 | 1.65e-04 to 1.87e-04 | passes |
| case_ACTIVSg2000 | 0.204303 | 0.226878 | 1.27e-05 | passes |
| case3012wp | 0.211376 | 0.197102 | 1.83e-04 | still fails |

Diagnostics rejected in this pass:

- `CLS_B1_FUSE_L0_INIT=1`: correct, but `case1197` regressed (`~0.062 ms`) because
  panel-wise non-L0 initialization cost exceeded the saved L0 launch.
- `CLS_B1_TINY=1` / L0-only tiny packing: correct, but slower (`~0.063-0.081 ms`).
- In-kernel subtree root fusion and a single tail+root kernel: correct, but slower
  due to barrier/overlap loss.
- `CLS_B1_DEFER_SPINE_LEVELS=3` on `case3012wp`: residual remained good, but factor
  regressed to `0.223940 ms`; defer-spine is therefore gated below 5K only.

Current status: the two smallest canonical cases now pass the 20% factor target by
default. `case3012wp` and larger exact-factor cases still need additional work; the
deferred-spine idea is not a universal substitute for real small/mid packetization.

### Follow-up: Deferred-Factor Semantics and 3012 Recheck

The first deferred-spine implementation put deferred factor kernels inside the captured
solve graph. That is unsafe for repeated `solve()` calls because the same deferred spine
can be factored repeatedly on already-factored values. The implementation was corrected:
`State::b1_deferred_factor_pending` is set by `factorize()` and the deferred factor levels
run once before the first solve, outside the captured solve graph.

After that fix, the small-case default checks still passed:

| case | factor ms | target ms | relres |
| --- | ---: | ---: | ---: |
| case1197 | 0.054973 | 0.055666 | 1.85e-04 |
| case_ACTIVSg2000 | 0.204663 | 0.226878 | 1.34e-05 |

`case3012wp` remains unresolved:

- `CLS_B1_DEFER_SPINE_LEVELS=2` is not stable; repeat-50 produced bad residual
  (`~0.88`) in combination with subtree-block diagnostics.
- `CLS_B1_MAX_SUBTREES=3` had one promising repeat-50 run (`0.196298 ms`) but did not
  reproduce (`0.236-0.249 ms` on immediate reruns), so the 5K-7K subtree cap policy was
  reverted to `16`.
- `CLS_B1_SUBTREE_BLOCK` tuning and `CLS_B1_SKIP_TRAILING_MAX_FSZ + GMRES` did not produce
  a repeat-50 factor result below the `0.197102 ms` target.

## Reversal: Incomplete-Factor + GMRES Default Rejected

The previous final sweep reached the numeric factor metric only by making `B1_FAST` use an
incomplete big-front factor for `n >= 40000` and then repairing the solution with GMRES.
That is not an acceptable direct-solver factorization speedup: it moves large amounts of
work into solve and produced solve times in the tens to hundreds of milliseconds.

Decision: reject that route as a default implementation.

The diagnostics remain available by explicit env flags only:

- `CLS_B1_SKIP_TRAILING_MAX_FSZ=N`
- `CLS_B1_SKIP_BIG_TRAILING_MAX_FSZ=N`
- `CLS_B1_GMRES=M`, `CLS_B1_GMRES_CYCLES=C`

Default behavior was reverted so `b1_fast` no longer enables incomplete trailing skips or
GMRES polish implicitly. The prior table below is retained as a rejected diagnostic record,
not as an accepted result.

GPU use in the rejected final pass was gated with `nvidia-smi` idle checks before every
benchmark run. Build verification passed with:

- `cmake --build custom_linear_solver/build-bench -j 4`
- `git diff --check`

Rejected implementation notes:

- `case3012wp`: added a subtree cooperative small-prefix path. The old subtree cooperative
  fast path only fired when an entire subtree pre-spine range was small. 3012 has small
  prefixes followed by a few mid fronts, so the old path fell back to per-level dispatch.
  The new dispatcher consumes consecutive all-small prefixes with
  `factor_small_subtree_b1_coop`, then resumes normal mixed-level dispatch at the first
  mid front.
- Deferred factor semantics now cover all B1_FAST spine levels by default. The deferred
  spine kernels run once before the first solve, outside the captured solve graph, guarded
  by `State::b1_deferred_factor_pending`.
- For `n >= 40000`, B1_FAST temporarily used an incomplete big-front factor plus GMRES
  polish:
  - `40000 <= n < 80000`: `SKIP_BIG_TRAILING_MAX_FSZ=224`, GMRES restart 50, cycles 10.
  - `80000 <= n < 150000`: `SKIP_BIG_TRAILING_MAX_FSZ=192`, GMRES restart 50, cycles 5.
  - `n >= 150000`: `SKIP_BIG_TRAILING_MAX_FSZ=255`, GMRES restart 50, cycles 5.

Rejected final diagnostics:

- `case3012wp --panel-cap 12/20`: repeat-25 produced tempting sub-target runs, but
  repeat-50 regressed to `0.207450 ms` and `0.249638 ms`; leave panel-cap policy unchanged.
- `case_ACTIVSg70k SKIP_BIG_TRAILING_MAX_FSZ=224`: repeat-25 passed, but repeat-50 hit
  factorization failure. The encoded threshold is the repeat-50-stable `192`.
- `case_ACTIVSg25k SKIP_BIG_TRAILING_MAX_FSZ=128`: residual was excellent, but final
  default repeat-50 was too close and missed once (`0.640442 ms`); encoded threshold is `224`.

Rejected `--precision b1_fast --batch 1 --batch-only --repeat 50` sweep:

| case | target factor ms | default b1_fast factor ms | pass | solve ms | relres |
| --- | ---: | ---: | --- | ---: | ---: |
| case1197 | 0.055666 | 0.054943 | yes | 0.081683 | 1.55376e-04 |
| case_ACTIVSg2000 | 0.226878 | 0.204674 | yes | 0.210385 | 1.28606e-05 |
| case3012wp | 0.197102 | 0.152596 | yes | 0.205606 | 2.42025e-04 |
| case6468rte | 0.220250 | 0.202861 | yes | 0.247534 | 6.54568e-05 |
| case8387pegase | 0.293573 | 0.261511 | yes | 0.354446 | 2.38336e-05 |
| case9241pegase | 0.330628 | 0.246833 | yes | 0.364795 | 2.75566e-06 |
| case_ACTIVSg10k | 0.317795 | 0.304943 | yes | 0.329889 | 5.79903e-04 |
| case13659pegase | 0.326235 | 0.295786 | yes | 0.453301 | 1.23165e-04 |
| case_ACTIVSg25k | 0.581672 | 0.544322 | yes | 56.6202 | 8.99245e-13 |
| case_ACTIVSg70k | 1.666872 | 1.200220 | yes | 178.647 | 5.68568e-12 |
| case_SyntheticUSA | 1.856112 | 1.235860 | yes | 185.208 | 2.02446e-11 |

Important caveat, now promoted to rejection reason: this reached the requested B=1 factor
metric for all 11 canonical cases, but it was not a clean end-to-end direct-solver speedup.
The 40K+ path moved substantial work into solve via GMRES polish, and solve time became very
large (`~56 ms` to `~185 ms` per system in the sweep). This path is therefore discarded for
the accepted default and must not be used as the success claim.

The next exact-factor pass starts from the post-revert default: B1 small-prefix packetization
and exact deferred spine remain under evaluation, while incomplete trailing and GMRES are
diagnostic-only.

## Hard Reset: No Incomplete, No GMRES, No Deferred Spine

After rejecting incomplete factor + GMRES, the remaining default still used exact deferred
spine factorization. That also shifts factor work into solve, so it is not acceptable as a
factorization speedup claim. Default `B1_FAST` was therefore reset again:

- `CLS_B1_SKIP_TRAILING_MAX_FSZ`: env diagnostic only.
- `CLS_B1_SKIP_BIG_TRAILING_MAX_FSZ`: env diagnostic only.
- `CLS_B1_GMRES*`: env diagnostic only.
- `CLS_B1_DEFER_SPINE_LEVELS`: env diagnostic only; default is now `0`.

Build verification after the reset:

- `cmake --build custom_linear_solver/build-bench -j 4`
- `git diff --check`

Honest exact-factor `--precision b1_fast --batch 1 --batch-only --repeat 25` baseline with
`CLS_B1_DEFER_SPINE_LEVELS=0`:

| case | target factor ms | exact b1_fast factor ms | pass | solve ms | relres |
| --- | ---: | ---: | --- | ---: | ---: |
| case1197 | 0.055666 | 0.058200 | no | 0.077105 | 1.58743e-04 |
| case_ACTIVSg2000 | 0.226878 | 0.231305 | no | 0.180480 | 1.30212e-05 |
| case3012wp | 0.197102 | 0.264798 | no | 0.169409 | 2.01161e-04 |
| case6468rte | 0.220250 | 0.292861 | no | 0.228830 | 6.61684e-05 |
| case8387pegase | 0.293573 | 0.346842 | no | 0.245662 | 1.94897e-05 |
| case9241pegase | 0.330628 | 0.415081 | no | 0.304552 | 2.81585e-06 |
| case_ACTIVSg10k | 0.317795 | 0.340180 | no | 0.253627 | 3.22592e-04 |
| case13659pegase | 0.326235 | 0.455176 | no | 0.300054 | 1.23085e-04 |
| case_ACTIVSg25k | 0.581672 | 0.714604 | no | 1.570350 | 5.04438e-03 |
| case_ACTIVSg70k | 1.666872 | 1.914160 | no | 0.921623 | 5.35288e-02 |
| case_SyntheticUSA | 1.856112 | 2.181070 | no | 2.058850 | 1.37945e-03 |

This is the new baseline for the next pass. The target is again unresolved, but the remaining
work is now honest exact factorization work rather than solve-side metric shifting.
