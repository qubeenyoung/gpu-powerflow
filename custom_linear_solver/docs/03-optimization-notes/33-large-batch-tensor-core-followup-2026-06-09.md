# Large-batch Tensor Core follow-up

**Date**: 2026-06-09
**Goal**: reach 1.2-1.4x factorize speedup over FP32 at B=64/256 with Tensor Cores as the enabler.
**Status**: not achieved. The best measured valid point in this round is a narrow 25K B=256
TF32-mid/big-fused configuration at ~1.20x with high TF32 residual; B=64 remains below target.

## Measurements

All timings are `batch_factor_per_sys_ms`; speedup is `fp32 / TC`.

### Forced FP16 mid+big TC

Build: `build-tc-midbig-force`, `CLS_MID_FP16_TC_FORCE_ALL=ON`.

| case | B | FP32 | FP16 mid+big TC | speedup |
|---|---:|---:|---:|---:|
| 8387 | 64 | 0.024966 | 0.032887 | 0.76x |
| 8387 | 256 | 0.019894 | 0.024794 | 0.80x |
| 25K | 64 | 0.101035 | 0.118454 | 0.85x |
| 25K | 256 | 0.102071 | 0.113829 | 0.90x |
| USA | 64 | 0.419834 | 0.441383 | 0.95x |
| USA | 256 | 0.449180 | 0.428978 | 1.05x |

Interpretation: mid FP16 TC is useful at B=1, but at large B the scalar/staged FP32 path is already
saturated and the TC staging/padding/drain overhead dominates.

### Big-only FP16 TC

Build: `build-tc-no-mid`, `CLS_MID_FP16_TC=OFF`.

| case | B | FP32 | FP16 big TC | speedup |
|---|---:|---:|---:|---:|
| 25K | 64 | 0.101627 | 0.101668 | 1.00x |
| 25K | 256 | 0.096614 | 0.094833 | 1.02x |
| USA | 64 | 0.414383 | 0.439442 | 0.94x |
| USA | 256 | 0.407837 | 0.434854 | 0.94x |

Interpretation: disabling mid avoids the worst 25K regression, but big FP16 TC alone does not move
the large-batch wall enough.

### TF32 mid TC + big fused drain

New experimental options:

- `CLS_MID_TF32_TC=ON`: shared-resident mid TF32 PTX `mma.m16n8k8` trailing.
- `CLS_FUSE_TF32_TRAIL_EXTEND=ON`: big TF32 direct parent drain.

Build: `build-tc-midtf32-fuse`.

| case | B | FP32 | TF32 mid TC + big fuse | speedup |
|---|---:|---:|---:|---:|
| 25K | 64 | 0.105237 | 0.094296 | 1.12x |
| 25K | 256 | 0.096289 | 0.093045 | 1.03x |
| USA | 64 | 0.432838 | 0.419297 | 1.03x |
| USA | 256 | 0.433006 | 0.408447 | 1.06x |

With `--panel-cap 64` on 25K:

| case | B | FP32 | TF32 mid TC + big fuse | speedup |
|---|---:|---:|---:|---:|
| 25K | 64 | 0.106511 | 0.102547 | 1.04x |
| 25K | 256 | 0.105604 | 0.087958 | 1.20x |

Caveat: TF32 residuals in these runs are high, typically around `4e-2` to `6e-2`, so this is not a
clean accuracy-preserving pass.

### cuBLAS grouped TF32 trailing

New experimental options:

- `CLS_CUBLAS_TF32_TRAILING=ON`: route big trailing through cuBLAS grouped-batched SGEMM with TF32
  tensor-op math.
- `CLS_CUBLAS_TF32_MID=ON`: optional mid routing. Default OFF after this round.

Findings:

- Mid+big cuBLAS was slower on 25K and failed correctness on USA.
- Big-only cuBLAS was correct but only small positive on USA:
  - B=64: `0.427994 -> 0.419510` = 1.02x
  - B=256: `0.420068 -> 0.410046` = 1.02x

cuBLAS capture requires pointer arrays to be materialized before graph capture. Building pointer
arrays inside the captured graph caused factorization failure because cuBLAS captures or validates
the pointer-array values at capture time.

## Conclusion

Current evidence still supports docs 28/29: at large B, TC trailing is not the dominant lever. The
factor wall is mostly panel LU/U-solve synchronization, front staging/writeback, and extend-add.
TC can help selected points, but it is not yet a broad 1.2-1.4x enabler for B=64/256.

## Follow-up: direct-shared and blocked TF32 mid/big

After the first round, two additional TC paths were tried.

### Direct-shared mid TF32

The original mid TF32 prototype re-staged L/U from shared `Fs` into padded `Ltf/Utf` scratch before
`mma.sync`. A direct-shared helper was added so the mid front reads L/U directly from `Fs` and
zeros out-of-bounds lanes in registers.

Build: `build-tc-midtf32-direct-fuse`, `CLS_MID_TF32_TC=ON`, `CLS_FUSE_TF32_TRAIL_EXTEND=ON`.

| case | B | FP32 | TF32 direct-shared mid + big fuse | speedup |
|---|---:|---:|---:|---:|
| 25K | 64 | 0.099508 | 0.097608 | 1.02x |
| 25K | 256 | 0.102650 | 0.096689 | 1.06x |
| USA | 64 | 0.420966 | 0.397357 | 1.06x |
| USA | 256 | 0.413001 | 0.394609 | 1.05x |

Direct-shared is directionally useful on USA but still far below 1.2x.

### Right-looking blocked TF32 update

A more aggressive mid prototype (`factorize_front_blocked_tf32`) used TC for each pivot-block update,
so TC covers remaining panel columns plus C, not only final C trailing. This makes TC a stronger
enabler, but the measured wall still did not reach the target.

Build: `build-tc-midtf32-blocked-fuse`, BK=8, `CLS_MID_TF32_TC=ON`,
`CLS_FUSE_TF32_TRAIL_EXTEND=ON`.

| case | B | FP32 | TF32 blocked mid + big fuse | speedup |
|---|---:|---:|---:|---:|
| 25K | 64 | 0.105226 | 0.099277 | 1.06x |
| 25K | 256 | 0.094178 | 0.095921 | 0.98x |
| USA | 64 | 0.426263 | 0.389621 | 1.09x |
| USA | 256 | 0.421980 | 0.404021 | 1.04x |

BK=16 was also tested to increase K per mma. It regressed badly on USA B=64
(`0.429844 -> 0.860653`) due to register/accumulator pressure and was reverted to BK=8.

Big-front blocked TF32 was tried under `CLS_BIG_TF32_BLOCKED_TC=ON`. It was not viable:

| case | B | FP32 | TF32 blocked mid+big | speedup |
|---|---:|---:|---:|---:|
| 25K | 64 | 0.098201 | 0.102152 | 0.96x |
| 25K | 256 | 0.099475 | 0.090169 | 1.10x |
| USA | 64 | 0.425523 | 0.563860 | 0.75x |
| USA | 256 | 0.415466 | 0.533482 | 0.78x |

This confirms the big front cannot simply switch to global-memory right-looking TC updates; without
a shared/full-front residency strategy, repeated global L/U traffic overwhelms the reduced sync count.

## Follow-up: shared-resident big-low split

The next variant split the big bucket and used a full-front shared-resident blocked TF32 kernel for
big-low fronts. The shared cutoff was raised to the 99 KiB opt-in limit, so fronts up to fsz=159 fit
(`159*159*4 = 101124` bytes). Larger big fronts still fall back to the existing TF32 path.

Build: `build-tc-mid-big-both`, `CLS_MID_TF32_TC=ON`, `CLS_BIG_LOW_SPLIT=ON`,
`CLS_BIG_TF32_BLOCKED_TC=ON`, `CLS_FUSE_TF32_TRAIL_EXTEND=OFF`, warmup=8, repeat=31.

| case | B | FP32 factor ms/sys | TF32 mid+big TC factor ms/sys | speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|
| 8387 | 1 | 0.336162 | 0.393048 | 0.855x | 3.32e-05 |
| 8387 | 64 | 0.0242008 | 0.0246826 | 0.980x | 1.65e-02 |
| 8387 | 256 | 0.0207004 | 0.0208549 | 0.993x | 3.34e-02 |
| 25K | 1 | 0.768274 | 0.795355 | 0.966x | 9.17e-03 |
| 25K | 64 | 0.103779 | 0.0949500 | 1.093x | 4.36e-02 |
| 25K | 256 | 0.0986256 | 0.0902209 | 1.093x | 4.31e-02 |
| USA | 1 | 2.44424 | 2.03013 | 1.204x | 5.47e-02 |
| USA | 64 | 0.426574 | 0.390988 | 1.091x | 4.79e-02 |
| USA | 256 | 0.433898 | 0.351679 | 1.234x | 4.82e-02 |

This is the first broad-ish TC result that crosses 1.2x on a large-batch point (USA B=256) while
also improving USA B=1. It is still not a general 1.2-1.4x result: 25K plateaus near 1.09x and 8387
does not benefit because it has no big fronts and is dominated by small/mid work. The TF32 residuals
on the large cases are also around 4e-2 to 5e-2, so this is a speed-enabler result rather than an
accuracy-neutral replacement for FP32.

## Follow-up: rejected mid/small extensions

The mid TF32 path was split into two variants:

- blocked mid update (`factorize_front_blocked_tf32`), better on USA,
- direct-shared trailing-only mid update, better on 25K in quick A/Bs.

An attempted policy build used direct-shared mid for `num_rows < 80000` and blocked mid otherwise,
with shared-resident big-low TF32 still enabled. Warmup=8, repeat=31:

| case | B | FP32 factor ms/sys | TF32 policy factor ms/sys | speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|
| 8387 | 64 | 0.0235602 | 0.0243854 | 0.966x | 3.25e-02 |
| 8387 | 256 | 0.0215140 | 0.0213413 | 1.008x | 3.25e-02 |
| 25K | 64 | 0.100801 | 0.0926695 | 1.088x | 6.00e-02 |
| 25K | 256 | 0.0965667 | 0.0897003 | 1.077x | 4.17e-02 |
| USA | 64 | 0.413925 | 0.362743 | 1.141x | 4.78e-02 |
| USA | 256 | 0.439883 | 0.372006 | 1.182x | 4.81e-02 |

This policy is directionally better for USA B64 but does not preserve the earlier USA B256 1.23x
point, and it does not lift 25K past 1.2x. The result is still target-miss, so the default
experimental mid TF32 path remains blocked; direct-shared is available only through the explicit
`CLS_MID_TF32_DIRECT_SHARED` option.

Two smaller-front extensions were also tried:

- `CLS_MID_TF32_LOW_TC`: lets 33..48 mid-low fronts use TF32 TC. Direct and blocked forms did not
  improve 25K B64/B256 enough; the best quick B256 point was about 1.14x, still below target.
- `CLS_SMALL_TF32_TC`: a warp-resident TF32 path for 17..32 small fronts. This was slower than the
  existing warp scalar fused path, confirming that the small tier is too latency/control-heavy for
  TC to amortize.
- `CLS_TF32_BLOCKED_BK4`: uses BK=4 instead of BK=8 for shared-resident blocked TF32. Quick runs
  looked slightly better, but repeat=31 regressed to 25K B64/B256 = 1.04x/1.06x and USA B64/B256 =
  1.09x/1.09x. BK=8 remains the best default.
- `CLS_BIG_TF32_SHARED_THREADS_128`: launches shared-resident big-low TF32 with 128 threads instead
  of 256. Repeat=31 gave 25K B64/B256 = 1.07x/1.11x and USA B64/B256 = 1.09x/1.15x. Not enough.
- `CLS_BIG_TF32_THREADS_256`: launches the global-resident big-high TF32 fallback with 256 threads
  instead of 512. USA repeat=31 gave B64/B256 = 1.13x/1.13x. This improves neither target point
  enough to justify changing the default.
- `CLS_BIG_TF32_THREADS_384`: later added for the global-resident big-high TF32 fallback. It is a
  useful compromise between the 256-thread and 512-thread variants, but only after combining it
  with cap-respect and the shared big-low 512-thread variant; see the continuation below.

For completeness, the TC build was also compared against a separate plain FP32 baseline build
without the TC-only big-low split. Quick B64/B256 speedups were 25K = 1.12x/1.11x and USA =
1.21x/1.07x. This is a useful presentation angle for USA B64, but it still does not satisfy the
broad B64/B256 target.

An FP16 shared-resident blocked update was also implemented as an experimental OFF-by-default path
(`CLS_FP16_BLOCKED_SHARED_TC`). It keeps the front in FP32 shared memory but feeds FP16
multiplicands to Tensor Cores for the blocked updates. Quick B64/B256 A/B was slower on 25K and only
barely positive on USA B256:

| case | B | FP32 | FP16 blocked shared | speedup | FP16 relres |
|---|---:|---:|---:|---:|---:|
| 25K | 64 | 0.0967397 | 0.115443 | 0.838x | 9.80e-03 |
| 25K | 256 | 0.0993732 | 0.106178 | 0.936x | 2.24e-02 |
| USA | 64 | 0.416173 | 0.424294 | 0.981x | 1.35e-01 |
| USA | 256 | 0.418909 | 0.403612 | 1.038x | 6.46e-02 |

The half conversion overhead and lower numerical quality do not justify using this path.

## Continuation: direct-low mid TF32 and 48 split

The next attempt targeted the B=64/B=256 gap in the mid tier:

- `CLS_MID_TF32_MIN_FSZ`: compile-time gate for the minimum `fsz_cap` routed to mid TF32.
- `CLS_MID_TF32_DIRECT_HIGH`: direct-shared only for `fsz_cap > 64`, blocked for 49..64.
- `CLS_MID_TF32_DIRECT_FUSE_EXTEND`: direct TC drain directly into the parent extend-add.
- `CLS_MID_LOW_SPLIT`: split 33..48 and 49..64 into separate dispatch buckets.
- `CLS_TF32_DIRECT_NTJ8_16`: direct-shared TF32 accumulates up to 16 N-tiles instead of 8.

The useful piece was `CLS_MID_LOW_SPLIT`: it reduces the shared `fsz_cap` for 33..48 fronts when
`CLS_MID_TF32_LOW_TC` routes them through the direct-shared TC kernel. The best candidate build was:

```text
CLS_BIG_LOW_SPLIT=ON
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON
CLS_MID_TF32_LOW_TC=ON
CLS_MID_TF32_MIN_FSZ=48
```

Warmup=8, repeat=31:

| case | B | FP32 factor ms/sys | TF32 direct-low+split48 ms/sys | speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|
| 8387 | 64 | 0.024708 | 0.024046 | 1.028x | 4.52e-02 |
| 8387 | 256 | 0.021972 | 0.021148 | 1.039x | 3.60e-02 |
| 25K | 64 | 0.098129 | 0.089413 | 1.097x | 5.37e-02 |
| 25K | 256 | 0.097810 | 0.085217 | 1.148x | 5.70e-02 |
| USA | 64 | 0.425087 | 0.372253 | 1.142x | 5.12e-02 |
| USA | 256 | 0.436514 | 0.388562 | 1.123x | 4.96e-02 |

This is a real improvement over the previous direct/blocked policy on 25K B256, but it still misses
the requested 1.2-1.4x band and it does not help 8387 enough. Quick sanity after rebuilding the
same candidate stayed in the same range: 25K B256 1.08x, USA B256 1.19x.

Rejected continuation A/Bs:

- `CLS_MID_TF32_MIN_FSZ=64/80`: thresholding away low-mid TC helps isolated points but hurts 25K
  B256; it is not a stable policy.
- `CLS_MID_TF32_DIRECT_HIGH`: mixing blocked low-mid with direct high-mid erased the B256 gain.
- `CLS_MID_TF32_DIRECT_FUSE_EXTEND`: interleaving parent atomics into the direct TC drain was slower
  than the separate extend loop.
- `CLS_TF32_DIRECT_NTJ8_16`: register pressure dominated; 25K B64/B256 fell below 0.9x.
- `CLS_SMALL_FRONT_MAX_16`: routing 17..32 fronts into mid/direct-low reduced speedup and produced
  an invalid 25K B256 residual near 1.0 in a quick run.
- `CLS_BIG_TF32_SHARED_THREADS_128/512` and `CLS_BIG_TF32_THREADS_256` remain case-point wins at
  best; combined with direct-low+split48 they did not lift the broad set.
- `--panel-cap 12/16/24/32` can produce isolated points (25K B256 at cap24 reached 1.26x in a quick
  run), but the optimum is case/B dependent and USA/25K B64 stay below target.
- `--serial-nd` stayed near 1.15x and worsened 25K B64 residual.

Kernel-profile check on 25K B256 explains the ceiling. With blocked TF32, `factor_mid_tf32_ptx`
reduced mid kernel time from about 75 ms to 66 ms over the profiled iterations, but scatter and
small-front kernels remained large. Direct-shared lowered the mid TF32 kernel further to about
58 ms, yet scatter plus small-front work still left the end-to-end factorize speedup around
1.1x. Multi-batch improves occupancy/latency hiding, but it does not remove the per-system
front traffic, scatter, small-front control work, or parent update traffic.

## Continuation: column-owned U-solve

The next structural attempt removed the per-row barrier chain in the U-panel solve for TF32 TC
fronts. Once panel LU is complete, each U column is independent, so one thread can own one trailing
column and solve all `nc` rows serially:

```text
for each column j owned by thread:
  for k in 1..nc-1:
    U[k,j] -= sum_i<k L[k,i] * U[i,j]
one block sync before trailing TC
```

This is controlled by `CLS_TF32_COLUMN_USOLVE`. It is applied to direct-shared mid TF32 and
shared-resident blocked TF32; a separate `CLS_TF32_GLOBAL_COLUMN_USOLVE` was added for global
big-high TF32 and left OFF because it regressed USA.

Best build:

```text
CLS_BIG_LOW_SPLIT=ON
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON
CLS_MID_TF32_LOW_TC=ON
CLS_TF32_COLUMN_USOLVE=ON
CLS_MID_TF32_MIN_FSZ=48
```

Warmup=8, repeat=31:

| case | B | FP32 factor ms/sys | TF32 + column U-solve ms/sys | speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|
| 8387 | 64 | 0.023811 | 0.023915 | 0.996x | 3.93e-02 |
| 8387 | 256 | 0.020539 | 0.020704 | 0.992x | 5.12e-02 |
| 25K | 64 | 0.104066 | 0.086015 | 1.210x | 4.80e-02 |
| 25K | 256 | 0.100155 | 0.087204 | 1.149x | 5.09e-02 |
| USA | 64 | 0.442168 | 0.371654 | 1.190x | 5.16e-02 |
| USA | 256 | 0.420116 | 0.377373 | 1.113x | 5.09e-02 |

This is the first stable B64 result at the target threshold on 25K, and USA B64 is close. It still
misses B256 and does not help 8387. Quick runs showed isolated B256 target hits (for example USA
B256 1.20x after adding blocked-TF32 column solve, and 25K B256 near 1.19x with panel-cap 10), but
repeat=31 did not hold a broad 1.2x result.

Rejected follow-ups:

- `CLS_TF32_GLOBAL_COLUMN_USOLVE`: global big-high column-owned solve was slower; USA B256 quick
  fell to about 1.10x.
- `CLS_SMALL_BUCKET_SPLIT_16`: keeps 17..32 in the small tier but splits `<=16` and `17..32`
  buckets. It improved some 25K B256 quick runs (~1.18x) but hurt B64/USA stability and produced a
  bad USA B256 residual in one gated run.
- `CLS_SMALL_TF32_TC` was retried with the column-owned warp U-solve and B>=128 dispatch gating.
  Quick runs briefly crossed B256 target on 25K/USA (1.24x/1.23x), but repeat=31 fell back to
  25K B64/B256 = 1.14x/1.16x and USA B64/B256 = 1.13x/1.11x. Without the small bucket split, quick
  B64/B256 stayed around 25K 1.19x/1.17x and USA 1.14x/1.15x. Small TF32 remains too
  latency/control-heavy to be a stable enabler.
- `CLS_MID_TF32_TC_THREADS_128`: reducing mid TF32 to 128 threads did not help. Quick 25K B64/B256
  = 1.15x/1.19x and USA B64/B256 = 1.12x/1.13x.
- `CLS_BIG_TF32_SHARED_THREADS_128/512` was also retried with column-owned U-solve. 512 threads
  gave a useful 25K B64 quick point (1.21x) but hurt USA B64 and still missed B256; 128 threads
  stayed around 1.16x to 1.19x. The default 256-thread shared big-low kernel remains the least bad
  broad policy.
- `CLS_CUBLAS_TF32_TRAILING=ON` + `CLS_CUBLAS_TF32_MID=ON` with column U-solve remained slower:
  quick 25K B64/B256 = 1.02x/1.07x and USA B64/B256 = 1.11x/1.08x.

## Continuation: respecting panel cap

One issue in the earlier cap sweeps was that large cases did not actually use the requested
`--panel-cap`: `compute_effective_panel_cap()` forced `cap=12` for `n>=16000` and `cap=20` for
`n>=80000`. `CLS_RESPECT_PANEL_CAP=ON` disables that override. With the column U-solve build, this
made cap tuning visible again:

```text
CLS_BIG_LOW_SPLIT=ON
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON
CLS_MID_TF32_LOW_TC=ON
CLS_TF32_COLUMN_USOLVE=ON
CLS_RESPECT_PANEL_CAP=ON
CLS_MID_TF32_MIN_FSZ=48
```

The best large-case cap was around 28..32. At `--panel-cap 32`, repeat=31 gave 25K
B64/B256 = 1.27x/1.25x and USA B64/B256 = 1.29x/1.23x in one run, but repeat=61 fell back to
USA B64/B256 = 1.18x/1.14x. The useful and stable part is 25K: repeat=61 still held
25K B64/B256 = 1.27x/1.28x with TF32 relres near `5e-2`.

The cap lever does not rescue 8387. A repeat=31 cap sweep over 8, 10, 12, 16, 20, 24, 28, 32, 40,
48, and 64 never produced a valid 1.2x point. The best valid B64 points stayed near 1.05x, and
B256 peaked around 1.09x. Caps 40+ produced invalid FP32/TF32 residuals on 8387 and are not usable.

## Continuation: big-high 256 threads with cap32

The remaining USA B256 profile still had global-resident `factor_big_tf32_ptx` for fsz>159, so the
global big-high kernel was retried with 256 threads, now combined with cap-respect and column
U-solve:

```text
CLS_BIG_LOW_SPLIT=ON
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON
CLS_BIG_TF32_THREADS_256=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON
CLS_MID_TF32_LOW_TC=ON
CLS_TF32_COLUMN_USOLVE=ON
CLS_RESPECT_PANEL_CAP=ON
CLS_MID_TF32_MIN_FSZ=48
```

Warmup=8, repeat=31, `--panel-cap 32`:

| case | B | FP32 factor ms/sys | TF32 cap32 + big-high256 ms/sys | speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|
| 25K | 64 | 0.115862 | 0.086959 | 1.332x | 5.01e-02 |
| 25K | 256 | 0.112856 | 0.090091 | 1.253x | 5.81e-02 |
| USA | 64 | 0.497735 | 0.394027 | 1.263x | 4.72e-02 |
| USA | 256 | 0.467153 | 0.388396 | 1.203x | 5.09e-02 |
| 8387 | 64 | 0.026050 | 0.023723 | 1.098x | 5.12e-02 |
| 8387 | 256 | 0.024412 | 0.020427 | 1.195x | 5.79e-03 |

This is the first repeat=31 configuration where both 25K and USA hit the 1.2x target for B64 and
B256 with Tensor Cores enabled. It is not a broad completion:

- Repeat=61 kept 25K strong (B64/B256 = 1.328x/1.285x) but USA dropped to
  B64/B256 = 1.123x/1.119x.
- 8387 remains below target because fsz<=16 dominates the front distribution; at cap32, fsz<=16 is
  7122 of 7334 fronts and about 60% of f2 work.
- The 13K case is absent from `/datasets/power_system/nr_linear_systems`, so it was not measured in
  this continuation.

Rejected follow-ups after this point:

- Allowing small TF32 at B64 with split `<=16` / `17..32` buckets did not help. 25K B64/B256 fell to
  1.19x/1.22x, USA B64 stayed 1.14x, USA B256 produced a bad relres (~1.06), and 8387 still missed.
- Combining `CLS_TF32_GLOBAL_COLUMN_USOLVE=ON` with cap32 + big-high256 regressed USA to
  B64/B256 = 1.16x/1.12x and worsened 25K residuals.
- Caps above 32 on USA printed `batch_relres=0` and did not improve speedup enough anyway; they are
  not considered valid evidence.

## Continuation: bighigh384 plus bigshared512

The next attempt paired the global big-high 384-thread variant with 512 threads for the
shared-resident big-low TF32 kernel:

```text
CLS_BIG_LOW_SPLIT=ON
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON
CLS_BIG_TF32_THREADS_384=ON
CLS_BIG_TF32_SHARED_THREADS_512=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON
CLS_MID_TF32_LOW_TC=ON
CLS_TF32_COLUMN_USOLVE=ON
CLS_RESPECT_PANEL_CAP=ON
CLS_MID_TF32_MIN_FSZ=48
```

Warmup=8, repeat=31, `--panel-cap 32`:

| case | B | FP32 factor ms/sys | TF32 cap32 + big384/shared512 ms/sys | speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|
| 25K | 64 | 0.121242 | 0.092218 | 1.315x | 5.70e-02 |
| 25K | 256 | 0.111106 | 0.088780 | 1.251x | 7.64e-02 |
| USA | 64 | 0.474559 | 0.384809 | 1.233x | 4.96e-02 |
| USA | 256 | 0.444993 | 0.370582 | 1.201x | 5.22e-02 |
| 8387 | 64 | 0.024737 | 0.026680 | 0.927x | 5.10e-02 |
| 8387 | 256 | 0.020386 | 0.020740 | 0.983x | 3.61e-02 |

This is the best repeat=31 large-case configuration so far: both 25K and USA hit the B64/B256
target with Tensor Cores enabled. It is still not a completion:

- Repeat=61 kept 25K strong again (B64/B256 = 1.305x/1.321x).
- Repeat=61 kept USA B256 at 1.207x, but USA B64 fell to 1.120x.
- 8387 regressed because the extra big-tier tuning does not touch its fsz<=16-dominant work.

Other follow-ups checked and rejected in this continuation:

- `CLS_SMALL_FRONT_MAX_16=ON` + `CLS_MID_TF32_MIN_FSZ=16` routes 17..32 fronts through the mid
  direct-shared TF32 path. It regressed 8387 and did not beat the cap32/big-high256 candidate on
  25K/USA.
- `--serial-nd` did not help 8387 and left USA near 1.17..1.20x.
- `--no-multistream` worsened absolute time and stayed below target.
- FP16 Tensor Core precision was not a viable large-batch replacement: 25K residuals became too
  large in some runs and speedups were below the TF32 candidates.
- `CLS_MID_HIGH_SPLIT=ON`, `CLS_MID_TF32_TC_THREADS_512=ON`, mid direct fuse-extend, and all TF32
  fuse-extend were all slower or produced bad residuals.
- The new `CLS_BIG_TF32_THREADS_128` experiment was removed from source after testing; USA B64
  collapsed to about 1.0x.
- A graph-node nsys profile of USA B256 cap32 showed the remaining TF32 factor kernels are still
  dominated by `factor_big_tf32_ptx`, `factor_small`, `factor_mid_tf32_ptx`, and
  `factor_big_shared_tf32_blocked`. The missing B64 margin is only a few percent, but it is in
  kernels where further block-size/fuse knobs have already become unstable.

## Continuation: stable 512-thread no-fuse cap policy

The previous all-fuse attempt mixed mid direct fuse-extend with big fuse-extend. Retrying only the
global big-high TF32 trail+extend fuse on the bighigh384/bigshared512 build initially looked good,
but repeat=61 rechecks of USA B256 were unstable around the 1.2x boundary. The more stable large-case
policy is the same bighigh/shared-low split with the default 512-thread global big-high TF32 fallback
and no TF32 trail+extend fuse:

```text
CLS_BIG_LOW_SPLIT=ON
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON
CLS_BIG_TF32_SHARED_THREADS_512=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON
CLS_MID_TF32_LOW_TC=ON
CLS_TF32_COLUMN_USOLVE=ON
CLS_RESPECT_PANEL_CAP=ON
CLS_FUSE_TF32_TRAIL_EXTEND=OFF
CLS_MID_TF32_DIRECT_FUSE_EXTEND=OFF
CLS_MID_TF32_MIN_FSZ=48
```

Warmup=8, repeat=61, with a batch-specific panel cap (`cap31` for B64, `cap32` for B256):

| case | B | panel cap | FP32 factor ms/sys | TF32 policy ms/sys | speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|---:|
| 25K | 64 | 31 | 0.111079 | 0.090940 | 1.222x | 5.42e-02 |
| 25K | 256 | 32 | 0.110488 | 0.086665 | 1.275x | 5.54e-02 |
| USA | 64 | 31 | 0.470502 | 0.378995 | 1.241x | 4.75e-02 |
| USA | 256 | 32 | 0.442969 | 0.359998 | 1.231x | 5.26e-02 |

This is the strongest repeat=61 result so far for the large cases: both 25K and USA land in the
1.2..1.4x target band at B64 and B256 with Tensor Cores as the active trailing-update enabler. The
policy is not a single universal cap, and the speedup partly depends on `CLS_RESPECT_PANEL_CAP`
making the cap visible to both FP32 and TF32 runs.

The same path still does not solve 8387. A repeat=31 cap sweep over 8, 16, 24, 28, 31, and 32
peaked near 1.16x at B64 and stayed near 1.0x at B256. A per-front dump at cap28 showed why:
within fsz<=16, the dominant tiny fronts are `(fsz,nc,uc)=(6,2,4)`, `(4,2,2)`, `(5,2,3)`,
and `(3,1,2)`. Their `nc=1/2` trailing updates cannot amortize a TF32 K=8 tile, so adding another
tiny-front TC path would mostly add padding and control overhead.

That was tested explicitly with an off-source-cleanup experiment: `CLS_SMALL_FRONT_MAX_8=ON`
routed 9..32 fronts through the mid kernel, and `CLS_MID_TF32_TINY_TC=ON` enabled direct-shared
TF32 TC for `fsz>8`, `uc>=8`, `nc<=4` fronts with `CLS_MID_TF32_MIN_FSZ=8`. It compiled, but
repeat=31 on 8387 regressed every cap:

| cap | B | FP32 ms/sys | TF32 tiny-TC ms/sys | speedup | TF32 relres |
|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 0.025594 | 0.030834 | 0.830x | 3.59e-02 |
| 8 | 256 | 0.020768 | 0.027194 | 0.764x | 3.81e-02 |
| 24 | 64 | 0.026275 | 0.029504 | 0.891x | 2.98e-02 |
| 24 | 256 | 0.020946 | 0.025886 | 0.809x | 5.01e-02 |
| 32 | 64 | 0.026130 | 0.032945 | 0.793x | 2.32e-02 |
| 32 | 256 | 0.021328 | 0.026232 | 0.813x | 3.86e-02 |

Those temporary options were removed from source. The result reinforces that 8387's remaining
non-GEMM work needs a small-front structural change, not a smaller TC tile gate.

Two non-GEMM follow-ups were also rejected:

- Reintroducing the analyze-time `a_pos_unique` scatter-store path correctly reduced absolute
  scatter cost, but it lowered the FP32 baseline more than the TF32 path on USA B256; repeat=61
  speedup fell to roughly 1.13..1.18x depending on cap. It was reverted because the active target is
  FP32-relative TC speedup, not absolute FP32 factor time.
- Splitting small buckets at `fsz<=8` exposed more `factor_small<8>` work and improved some B256
  absolute times, but the ratio did not move: 8387 stayed around 1.0..1.08x and the `mid32`
  combination regressed B256. That option was removed after the experiment.
- A no-extra-launch variable-lane small kernel (`sg8/sg16/sg32` in one kernel over sorted
  `<=8/<=16/<=32` buckets) also compiled and produced correct residuals, but 8387 repeat=31 still
  peaked at only about 1.13x for B64 and stayed near 1.0..1.05x for B256. It was removed from source.
- A one-thread-per-front tiny-lane path for `fsz<=8` was also tested to attack the dominant
  `(fsz,nc,uc)=(6,2,4),(4,2,2),(5,2,3)` leaves without extra launches. It used one scalar thread
  per tiny front with a private 8x8 shared slab, then dispatched the remaining small/mid/big buckets
  normally. Repeat=31 on 8387 stayed below target and worsened B256: cap28 B64/B256 =
  `0.028410/0.027515 = 1.03x` and `0.025247/0.026115 = 0.97x`; cap32 B64/B256 =
  `0.034035/0.032247 = 1.06x` and `0.025070/0.025199 = 0.99x`. The option was removed from source.
- A no-extend diagnostic (`CLS_DISABLE_EXTEND_ADD=ON`) skipped all contribution-block parent
  updates to test whether a perfect parent-update redesign could close the 8387 gap. It could not:
  repeat=31 no-extend speedups were cap28 B64/B256 = `1.05x/1.06x` and cap32 B64/B256 =
  `1.10x/0.99x`, with invalid solves as expected. The parent update is real work, but removing it
  improves FP32 too much to become a TC-ratio enabler.
- `CLS_SMALL_TF32_TC=ON` was rechecked on top of the stable large-case policy. A repeat=31 cap24
  B256 run briefly showed `1.20x`, but repeat=61 did not hold: cap24 B64/B256 =
  `0.024096/0.025661 = 0.94x` and `0.022016/0.021173 = 1.04x`. The split16 combination also missed:
  repeat=31 cap24/28/30 stayed below target or regressed B256. Small TF32 remains rejected.
- A broader 8387 upper-bound check (`docs/36`) then removed the entire small tier and parent
  extend-add as timing diagnostics. Even that did not produce a stable B256 target point:
  `CLS_DISABLE_SMALL_FACTOR=ON` peaked around B256 `1.12x`; combined
  `CLS_DISABLE_SMALL_FACTOR=ON` + `CLS_DISABLE_EXTEND_ADD=ON` swept caps 16..32 and peaked at B256
  `1.16x`. This strongly suggests that 8387 needs a structural ordering/panelization or packed
  many-front TC design, not another local small/mid/extend knob.

The new ceiling is clearer from B256 profiles:

- 25K B256 TF32 still spends large time in `factor_small` plus `scatter_values`; mid TC is no
  longer the only dominant kernel.
- USA B256 still has `factor_big_tf32_ptx` for fsz>159, shared big-low, small fronts, and scatter.
  Full-front shared memory cannot extend beyond fsz=159 on sm_86, so the remaining big-high path
  needs a different tiling strategy rather than more shared-resident tuning.

Next viable direction is not another thin-GEMM micro-optimization. It should make TC affect more of
the non-GEMM path, for example:

- blocked or warp-specialized panel LU so trailing becomes a larger wall fraction,
- pipelined mid-front packing that overlaps stage-in with factor work,
- a parent-update redesign that removes more global C traffic without paying extra atomics.
