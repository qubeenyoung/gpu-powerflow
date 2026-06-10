# Parent Update Fan-in Profile

**Date**: 2026-06-09

**Context**: After the large-case TF32 policy reached the B64/B256 target on 25K and USA, 8387
remained below target. The literature survey suggested parent assembly/`extend_add` as the next
algorithmic object to inspect. This note adds a concrete fan-in/update-size profile before attempting
another parent-update kernel.

## Tooling Change

`--dump-fronts` now writes the original per-front fields plus parent-update metadata:

```text
q,p,fsz,nc,uc,level,parent,asm_len,extend_elems
```

Where:

- `parent` is the parent panel id (`-1` for root),
- `asm_len` is the number of update rows assembled into the parent,
- `extend_elems = uc * uc` for non-root fronts.

This is analysis-only host metadata. It does not change factor/solve dispatch or device kernels.

A separate diagnostic build option was also added:

```text
CLS_DISABLE_EXTEND_ADD=ON
```

It sets the factor dispatch `do_extend` flag to zero and skips the split cuBLAS extend kernels. This
intentionally produces invalid solves, but the factor timing gives a hard upper bound on what a
perfect parent-update removal could do. The option is OFF by default.

## Commands

Generated with the current stable large-case build:

```bash
build-tc-colusolve-respectcap-bighigh512-bigshared512-current/custom_linear_solver_run \
  /datasets/power_system/nr_linear_systems/case8387pegase \
  --batch 64 --batch-only --precision fp32 --panel-cap 28 \
  --warmup 0 --repeat 1 --dump-fronts /tmp/cls_front_dumps/8387_cap28.csv
```

The same command was run for:

- 25K cap31 and cap32,
- USA cap31 and cap32.

## Summary

| case/cap | fronts | parent fronts | parents | total `uc^2` extend elems | max children | max parent elems |
|---|---:|---:|---:|---:|---:|---:|
| 8387 cap28 | 7,282 | 7,281 | 3,164 | 229,360 | 32 | 5,280 |
| 25K cap31 | 22,540 | 22,539 | 9,835 | 1,089,910 | 21 | 17,342 |
| 25K cap32 | 22,574 | 22,573 | 9,759 | 1,125,618 | 29 | 15,900 |
| USA cap31 | 74,178 | 74,175 | 32,455 | 4,704,292 | 35 | 50,053 |
| USA cap32 | 74,121 | 74,118 | 32,407 | 4,564,068 | 31 | 56,473 |

## Update Size Distribution

Percent of total `extend_elems`:

| case/cap | `<=16` | `17..64` | `65..256` | `>256` |
|---|---:|---:|---:|---:|
| 8387 cap28 | 25.0% | 19.1% | 23.4% | 32.5% |
| 25K cap31 | 12.7% | 15.7% | 15.5% | 56.1% |
| 25K cap32 | 12.4% | 15.0% | 15.9% | 56.7% |
| USA cap31 | 11.0% | 12.4% | 12.7% | 63.8% |
| USA cap32 | 11.3% | 13.0% | 12.5% | 63.2% |

By `uc`, 8387 has only 11.7% of update elems at `uc>32`; USA has about 49%. This matches the
factorization evidence: large cases expose more mid/big dense work, while 8387 is tiny/small-update
heavy.

## Parent Fan-in Distribution

Percent of total `extend_elems` grouped by number of children that update the same parent:

| case/cap | fan-in 1 | fan-in 2 | fan-in 3..4 | fan-in 5..8 | fan-in 9..16 | fan-in >16 |
|---|---:|---:|---:|---:|---:|---:|
| 8387 cap28 | 8.9% | 25.6% | 24.3% | 20.0% | 17.7% | 3.5% |
| 25K cap31 | 4.5% | 17.1% | 21.1% | 19.6% | 29.9% | 7.9% |
| 25K cap32 | 6.0% | 16.6% | 22.4% | 14.6% | 28.9% | 11.4% |
| USA cap31 | 8.8% | 14.3% | 20.3% | 17.7% | 30.5% | 8.2% |
| USA cap32 | 7.6% | 14.1% | 20.2% | 17.6% | 33.0% | 7.7% |

This does not look like a simple "many children collide on one parent" bottleneck. Even on 25K/USA,
the highest fan-in buckets are important but not dominant enough to justify a broad two-stage parent
reduction without more kernel timing evidence. On 8387, fan-in 9+ is only about 21% of update elems.

## Interpretation

1. **8387 parent update is mostly tiny/small traffic, not high-conflict reduction.** A parent-grouped
   reduction might help selected parents, but it will not cover most of the 8387 wall.

2. **Large cases have more useful parent-update size.** 25K/USA have `>256` update elems at about
   56..64% of total. If parent-update redesign is attempted, it should start there, not on 8387.

3. **Unique scatter failure is consistent with this profile.** Optimizing mostly-uncontended writes
   can improve absolute time but may improve FP32 more than TF32, lowering the requested speedup
   ratio.

4. **No immediate parent-reduce implementation is justified.** The next implementation should first
   collect kernel-level timing by update-size bucket or parent fan-in bucket. Otherwise we risk
   adding another optimization that moves absolute time but not the TC-enabler ratio.

## Next Gate

Before writing a new parent-update kernel, collect one of:

- Nsight/CUPTI kernel timing split by `extend_add` shape bucket,
- an instrumented build that dispatches extend ranges by update-size/fan-in bucket,
- a selective A/B that only changes parents with fan-in `>=9` and `uc>=16`.

Go criterion: the targeted bucket must be at least `~10%` of factor wall and must improve TF32
policy speedup, not only absolute FP32 time.

## No-extend Upper Bound

Build:

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
CLS_MID_TF32_MIN_FSZ=48
CLS_DISABLE_EXTEND_ADD=ON
```

8387 repeat=31:

| cap | B | normal FP32/TF32 | normal speedup | no-extend FP32/TF32 | no-extend speedup |
|---:|---:|---:|---:|---:|---:|
| 28 | 64 | `0.026261 / 0.022922` | 1.146x | `0.022768 / 0.021696` | 1.049x |
| 28 | 256 | `0.022472 / 0.019986` | 1.124x | `0.019599 / 0.018542` | 1.057x |
| 32 | 64 | `0.023771 / 0.024672` | 0.963x | `0.023293 / 0.021136` | 1.102x |
| 32 | 256 | `0.022931 / 0.020570` | 1.115x | `0.018865 / 0.019155` | 0.985x |

This rules out a broad parent-update removal as the missing 8387 enabler. Removing all extend-add
traffic improves absolute times, especially FP32, but it does not create a stable 1.2x TF32/FP32
ratio. Parent redesign may still help absolute large-case time, but it is unlikely to close the
8387 TC-ratio gap by itself.
