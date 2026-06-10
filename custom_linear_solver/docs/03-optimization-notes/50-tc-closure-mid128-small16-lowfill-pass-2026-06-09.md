# TC-closure + mid128 + small16 Low-fill Pass

**Date**: 2026-06-09

## Question

Can 8387/13K, previously failed by local per-front TC toggles, reach B64/B256 `1.2..1.4x`
factorize speedup versus FP32 with Tensor Cores as the enabler?

## New Diagnostic Policy

Build:

```text
build-tc-colusolve-respectcap-bighigh512-bigshared512-tcclosure32-mid128-small16-current
```

Relevant CMake flags:

```text
CLS_BIG_LOW_SPLIT=ON
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON
CLS_BIG_TF32_SHARED_THREADS_512=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON
CLS_MID_TF32_TC_THREADS_128=ON
CLS_MID_TF32_LOW_TC=ON
CLS_MID_TF32_LOW_TC_FORCE_ALL=ON
CLS_TF32_COLUMN_USOLVE=ON
CLS_SMALL_FRONT_MAX_16=ON
CLS_RESPECT_PANEL_CAP=ON
CLS_MID_TF32_MIN_FSZ=48
CLS_TC_CLOSURE_PANEL_AMALGAMATE=ON
CLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP=32
```

Mechanism:

- `CLS_TC_CLOSURE_PANEL_AMALGAMATE` merges only consecutive panels that:
  - pass base parent-front containment,
  - pass final `multifrontal_symbolic` validation,
  - become Tensor-Core-routable: `fsz>32`, `uc>=16`, `4<=nc<=32`.
- `CLS_MID_TF32_TC_THREADS_128` reduces thread count for the new 49..64-dominant mid-TC work.
- `CLS_SMALL_FRONT_MAX_16` routes 17..32 fronts out of the small tier so low-mid TF32 dispatch can
  see more of the newly created work.

This is the first structural change that improves 8387/13K rather than merely changing noise.

## Structure Shift

At seed42/cap32, `--analyze-info` showed:

| case | panels before | panels after | groups | extra panels | padded fill | key shift |
|---|---:|---:|---:|---:|---:|---|
| 8387 | 7322 | 6354 | 84 | 968 | 1.152x | `49..64` f2 = 33.8% |
| 13K | 12339 | 10391 | 178 | 1948 | 1.213x | `49..64` f2 = 42.7% |

Unlike plain closure amalgamation, this does not merge every locally valid group. It spends fill
only when the resulting group is expected to hit the mid TF32 path.

## Accepted Low-fill Results

Runtime:

```text
--serial-nd --batch <64|256> --batch-only --repeat 61 --warmup 8
--single-precision fp64
```

Raw:

```text
/tmp/cls_tcclosure32_mid128_small16_serial_accept_r61_20260609.tsv
```

| case | seed | cap | B | fp32 ms/sys | tf32 ms/sys | fp32/tf32 | fp32 relres | tf32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case8387pegase | 7 | 30 | 64 | 0.0331602 | 0.0266655 | 1.244 | 3.34e-05 | 3.97e-02 |
| case8387pegase | 7 | 30 | 256 | 0.0289548 | 0.0239879 | 1.207 | 3.39e-05 | 3.97e-02 |
| case13659pegase | 99 | 32 | 64 | 0.0679609 | 0.0541622 | 1.255 | 1.21e-04 | 6.47e-03 |
| case13659pegase | 99 | 32 | 256 | 0.0572868 | 0.0461665 | 1.241 | 1.24e-04 | 6.47e-03 |

Summary:

| case | seed | cap | B64 speedup | B256 speedup | paired min |
|---|---:|---:|---:|---:|---:|
| case8387pegase | 7 | 30 | 1.244 | 1.207 | 1.207 |
| case13659pegase | 99 | 32 | 1.255 | 1.241 | 1.241 |

## Rejected Variants

### Plain closure amalgamation

Build:

```text
build-tc-colusolve-respectcap-bighigh512-bigshared512-closure32-current
```

It increased TC candidates but also merged non-TC useful work, raising fill and slowing both FP32
and TF32:

| case | config | B64 speedup | B256 speedup |
|---|---|---:|---:|
| 8387 seed42 cap32 | plain closure | ~0.98..1.00 | ~1.00..1.10 |
| 13K seed42 cap32 | plain closure | ~1.07 | ~1.05 |

### TC-closure without small16

Build:

```text
build-tc-colusolve-respectcap-bighigh512-bigshared512-tcclosure32-mid128-current
```

This was close but not stable enough:

| mode | case | seed | cap | repeat | B64 | B256 | result |
|---|---|---:|---:|---:|---:|---:|---|
| parallel ND | 8387 | 7 | 31 | 61 | 1.173 | 1.157 | fail |
| serial ND | 8387 | 7 | 28 | 61 | 1.218 | 1.197 | fail by 0.3% |
| serial ND | 13K | 99 | 32 | 61 | 1.246 | 1.242 | pass |

`CLS_SMALL_FRONT_MAX_16` was the small extra dispatch change that pushed 8387 B256 over the target.

### TC-closure cap above 32

Build:

```text
build-tc-colusolve-respectcap-bighigh512-bigshared512-tcclosure48-current
```

Runtime caps 34, 36, 40, and 48 are invalid for 8387/13K: FP32 residual already grows to
`1e35..inf`. The safe low-fill cap remains <=32.

### NTJ8_16

`CLS_TF32_DIRECT_NTJ8_16=ON` regressed both 8387 and 13K under the low-fill policy. Do not combine
it with this path.

## Large-case Interaction

The low-fill build is not a universal replacement for the stable large-case policy.

Observed with the low-fill build:

- USA still passes repeat=61: B64/B256 about `1.30/1.30x`.
- 25K is unstable under this build: a repeat=61 candidate dropped to B64 `1.192x` and had poor
  TF32 residual in some serial/parallel orderings.

Therefore keep the policies split:

1. low-fill cases 8387/13K: TC-closure + mid128 + small16, deterministic serial ND;
2. large cases 25K/70K/USA: existing stable large-case TF32 policy from docs/44.

## Decision

This is the first repeat=61 evidence that 8387 and 13K can meet the B64/B256 Tensor Core target.
The enabler is not a pure kernel switch; it is a structural TC-routable panelization plus a
mid-TF32 launch-shape change.

Do not mark this as a single universal policy yet. The correct current framing is a two-policy
result:

- large dense-update regime: stable mid/big TC policy;
- low-fill regime: TC-routable closure panelization + mid128 + small16.
