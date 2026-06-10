# Nonlocal Panelization and Parallel-ND Threshold Diagnostic

**Date**: 2026-06-09

## Question

After local TC toggles, conservative sibling/chain amalgamation, and batch-dimension TC packing
failed for 8387/13K, can a more structural but still cheap change expose enough Tensor Core work?

Two diagnostic levers were tested:

1. validated greedy merge of arbitrary consecutive relaxed panels;
2. parallel nested-dissection recursion/base-threshold changes.

Both were default-OFF / diagnostic-only changes.

## Build-time Knobs Added

Panelization:

```text
CLS_VALIDATED_RUN_PANEL_AMALGAMATE=OFF
CLS_VALIDATED_RUN_PANEL_AMALGAMATE_CAP=32
```

This tries to merge consecutive relaxed panels into wider runs up to the cap, then recomputes
`multifrontal_symbolic` and accepts the candidate only if:

```text
panel_parent[p] > p for every non-root panel
asm_idx >= 0 for every contribution row
```

Ordering:

```text
CLS_PAR_ND_DEPTH=4
CLS_PAR_ND_SMALL_BASE_THR=4000
CLS_PAR_ND_LARGE_BASE_THR=20000
```

The defaults preserve the current behavior. The diagnostics build alternate parallel-ND shapes
without changing the numeric kernels.

## Validated Run Amalgamation Result

Build:

```text
build-tc-colusolve-respectcap-bighigh512-bigshared512-validrun32-current
```

Core TC policy:

```text
CLS_BIG_LOW_SPLIT=ON
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON
CLS_BIG_TF32_SHARED_THREADS_512=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON
CLS_MID_TF32_LOW_TC=ON
CLS_MID_TF32_LOW_TC_FORCE_ALL=ON
CLS_TF32_COLUMN_USOLVE=ON
CLS_RESPECT_PANEL_CAP=ON
CLS_MID_TF32_MIN_FSZ=48
CLS_VALIDATED_RUN_PANEL_AMALGAMATE=ON
CLS_VALIDATED_RUN_PANEL_AMALGAMATE_CAP=32
```

Observed with `--analyze-info`:

| case | runtime cap | result |
|---|---:|---|
| 8387 | 2 | candidate invalid, fallback base |
| 8387 | 8 | candidate invalid, fallback base |
| 8387 | 16 | candidate invalid, fallback base |
| 8387 | 32 | candidate invalid, fallback base |
| 13K | 2 | candidate invalid, fallback base |
| 13K | 8 | candidate invalid, fallback base |
| 13K | 16 | candidate invalid, fallback base |
| 13K | 32 | candidate invalid, fallback base |

Interpretation:

- Arbitrary consecutive panel runs are not a safe nonlocal panelization rule.
- Even very small cap-2 greedy runs can create contribution rows that are not contained in the
  selected parent front, or otherwise violate the panel-parent ordering invariant.
- The earlier sibling/chain rules are conservative for a real reason: they preserve assembly
  containment. A useful nonlocal rule needs a stronger closure/containment criterion, not just
  a wider contiguous window.

## Parallel-ND Threshold Builds

Deep recursion:

```text
build-tc-colusolve-respectcap-bighigh512-bigshared512-parnddeep-current
CLS_PAR_ND_SMALL_BASE_THR=1000
CLS_PAR_ND_LARGE_BASE_THR=4000
```

Shallow recursion:

```text
build-tc-colusolve-respectcap-bighigh512-bigshared512-parndshallow-current
CLS_PAR_ND_SMALL_BASE_THR=8000
CLS_PAR_ND_LARGE_BASE_THR=30000
```

Both used the same TC policy as above, with low-mid TF32 force-all enabled so 8387/13K would not be
hidden behind the medium-case size gate.

### Structure Check

At `seed=42`, `cap=32`, the front distributions still stayed in the low-fill regime.

| build | case | P | levels | `fsz<=16` count | `fsz<=16` f2% | 33..96 f2% |
|---|---|---:|---:|---:|---:|---:|
| deep | 8387 | 7275 | 21 | 7052 | 57.7 | 21.8 |
| shallow | 8387 | 7329 | 18 | 7115 | 59.2 | 21.4 |
| deep | 13K | 12291 | 20 | 11987 | 53.7 | 26.7 |
| shallow | 13K | 12334 | 21 | 12011 | 53.3 | 28.5 |

The threshold change perturbs ordering and front buckets, but it does not move enough work into
large, high-`nc` TC-friendly fronts.

### Repeat=7 Mini-sweep

Raw table:

```text
/tmp/cls_parnd_threshold_minisweep_20260609.tsv
```

Best paired results from the measured seed/cap candidates:

| build | case | seed | cap | B64 speedup | B256 speedup | paired min |
|---|---|---:|---:|---:|---:|---:|
| deep | 8387 | 7 | 28 | 1.089 | 0.998 | 0.998 |
| deep | 8387 | 42 | 32 | 1.035 | 1.052 | 1.035 |
| shallow | 8387 | 7 | 28 | 1.043 | 0.996 | 0.996 |
| shallow | 8387 | 42 | 32 | 0.983 | 0.997 | 0.983 |
| deep | 13K | 99 | 28 | 1.112 | 1.188 | 1.112 |
| deep | 13K | 42 | 28 | 1.050 | 1.147 | 1.050 |
| shallow | 13K | 99 | 31 | 1.097 | 1.071 | 1.071 |
| shallow | 13K | 7 | 31 | 1.064 | 1.043 | 1.043 |

The best 13K point improves B256 but still misses B64. 8387 regresses or stays near 1.0.

## Decision

Close these cheap structural diagnostics:

- arbitrary validated consecutive-run amalgamation is invalid even at tiny caps;
- parallel-ND recursion/base-threshold changes do not create a paired B64/B256 `1.2x` Tensor Core
  result for 8387/13K.

The remaining work needed to include 8387/13K is not another local switch. It would require either:

1. a real symbolic closure-based panelization algorithm that proves parent containment while
   increasing `nc`/`uc` enough for TC; or
2. a low-fill sparse-LU branch, likely improving FP32 and TF32 together rather than making Tensor
   Cores the ratio enabler.

The current repeat-backed Tensor Core claim remains the large-case one: 25K, 70K, and USA.
