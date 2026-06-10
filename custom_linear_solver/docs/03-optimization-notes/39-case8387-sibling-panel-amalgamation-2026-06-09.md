# 8387 Sibling Panel Amalgamation Check

**Date**: 2026-06-09

## Question

Docs 36-38 closed the cheap local paths for 8387:

- small/tiny Tensor Core variants do not hold,
- packed tiny Tensor Core has poor tile efficiency,
- METIS seed/cap variance does not create a stable B64/B256 `1.2x` pair.

The next structural question was whether we can safely merge sibling panels that share the same
parent, creating wider fronts without arbitrary invalid panelization.

## Diagnostic Implementation

Added default-OFF CMake option:

```text
CLS_SIBLING_PANEL_AMALGAMATE=OFF
CLS_SIBLING_PANEL_AMALGAMATE_CAP=8
```

When enabled, the analyzer:

1. Builds the normal `relaxed_panels` partition.
2. Builds a base `multifrontal_symbolic`.
3. Greedily merges consecutive relaxed panels while:
   - they have the same base `panel_parent`,
   - the common parent is outside/above the merge group,
   - merged `ncols <= CLS_SIBLING_PANEL_AMALGAMATE_CAP`.
4. Rebuilds symbolic metadata for the candidate.
5. Accepts it only if:
   - every non-root `panel_parent[p] > p`,
   - every `asm_idx >= 0`.
6. Falls back to the base partition if the candidate violates an invariant.

This is intentionally a diagnostic structural knob. Default OFF preserves the stable large-case
policy.

## Shape Effect

On 8387, the candidate is valid and does change the shape.

Example, sibling cap 8, runtime cap 28, parallel ND seed 42:

```text
[analyze] sibling-amalgamate: panels 7306 -> 6622, padded_fill 1.025x
```

The same run shifted the front histogram:

| Bucket | Base count | Sibling8 count |
|---|---:|---:|
| fsz 1..16 | 7092 | 6329 |
| fsz 17..32 | 174 | 234 |
| fsz 33..48 | 32 | 41 |
| fsz 49..64 | 9 | 14 |
| fsz 65..96 | 6 | 4 |

Example, sibling cap 16, runtime cap 32, parallel ND seed 42:

```text
[analyze] sibling-amalgamate: panels 7321 -> 6662, padded_fill 1.026x
```

So the idea is not blocked by multifrontal invariants. It really reduces tiny-panel count with only
about 2.5-2.6% padded-fill growth in these samples.

## Timing Results

All timings use the stable large-case TF32 policy plus sibling amalgamation. Values are
`batch_factor_per_sys_ms`; speedup is `fp32 / tf32`.

### Sibling cap 8

Initial repeat=7 sweep over seeds `{1,7,42}`, caps `{20,24,28,30,32}`:

| Best paired setting | B64 speedup | B256 speedup |
|---|---:|---:|
| seed 42, cap 32 | 1.105 | 1.160 |

Repeat=31 verification:

| ND mode | Seed | Cap | B64 speedup | B256 speedup | Note |
|---|---:|---:|---:|---:|---|
| parallel | 42 | 32 | 0.997 | 1.312 | B256-only win; B64 fails |
| serial | 42 | 32 | 1.037 | 1.015 | deterministic structural effect is small |

### Sibling cap 16

Repeat=31, seed 42, cap 32:

| ND mode | B64 speedup | B256 speedup |
|---|---:|---:|
| serial | 1.037 | 1.015 |
| parallel | 1.127 | 1.102 |

Sibling cap 16 also produced a slow single-repeat TF32 sample (`batch_factor_per_sys_ms=0.0449`) in
an analyze-info run, so this path has nontrivial scheduling/fill risk.

Raw files:

- `/tmp/cls_8387_sibling8_sweep_20260609_161549.tsv`
- `/tmp/cls_8387_sibling8_verify_20260609_161655.tsv`
- `/tmp/cls_8387_sibling8_serial_verify_20260609_161713.tsv`
- `/tmp/cls_8387_sibling16_parallel_verify_20260609_161805.tsv`

## Conclusion

Invariant-safe sibling amalgamation is feasible but does not solve 8387.

The most important observation is that the deterministic serial-ND effect is tiny: about
`1.04/1.02x` for B64/B256 at cap32. Parallel-ND can show larger one-batch or one-repeat wins, but
not a stable paired `1.2x` result. The likely reason is that sibling merging reduces tiny-panel
overhead while also increasing common FP32 work, front fill, and scheduling variance. It does not
specifically amplify Tensor Core-covered work enough to make TF32 the clear enabler.

This closes the simple safe-amalgamation path. A stronger panelization redesign would need to be
more deliberate than "merge same-parent siblings":

1. target shapes that cross the TC gate (`nc` and `uc`, not just `fsz`),
2. preserve parent contribution invariants,
3. cap padded fill and level scheduling damage,
4. prove it improves TF32 more than FP32.
