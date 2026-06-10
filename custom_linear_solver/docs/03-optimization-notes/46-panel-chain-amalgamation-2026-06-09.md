# Panel-chain amalgamation diagnostic

**Date**: 2026-06-09

**Goal**: Check whether safe chain-wise panel amalgamation can create enough larger mid fronts for
8387/13K to reach the B64/B256 Tensor Core speedup target.

## Implementation

Added a default-off symbolic diagnostic:

- `CLS_PANEL_CHAIN_AMALGAMATE`
- `CLS_PANEL_CHAIN_AMALGAMATE_CAP`, default `32`

The pass starts from the relaxed panel partition, builds the base multifrontal symbolic structure,
then greedily merges consecutive panel-tree chains while:

- the previous panel's parent is the next panel,
- cumulative `ncols <= cap`,
- the rebuilt multifrontal symbolic passes invariant checks:
  - every non-root `panel_parent[p] > p`,
  - every `asm_idx >= 0`.

This is intentionally conservative. It does not merge arbitrary ancestors, siblings, or nonlocal
subtrees. The point was to test whether a safe local tree-chain merge is enough before attempting a
larger symbolic rewrite.

Build:

```bash
cmake -S custom_linear_solver \
  -B build-tc-colusolve-respectcap-bighigh512-bigshared512-chainamalg-current \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLS_BIG_LOW_SPLIT=ON \
  -DCLS_MID_LOW_SPLIT=ON \
  -DCLS_BIG_TF32_BLOCKED_TC=ON \
  -DCLS_BIG_TF32_SHARED_THREADS_512=ON \
  -DCLS_MID_TF32_TC=ON \
  -DCLS_MID_TF32_DIRECT_SHARED=ON \
  -DCLS_MID_TF32_LOW_TC=ON \
  -DCLS_MID_TF32_LOW_TC_FORCE_ALL=ON \
  -DCLS_TF32_COLUMN_USOLVE=ON \
  -DCLS_RESPECT_PANEL_CAP=ON \
  -DCLS_FUSE_TF32_TRAIL_EXTEND=OFF \
  -DCLS_MID_TF32_DIRECT_FUSE_EXTEND=OFF \
  -DCLS_MID_TF32_MIN_FSZ=48 \
  -DCLS_PANEL_CHAIN_AMALGAMATE=ON \
  -DCLS_PANEL_CHAIN_AMALGAMATE_CAP=32
cmake --build build-tc-colusolve-respectcap-bighigh512-bigshared512-chainamalg-current -j
```

## Structural effect

Representative `--analyze-info`, `--panel-cap 32`, `--metis-seed 42`:

```text
8387:
[analyze] panel-chain-amalgamate: panels 7274 -> 7162, padded_fill 1.019x
[analyze] panel-chain-amalgamate: panels 7317 -> 7219, padded_fill 1.014x

13K:
[analyze] panel-chain-amalgamate: panels 12336 -> 12181, padded_fill 1.020x
[analyze] panel-chain-amalgamate: panels 12325 -> 12152, padded_fill 1.026x
```

The runner performs more than one analysis in this path, so the exact counts vary slightly between
the setup and validation phases. The important point is stable: chain amalgamation reduces panel
count by only about `1.3..1.7%` with `~1.01..1.03x` padded fill.

Final front dumps from the same diagnostic build:

| Case | fronts | `fsz<=16` | `17..32` | `33..48` | `49..64` | `65..96` | `97..160` | low/mid TC candidates | high TC candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8387 cap32 seed42 | 7189 | 6987 | 155 | 25 | 18 | 4 | 0 | 63 | 11 |
| 13K cap32 seed42 | 12151 | 11871 | 186 | 56 | 24 | 12 | 2 | 135 | 20 |

Candidate definitions:

- low/mid TC: `fsz > 16 && uc >= 16 && 4 <= nc <= 32`
- high TC: `fsz > 48 && uc >= 32 && 8 <= nc <= 32`

This is the decisive structural result. The safe chain merge barely moves the tiny-front dominance:
8387 still has `6987/7189 = 97.2%` fronts at `fsz<=16`, and 13K still has
`11871/12151 = 97.7%` fronts at `fsz<=16`.

## Timing sweep, repeat=7

Raw output: `/tmp/cls_chainamalg_sweep_r7_20260609.tsv`

Same build for FP32 and TF32; speedup is `fp32_ms / tf32_ms`.

### 8387

| seed | cap | B64 | B256 | paired min |
|---:|---:|---:|---:|---:|
| 7 | 28 | 1.244 | 0.971 | 0.971 |
| 7 | 31 | 1.058 | 1.003 | 1.003 |
| 7 | 32 | 1.103 | 1.001 | 1.001 |
| 42 | 28 | 0.975 | 0.986 | 0.975 |
| 42 | 31 | 1.066 | 0.977 | 0.977 |
| 42 | 32 | 1.027 | 1.011 | 1.011 |
| 99 | 28 | 1.030 | 0.998 | 0.998 |
| 99 | 31 | 1.015 | 1.038 | 1.015 |
| 99 | 32 | 1.050 | 0.998 | 0.998 |

### 13K

| seed | cap | B64 | B256 | paired min |
|---:|---:|---:|---:|---:|
| 7 | 28 | 1.025 | 0.974 | 0.974 |
| 7 | 31 | 1.107 | 1.111 | 1.107 |
| 7 | 32 | 1.036 | 1.049 | 1.036 |
| 42 | 28 | 1.042 | 0.978 | 0.978 |
| 42 | 31 | 1.022 | 1.055 | 1.022 |
| 42 | 32 | 0.995 | 0.938 | 0.938 |
| 99 | 28 | 0.932 | 1.123 | 0.932 |
| 99 | 31 | 1.160 | 1.008 | 1.008 |
| 99 | 32 | 1.046 | 1.069 | 1.046 |

No candidate reaches the paired B64/B256 target. 8387 has one noisy B64-only point at `1.244x`,
but B256 is slower. 13K has no `>=1.2x` point and the best paired min is only `1.107x`.

## Timing verification, repeat=31

Raw output: `/tmp/cls_chainamalg_verify_r31_20260609.tsv`

| Case | seed | cap | B | FP32 ms/sys | TF32 ms/sys | speedup | FP32 relres | TF32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8387 | 7 | 28 | 64 | 0.0266097 | 0.0243851 | 1.091 | 3.28e-05 | 4.50e-02 |
| 8387 | 7 | 28 | 256 | 0.0208174 | 0.0221174 | 0.941 | 2.25e-05 | 5.35e-02 |
| 13K | 7 | 31 | 64 | 0.0425122 | 0.0382182 | 1.112 | 1.19e-04 | 6.37e-03 |
| 13K | 7 | 31 | 256 | 0.0358956 | 0.0357545 | 1.004 | 1.20e-04 | 7.11e-03 |
| 13K | 99 | 31 | 64 | 0.0444998 | 0.0427152 | 1.042 | 1.18e-04 | 6.96e-03 |
| 13K | 99 | 31 | 256 | 0.0398692 | 0.0359305 | 1.110 | 1.24e-04 | 6.89e-03 |

The repeat=31 verification rejects the only suspicious repeat=7 wins. The paired minimum is:

- 8387 seed7 cap28: `0.941x`
- 13K seed7 cap31: `1.004x`
- 13K seed99 cap31: `1.042x`

## Conclusion

Close safe panel-chain amalgamation as a Tensor Core enabler for 8387/13K.

It is invariant-safe and cheap in fill, but it is too weak structurally. It reduces panel count by
only `~1..2%` and leaves both cases with about `97%` tiny `fsz<=16` fronts. That is not enough to
materially increase Tensor Core-covered work, and same-build FP32/TF32 timing confirms it does not
reach the paired B64/B256 `1.2..1.4x` target.

The remaining credible structural routes are stronger and riskier:

1. A nonlocal panelization/tree rewrite with an explicit target of moving many `fsz<=16` fronts into
   useful `nc/uc` bands.
2. A separate low-fill sparse-LU branch for 8387/13K absolute performance.
3. Excluding 8387/13K from the raw factorize Tensor Core claim and keeping the accepted
   25K/70K/USA large-case evidence.
