# Sibling + chain amalgamation combo diagnostic

**Date**: 2026-06-09

**Goal**: Test the strongest conservative symbolic panelization combination before moving to a
nonlocal rewrite: sibling-panel amalgamation followed by parent-child chain amalgamation.

This combines the two invariant-safe diagnostics:

- `CLS_SIBLING_PANEL_AMALGAMATE`
- `CLS_PANEL_CHAIN_AMALGAMATE`

The question is whether the combination can move enough 8387/13K fronts out of the tiny
`fsz<=16` regime to make the TF32 Tensor Core path the paired B64/B256 winner.

## Builds

Valid cap32 build:

```bash
cmake -S custom_linear_solver \
  -B build-tc-colusolve-respectcap-bighigh512-bigshared512-sibling32-chain-current \
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
  -DCLS_SIBLING_PANEL_AMALGAMATE=ON \
  -DCLS_SIBLING_PANEL_AMALGAMATE_CAP=32 \
  -DCLS_PANEL_CHAIN_AMALGAMATE=ON \
  -DCLS_PANEL_CHAIN_AMALGAMATE_CAP=32
cmake --build build-tc-colusolve-respectcap-bighigh512-bigshared512-sibling32-chain-current -j
```

Aggressive cap48 build was also tested:

```bash
-DCLS_SIBLING_PANEL_AMALGAMATE_CAP=48
-DCLS_PANEL_CHAIN_AMALGAMATE_CAP=48
```

That build compiles, but `--panel-cap 40/48` is numerically invalid for both 8387 and 13K: FP32
`batch_relres` jumps from the normal `1e-5..1e-4` range to hundreds or worse. Those results are not
eligible timing candidates.

## Structure, cap32 seed42

Raw logs:

- `/tmp/cls_8387_sibling32_chain_analyze_20260609.log`
- `/tmp/cls_13k_sibling32_chain_analyze_20260609.log`

The runner performs more than one analysis in this command path, so the exact panel count varies
between setup and validation. Representative final pass:

```text
8387:
[analyze] sibling-amalgamate: panels 7325 -> 6608, padded_fill 1.044x
[analyze] panel-chain-amalgamate: panels 6608 -> 6404, padded_fill 1.037x

13K:
[analyze] sibling-amalgamate: panels 12286 -> 9102, padded_fill 1.052x
[analyze] panel-chain-amalgamate: panels 9102 -> 8776, padded_fill 1.048x
```

Final front dumps:

- `/tmp/cls_8387_sibling32_chain_cap32_seed42_fronts_20260609.csv`
- `/tmp/cls_13k_sibling32_chain_cap32_seed42_fronts_20260609.csv`

| Case | fronts | `fsz<=16` | tiny pct | `17..32` | `33..48` | `49..64` | `65..96` | `97..160` | low/mid TC candidates | high TC candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8387 cap32 seed42 | 6404 | 6205 | 96.89% | 120 | 55 | 16 | 7 | 1 | 85 | 16 |
| 13K cap32 seed42 | 8776 | 8456 | 96.35% | 180 | 84 | 43 | 12 | 1 | 184 | 25 |

Compared with chain-only, the combo moves more structure:

- 8387: `~7189 -> 6404` fronts.
- 13K: `~12151 -> 8776` fronts.

But the decisive distribution is still tiny-front dominated. Even after the strongest conservative
safe merge, both cases remain above `96%` at `fsz<=16`.

## Timing sweep, cap32 build, repeat=7

Raw output: `/tmp/cls_sibling32_chain_sweep_r7_20260609.tsv`

Same build for FP32 and TF32; speedup is `fp32_ms / tf32_ms`.

### 8387

| seed | cap | B64 | B256 | paired min |
|---:|---:|---:|---:|---:|
| 7 | 28 | 1.062 | 0.949 | 0.949 |
| 7 | 31 | 1.036 | 1.024 | 1.024 |
| 7 | 32 | 1.015 | 1.008 | 1.008 |
| 42 | 28 | 1.137 | 1.096 | 1.096 |
| 42 | 31 | 1.025 | 0.966 | 0.966 |
| 42 | 32 | 1.038 | 0.977 | 0.977 |
| 99 | 28 | 1.029 | 0.981 | 0.981 |
| 99 | 31 | 1.082 | 0.966 | 0.966 |
| 99 | 32 | 1.070 | 1.151 | 1.070 |

### 13K

| seed | cap | B64 | B256 | paired min |
|---:|---:|---:|---:|---:|
| 7 | 28 | 0.995 | 0.926 | 0.926 |
| 7 | 31 | 1.037 | 1.021 | 1.021 |
| 7 | 32 | 0.997 | 0.995 | 0.995 |
| 42 | 28 | 1.122 | 1.100 | 1.100 |
| 42 | 31 | 0.951 | 1.037 | 0.951 |
| 42 | 32 | 0.994 | 1.172 | 0.994 |
| 99 | 28 | 1.132 | 1.149 | 1.132 |
| 99 | 31 | 1.072 | 1.060 | 1.060 |
| 99 | 32 | 1.157 | 1.065 | 1.065 |

No repeat=7 candidate reaches the paired target. The closest 13K point, seed99 cap28, is
`1.132/1.149x`, still below `1.2x`.

## Timing verification, cap32 build, repeat=31

Raw output: `/tmp/cls_sibling32_chain_verify_r31_20260609.tsv`

| Case | seed | cap | B | FP32 ms/sys | TF32 ms/sys | speedup | FP32 relres | TF32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8387 | 42 | 28 | 64 | 0.0275362 | 0.0280272 | 0.982 | 1.26e-05 | 5.70e-02 |
| 8387 | 42 | 28 | 256 | 0.0246629 | 0.0244953 | 1.007 | 3.33e-05 | 6.04e-02 |
| 8387 | 99 | 32 | 64 | 0.0291952 | 0.0289266 | 1.009 | 1.04e-05 | 5.81e-02 |
| 8387 | 99 | 32 | 256 | 0.0258786 | 0.0250866 | 1.032 | 2.61e-05 | 9.94e-03 |
| 13K | 42 | 28 | 64 | 0.0480916 | 0.0468026 | 1.028 | 1.09e-04 | 6.48e-03 |
| 13K | 42 | 28 | 256 | 0.0484098 | 0.0433440 | 1.117 | 1.17e-04 | 6.34e-03 |
| 13K | 99 | 28 | 64 | 0.0488150 | 0.0497937 | 0.980 | 1.13e-04 | 5.96e-03 |
| 13K | 99 | 28 | 256 | 0.0454229 | 0.0465399 | 0.976 | 1.12e-04 | 6.62e-03 |

The repeat=31 pass removes the apparent repeat=7 gains. The paired minima are:

- 8387 seed42 cap28: `0.982x`
- 8387 seed99 cap32: `1.009x`
- 13K seed42 cap28: `1.028x`
- 13K seed99 cap28: `0.976x`

## Aggressive cap40/48 validity check

Build:

- `build-tc-colusolve-respectcap-bighigh512-bigshared512-sibling48-chain48-current`

Command shape:

```bash
custom_linear_solver_run <case> \
  --batch 1 --batch-only --repeat 1 \
  --precision <fp32|tf32> --single-precision fp64 \
  --panel-cap <40|48> --metis-seed <42|99> --analyze-info
```

Representative invalid results:

| Case | cap | seed | precision | batch relres |
|---|---:|---:|---|---:|
| 8387 | 40 | 42 | fp32 | 3.98e+05 |
| 8387 | 48 | 42 | tf32 | 1.31e+32 |
| 13K | 40 | 42 | fp32 | 8.92e+02 |
| 13K | 40 | 99 | tf32 | 5.64e+31 |
| 13K | 48 | 99 | fp32 | 2.24e+03 |

So cap40/48 is not a valid route. It breaks FP32 before a Tensor Core speedup claim can be made.

## Conclusion

Close the conservative symbolic-amalgamation path for 8387/13K.

The sibling+chain combination is stronger than either diagnostic alone and remains invariant-safe
at cap32. It still leaves both cases above `96%` tiny fronts and does not produce stable paired
B64/B256 Tensor Core speedups. Raising the cap to 40/48 is numerically invalid, including for FP32.

At this point, "slightly safer amalgamation" is exhausted. A remaining panelization attempt would
need to be a real nonlocal symbolic redesign with an explicit correctness strategy, not another
local merge heuristic. Otherwise, the credible performance route for 8387/13K is a low-fill
sparse-LU branch, while the Tensor Core factorize claim should stay anchored on 25K/70K/USA.
