# 8387 Ordering Seed Sweep

**Date**: 2026-06-09

## Question

Could METIS ordering variance create a 8387 front distribution where the existing TF32 Tensor Core
policy reaches the B64/B256 `1.2x` target versus FP32?

This is the low-cost version of the "deeper ordering/panelization" idea from docs 36-37. It does
not change the symbolic algorithm, only the METIS seed and panel cap.

## Build And Method

Build:

```bash
build-tc-colusolve-respectcap-bighigh512-bigshared512-current
```

Relevant build policy:

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

Runtime seed support was added through:

- `SolverConfig::metis_seed`
- `PlanBuildOptions::metis_seed`
- CLI `--metis-seed N`
- METIS `METIS_OPTION_SEED` in serial NodeND and the parallel separator/base recursion

Initial sweep:

```bash
case=/datasets/power_system/nr_linear_systems/case8387pegase
seeds=(1 7 17 42 99)
caps=(20 24 28 30 32)
B=(64 256)
repeat=7
warmup=2
```

Verification:

```bash
repeat=31
warmup=4
```

All numbers below are `batch_factor_per_sys_ms`; speedup is `fp32 / tf32`.

Raw sweep files:

- `/tmp/cls_8387_seed_sweep_20260609_160639.tsv`
- `/tmp/cls_8387_parallel_seed_sweep_20260609_160836.tsv`
- `/tmp/cls_8387_parallel_seed_verify_20260609_161021.tsv`

## Serial ND Result

Serial NodeND with seed variation did not improve the ratio. Best repeat=7 rows:

| Mode | Seed | Cap | B | FP32 ms | TF32 ms | Speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|
| best B64 | 7 | 32 | 64 | 0.027435 | 0.025274 | 1.0855 | 3.87e-02 |
| best B256 | 7 | 32 | 256 | 0.022272 | 0.021449 | 1.0384 | 3.87e-02 |
| best min(B64,B256) | 7 | 32 | both | - | - | 1.0384 | - |

Observation: serial ND tends to lower the FP32/TF32 ratio compared with the best parallel-ND
measurements. It can improve TF32 residual for some seeds, but it does not create enough TC-covered
work.

## Parallel ND Initial Sweep

Parallel ND can produce higher one-batch ratios, but the good B64 and B256 points are different
orderings/caps:

| Mode | Seed | Cap | B | FP32 ms | TF32 ms | Speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|
| best B64 | 1 | 32 | 64 | 0.026654 | 0.022526 | 1.1833 | 2.96e-02 |
| best B256 | 42 | 30 | 256 | 0.023225 | 0.019656 | 1.1816 | 1.72e-02 |
| best min(B64,B256) | 42 | 32 | both | - | - | 1.0932 | - |

Best repeat=7 paired rows by `min(B64,B256)`:

| Seed | Cap | B64 speedup | B256 speedup | min |
|---:|---:|---:|---:|---:|
| 42 | 32 | 1.0932 | 1.1189 | 1.0932 |
| 17 | 24 | 1.0423 | 1.0425 | 1.0423 |
| 1 | 30 | 1.0999 | 1.0359 | 1.0359 |
| 99 | 20 | 1.0318 | 1.0315 | 1.0315 |
| 42 | 28 | 1.0249 | 1.0708 | 1.0249 |

Observation: the sweep can find isolated ~1.18x points, but not a single ordering/cap that moves
both B64 and B256 into the 1.2x band.

## Repeat=31 Verification

The four most informative parallel-ND points were rerun at repeat=31:

| Seed | Cap | B | FP32 ms | TF32 ms | Speedup | FP32 relres | TF32 relres |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 32 | 64 | 0.025802 | 0.023721 | 1.0877 | 3.11e-05 | 5.93e-02 |
| 42 | 32 | 256 | 0.021672 | 0.020538 | 1.0552 | 4.98e-05 | 3.29e-02 |
| 1 | 32 | 64 | 0.024132 | 0.026136 | 0.9233 | 3.07e-05 | 3.33e-02 |
| 1 | 32 | 256 | 0.021349 | 0.020144 | 1.0598 | 3.10e-05 | 3.09e-02 |
| 42 | 30 | 64 | 0.024333 | 0.024309 | 1.0010 | 1.72e-05 | 1.56e-02 |
| 42 | 30 | 256 | 0.020983 | 0.020141 | 1.0418 | 2.18e-05 | 5.47e-02 |
| 7 | 28 | 64 | 0.024605 | 0.022821 | 1.0782 | 2.76e-05 | 3.35e-02 |
| 7 | 28 | 256 | 0.022384 | 0.020017 | 1.1182 | 2.75e-05 | 5.08e-02 |

The repeat=7 one-batch peaks were not stable enough to become target-satisfying paired settings.
The best verified pair here is still below target:

```text
seed=7 cap=28: B64/B256 = 1.078/1.118x
seed=42 cap=32: B64/B256 = 1.088/1.055x
```

## Conclusion

METIS seed variation is not the missing 8387 enabler. It changes residuals and can make one batch
size look better in short repeats, but it does not create a stable B64/B256 `1.2x` TF32-vs-FP32
factorize speedup.

This closes the cheap ordering-variance path. The remaining 8387 options are more structural:

1. Change symbolic panelization/amalgamation so dominant leaves no longer have `nc=1/2` and
   `uc<=8`.
2. Build a low-fill sparse-LU/circuit-style branch for absolute time, accepting that it may improve
   FP32 and TF32 together and therefore not satisfy the TC-enabler ratio.
3. Keep 8387 out of the TC-speedup claim and frame the Tensor Core result around cases with enough
   mid/big dense-update work.
