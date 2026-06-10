# 8387 Low-mid TF32 Force-all Diagnostic

**Date**: 2026-06-09

**Question**: Did we accidentally leave a useful 8387 mid-tier Tensor Core path disabled because
`CLS_MID_TF32_LOW_TC` only dispatches on `20000 <= n < 80000`, while 8387 has `n=14908`?

## Tooling Change

Added a default-OFF diagnostic CMake option:

```text
CLS_MID_TF32_LOW_TC_FORCE_ALL=ON
```

When enabled, the mid TF32 dispatch treats the low-mid TC gate as enabled for all matrix sizes.
The option also defines `CLS_MID_TF32_LOW_TC` so that the kernel-side low-mid `use_tc` predicate is
active even if the user only sets the force-all option.

Default behavior is unchanged.

## Build

Stable large-case policy plus the force-all diagnostic:

```bash
cmake -S custom_linear_solver \
  -B build-tc-colusolve-respectcap-bighigh512-bigshared512-lowtcforce-current \
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
  -DCLS_MID_TF32_MIN_FSZ=48
cmake --build build-tc-colusolve-respectcap-bighigh512-bigshared512-lowtcforce-current -j
```

## Front Coverage Check

Command:

```bash
build-tc-colusolve-respectcap-bighigh512-bigshared512-lowtcforce-current/custom_linear_solver_run \
  /datasets/power_system/nr_linear_systems/case8387pegase \
  --batch 64 --batch-only --precision fp32 \
  --panel-cap 32 --metis-seed 42 \
  --warmup 0 --repeat 1 \
  --dump-fronts /tmp/cls_8387_lowtcforce_cap32_seed42_fronts_20260609_163246.csv
```

Final batched analyze dump:

| bucket / predicate | fronts |
|---|---:|
| total | 7,274 |
| `fsz <= 16` | 7,071 |
| `17 <= fsz <= 32` | 161 |
| `33 <= fsz <= 48` | 29 |
| `49 <= fsz <= 64` | 7 |
| `fsz > 64` | 6 |
| low-TC predicate: `fsz>16, uc>=16, 4<=nc<=32` | 69 |
| 33..48 low-TC predicate | 18 |
| high-TC predicate: `fsz>48, uc>=32, 8<=nc<=32` | 7 |

Interpretation: the previous dispatch gate was a real blind spot, but the newly exposed work is
small. Force-all mainly gives TC access to a few dozen fronts; it does not change the dominant
`fsz<=16` workload.

## Repeat=7 Sweep

Input:

```text
/datasets/power_system/nr_linear_systems/case8387pegase
```

Common options:

```text
--batch <B> --batch-only --warmup 2 --repeat 7 --precision fp32|tf32
```

Raw TSVs:

- `/tmp/cls_8387_lowtcforce_sweep_seed42_20260609_163136.tsv`
- `/tmp/cls_8387_lowtcforce_sweep_seed1_7_20260609_163159.tsv`

### seed 42, caps 20/24/28/30/32

| seed | cap | B | FP32 ms/sys | TF32 ms/sys | speedup | TF32 relres |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 20 | 64 | 0.026193 | 0.024813 | 1.056x | 3.95e-02 |
| 42 | 20 | 256 | 0.020883 | 0.024375 | 0.857x | 3.35e-02 |
| 42 | 24 | 64 | 0.024338 | 0.023941 | 1.017x | 2.37e-02 |
| 42 | 24 | 256 | 0.021214 | 0.020962 | 1.012x | 4.54e-02 |
| 42 | 28 | 64 | 0.025901 | 0.023590 | 1.098x | 3.30e-02 |
| 42 | 28 | 256 | 0.021561 | 0.021343 | 1.010x | 2.87e-02 |
| 42 | 30 | 64 | 0.024759 | 0.024103 | 1.027x | 2.89e-02 |
| 42 | 30 | 256 | 0.021287 | 0.020531 | 1.037x | 4.03e-02 |
| 42 | 32 | 64 | 0.024499 | 0.024333 | 1.007x | 3.32e-02 |
| 42 | 32 | 256 | 0.022403 | 0.021036 | 1.065x | 1.37e-02 |

### seed 1/7, caps 28/30/32

| seed | cap | B | FP32 ms/sys | TF32 ms/sys | speedup | TF32 relres |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28 | 64 | 0.025213 | 0.024705 | 1.021x | 5.01e-02 |
| 1 | 28 | 256 | 0.022052 | 0.020768 | 1.062x | 3.31e-02 |
| 1 | 30 | 64 | 0.026620 | 0.022745 | 1.170x | 3.18e-02 |
| 1 | 30 | 256 | 0.021647 | 0.020812 | 1.040x | 3.26e-02 |
| 1 | 32 | 64 | 0.026004 | 0.023339 | 1.114x | 5.72e-02 |
| 1 | 32 | 256 | 0.021083 | 0.020957 | 1.006x | 3.38e-02 |
| 7 | 28 | 64 | 0.028727 | 0.025872 | 1.110x | 5.10e-02 |
| 7 | 28 | 256 | 0.023818 | 0.021118 | 1.128x | 8.35e-03 |
| 7 | 30 | 64 | 0.025832 | 0.025199 | 1.025x | 3.64e-02 |
| 7 | 30 | 256 | 0.022319 | 0.020754 | 1.075x | 1.58e-02 |
| 7 | 32 | 64 | 0.026216 | 0.024630 | 1.064x | 5.46e-02 |
| 7 | 32 | 256 | 0.020965 | 0.023239 | 0.902x | 3.37e-02 |

The best repeat=7 paired point is seed7/cap28 at about `1.11/1.13x`, still below the target.

## Repeat=31 Verification

Common options:

```text
--batch <B> --batch-only --warmup 3 --repeat 31 --precision fp32|tf32
```

Raw TSV:

- `/tmp/cls_8387_lowtcforce_verify_20260609_163224.tsv`

| seed | cap | B | FP32 ms/sys | TF32 ms/sys | speedup | TF32 relres |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 28 | 64 | 0.023551 | 0.023648 | 0.996x | 4.20e-02 |
| 7 | 28 | 256 | 0.021536 | 0.020410 | 1.055x | 1.77e-02 |
| 1 | 30 | 64 | 0.025279 | 0.025396 | 0.995x | 5.29e-02 |
| 1 | 30 | 256 | 0.021454 | 0.021582 | 0.994x | 5.49e-02 |
| 42 | 32 | 64 | 0.025886 | 0.022892 | 1.131x | 6.10e-02 |
| 42 | 32 | 256 | 0.020402 | 0.020721 | 0.985x | 5.51e-02 |

Repeat=31 removes the repeat=7 apparent wins. There is no paired B64/B256 target-range result.

## Conclusion

`CLS_MID_TF32_LOW_TC_FORCE_ALL` closes the dispatch-gate ambiguity. The prior medium-case size gate
did hide the low-mid TF32 path from 8387, but enabling it for 8387 is not enough:

- newly eligible fronts are only a few dozen,
- repeat=31 paired speedups are at best `1.131/0.985x` for seed42/cap32 and `0.996/1.055x` for
  seed7/cap28,
- TF32 residual remains in the expected diagnostic range (`~1e-2..6e-2`).

Decision: do not pursue low-mid 8387 per-front TF32 further unless ordering/panelization first
creates materially more `nc/uc`-large fronts. This leaves the remaining 8387 options as structural
panelization/order changes, parent/assembly timing work, or a low-fill sparse-LU branch.
