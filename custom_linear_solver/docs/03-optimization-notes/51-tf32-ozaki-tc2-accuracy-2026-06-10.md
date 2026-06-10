# TF32 Ozaki TC2 Accuracy Check

**Date**: 2026-06-10

## Question

Can the raw TF32 Tensor Core factorization residual be brought back to FP32 level without falling
back to scalar FP32 trailing updates?

## Implementation

New default-OFF CMake option:

```text
CLS_TF32_OZAKI_TC2=ON
```

When enabled, TF32 MMA inputs are explicitly split into two TF32-representable components:

```text
x0 = cvt.rna.tf32.f32(x)
x1 = cvt.rna.tf32.f32(x - x0)
```

Each `L*U` product is then accumulated with Tensor Cores as:

```text
L0*U0 + L1*U0 + L0*U1 + L1*U1
```

The option is applied to the TF32 global trailing path, mid direct-shared trailing path,
right-looking blocked TF32 update, and small warp TF32 path. With the option OFF, the original
single-pass TF32 MMA path remains compiled and tested.

Diagnostic build:

```text
build-tc-ozaki-tc2-lowfill-current
```

Base policy:

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
CLS_TF32_OZAKI_TC2=ON
```

## Low-fill Accuracy Results

Runtime:

```text
--serial-nd --batch <64|256> --batch-only --repeat 31 --warmup 6 --single-precision fp64
```

| case | seed | cap | B | FP32 ms/sys | Ozaki TF32 ms/sys | speedup | FP32 relres | Ozaki TF32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case8387pegase | 7 | 30 | 64 | 0.0331604 | 0.0280826 | 1.181 | 3.33e-05 | 4.77e-05 |
| case8387pegase | 7 | 30 | 256 | 0.0289942 | 0.0254547 | 1.139 | 3.41e-05 | 4.77e-05 |
| case13659pegase | 99 | 32 | 64 | 0.0679608 | 0.0560004 | 1.214 | 1.24e-04 | 1.33e-04 |
| case13659pegase | 99 | 32 | 256 | 0.0572863 | 0.0480058 | 1.193 | 1.23e-04 | 1.35e-04 |

Previous raw TF32 residuals for the same accepted low-fill policy were:

| case | B64 raw TF32 relres | B256 raw TF32 relres |
|---|---:|---:|
| case8387pegase | 3.97e-02 | 3.97e-02 |
| case13659pegase | 6.47e-03 | 6.47e-03 |

So Ozaki TC2 reduces the TF32 residual by roughly `10^2..10^3x` and brings it to the FP32 band.
The cost is visible: the low-fill speedup margin drops, especially for 8387 B256.

## Large-case Sanity Check

Runtime: repeat 7 / warmup 2, default parallel ND.

| case | seed | cap | B | FP32 ms/sys | Ozaki TF32 ms/sys | speedup | FP32 relres | Ozaki TF32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg25k | 42 | 31 | 64 | 0.163322 | 0.126906 | 1.287 | 2.03e-04 | 2.31e-04 |
| case_ACTIVSg25k | 42 | 32 | 256 | 0.168313 | 0.124993 | 1.347 | 1.62e-04 | 2.45e-04 |
| case_SyntheticUSA | 42 | 31 | 64 | 0.629164 | 0.497960 | 1.264 | 2.91e-02 | 2.68e-02 |
| case_SyntheticUSA | 42 | 32 | 256 | 0.603708 | 0.478246 | 1.262 | 2.69e-02 | 9.45e-03 |

25K also lands in the FP32 residual band. USA remains at a large absolute residual, but the FP32
baseline is already large under the same ordering/cap; Ozaki TF32 is not worse than FP32 there.
This matches the earlier conditioning note for USA.

## Conclusion

`CLS_TF32_OZAKI_TC2` is a valid accuracy lever: it keeps the product update on Tensor Cores and
brings TF32 factor residuals back to FP32 level on 8387, 13K, and 25K; on USA it matches or improves
the same-condition FP32 residual floor.

It is not a free replacement for the raw-TF32 speed policy. The extra three MMA passes plus TF32
head/tail conversion reduce the low-fill speed margin. The next speed-oriented step would be a
first-order variant (`L0*U0 + L1*U0 + L0*U1`, omitting `L1*U1`) or a selective policy that enables
Ozaki only for cases where raw TF32 residual is outside the acceptable band.
