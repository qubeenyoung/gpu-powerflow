# cuSPARSE Ibus Baseline Results

Date: 2026-04-22 UTC
GPU: NVIDIA GeForce RTX 3090, sm_86
Runs: warmup 3, repeats 20

Raw CSV:

```text
exp/20260422/cusparse_ibus/results/ibus_cusparse_spmm.csv
```

## Result Summary

The direct cuPF mixed contract is not supported by the local cuSPARSE runtime:

```text
Ybus FP32 * V FP64 -> Ibus FP64
CUSPARSE_STATUS_NOT_SUPPORTED(10): operation not supported
```

Therefore a production cuSPARSE path must choose one of these compromises:

- `cusparse_fp64`: keep mismatch precision by converting/storing Ybus as FP64 for cuSPARSE.
- `cusparse_fp32`: run Ibus in FP32 and cast to the current FP64/Jacobian outputs afterward.

## Mean Speedup vs Custom Ibus

Speedup uses the cuSPARSE timing including the post-pack/cast kernel.

| batch | fp64 cuSPARSE | fp32 cuSPARSE |
|---:|---:|---:|
| 1 | 1.03x | 1.32x |
| 4 | 1.70x | 3.52x |
| 8 | 2.03x | 5.13x |
| 16 | 2.30x | 6.13x |
| 64 | 3.38x | 9.09x |
| 256 | 4.04x | 11.29x |

## Key Rows

| case | batch | custom ms | fp64 ms | fp64 speedup | fp32 ms | fp32 speedup |
|---|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | 1 | 0.0040 | 0.0380 | 0.10x | 0.0311 | 0.13x |
| case_ACTIVSg200 | 256 | 0.1412 | 0.0810 | 1.74x | 0.0454 | 3.11x |
| case_ACTIVSg2000 | 1 | 0.0093 | 0.0408 | 0.23x | 0.0339 | 0.27x |
| case_ACTIVSg2000 | 256 | 1.3795 | 0.3454 | 3.99x | 0.1005 | 13.73x |
| Texas7k_20220923 | 1 | 0.0217 | 0.0440 | 0.49x | 0.0369 | 0.59x |
| Texas7k_20220923 | 256 | 4.6339 | 1.0128 | 4.58x | 0.3611 | 12.83x |
| case_ACTIVSg25k | 1 | 0.0709 | 0.0499 | 1.42x | 0.0406 | 1.75x |
| case_ACTIVSg25k | 256 | 16.9987 | 3.3862 | 5.02x | 1.2287 | 13.83x |
| case_ACTIVSg70k | 1 | 0.1908 | 0.0659 | 2.90x | 0.0493 | 3.87x |
| case_ACTIVSg70k | 256 | 44.0868 | 9.0337 | 4.88x | 3.4090 | 12.93x |

## Accuracy

Against the custom kernel output:

| variant | max Ibus abs diff | max J_Ibus abs diff |
|---|---:|---:|
| fp64 cuSPARSE | 1.48e-11 | 4.66e-10 |
| fp32 cuSPARSE | 5.89e-03 | 5.89e-03 |

The FP64 path is numerically equivalent for this benchmark. The FP32 path is
much faster but changes the mismatch-side Ibus precision enough that it should
not replace the current mixed mismatch path without a solver-level convergence
check.

## Recommendation

Use `cusparse_fp64` as the production candidate for large cases/batches. It
requires either an FP64 Ybus mirror or a conversion step, but it preserves the
current mismatch precision contract and is already faster for large systems.

Do not use the direct mixed cuSPARSE path; this CUDA/cuSPARSE runtime rejects
the required type combination.
