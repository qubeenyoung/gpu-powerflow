# TF32 Ozaki Speed Follow-up

**Date**: 2026-06-10

## Question

After `CLS_TF32_OZAKI_TC2` restored TF32 residuals to the FP32 band, can we recover some of the raw
TF32 speed?

## Implemented Variants

### First-order Ozaki

New default-OFF option:

```text
CLS_TF32_OZAKI_TC2_FIRST_ORDER=ON
```

This implies `CLS_TF32_OZAKI_TC2=ON` and uses:

```text
L0*U0 + L1*U0 + L0*U1
```

It omits the second-order `L1*U1` term, saving one of the four Ozaki MMA passes.

### Staged direct-shared operands

New default-OFF diagnostic option:

```text
CLS_TF32_OZAKI_STAGE_DIRECT=ON
```

For direct-shared mid fronts with `fsz_cap<=64`, it precomputes L/U head/tail operands in shared
scratch instead of recomputing the conversion inside each MMA tile. This was expected to reduce
`cvt.rna.tf32.f32` work, but it increases dynamic shared memory and adds staging traffic.

## Low-fill First-order Results

Runtime:

```text
--serial-nd --batch <64|256> --batch-only --repeat 31 --warmup 6 --single-precision fp64
```

| case | seed | cap | B | FP32 ms/sys | first-order Ozaki ms/sys | speedup | FP32 relres | first-order relres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case8387pegase | 7 | 30 | 64 | 0.0331604 | 0.0278644 | 1.190 | 3.33e-05 | 4.77e-05 |
| case8387pegase | 7 | 30 | 256 | 0.0289942 | 0.0251996 | 1.151 | 3.41e-05 | 4.77e-05 |
| case13659pegase | 99 | 32 | 64 | 0.0679608 | 0.0553270 | 1.228 | 1.24e-04 | 1.32e-04 |
| case13659pegase | 99 | 32 | 256 | 0.0572863 | 0.0475258 | 1.205 | 1.23e-04 | 1.33e-04 |

Compared with full four-term Ozaki TC2:

| case | B | full Ozaki ms/sys | first-order ms/sys | time change |
|---|---:|---:|---:|---:|
| case8387pegase | 64 | 0.0280826 | 0.0278644 | -0.8% |
| case8387pegase | 256 | 0.0254547 | 0.0251996 | -1.0% |
| case13659pegase | 64 | 0.0560004 | 0.0553270 | -1.2% |
| case13659pegase | 256 | 0.0480058 | 0.0475258 | -1.0% |

So `L1*U1` is not important for accuracy on these cases, but it is also not the main speed cost.
The main overhead is the two-component conversion path plus the two first-order correction MMA
passes.

## Staging Result

Runtime: repeat 7 / warmup 2.

| case | B | first-order ms/sys | staged first-order ms/sys | staged relres | result |
|---|---:|---:|---:|---:|---|
| case8387pegase | 64 | 0.0279017 | 0.0332889 | 4.77e-05 | reject |
| case8387pegase | 256 | 0.0251990 | 0.0302461 | 4.77e-05 | reject |
| case13659pegase | 64 | 0.0555593 | 0.0625493 | 1.32e-04 | reject |
| case13659pegase | 256 | 0.0475693 | 0.0557039 | 1.32e-04 | reject |

The staged version preserves accuracy but regresses speed badly. The extra shared footprint reduces
occupancy and the staging loops outweigh the saved conversion work. Keep
`CLS_TF32_OZAKI_STAGE_DIRECT` as a diagnostic only; do not use it in the policy stack.

## 8387 Cap/Seed Sweep

First-order Ozaki was swept on 8387 with repeat 7:

- seed7, cap `28..32`: no paired B64/B256 target recovery. Best paired min was about `1.15x`.
- seed42, cap `28/30/32`: residuals improved to `~1.7e-5..2.2e-5`, but speedup stayed below target.
- seed99, cap `28/30/32`: residuals stayed FP32-band, but paired speedup stayed around `1.13x`.

Conclusion: after Ozaki correction, 8387's prior raw-TF32 `1.207x` B256 margin is too thin. Cheap
ordering/cap retuning does not recover the lost speed ratio.

## Large-case First-order Sanity

Runtime: repeat 7 / warmup 2, default parallel ND.

| case | seed | cap | B | FP32 ms/sys | first-order Ozaki ms/sys | speedup | FP32 relres | first-order relres |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg25k | 42 | 31 | 64 | 0.159254 | 0.128387 | 1.240 | 5.44e-04 | 1.72e-04 |
| case_ACTIVSg25k | 42 | 32 | 256 | 0.164009 | 0.130389 | 1.258 | 1.61e-04 | 1.85e-04 |
| case_SyntheticUSA | 42 | 31 | 64 | 0.632864 | 0.489975 | 1.292 | 1.11e-03 | 2.73e-03 |
| case_SyntheticUSA | 42 | 32 | 256 | 0.634600 | 0.488702 | 1.299 | 3.38e-02 | 1.16e-03 |

Large cases retain good speed with first-order Ozaki and stay within the same FP32 conditioning band
or better. This makes first-order Ozaki a better default candidate than full TC2 for accuracy-aware
large-case measurements.

## Conclusion

`CLS_TF32_OZAKI_TC2_FIRST_ORDER` is the only useful speed follow-up from this round. It preserves the
FP32-band residual and recovers about `1%` versus full TC2. It is enough for 13K and large cases, but
not enough to keep 8387 B256 above `1.2x`.

`CLS_TF32_OZAKI_STAGE_DIRECT` failed because the shared-memory/occupancy cost dominates.

Next useful speed directions are not more local Ozaki algebra:

1. selective Ozaki only on residual-sensitive fronts or matrices, falling back to raw TF32 where the
   downstream tolerance allows it;
2. solve-level or Newton-level iterative refinement so raw TF32 factorization speed can be kept while
   final residual is corrected outside factorization;
3. a new low-fill branch that reduces non-GEMM overhead, because 8387 has too little speed margin for
   an always-corrected Tensor Core update.
