# Many-front Packed Tensor Core Feasibility

**Date**: 2026-06-09

**Question**: After per-front tiny/small Tensor Core attempts failed on 8387, could we pack many
independent tiny fronts into one TF32 `mma.m16n8k8` tile and make Tensor Cores the B64/B256 enabler?

## Packing Model

For a front with trailing update:

```text
C[uc x uc] -= L[uc x nc] * U[nc x uc]
```

One TF32 MMA tile computes:

```text
D[16 x 8] += A[16 x 8] * B[8 x 8]
```

For multiple independent fronts in one tile, the only safe simple layout is block-diagonal packing:

- give each front its own row block in `D`,
- give each front its own column block in `D`,
- share the K lanes for the same pivot slice,
- ignore cross-front output blocks.

For `uc <= 8`, the maximum fronts per tile is:

```text
fpack = min(floor(16 / uc), floor(8 / uc))
```

For `uc > 8`, only one front fits in the N dimension, so many-front packing does not help.

The packed efficiency estimate is:

```text
useful_fma / issued_fma
```

where one `m16n8k8` tile issues `16*8*8 = 1024` FMA-equivalent multiply-add slots per K tile.

## 8387 Shape Results

Using the cap24/28/30 front dumps from `docs/36`, grouped by `(fsz,nc,uc)`:

| cap | all-front one-front TC eff | all-front packed TC eff | small<=16 one-front eff | small<=16 packed eff | small<=16 useful share |
|---:|---:|---:|---:|---:|---:|
| 24 | 13.41% | 21.02% | 3.54% | 6.51% | 21.0% |
| 28 | 12.82% | 19.97% | 3.57% | 6.56% | 21.9% |
| 30 | 12.98% | 20.48% | 3.52% | 6.47% | 21.8% |

The all-front number is not an implementation proposal because mid/big fronts already have working
TC paths. The new opportunity would be the `small<=16` region. There, packed TC efficiency is only
about `6.5%`.

## Dominant Small Shapes

Cap28, top `fsz<=16` shapes by useful work:

| count | useful work | fpack | packed eff | shape `(fsz,nc,uc)` |
|---:|---:|---:|---:|---|
| 2,288 | 73,216 | 2 | 6.25% | `(6,2,4)` |
| 387 | 27,864 | 1 | 7.03% | `(8,2,6)` |
| 138 | 17,664 | 1 | 12.50% | `(10,2,8)` |
| 823 | 14,814 | 2 | 3.51% | `(5,2,3)` |
| 74 | 14,800 | 1 | 9.77% | `(12,2,10)` |
| 1,111 | 8,888 | 4 | 3.12% | `(4,2,2)` |
| 527 | 2,108 | 4 | 1.56% | `(3,1,2)` |

The dominant tiny shapes are exactly the bad cases: `nc=1/2`, `uc=2..6`. Packing improves tile
fill versus one-front TC, but not enough.

## Hardware Implication

For the `small<=16` region:

- if TF32 TC peak is treated as `4x` FP32 scalar peak, `6.5%` tile efficiency is equivalent to only
  about `0.26x` of FP32 scalar peak;
- even if TF32 TC peak is treated as `8x`, it is only about `0.52x`.

This ignores packing/staging/drain overhead, so it is an optimistic bound. A real packed-TC small
kernel would likely be worse.

## Conclusion

Do not implement a many-front packed TC small-tier prototype for 8387 under the current front
shapes. It is mathematically underfilled before considering memory movement or control overhead.

The only packed-TC variant that could be credible would require changing the symbolic structure so
many fronts have `uc≈8` and `nc≈8` or larger. That is no longer "packing the current tiny fronts";
it is an ordering/panelization problem.
