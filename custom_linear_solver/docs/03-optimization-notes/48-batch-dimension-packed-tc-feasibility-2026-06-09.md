# Batch-dimension Packed Tensor Core Feasibility

**Date**: 2026-06-09

**Question**: If B=64/256 gives many independent systems, can the batch dimension itself fill
Tensor Core tiles for the 8387/13K mid/small-front region and make TF32 the enabler?

## Short Answer

No, not with the current front shapes.

More batches increase the number of independent front instances, but they do not change the
per-instance update:

```text
C_b[uc x uc] -= L_b[uc x nc] * U_b[nc x uc]
```

for each batch `b`. A standard GEMM that treats batch as a dense matrix dimension would introduce
cross-batch products such as `L_b * U_{b'}`. Those terms are mathematically invalid because the
systems are independent.

The safe Tensor Core mapping is therefore not "batch as GEMM". It is block-diagonal packing of
independent tiny GEMMs, where off-diagonal outputs are ignored. That reduces to the same tile-fill
model as docs/37.

## Current Code Shape

The current factor arena is batch-major over identical symbolic fronts:

```text
front = frontB + bb * front_total
```

Small tier dispatch maps one sub-group to one `(front, batch)` instance:

```text
slot = global_sub_group
fl = slot % level_size
bb = slot / level_size
```

Mid/big tier kernels similarly own one `(front, batch)` per block. This is the correct layout for
independent batched sparse direct solves: batches share symbolic structure but not numerical values.

## Why Batch-as-GEMM Is Invalid

For two batches, the desired work is:

```text
C_0 -= L_0 * U_0
C_1 -= L_1 * U_1
```

If batch is folded into a dense GEMM dimension, the tile also computes or implies cross terms:

```text
L_0 * U_1
L_1 * U_0
```

Those terms have no place in the factorization. They must be zeroed, masked, or discarded. Tensor
Core MMA does not provide arbitrary element masks for these off-diagonal products, so the practical
safe layout is block-diagonal packing.

## Block-diagonal Packed Batch Model

One TF32 Ampere tile:

```text
mma.m16n8k8: D[16 x 8] += A[16 x 8] * B[8 x 8]
```

For an independent front shape `(nc, uc)`, useful work per packed item is:

```text
uc * uc * nc
```

The issued tile capacity is:

```text
16 * 8 * 8 = 1024 FMA slots
```

If each independent batch/front occupies its own `uc x uc` output block, the maximum number of
items in one tile is:

```text
fpack = min(floor(16 / uc), floor(8 / uc))
```

This is identical whether the packed items are different fronts from the same batch or the same
front from different batches. Batch count gives more items to choose from, but the `16 x 8` tile
geometry still limits useful fill.

## 8387 Dominant Tiny Shapes

Using the dominant cap28 small-front shapes from docs/37:

| shape `(fsz,nc,uc)` | `fpack` | useful FMA/tile | tile efficiency |
|---|---:|---:|---:|
| `(6,2,4)` | 2 | 64 | 6.25% |
| `(8,2,6)` | 1 | 72 | 7.03% |
| `(10,2,8)` | 1 | 128 | 12.50% |
| `(5,2,3)` | 2 | 36 | 3.52% |
| `(4,2,2)` | 4 | 32 | 3.12% |
| `(3,1,2)` | 4 | 16 | 1.56% |

The dominant shapes have `nc=1/2` and `uc=2..6`. Extra batches make it easy to find enough
instances, but each MMA tile is still mostly off-diagonal or padded space.

## Implication for B=64/256

Multi-batch helps saturate launches and SM scheduling for kernels that already have enough useful
work per block. It does not fix the Tensor Core underfill of tiny updates.

For the small/tiny region:

- current scalar/sub-group code already packs multiple `(front,batch)` instances per warp to reduce
  scheduling overhead;
- packed TC would add staging, layout packing, MMA drain, and off-diagonal discard work;
- even an optimistic `4x..8x` TF32 peak advantage is multiplied by only `~1.5..12.5%` useful tile
  efficiency on the dominant shapes.

That is below the threshold needed to beat the existing FP32 scalar path, before considering memory
traffic and control overhead.

Panel LU and U-solve have the same independence constraint. They are sequential per front/batch,
and cross-batch mixing would be invalid. Tensor Core acceleration across batch is therefore not a
safe shortcut for the non-GEMM portion either.

## Decision

Do not implement a batch-dimension Tensor Core prototype for 8387/13K under the current symbolic
shape.

The remaining credible paths are:

1. nonlocal ordering/panelization that raises `uc` and especially `nc` before TC dispatch;
2. a separate low-fill sparse-LU branch for 8387/13K-style cases;
3. keeping the Tensor Core claim scoped to large cases where mid/big updates already expose enough
   useful work: 25K, 70K, and USA.
