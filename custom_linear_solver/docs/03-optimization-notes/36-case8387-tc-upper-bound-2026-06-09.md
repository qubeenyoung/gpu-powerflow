# 8387 Tensor Core Upper-bound Check

**Date**: 2026-06-09

**Question**: Can 8387 still plausibly reach the B64/B256 `1.2..1.4x` FP32-relative factorize
speedup target with the current multifrontal structure, if Tensor Cores are the enabler?

## Front-shape Coverage

Generated with:

```bash
build-tc-colusolve-respectcap-bighigh512-bigshared512-current/custom_linear_solver_run \
  /datasets/power_system/nr_linear_systems/case8387pegase \
  --batch 64 --batch-only --precision fp32 --panel-cap 24 \
  --warmup 0 --repeat 1 --dump-fronts /tmp/cls_front_dumps/8387_cap24.csv
```

The same dump was collected at caps 28 and 30.

| cap | fronts | trailing flop proxy | `<=8` flop | `<=16` flop | `17..32` flop | `33..48` flop | `49+` flop | strong TC fronts (`nc>=8,uc>=16`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 7,336 | 2,574,912 | 12.0% | 9.0% | 14.9% | 14.9% | 49.2% | 56 fronts / 66.0% flop |
| 28 | 7,334 | 2,486,854 | 12.4% | 9.5% | 17.6% | 18.0% | 42.5% | 52 fronts / 61.2% flop |
| 30 | 7,328 | 2,466,676 | 12.4% | 9.4% | 15.5% | 16.0% | 46.6% | 49 fronts / 59.6% flop |

FLOP proxy alone is misleading here. A few 49+ fronts account for most dense-update FLOPs, but the
factor wall still contains thousands of tiny small-tier fronts, parent update traffic, scatter, and
front memory movement. Naive TF32 tile efficiency over all fronts is only about `13%` because most
fronts have small `uc` and `nc`.

## Diagnostic Options

Two OFF-by-default timing diagnostics were added:

```text
CLS_DISABLE_EXTEND_ADD=ON
CLS_DISABLE_SMALL_FACTOR=ON
```

They intentionally produce invalid solves and are only used to estimate timing upper bounds.

## Small-tier Removal Upper Bound

`CLS_DISABLE_SMALL_FACTOR=ON`, stable large-case TC policy otherwise unchanged, 8387 repeat=31:

| cap | B | FP32 | TF32 | speedup |
|---:|---:|---:|---:|---:|
| 24 | 64 | 0.014792 | 0.015013 | 0.985x |
| 24 | 256 | 0.011221 | 0.010702 | 1.048x |
| 28 | 64 | 0.014103 | 0.015048 | 0.937x |
| 28 | 256 | 0.012212 | 0.010947 | 1.116x |
| 30 | 64 | 0.016425 | 0.014589 | 1.126x |
| 30 | 256 | 0.011307 | 0.010457 | 1.081x |

Even removing the entire small tier does not reliably create a target-range TC ratio.

## Small-tier + Extend-add Removal Upper Bound

`CLS_DISABLE_SMALL_FACTOR=ON`, `CLS_DISABLE_EXTEND_ADD=ON`, stable large-case TC policy otherwise
unchanged, 8387 repeat=31:

| cap | B | FP32 | TF32 | speedup |
|---:|---:|---:|---:|---:|
| 24 | 64 | 0.015944 | 0.012551 | 1.270x |
| 24 | 256 | 0.011058 | 0.009981 | 1.108x |
| 28 | 64 | 0.013815 | 0.012830 | 1.077x |
| 28 | 256 | 0.011595 | 0.010425 | 1.112x |
| 30 | 64 | 0.015073 | 0.013798 | 1.092x |
| 30 | 256 | 0.011044 | 0.010353 | 1.067x |

B256 cap sweep for the same diagnostic:

| cap | FP32 | TF32 | speedup |
|---:|---:|---:|---:|
| 16 | 0.010283 | 0.009915 | 1.037x |
| 20 | 0.010896 | 0.009421 | 1.157x |
| 22 | 0.011003 | 0.010206 | 1.078x |
| 24 | 0.010112 | 0.010064 | 1.005x |
| 26 | 0.010980 | 0.010666 | 1.029x |
| 28 | 0.010671 | 0.011083 | 0.963x |
| 30 | 0.010809 | 0.010720 | 1.008x |
| 32 | 0.011105 | 0.010359 | 1.072x |

The combined diagnostic can cross B64 at one cap, but B256 still does not reach 1.2x. This is a
strong negative result: even after removing two large common walls, the remaining 8387 mid/big
Tensor Core path does not have enough stable advantage at B256.

## Implication

For 8387, the current multifrontal structure does not have a clear remaining local lever that can
make Tensor Cores a broad B64/B256 `1.2..1.4x` enabler:

- per-front tiny/small TC is rejected by direct measurement,
- parent update removal is not enough,
- small-tier removal is not enough,
- small-tier plus parent-update removal is not enough at B256.

The remaining credible ways to change this are structural, not local:

1. Change ordering/panelization to create fewer, larger TC-suitable fronts while preserving residuals.
2. Use a fundamentally different tiny-front aggregation scheme that batches many independent small
   GEMMs into one Tensor Core tile, not one front per TC tile.
3. Treat 8387 as a low-fill sparse-LU branch and optimize absolute time separately, accepting that
   the TC-enabler ratio may not be the right metric for this case.

## Structural Cap Check: Serial ND + Larger Panels

The cheapest structural attempt is to keep the safe chain-only `relaxed_panels()` invariant but
increase panel cap under deterministic serial METIS ND. Parallel-ND cap34+ already produced invalid
FP32 residuals; this checked whether serial ND made larger panels numerically usable.

8387, stable large-case TC policy, `--serial-nd`, repeat=31:

| cap | B | FP32 | FP32 relres | TF32 | TF32 relres | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 0.024322 | 3.01e-05 | 0.023102 | 8.22e-03 | 1.053x |
| 32 | 256 | 0.020689 | 3.03e-05 | 0.020272 | 8.22e-03 | 1.021x |
| 36 | 64 | 0.025092 | 9.17e37 | 0.023976 | 1.07e39 | invalid |
| 36 | 256 | 0.021178 | 1.43e39 | 0.020935 | inf | invalid |
| 40 | 64 | 0.026015 | 3.63e38 | 0.025293 | 1.52e39 | invalid |
| 40 | 256 | 0.021472 | 7.23e38 | 0.021162 | 9.28e38 | invalid |
| 48 | 64 | 0.026166 | 2.36e37 | 0.025349 | 6.33e37 | invalid |
| 48 | 256 | 0.021715 | inf | 0.021215 | 3.24e38 | invalid |
| 56 | 64 | 0.026160 | 1.22e37 | 0.025433 | 6.04e37 | invalid |
| 56 | 256 | 0.021920 | inf | 0.021418 | inf | invalid |
| 64 | 64 | 0.025665 | 1.29e37 | 0.025468 | 2.61e37 | invalid |
| 64 | 256 | 0.021669 | inf | 0.021358 | inf | invalid |

Serial ND does not make larger chain panels usable. The valid cap32 point is still far below target.
So the simple "increase panel cap to create larger TC fronts" structural route is closed for 8387.

## Many-front Packed TC Check

`docs/37` evaluates the remaining "pack many tiny fronts into one MMA tile" idea. The simple safe
block-diagonal packing layout can only pack:

```text
min(floor(16/uc), floor(8/uc))
```

fronts per `m16n8k8` tile when `uc<=8`; `uc>8` does not pack in the N dimension. On 8387 cap24/28/30,
the `small<=16` region reaches only about `6.5%` packed tile efficiency and accounts for about
`21..22%` of useful trailing work. Dominant shapes such as `(6,2,4)` and `(4,2,2)` remain far too
underfilled. A packed tiny-front TC prototype is therefore not justified without first changing the
symbolic/front structure.
