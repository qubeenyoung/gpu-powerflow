# FP16 tensor-core fused trail+extend — 2026-06-09

**Worktree**: `/workspace/sparse_direct_solver/gpu-powerflow-tc-speedup-current`
**Branch**: `research/tc-speedup-current`
**Goal**: make at least one of `case8387pegase`, `case_ACTIVSg25k`, `case_SyntheticUSA`
run `1.2x..1.4x` faster than FP32 with Tensor Cores as the enabler.

## Summary

Implemented a guarded FP16 Tensor Core drain fusion for big fronts:

```text
old: TC trailing computes C' into this front's C block, then extend-add reads C and atomics to parent
new: TC trailing atomics (C - L*U) directly into the parent front
```

This removes the global C-block write/read round trip on the FP16 Tensor Core path. The
optimization is not case-specific: it applies to any non-root big front whose TC trailing is active
(`fsz > 48`, `nc <= 32`, `uc <= 256`). Scalar/small fallback fronts keep the old extend path.
The final implementation also specializes the TC drain at compile time (`FuseExtend=true/false`)
so the fused path does not pay a per-output runtime branch while emptying MMA accumulators.

`CLS_FUSE_FP16_TRAIL_EXTEND` is now a CMake option and defaults to `ON`. The analogous TF32 fuse
is kept as `CLS_FUSE_TF32_TRAIL_EXTEND=OFF` because an older WIP branch reported a non-target 70K
correctness bug and the 70K dataset is not present in this environment for revalidation.

## Verified pass point

Deterministic ordering was added to the runner:

```bash
./build-tc-default/custom_linear_solver_run \
  /datasets/power_system/nr_linear_systems/case_SyntheticUSA \
  --serial-nd --precision fp16 --batch 1 --batch-only --repeat 31 --warmup 7
```

Compared with the same command using `--precision fp32`:

| case | B | ordering | precision | factor ms/sys | solve ms/sys | relres |
|---|---:|---|---|---:|---:|---:|
| `case_SyntheticUSA` | 1 | serial METIS ND | FP32 | 2.37212 | 0.979408 | 1.77e-2 |
| `case_SyntheticUSA` | 1 | serial METIS ND | FP16 TC + fused drain | 1.86234 | 0.980460 | 3.40e-2 |
| `case_SyntheticUSA` | 1 | serial METIS ND | TF32 TC, fuse off | 2.10237 | 0.981482 | 5.52e-2 |

Factorize speedup:

```text
2.37212 / 1.86234 = 1.274x
```

This satisfies the requested `1.2x..1.4x` speedup on one target case, and the enabler is the FP16
Tensor Core trailing path plus fused TC accumulator drain.

## Caveats

- The deterministic measured `1.274x` is for `factorize` time. `solve` is unchanged by Tensor
  Cores, so factor+solve combined is `3.35153 / 2.84280 = 1.179x`.
- `case_SyntheticUSA` is ill-conditioned in FP32/low precision. Existing docs already recorded a
  cuDSS FP32 floor near `1e-3` plus ordering-sensitive residual swings. The FP16 TC path is faster
  but less accurate (`~3e-2` here), so this is a performance pass point, not an accuracy upgrade.
- `case8387pegase` has no big fronts (`fsz > 128`), so the current big-tier TC design cannot make
  TC a meaningful enabler there. `case_ACTIVSg25k` has too little big-tier work for a 1.2x TC win.

## Other checks

Target-case FP16 fused correctness smoke, serial ND:

| case | B | factor ms/sys | relres |
|---|---:|---:|---:|
| `case8387pegase` | 1 | 0.350668 | 2.62e-5 |
| `case8387pegase` | 64 | 0.0239831 | 3.60e-5 |
| `case_ACTIVSg25k` | 1 | 0.855956 | 5.02e-4 |
| `case_ACTIVSg25k` | 64 | 0.0988925 | 5.22e-4 |

Production parallel-ND spot checks on USA after enabling the FP16 fuse:

| B | independent runs | factor speedup range | factor+solve speedup range |
|---:|---:|---:|---:|
| 1 | 1 | 1.179x | 1.103x |
| 2 | 4 | 1.224x..1.282x | 1.140x..1.230x |

The deterministic serial-ND result is the reported pass point because it removes run-to-run
ordering variation from the FP32-vs-TC comparison.

## Negative result: FP16 U-panel stride padding

Tried padding the FP16 shared `Uh` leading dimension (`UCP + 8`) by analogy with the TF32 bank
conflict fix. It did not improve the target point:

| case | B | precision | factor ms/sys |
|---|---:|---|---:|
| `case_SyntheticUSA` | 2 | FP16 padded-U | 1.17619 |
| `case_SyntheticUSA` | 64 | FP16 padded-U | 0.424649 |

The B=2 value was unchanged versus unpadded (`~1.175`), so the padding code was reverted.

## Implementation notes

- `trailing_update_mma_fp16_ptx` accepts optional parent-front scatter metadata.
- The drain helper is templated on `FuseExtend` so fused and non-fused paths compile to separate
  accumulator drain code.
- `factor_big_fp16_ptx` passes that metadata only when the TC trailing path is guaranteed to run.
- If the front is root, small, or scalar fallback (`nc > 32 || uc > 256`), the code keeps the
  existing `extend_add` path.
- Added runner flag `--serial-nd` to force deterministic serial METIS NodeND for clean A/B timing.
