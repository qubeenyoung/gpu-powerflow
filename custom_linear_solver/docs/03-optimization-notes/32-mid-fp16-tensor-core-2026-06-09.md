# Mid-tier FP16 Tensor Core trailing - 2026-06-09

**Worktree**: `/workspace/sparse_direct_solver/gpu-powerflow-tc-speedup-current`
**Branch**: `research/tc-speedup-current`

## Summary

Implemented a guarded mid-tier FP16 Tensor Core path for fronts where the trailing GEMM has enough
shape to amortize TC staging:

```text
33 <= fsz <= 128, fsz > 48, uc >= 32, 10 <= nc <= 32
```

The kernel is enabled by `CLS_MID_FP16_TC=ON` and currently routes only systems with
`20000 <= n < 80000`. That keeps `case8387pegase` on the old scalar mid path and avoids the
ill-conditioned `case_SyntheticUSA` parallel-ND residual spike observed when mid FP16 TC was forced
there.

Implementation details:

- Added `factor_mid_fp16_ptx`.
- Shared layout is compact: `Fs | Lh | Uh`.
- Non-TC fallback inside the kernel uses direct scalar trailing from shared `Fs`, so scalar fallback
  does not force permanent float L/U staging scratch.
- Non-root TC fronts drain the FP16 MMA accumulator directly into parent extend-add, matching the
  big-tier fused trail+extend idea.
- Launch is fixed at 512 threads. 1024 threads failed setup with `too many resources requested`.

## Candidate front counts

Criteria: `mid = 33 <= fsz <= 128`, candidate = `uc >= 32 && nc <= 32`.

| case | total fronts | mid fronts | candidates | candidate mid work share |
|---|---:|---:|---:|---:|
| `case8387pegase` | 7,408 | 55 | 37 | 84.2% |
| `case13659pegase` | 12,392 | 115 | 61 | 74.0% |
| `case_ACTIVSg25k` | 22,717 | 371 | 293 | 94.3% |
| `case_SyntheticUSA` | 74,129 | 901 | 643 | 93.4% |

Adding `nc >= 10` removes all `case8387pegase` candidates while preserving most work in larger
cases:

| case | `nc>=10` candidate count | candidate work kept |
|---|---:|---:|
| `case8387pegase` | 0 | 0.0% |
| `case13659pegase` | 43 | 88.5% |
| `case_ACTIVSg25k` | 194 | 90.1% |
| `case_SyntheticUSA` | 463 | 94.0% |

`case13659pegase` is not present under `/datasets/power_system/nr_linear_systems` in this
environment, so its count is from the existing docs front CSV.

## Verified pass point

Target: `case_ACTIVSg25k`, default parallel METIS ND, B=1, `--batch-only`, repeat 31, warmup 9.

Three independent runs:

| run | FP32 factor ms/sys | FP16 mid TC factor ms/sys | speedup | FP16 relres |
|---:|---:|---:|---:|---:|
| 1 | 0.803988 | 0.673593 | 1.194x | 9.29e-3 |
| 2 | 0.787466 | 0.676408 | 1.164x | 1.27e-2 |
| 3 | 0.806232 | 0.656370 | 1.228x | 2.05e-2 |

Average:

```text
FP32: 0.799229 ms/sys
FP16 mid TC: 0.668790 ms/sys
speedup: 1.195x
```

This is the current `~1.2x` factorize pass point.

To isolate the mid-TC contribution, `CLS_MID_FP16_TC=OFF` on the same case/runner gave:

| build | precision | factor ms/sys | relres |
|---|---|---:|---:|
| no mid TC | FP32 | 0.786004 | 1.48e-4 |
| no mid TC | FP16 | 0.759554 | 2.20e-4 |

Compared with the `0.668790` FP16 mid-TC average above, the new mid path reduces FP16 factor time
by about `1.14x` on the same target.

## Deterministic serial check

`case_ACTIVSg25k`, `--serial-nd`, B=1, repeat 31, warmup 9:

| precision | factor ms/sys | solve ms/sys | relres |
|---|---:|---:|---:|
| FP32 | 0.878437 | 0.606558 | 1.28e-4 |
| FP16 mid TC | 0.750316 | 0.606397 | 1.21e-2 |

Speedup is `1.171x`. The default parallel-ND pass point is therefore the reported `~1.2x`; serial
ND is slightly below target but confirms that the effect is not only a single noisy run.

## Guardrail checks

Policy smoke, default parallel ND:

| case | precision | factor ms/sys | relres | note |
|---|---|---:|---:|---|
| `case8387pegase` | FP32 | 0.358132 | 1.27e-5 | mid TC skipped by row/nc gate |
| `case8387pegase` | FP16 | 0.336390 | 3.22e-5 | preserved residual |
| `case_SyntheticUSA` | FP32 | 2.33649 | 1.05e-3 | mid TC skipped by row gate |
| `case_SyntheticUSA` | FP16 | 1.81909 | 4.82e-2 | big FP16 TC still active |

## Negative results

- Forcing mid FP16 TC on `case8387pegase` produced `relres ~= 2e-2`; the `nc>=10` gate removes
  that route.
- Forcing mid FP16 TC on `case_SyntheticUSA` default parallel ND produced a residual spike
  (`relres ~= 5.86` in one run); the row-count gate skips USA mid TC.
- 1024-thread launch for the fused mid FP16 kernel failed graph setup with
  `too many resources requested`; 512 threads is the working launch shape.
- A trial WMMA `m16n16k16` mid path failed factorization and was reverted. The current path uses
  the verified PTX `mma.m16n8k8` helper.
