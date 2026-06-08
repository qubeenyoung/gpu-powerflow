# B=1 factorize 10% progress (Codex worktree)

Date: 2026-06-07
Branch: `perf/factorize-b1-10pct-codex`
Worktree: `/workspace/sparse_direct_solver/gpu-powerflow-factorize-10pct`
Baseline worktree: `/workspace/sparse_direct_solver/gpu-powerflow-baseline-7394`
Baseline commit: `7394c28`

## Goal

Accelerate `custom_linear_solver/src/factorize` at `B=1` versus current custom
`fp32`/`fp16`, with at least one `<10K bus` and one `>10K bus` case reaching
10% lower factor time. No skipped trailing update, incomplete factorization, or
factor work shifted into solve is used.

Status: the accepted code clears 10% against the best baseline mode
(`min(fp32, fp16)`) for one `<10K` case (`case3012wp`) and one `>10K`
case (`case13659pegase`). It also clears 10% against the current `fp16`
path across the measured `<10K` / `>10K` representatives.

## Accepted changes

| change | exactness / safety | impact |
| --- | --- | --- |
| Analyze-time `a_pos` uniqueness check + `scatter_values_unique` | Copies `a_pos` once after analyze, sorts it, and only replaces scatter `atomicAdd` with a store when no destination duplicate exists. Falls back to the old atomic path otherwise. | Removes unnecessary atomics from numeric scatter on all tested NR cases. |
| B=1 full factor graph replay | Captures scatter/init plus factor levels into one graph keyed by values pointer, `o2c`, and value type. Replays still read the current device values. | Reduces B=1 graph-launch overhead and keeps scatter inside the measured replay. |
| B=1 `factor_small_single` | Same small-front LU/extend math as `factor_small`, but removes batch indexing and modulo arithmetic when `B==1`. | Small but positive in small-heavy cases. |
| FP16 B=1 scalar mid for `n < 24000` | Uses the existing FP32 scalar mid kernel instead of FP16 WMMA mid when WMMA staging overhead dominates. Big-tier FP16 path is unchanged. | Large fp16 improvement and better residuals on 3012/8387/9241/13659. |
| B=1 spine panel-chain kernel | Fuses only the single-panel top spine into one staged scalar kernel. The chain is already sequential. The extend path uses plain `+=` for `n<20000` and the original atomic path for larger cases, matching the faster measured policy. | Saves several graph child kernels without serializing broad levels. |
| Float-front cap18 policy for `5000 <= n < 8000` | A narrow analyze policy for the 3K-bus / 6K-unknown class. It keeps exact factorization and only changes amalgamation width. | Moves `case3012wp` over the strict best-mode 10% target. |
| Narrow FP32 plain extend gate for very large single-front main-stream levels | Kept only for `B==1`, `fp32`, `n>=80000`, main stream, `level_size==1`. | Diagnostic large-case improvement without affecting small/medium cases. |

## Verification

Command shape for the final measurements:

```bash
build/custom_linear_solver/custom_linear_solver_run <case-dir> \
  --batch 1 --repeat 101 --precision <fp32|fp16> --single-precision fp64
```

The table compares the best baseline mode (`min(fp32, fp16)`) against the best
candidate mode for the same case. Baseline values are 3-process medians of the
baseline best mode where measured. Candidate values are 3-process medians of
`fp16` after the accepted changes. Factor times are `batch_factor_per_sys_ms`.

| case | bus class | baseline fp32 | baseline fp16 | candidate mode | candidate | reduction vs baseline best | relres |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `case3012wp` | `<10K` | 0.222217 | 0.283132 | fp16 | 0.196111 | 11.8% | 1.96e-04 |
| `case8387pegase` | `<10K` | 0.333045 | 0.439595 | fp16 | 0.314470 | 5.6% | 2.17e-05 |
| `case9241pegase` | `<10K` | 0.344407 | 0.461587 | fp16 | 0.316234 | 8.2% | 2.62e-06 |
| `case13659pegase` | `>10K` | 0.421932 | 0.579688 | fp16 | 0.363716 | 13.8% | 1.16e-04 |

Mode-specific `fp16` baseline comparison:

| case | bus class | baseline fp16 | candidate fp16 | reduction vs fp16 |
| --- | --- | ---: | ---: | ---: |
| `case3012wp` | `<10K` | 0.283132 | 0.196111 | 30.7% |
| `case8387pegase` | `<10K` | 0.439595 | 0.314470 | 28.5% |
| `case9241pegase` | `<10K` | 0.461587 | 0.316234 | 31.5% |
| `case13659pegase` | `>10K` | 0.579688 | 0.363716 | 37.3% |

Additional observations:

| case | baseline best | candidate best | note |
| --- | ---: | ---: | --- |
| `case1197` | 0.069180 | 0.065092 | Improved, but below the strict 10% target. |
| `case6468rte` | 0.271690 | 0.257443 | Improved 5.2% vs best baseline; fp16 baseline itself improves 34.0%. |
| `case_ACTIVSg10k` | 0.342312 | 0.349477 | fp16 improves versus baseline fp16 but does not beat baseline fp32. |
| `case_ACTIVSg70k` | 2.185720 | 2.079370 | Improved versus baseline fp16, not a strict best-mode 10% win. |

## Rejected experiments in this worktree

| experiment | result | decision |
| --- | --- | --- |
| `fsz 33..48` split path (`factorize_front` cutoff 48 -> 32) | No stable win on 9241/13659. | Discarded |
| FP32 mid dynamic block size 64/128/256 | Severe regression (`case9241 fp32` around milliseconds). | Discarded |
| B=1 small tier 4 or 16 warps/block | Some single-run wins, median/regression risk too high. | Discarded |
| reciprocal multiply in small LU | Regressed 13659. | Discarded |
| global `ROWFUSED_NC_MAX=8` or FP32-only narrow helper | Unstable; helped one run, hurt others. | Discarded |
| `factor_mid` `__launch_bounds__(256,2)` | Regressed 13659. | Discarded |
| subtree stream cap increase / serial nested dissection | No stable factor win. | Discarded |
| full tail-chain fusion up to 48 panels | Correct but much slower because it serialized too much useful parallelism. | Discarded |
| `factor_small_single` launch bound 512/2 and L1 preference | Regressed 6468/9241. | Discarded |
| spine-chain block size 128 | Regressed 13659/8387 versus 256 threads. | Discarded |
| move `cap=12` threshold from `n>=16000` to `n>=20000` | Correct but did not close the `<10K` strict gap and hurt 13659 variance. | Discarded |
| TF32 / TF32 WMMA for this target | Slower and less accurate on 9241/13659. | Discarded |

## Literature notes

- Davis/Duff unsymmetric-pattern multifrontal LU: dense frontal kernels are the
  right abstraction, but small/irregular fronts limit efficient GPU utilization.
- Rennich/Stosic/Davis CHOLMOD GPU work: small frontal matrices need subtree-level
  strategies; simple per-front GPU kernels hit launch/occupancy limits.
- SuperLU_DIST batched sparse direct solver (2024): batching and batched scatter
  amortize launch overhead; this helps large batches but does not directly solve
  single-system B=1.

## Notes

The accepted changes stay inside exact numeric factorization. The main speedups come from
removing overhead that is unnecessary for `B=1` and from avoiding tensor-core staging on
medium fronts where FP16 WMMA is slower than scalar FP32 for a single system.
