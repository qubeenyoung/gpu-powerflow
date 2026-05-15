# BiCGSTAB(2), block size 8, bootstrap 2, all cases

## Setting
- middle solver: `bicgstab_block_jacobi`
- fixed BiCGSTAB iterations: `2`
- preconditioner: `metis_block_jacobi`, block size `8`, FP32 `inverse_gemv`
- bootstrap cuDSS iterations: `2`
- polish threshold: `1e-4`, accept `0.9`, reject `1.05`, fallback immediate
- warmup: `1`

Raw summary columns still use legacy `gmres_*` names; here they mean BiCGSTAB middle calls/accepted steps.

## Bootstrap 1 vs 2

| metric | bootstrap 1 | bootstrap 2 |
|---|---:|---:|
| cases | 78 | 78 |
| converged | 78 | 78 |
| aggregate speedup | 0.842 | 0.899 |
| mean speedup | 0.869 | 0.944 |
| median speedup | 0.889 | 0.996 |
| speedup > 1 cases | 5 | 28 |
| speedup >= 0.95 cases | 12 | 45 |
| total hybrid ms | 2141.6 | 2014.4 |
| total pure cuDSS ms | 1804.0 | 1811.3 |
| total NR iterations | 378 | 325 |
| cuDSS calls | 244 | 263 |
| BiCGSTAB middle calls | 227 | 103 |
| accepted BiCGSTAB steps | 134 | 62 |
| fallback calls | 93 | 41 |
| cases with accepted BiCGSTAB | 49 | 29 |

## Bootstrap 2 Size Bins

| bin | cases | aggregate speedup | median speedup | wins | accepted BiCGSTAB cases |
|---|---:|---:|---:|---:|---:|
| <=100 buses | 41 | 0.952 | 0.998 | 17 | 12 |
| 101-1000 buses | 10 | 0.977 | 1.000 | 4 | 2 |
| 1001-10000 buses | 23 | 0.907 | 0.929 | 7 | 12 |
| >10000 buses | 4 | 0.863 | 0.861 | 0 | 3 |

## Top Improved vs Bootstrap 1

| case | buses | speedup b1 -> b2 | NR b1 -> b2 | cuDSS b1 -> b2 | middle b1 -> b2 | accepted b1 -> b2 | fallback b1 -> b2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg500 | 500 | 0.640 -> 0.999 | 10 -> 3 | 3 -> 3 | 8 -> 0 | 7 -> 0 | 1 -> 0 |
| case16ci | 16 | 0.702 -> 0.999 | 8 -> 3 | 2 -> 3 | 6 -> 0 | 6 -> 0 | 0 -> 0 |
| case18 | 18 | 0.627 -> 0.875 | 10 -> 5 | 3 -> 4 | 9 -> 2 | 7 -> 1 | 2 -> 1 |
| case145 | 145 | 0.772 -> 1.002 | 6 -> 3 | 3 -> 3 | 4 -> 0 | 3 -> 0 | 1 -> 0 |
| case_ACTIVSg2000 | 2000 | 0.716 -> 0.943 | 9 -> 4 | 3 -> 3 | 7 -> 1 | 6 -> 1 | 1 -> 0 |
| case118 | 118 | 0.788 -> 1.001 | 6 -> 3 | 3 -> 3 | 4 -> 0 | 3 -> 0 | 1 -> 0 |
| case6ww | 6 | 0.952 -> 1.148 | 3 -> 3 | 2 -> 3 | 1 -> 0 | 1 -> 0 | 0 -> 0 |
| case17me | 17 | 0.815 -> 1.001 | 6 -> 4 | 2 -> 4 | 4 -> 0 | 4 -> 0 | 0 -> 0 |
| case38si | 38 | 0.831 -> 1.014 | 5 -> 5 | 4 -> 4 | 3 -> 2 | 1 -> 1 | 2 -> 1 |
| case300 | 300 | 0.719 -> 0.902 | 9 -> 7 | 3 -> 4 | 7 -> 4 | 6 -> 3 | 1 -> 1 |
| case59 | 59 | 0.641 -> 0.811 | 11 -> 7 | 4 -> 4 | 9 -> 4 | 7 -> 3 | 2 -> 1 |
| case2746wp | 2746 | 0.650 -> 0.815 | 8 -> 7 | 4 -> 4 | 7 -> 5 | 4 -> 3 | 3 -> 2 |

## 1K+ Accepted Middle Cases, Bootstrap 2

| case | buses | speedup | NR | cuDSS | middle | accepted | fallback | accepted mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg2000 | 2000 | 0.943 | 4 | 3 | 1 | 1 | 0 | 0.094 |
| case2868rte | 2868 | 0.894 | 5 | 4 | 3 | 1 | 2 | 0.317 |
| case2736sp | 2736 | 0.883 | 5 | 4 | 2 | 1 | 1 | 0.297 |
| case_ACTIVSg10k | 10000 | 0.866 | 6 | 4 | 3 | 2 | 1 | 0.635 |
| case_ACTIVSg70k | 70000 | 0.861 | 7 | 6 | 4 | 1 | 3 | 0.821 |
| case2746wop | 2746 | 0.855 | 6 | 4 | 3 | 2 | 1 | 0.594 |
| case2737sop | 2737 | 0.850 | 7 | 4 | 4 | 3 | 1 | 0.592 |
| case_SyntheticUSA | 82000 | 0.849 | 7 | 6 | 4 | 1 | 3 | 0.419 |
| case3120sp | 3120 | 0.835 | 7 | 6 | 4 | 1 | 3 | 0.565 |
| case9241pegase | 9241 | 0.819 | 8 | 5 | 5 | 3 | 2 | 0.388 |
| case2746wp | 2746 | 0.815 | 7 | 4 | 5 | 3 | 2 | 0.767 |
| case2383wp | 2383 | 0.813 | 7 | 5 | 5 | 2 | 3 | 0.399 |
| case13659pegase | 13659 | 0.795 | 7 | 5 | 4 | 2 | 2 | 0.310 |
| case2869pegase | 2869 | 0.794 | 9 | 4 | 6 | 5 | 1 | 0.376 |
| case1354pegase | 1354 | 0.764 | 8 | 4 | 5 | 4 | 1 | 0.577 |

## Timing Snapshot

- Average BiCGSTAB middle/fallback solve time: `0.279 ms` over 103 timing rows.
- Average preconditioner total time: `0.057 ms`.
- Average BiCGSTAB SpMV time: `0.033 ms`.
- Average BiCGSTAB dot/reduction time: `0.152 ms`.

## Judgment

- Bootstrap 2 improves the aggregate number from `0.842` to `0.899`, but it does this mostly by suppressing middle BiCGSTAB attempts, not by making accepted BiCGSTAB steps faster.
- BiCGSTAB middle calls drop from `227` to `103`, fallback calls from `93` to `41`, and total NR iterations from `378` to `325`.
- cuDSS calls increase from `244` to `263`, so this is closer to pure cuDSS behavior.
- Speedup > 1 cases are mostly cases with no accepted BiCGSTAB middle step. That is not evidence that the middle solver accelerates NR.
- For 1K+ cases with real accepted middle steps, the best bootstrap 2 speedups are still below 1 except trivial/no-middle cases.
