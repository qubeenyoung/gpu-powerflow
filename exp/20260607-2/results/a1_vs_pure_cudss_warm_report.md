# A1 vs Pure cuDSS Warm Comparison

Both sides are warmed: pure cuDSS uses `--full-cudss-analyze-before-loop true --warmup 1`; A1 hybrid also uses `--full-cudss-analyze-before-loop true --warmup 1`. A1 timing is the middle-call time: BJ cache check/setup + BiCGSTAB(2) + device A1 field correction.

## strict_cap2_accept0.5
| case | NR iters | calls cuDSS/A1/fallback | pure cuDSS avg ms | pure cuDSS median ms | A1 avg ms | A1 median ms | A1 / pure median |
|---|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 6 | 4/2/0 | 1.577 | 1.559 | 1.981 | 1.981 | 1.27 |
| case2383wp | 6 | 4/2/0 | 0.516 | 0.503 | 1.137 | 1.137 | 2.26 |
| case3120sp | 6 | 4/2/0 | 0.639 | 0.627 | 1.250 | 1.250 | 1.99 |
| case6468rte | 3 | 2/1/0 | 0.847 | 0.821 | 1.656 | 1.656 | 2.02 |
| case9241pegase | 6 | 4/2/0 | 1.278 | 1.261 | 1.717 | 1.717 | 1.36 |

## nocap_accept0.9
| case | NR iters | calls cuDSS/A1/fallback | pure cuDSS avg ms | pure cuDSS median ms | A1 avg ms | A1 median ms | A1 / pure median |
|---|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 6 | 3/4/1 | 1.577 | 1.559 | 1.955 | 1.923 | 1.23 |
| case2383wp | 9 | 2/7/0 | 0.516 | 0.503 | 1.094 | 1.083 | 2.15 |
| case3120sp | 6 | 3/4/1 | 0.639 | 0.627 | 1.237 | 1.205 | 1.92 |
| case6468rte | 3 | 2/1/0 | 0.847 | 0.821 | 1.643 | 1.643 | 2.00 |
| case9241pegase | 6 | 3/5/2 | 1.278 | 1.261 | 2.228 | 1.852 | 1.47 |

## Bottom line
With both sides warmed, A1 middle is still slower than full-J cuDSS factor+solve per call on every selected case. The median A1/full-cuDSS ratio ranges from about 1.17x to 2.09x depending on policy/case.
