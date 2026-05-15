# Block Jacobi vs Block ILU(0) Standalone Comparison

## Scope

Selected cases:

- `case2383wp`
- `case3120sp`
- `case6468rte`
- `case9241pegase`
- `case13659pegase`

Experiment:

- Linear system: `J1/F1`
- Outer method: fixed BiCGSTAB(2)
- Baseline: block-Jacobi
- Candidate: block ILU(0), block-coloring order
- Block sizes: 16, 32
- Reference solution: full-J cuDSS

Output CSV:

- `results/block_jacobi_vs_block_ilu_compare/block_ilu0_standalone_quality.csv`
- `results/block_jacobi_vs_block_ilu_compare/block_ilu0_timing.csv`

## Summary

| block size | residual improved | dx cosine improved | dx norm ratio improved | mean BiCGSTAB time increase | mean apply time increase |
|---:|---:|---:|---:|---:|---:|
| 16 | 5 / 5 | 5 / 5 | 5 / 5 | 4.43x | 10.74x |
| 32 | 4 / 5 | 4 / 5 | 5 / 5 | 5.40x | 8.77x |

## Case Detail

| case | bs | BJ relres | ILU relres | BJ cosine | ILU cosine | BJ dx ratio | ILU dx ratio | BJ BiCGSTAB ms | ILU BiCGSTAB ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 16 | 1.916e-2 | 1.580e-2 | 0.427 | 0.868 | 0.017 | 0.127 | 0.307 | 1.246 |
| case2383wp | 32 | 1.820e-2 | 2.068e-2 | 0.561 | 0.806 | 0.031 | 0.137 | 0.431 | 2.220 |
| case3120sp | 16 | 1.656e-1 | 7.921e-2 | 0.293 | 0.757 | 0.021 | 0.108 | 0.400 | 1.608 |
| case3120sp | 32 | 2.189e-1 | 9.861e-2 | 0.450 | 0.891 | 0.051 | 0.195 | 0.581 | 2.943 |
| case6468rte | 16 | 6.631e-1 | 2.326e-1 | -0.137 | 0.644 | 0.145 | 0.530 | 0.920 | 4.048 |
| case6468rte | 32 | 6.573e-1 | 5.253e-1 | 0.526 | 0.600 | 0.288 | 0.509 | 1.311 | 7.098 |
| case9241pegase | 16 | 3.397e-2 | 7.796e-3 | 0.256 | 0.271 | 0.223 | 0.261 | 1.255 | 5.454 |
| case9241pegase | 32 | 2.217e-2 | 1.257e-3 | 0.258 | 0.297 | 0.228 | 0.262 | 1.777 | 9.770 |
| case13659pegase | 16 | 3.018e-1 | 9.711e-2 | 0.020 | 0.025 | 0.006 | 0.012 | 1.647 | 8.797 |
| case13659pegase | 32 | 2.075e-1 | 3.699e-2 | 0.040 | 0.032 | 0.009 | 0.015 | 2.410 | 14.139 |

## Interpretation

- Block ILU(0) improves standalone correction quality over block-Jacobi in most cases.
- The strongest quality gain is on `case6468rte`: cosine changes from negative or weak to clearly positive, and dx norm ratio increases sharply.
- The best consistency is block size 16: it improves all three quality metrics on all five cases.
- The cost increase is large. Block ILU(0) apply is about 9-11x more expensive than block-Jacobi apply in this CPU pilot, and BiCGSTAB total time grows about 4-5x.
- This supports the current thesis: block ILU(0) has useful preconditioner quality, but the implementation must be accelerated before it is competitive.

## Caveat

This comparison uses the existing CPU dense block ILU(0) pilot for solver-quality measurement. The separate GPU block ILU(0) phase pilot measures GPU factor/apply mechanics, but is not yet integrated into the BiCGSTAB solver path.
