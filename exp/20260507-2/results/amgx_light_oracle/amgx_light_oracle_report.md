# AMGX Light Correction Oracle

Input: selected five `J1/F1` standalone linear systems. AMGX configs use `AGGREGATION`, `BLOCK_JACOBI`, one V-cycle, `NOSOLVER` coarse solver, `min_coarse_rows` in `{256,512}`, and `max_levels` in `{5,10}`. No ILU/DILU and no `DENSE_LU_SOLVER` were used.

## Mode Averages

| mode | mean solve/cycle ms | median solve/cycle ms | mean dx ratio | mean dx cosine | mean true rel residual |
|---|---:|---:|---:|---:|---:|
| MR1 block-Jacobi | 0.4362 | 0.1429 | 0.1178 | 0.3109 | 0.2986 |
| AMG-only 1 V-cycle | 0.4867 | 0.2548 | 17.95 | 0.1553 | 1.461e+03 |
| FGMRES(1)+AMG(1V) | 0.1171 | 0.1121 | 0 | 0 | 1 |
| FGMRES(2)+AMG(1V) | 0.4517 | 0.407 | nan | nan | nan |

## Fastest Stable-Looking AMG-Only Config: `amg1v_nosolver_mc512_ml5`

| case | MR1 ms | AMG cycle ms | MR1 dx ratio | AMG dx ratio | MR1 cosine | AMG cosine | AMG theta ratio | AMG |V| ratio | AMG true rel residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 0.1559 | 0.288 | 0.004865 | 0.007788 | -0.02776 | 0.1207 | 0.005791 | 0.3527 | 0.4426 |
| case2383wp | 1.641 | 0.1711 | 0.02206 | 0.01328 | 0.4699 | 0.3824 | 0.01125 | 0.7293 | 0.08605 |
| case3120sp | 0.1135 | 0.2054 | 0.02175 | 0.1594 | 0.3512 | 0.01107 | 0.1054 | 5.411 | 15.06 |
| case6468rte | 0.1283 | 0.2182 | 0.2635 | 89.2 | 0.514 | 0.009229 | 62.67 | 1.791e+02 | 7.271e+03 |
| case9241pegase | 0.1429 | 0.2798 | 0.2769 | 0.1262 | 0.2472 | 0.1997 | 0.1011 | 0.3218 | 0.3299 |

## FGMRES Fixed-Step Sanity

- FGMRES(1)+AMG(1V) returned zero correction in `20/20` runs, with true relative residual `1`.
- FGMRES(2)+AMG(1V) produced NaN residual/correction in `20/20` runs.
- A sanity rerun with `tolerance=1e-12` showed the same behavior, while the previous full AMGX config only produced a nonzero FGMRES correction when `DENSE_LU_SOLVER` was used.

## Gate Decision

- AMG-only 1 V-cycle can be in the MR1 timing neighborhood for the lighter configs, but the correction quality is poor: average cosine is lower than MR1 and several cases blow up the true residual.
- FGMRES(1/2)+AMG(1V)+`NOSOLVER` is not usable in this AMGX configuration because it returns zero or NaN corrections.
- Gate result: `reject`. AMG light correction does not improve dx cosine without norm/residual collapse.
