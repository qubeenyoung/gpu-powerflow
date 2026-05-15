# AMGX Oracle Report

Standalone input: `J1/F1` dumps for the selected five cases. Nonlinear mismatch rollback was not evaluated here because the J/F dump does not include the full voltage state needed to apply and restore a trial step.

## 1. Does AMGX improve dx direction/scale versus MR1?

- AMGX mean dx norm ratio vs cuDSS: `0.00803743275554`.
- MR1 mean dx norm ratio vs cuDSS: `0.0220620623425`.
- AMGX mean dx cosine vs cuDSS: `0.253620883505`.
- MR1 mean dx cosine vs cuDSS: `0.469919464223`.

| case | AMGX dx ratio | MR1 dx ratio | AMGX cosine | MR1 cosine | AMGX theta ratio | AMGX |V| ratio |
|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 0 | 0.0220620623425 | 0 | 0.469919464223 | 0 | 0 |
| case2383wp | nan | 0.0220620623425 | nan | 0.469919464223 | nan | nan |
| case2383wp | 0.0160748655111 | 0.0220620623425 | 0.507241767011 | 0.469919464223 | 0.0131081907719 | 0.962155062045 |

## 2. Does AMGX reduce nonlinear mismatch more than MR1?

- Not answered by this standalone run. The linear/dx gate failed, so hybrid shadow integration was not added.

## 3. Is AMGX setup/solve cost compatible with hybrid middle use?

- Mean AMGX setup ms: `7.18546433333`.
- Mean AMGX solve ms: `5.540347`.
- Mean cuDSS factorize+solve ms: `15.341107`.

| case | AMGX rel residual | MR1 rel residual | AMGX setup ms | AMGX solve ms | cuDSS factorize+solve ms |
|---|---:|---:|---:|---:|---:|
| case2383wp | 1 | 0.029952904578 | 12.309178 | 3.417098 | 15.341107 |
| case2383wp | nan | 0.029952904578 | 1.899577 | 11.12989 | 15.341107 |
| case2383wp | 0.0869151460467 | 0.029952904578 | 7.347638 | 2.074053 | 15.341107 |

## 4. Should AMGX be integrated into hybrid NR?

- Gate score from standalone metrics: `1 / 3`.
- Hybrid integration recommendation: `reject`.
