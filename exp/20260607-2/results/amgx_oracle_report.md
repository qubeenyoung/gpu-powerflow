# AMGX Oracle Report

Standalone input: `J1/F1` dumps for the selected five cases. Nonlinear mismatch rollback was not evaluated here because the J/F dump does not include the full voltage state needed to apply and restore a trial step.

## 1. Does AMGX improve dx direction/scale versus MR1?

- AMGX mean dx norm ratio vs cuDSS: `0.0834328563772`.
- MR1 mean dx norm ratio vs cuDSS: `0.117826865774`.
- AMGX mean dx cosine vs cuDSS: `0.432559778344`.
- MR1 mean dx cosine vs cuDSS: `0.310913293874`.

| case | AMGX dx ratio | MR1 dx ratio | AMGX cosine | MR1 cosine | AMGX theta ratio | AMGX |V| ratio |
|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 0.0160748655111 | 0.0220620623425 | 0.507241767011 | 0.469919464223 | 0.0131081907719 | 0.962155062045 |
| case3120sp | 0.0165772383668 | 0.0217495193066 | 0.53823912759 | 0.351205379859 | 0.0118391094236 | 0.525288465161 |
| case9241pegase | 0.241936114416 | 0.276918794375 | 0.255327701705 | 0.247239770181 | 0.148794553574 | 0.784734576322 |
| case13659pegase | 0.0170972407419 | 0.00486466701122 | 0.310333392039 | -0.0277580835571 | 0.0122927096063 | 0.804739620542 |
| case6468rte | 0.12547882285 | 0.263539285832 | 0.551656903374 | 0.513959938663 | 0.0746824340959 | 0.276866924852 |

## 2. Does AMGX reduce nonlinear mismatch more than MR1?

- Not answered by this standalone run. The linear/dx gate failed, so hybrid shadow integration was not added.

## 3. Is AMGX setup/solve cost compatible with hybrid middle use?

- Mean AMGX setup ms: `9.3296458`.
- Mean AMGX solve ms: `6.9155102`.
- Mean cuDSS factorize+solve ms: `4.0239862`.

| case | AMGX rel residual | MR1 rel residual | AMGX setup ms | AMGX solve ms | cuDSS factorize+solve ms |
|---|---:|---:|---:|---:|---:|
| case2383wp | 0.0869151460467 | 0.029952904578 | 31.091002 | 7.923866 | 15.166887 |
| case3120sp | 0.345737352644 | 0.256996272353 | 3.237321 | 7.896365 | 0.707866 |
| case9241pegase | 0.0812463406683 | 0.155972926023 | 4.076272 | 1.739808 | 1.431059 |
| case13659pegase | 0.0980783961485 | 0.553976388557 | 4.501478 | 8.826666 | 1.820247 |
| case6468rte | 0.64077681679 | 0.495915403417 | 3.742156 | 8.190846 | 0.993872 |

## 4. Should AMGX be integrated into hybrid NR?

- Gate score from standalone metrics: `0 / 3`.
- Hybrid integration recommendation: `reject`.
