# AMGX Oracle Report

Standalone input: `J1/F1` dumps for the selected five cases. Nonlinear mismatch rollback was not evaluated here because the J/F dump does not include the full voltage state needed to apply and restore a trial step.

## 1. Does AMGX improve dx direction/scale versus MR1?

- AMGX mean dx norm ratio vs cuDSS: `0`.
- MR1 mean dx norm ratio vs cuDSS: `0.117826865774`.
- AMGX mean dx cosine vs cuDSS: `0`.
- MR1 mean dx cosine vs cuDSS: `0.310913293874`.

| case | AMGX dx ratio | MR1 dx ratio | AMGX cosine | MR1 cosine | AMGX theta ratio | AMGX |V| ratio |
|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 0 | 0.0220620623425 | 0 | 0.469919464223 | 0 | 0 |
| case2383wp | nan | 0.0220620623425 | nan | 0.469919464223 | nan | nan |
| case2383wp | 0 | 0.0220620623425 | 0 | 0.469919464223 | 0 | 0 |
| case2383wp | nan | 0.0220620623425 | nan | 0.469919464223 | nan | nan |
| case2383wp | 0 | 0.0220620623425 | 0 | 0.469919464223 | 0 | 0 |
| case2383wp | nan | 0.0220620623425 | nan | 0.469919464223 | nan | nan |
| case2383wp | 0 | 0.0220620623425 | 0 | 0.469919464223 | 0 | 0 |
| case2383wp | nan | 0.0220620623425 | nan | 0.469919464223 | nan | nan |
| case3120sp | 0 | 0.0217495193066 | 0 | 0.351205379859 | 0 | 0 |
| case3120sp | nan | 0.0217495193066 | nan | 0.351205379859 | nan | nan |
| case3120sp | 0 | 0.0217495193066 | 0 | 0.351205379859 | 0 | 0 |
| case3120sp | nan | 0.0217495193066 | nan | 0.351205379859 | nan | nan |
| case3120sp | 0 | 0.0217495193066 | 0 | 0.351205379859 | 0 | 0 |
| case3120sp | nan | 0.0217495193066 | nan | 0.351205379859 | nan | nan |
| case3120sp | 0 | 0.0217495193066 | 0 | 0.351205379859 | 0 | 0 |
| case3120sp | nan | 0.0217495193066 | nan | 0.351205379859 | nan | nan |
| case9241pegase | 0 | 0.276918794375 | 0 | 0.247239770181 | 0 | 0 |
| case9241pegase | nan | 0.276918794375 | nan | 0.247239770181 | nan | nan |
| case9241pegase | 0 | 0.276918794375 | 0 | 0.247239770181 | 0 | 0 |
| case9241pegase | nan | 0.276918794375 | nan | 0.247239770181 | nan | nan |
| case9241pegase | 0 | 0.276918794375 | 0 | 0.247239770181 | 0 | 0 |
| case9241pegase | nan | 0.276918794375 | nan | 0.247239770181 | nan | nan |
| case9241pegase | 0 | 0.276918794375 | 0 | 0.247239770181 | 0 | 0 |
| case9241pegase | nan | 0.276918794375 | nan | 0.247239770181 | nan | nan |
| case13659pegase | 0 | 0.00486466701122 | 0 | -0.0277580835571 | 0 | 0 |
| case13659pegase | nan | 0.00486466701122 | nan | -0.0277580835571 | nan | nan |
| case13659pegase | 0 | 0.00486466701122 | 0 | -0.0277580835571 | 0 | 0 |
| case13659pegase | nan | 0.00486466701122 | nan | -0.0277580835571 | nan | nan |
| case13659pegase | 0 | 0.00486466701122 | 0 | -0.0277580835571 | 0 | 0 |
| case13659pegase | nan | 0.00486466701122 | nan | -0.0277580835571 | nan | nan |
| case13659pegase | 0 | 0.00486466701122 | 0 | -0.0277580835571 | 0 | 0 |
| case13659pegase | nan | 0.00486466701122 | nan | -0.0277580835571 | nan | nan |
| case6468rte | 0 | 0.263539285832 | 0 | 0.513959938663 | 0 | 0 |
| case6468rte | nan | 0.263539285832 | nan | 0.513959938663 | nan | nan |
| case6468rte | 0 | 0.263539285832 | 0 | 0.513959938663 | 0 | 0 |
| case6468rte | nan | 0.263539285832 | nan | 0.513959938663 | nan | nan |
| case6468rte | 0 | 0.263539285832 | 0 | 0.513959938663 | 0 | 0 |
| case6468rte | nan | 0.263539285832 | nan | 0.513959938663 | nan | nan |
| case6468rte | 0 | 0.263539285832 | 0 | 0.513959938663 | 0 | 0 |
| case6468rte | nan | 0.263539285832 | nan | 0.513959938663 | nan | nan |

## 2. Does AMGX reduce nonlinear mismatch more than MR1?

- Not answered by this standalone run. The linear/dx gate failed, so hybrid shadow integration was not added.

## 3. Is AMGX setup/solve cost compatible with hybrid middle use?

- Mean AMGX setup ms: `2.487714625`.
- Mean AMGX solve ms: `0.2843952`.
- Mean cuDSS factorize+solve ms: `4.0462222`.

| case | AMGX rel residual | MR1 rel residual | AMGX setup ms | AMGX solve ms | cuDSS factorize+solve ms |
|---|---:|---:|---:|---:|---:|
| case2383wp | 1 | 0.029952904578 | 1.894447 | 0.179857 | 15.313924 |
| case2383wp | nan | 0.029952904578 | 1.897723 | 1.397176 | 15.313924 |
| case2383wp | 1 | 0.029952904578 | 1.897083 | 0.107251 | 15.313924 |
| case2383wp | nan | 0.029952904578 | 1.891392 | 0.354694 | 15.313924 |
| case2383wp | 1 | 0.029952904578 | 1.453492 | 0.110087 | 15.313924 |
| case2383wp | nan | 0.029952904578 | 1.423996 | 0.319258 | 15.313924 |
| case2383wp | 1 | 0.029952904578 | 1.438955 | 0.10661 | 15.313924 |
| case2383wp | nan | 0.029952904578 | 1.442501 | 0.312555 | 15.313924 |
| case3120sp | 1 | 0.256996272353 | 1.984296 | 0.106339 | 0.709208 |
| case3120sp | nan | 0.256996272353 | 1.981771 | 0.356067 | 0.709208 |
| case3120sp | 1 | 0.256996272353 | 1.984646 | 0.108352 | 0.709208 |
| case3120sp | nan | 0.256996272353 | 1.984486 | 0.354393 | 0.709208 |
| case3120sp | 1 | 0.256996272353 | 1.981941 | 0.10676 | 0.709208 |
| case3120sp | nan | 0.256996272353 | 1.987782 | 0.353973 | 0.709208 |
| case3120sp | 1 | 0.256996272353 | 2.005956 | 0.109105 | 0.709208 |
| case3120sp | nan | 0.256996272353 | 1.982672 | 0.355225 | 0.709208 |
| case9241pegase | 1 | 0.155972926023 | 2.962737 | 0.124473 | 1.422795 |
| case9241pegase | nan | 0.155972926023 | 2.955614 | 0.444502 | 1.422795 |
| case9241pegase | 1 | 0.155972926023 | 3.634916 | 0.124473 | 1.422795 |
| case9241pegase | nan | 0.155972926023 | 3.648792 | 0.483284 | 1.422795 |
| case9241pegase | 1 | 0.155972926023 | 2.962166 | 0.125535 | 1.422795 |
| case9241pegase | nan | 0.155972926023 | 2.950665 | 0.445323 | 1.422795 |
| case9241pegase | 1 | 0.155972926023 | 3.184082 | 0.122138 | 1.422795 |
| case9241pegase | nan | 0.155972926023 | 3.153425 | 0.459711 | 1.422795 |
| case13659pegase | 1 | 0.553976388557 | 2.93735 | 0.117159 | 1.784501 |
| case13659pegase | nan | 0.553976388557 | 2.956115 | 0.456324 | 1.784501 |
| case13659pegase | 1 | 0.553976388557 | 3.609108 | 0.117389 | 1.784501 |
| case13659pegase | nan | 0.553976388557 | 3.636359 | 0.490178 | 1.784501 |
| case13659pegase | 1 | 0.553976388557 | 2.932611 | 0.11725 | 1.784501 |
| case13659pegase | nan | 0.553976388557 | 2.925568 | 0.451555 | 1.784501 |
| case13659pegase | 1 | 0.553976388557 | 3.128908 | 0.118322 | 1.784501 |
| case13659pegase | nan | 0.553976388557 | 3.138908 | 0.449822 | 1.784501 |
| case6468rte | 1 | 0.495915403417 | 2.501644 | 0.11219 | 1.000683 |
| case6468rte | nan | 0.495915403417 | 2.510541 | 0.404027 | 1.000683 |
| case6468rte | 1 | 0.495915403417 | 2.704744 | 0.11204 | 1.000683 |
| case6468rte | nan | 0.495915403417 | 2.720103 | 0.409967 | 1.000683 |
| case6468rte | 1 | 0.495915403417 | 2.274719 | 0.109495 | 1.000683 |
| case6468rte | nan | 0.495915403417 | 2.280671 | 0.369743 | 1.000683 |
| case6468rte | 1 | 0.495915403417 | 2.289477 | 0.107962 | 1.000683 |
| case6468rte | nan | 0.495915403417 | 2.276223 | 0.365244 | 1.000683 |

## 4. Should AMGX be integrated into hybrid NR?

- Gate score from standalone metrics: `1 / 3`.
- Hybrid integration recommendation: `reject`.
