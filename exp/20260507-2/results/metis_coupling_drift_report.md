# METIS Coupling Retention and Jacobian Drift Diagnostic

## 1. Does current METIS cut strong coupling?

- Mean off-block NNZ ratio is `0.249`, but mean off-block abs ratio is only `0.0787` and mean off-block Frobenius ratio is `0.205`.
- This means many entries are outside the diagonal blocks, but most large-magnitude coupling remains inside.

| case | offblock nnz | offblock abs | offblock fro |
|---|---:|---:|---:|
| case13659pegase | 0.289 | 0.0975 | 0.212 |
| case2383wp | 0.215 | 0.0702 | 0.252 |
| case3120sp | 0.206 | 0.0845 | 0.254 |
| case6468rte | 0.238 | 0.06 | 0.119 |
| case9241pegase | 0.297 | 0.0814 | 0.189 |

## 2. Where is the cut coupling concentrated?

- By cuDSS-dx effect, the largest off-block fractions are in the cross terms `J12` and `J21`.
- Mean field effect ratios: J11 `0.0593`, J12 `0.358`, J21 `0.329`, J22 `0.0388`.

| case | J11 effect | J12 effect | J21 effect | J22 effect |
|---|---:|---:|---:|---:|
| case13659pegase | 0.0877 | 0.452 | 0.395 | 0.0506 |
| case2383wp | 0.068 | 0.275 | 0.27 | 0.0369 |
| case3120sp | 0.0684 | 0.314 | 0.338 | 0.0463 |
| case6468rte | 0.0218 | 0.153 | 0.203 | 0.0205 |
| case9241pegase | 0.0507 | 0.594 | 0.44 | 0.0399 |

## 3. Is important cuDSS-dx coupling off-block?

- Overall mean offblock_effect_ratio is `0.085`.
- It is not large enough to say that unknown-level METIS is discarding most of the correction-driving coupling.

| case | offblock effect | top 5% effect kept | top 5% coupling kept | theta/V split |
|---|---:|---:|---:|---:|
| case13659pegase | 0.118 | 0.944 | 0.986 | 3283 |
| case2383wp | 0.0735 | 0.939 | 0.959 | 585 |
| case3120sp | 0.0971 | 0.906 | 0.912 | 765 |
| case6468rte | 0.0545 | 0.992 | 0.993 | 1953 |
| case9241pegase | 0.0818 | 0.971 | 0.983 | 3157 |

## 4. Are top bus-pair couplings preserved?

- Mean top-5% coupling kept ratio is `0.967`.
- Mean top-5% effect kept ratio is `0.95`.
- The lowest top-5% effect kept case is `case3120sp`, but even there the mean is about `0.906`.

## 5. How much does J change from J0 to J2?

| case | rel J0->J1 | rel J1->J2 | offblock J0->J1 | offblock J1->J2 |
|---|---:|---:|---:|---:|
| case13659pegase | 0.0204 | 0.0146 | 0.0241 | 0.018 |
| case2383wp | 0.106 | 0.0114 | 0.0871 | 0.00772 |
| case3120sp | 0.0864 | 0.0667 | 0.0628 | 0.0734 |
| case6468rte | 2.48e-05 | 1.26e-06 | 2.69e-05 | 9.83e-07 |
| case9241pegase | 0.00298 | 0.00168 | 0.00579 | 0.00333 |

- Drift is case-dependent: `case2383wp` and `case3120sp` move noticeably early, while `case6468rte` and `case9241pegase` are nearly static.
- Off-block drift is not systematically larger than in-block drift, so the static partition assumption is not the main suspect from this diagnostic alone.

## 6. Does this justify bus-aware weighted METIS?

- Weakly. Same-bus theta/V splits are common, and J12/J21 off-block effect is high, so the idea was worth checking.
- But top effect bus-pair retention is already high and overall offblock_effect_ratio is modest, so the diagnostic does not strongly support the hypothesis that unknown-level METIS is cutting the most important bus couplings.
- This matches the follow-up bus-weighted experiment: structural retention improved, but MR1 dx quality and fallback behavior did not.
