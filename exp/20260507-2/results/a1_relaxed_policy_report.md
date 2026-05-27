# A1 relaxed fallback comparison

Compared against previous best implementation path: `optimized_numeric_reuse`.

| policy | case | NR / pure | full cuDSS | A1 calls | fallback | total speedup | linear speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| strict_cap2_accept0.5 | case2383wp | 6 / 6 | 4 | 2 | 0 | 1.560 | 1.422 |
| strict_cap2_accept0.5 | case3120sp | 6 / 6 | 4 | 2 | 0 | 0.992 | 0.858 |
| strict_cap2_accept0.5 | case9241pegase | 6 / 6 | 4 | 2 | 0 | 1.053 | 0.862 |
| strict_cap2_accept0.5 | case13659pegase | 6 / 5 | 4 | 2 | 0 | 1.008 | 0.840 |
| strict_cap2_accept0.5 | case6468rte | 3 / 3 | 2 | 1 | 0 | 1.005 | 0.899 |
| nocap_accept0.5 | case2383wp | 9 / 6 | 2 | 7 | 0 | 1.318 | 1.184 |
| nocap_accept0.5 | case3120sp | 6 / 6 | 3 | 4 | 1 | 0.970 | 0.808 |
| nocap_accept0.5 | case9241pegase | 6 / 6 | 3 | 5 | 2 | 1.059 | 0.822 |
| nocap_accept0.5 | case13659pegase | 6 / 5 | 3 | 4 | 1 | 1.038 | 0.840 |
| nocap_accept0.5 | case6468rte | 3 / 3 | 2 | 1 | 0 | 1.001 | 0.893 |
| nocap_accept0.9 | case2383wp | 9 / 6 | 2 | 7 | 0 | 1.340 | 1.198 |
| nocap_accept0.9 | case3120sp | 6 / 6 | 3 | 4 | 1 | 0.963 | 0.808 |
| nocap_accept0.9 | case9241pegase | 6 / 6 | 3 | 5 | 2 | 1.050 | 0.824 |
| nocap_accept0.9 | case13659pegase | 6 / 5 | 3 | 4 | 1 | 1.034 | 0.840 |
| nocap_accept0.9 | case6468rte | 3 / 3 | 2 | 1 | 0 | 1.005 | 0.899 |

Finding: removing `max_a1_middle_accepts=2` reduces full cuDSS calls on 3120/9241/13659 and dramatically on 2383, but it also accepts weaker late A1 steps. That increases NR length for case2383wp from 6 to 9 and does not improve linear-solve speed on most cases.

Accept ratio 0.5 vs 0.9 makes almost no difference here; the accepted A1 steps mostly reduce mismatch by far more than 50%. The active strictness was the accept-count cap, not the mismatch ratio.
