# A1 timing reconciled report

This report separates four different quantities that were previously mixed: end-to-end NR wall time, cold-included linear total, warm full-J cuDSS per-call time, and A1 middle per-call time.

## strict_cap2
| case | NR iters hybrid/pure | calls full cuDSS/A1/fallback | pure NR total ms | hybrid NR total ms | pure linear cold-included ms | hybrid full cuDSS linear ms | hybrid A1 linear ms | pure warm cuDSS median ms | A1 middle median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 6/5 | 4/2/0 | 74.456 | 73.841 | 61.232 | 66.945 | 3.914 | 1.554 | 1.957 |
| case2383wp | 6/6 | 4/2/0 | 46.562 | 29.839 | 40.512 | 20.789 | 6.919 | 0.505 | 3.459 |
| case3120sp | 6/6 | 4/2/0 | 28.343 | 28.576 | 23.530 | 23.890 | 2.542 | 0.622 | 1.271 |
| case6468rte | 3/3 | 2/1/0 | 37.598 | 37.397 | 33.130 | 34.526 | 1.656 | 0.823 | 1.656 |
| case9241pegase | 6/6 | 4/2/0 | 62.795 | 59.641 | 50.601 | 53.590 | 3.449 | 1.260 | 1.724 |

## nocap_accept0.9
| case | NR iters hybrid/pure | calls full cuDSS/A1/fallback | pure NR total ms | hybrid NR total ms | pure linear cold-included ms | hybrid full cuDSS linear ms | hybrid A1 linear ms | pure warm cuDSS median ms | A1 middle median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 6/5 | 3/3/1 | 74.456 | 71.982 | 61.232 | 63.940 | 5.821 | 1.554 | 1.920 |
| case2383wp | 9/6 | 2/7/0 | 46.252 | 34.525 | 40.512 | 18.826 | 12.257 | 0.505 | 1.085 |
| case3120sp | 6/6 | 3/3/1 | 28.428 | 29.509 | 23.530 | 23.888 | 3.709 | 0.622 | 1.204 |
| case6468rte | 3/3 | 2/1/0 | 37.674 | 37.468 | 33.130 | 34.597 | 1.656 | 0.823 | 1.656 |
| case9241pegase | 6/6 | 3/3/2 | 63.296 | 60.285 | 50.601 | 52.200 | 6.660 | 1.260 | 1.830 |

## Correct reading
- Trust `pure warm cuDSS median ms` when asking: “is one A1 middle call faster than one already-analyzed full-J cuDSS factor+solve?” On these runs, A1 is generally slower than warm full cuDSS.
- Trust `pure NR total ms` vs `hybrid NR total ms` when asking: “what happened end-to-end including first cold analyze/setup?” These can be close because the first cold cuDSS call dominates pure total and hybrid still uses/reuses some full cuDSS calls.
- Trust `hybrid full cuDSS linear ms` + `hybrid A1 linear ms` when asking: “where did hybrid time go?” In hybrid, a lot of time is still full cuDSS, and A1 adds extra time rather than being a universally cheaper replacement for warm cuDSS.

## Bottom line
The consistent conclusion is: A1 can reduce full cuDSS call count, but one A1 middle step is not faster than a warm full-J cuDSS factor+solve in this measurement. Similar end-to-end totals come from cold-start/analyze effects and reduced call counts, not because A1 is intrinsically faster per warm solve.
