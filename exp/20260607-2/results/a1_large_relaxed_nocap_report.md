# A1 relaxed/no-cap on larger cases

Setting: `bicgstab_block_jacobi_a1_device`, `block_size=16`, `BiCGSTAB(2)`, `accept=0.9`, no A1 accept cap, `fallback=immediate`, `bj_setup=numeric_reuse_after_full_cudss`, `a1_event_dag=true`.

Timing comparison uses the embedded pure-cuDSS baseline from the same A1 benchmark run. The separate pure-only run is used only to record pure NR iteration counts.

| case | buses | pure NR | A1 NR | full cuDSS calls | A1 calls | fallback | pure ms | A1 hybrid ms | speedup | median A1 middle ms | median field ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case8387pegase | 8387 | 3 | 4 | 2 | 2 | 0 | 31.223 | 14.002 | 2.230 | 3.118 | 2.417 |
| case_ACTIVSg10k | 10000 | 4 | 9 | 2 | 7 | 0 | 12.389 | 21.470 | 0.577 | 1.689 | 1.173 |
| case_ACTIVSg25k | 25000 | 4 | 5 | 3 | 3 | 1 | 28.960 | 28.429 | 1.019 | 2.932 | 2.250 |
| case_ACTIVSg70k | 70000 | 6 | 9 | 3 | 7 | 1 | 107.999 | 96.087 | 1.124 | 6.305 | 5.223 |
| case_SyntheticUSA | 82000 | 6 | 9 | 4 | 7 | 2 | 121.394 | 126.474 | 0.960 | 6.884 | 5.674 |

## Observations
- Faster than embedded pure cuDSS on `3/5` large cases: case8387pegase, case_ACTIVSg25k, case_ACTIVSg70k.
- `case8387pegase` is the clean win: pure NR is 3 iterations, A1 hybrid is 4 iterations, no fallback, full cuDSS calls drop to 2, speedup about `2.23x`.
- `case_ACTIVSg70k` wins despite one fallback: pure NR is 6, A1 hybrid is 9, but full cuDSS calls drop from 6 to 3 and total speedup is about `1.12x`.
- `case_ACTIVSg10k` loses because A1 creates many weak accepted steps: pure NR is 4, A1 hybrid is 9, with no fallback but too many middle steps.
- `case_ACTIVSg25k` is roughly break-even; `case_SyntheticUSA` is slightly slower because fallback remains and NR grows to 9.
