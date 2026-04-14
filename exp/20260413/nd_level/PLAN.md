# CUDSS_CONFIG_ND_NLEVELS Follow-up Sweep

## Goal

Extend the 2026-04-12 ND_NLEVELS sweep with lower values because `nd_8`
was the best value in that run.

This follow-up keeps the previous benchmark setup fixed:

- profile: `cuda_edge` (`cuda_mixed_edge`)
- reordering: `CUDSS_ALG_DEFAULT`
- MT mode: disabled
- modes: `end2end`, `operators`
- warmup/repeats: `1 / 10`
- cases: see `cases.txt`

## Sweep Matrix

| label | `CUPF_CUDSS_ND_NLEVELS` |
|---|---:|
| `nd_5` | `5` |
| `nd_6` | `6` |
| `nd_7` | `7` |

The comparison report will include the previous `/workspace/exp/20260412/nd_level`
results for `auto`, `8`, `9`, `10`, and `11` when available.
