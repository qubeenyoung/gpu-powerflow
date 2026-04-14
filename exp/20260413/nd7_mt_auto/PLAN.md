# ND7 MT Auto Run Plan

## Setup

- profile: `cuda_edge`
- reordering: `CUDSS_ALG_DEFAULT`
- `CUDSS_CONFIG_ND_NLEVELS`: `7`
- cuDSS MT mode: enabled
- `CUDSS_CONFIG_HOST_NTHREADS`: `AUTO`
- cases: see `cases.txt`

## Runs

| folder | mode | repeats | residual dump | purpose |
|---|---|---:|---|---|
| `end2end` | `end2end` | 10 | off | clean end-to-end timing |
| `operator` | `operators` | 10 | off | clean operator timing |
| `dump` | `operators` | 1 | on | residual vector dump only |

`dump` is not used for timing comparison. Residual file I/O would contaminate
both end2end elapsed time and `NR.iteration.mismatch` operator timing, so it is
kept as a separate one-repeat diagnostic run.

## Output Layout

```text
/workspace/exp/20260413/nd7_mt_auto/
  build/
    end2end/
    operator/
    dump/
  results/
    end2end/
    operator/
    dump/
```

Residual files are written under:

```text
/workspace/exp/20260413/nd7_mt_auto/results/dump/residuals/operators/cuda_edge/<case>/repeat_00/residual_iterY.txt
```
