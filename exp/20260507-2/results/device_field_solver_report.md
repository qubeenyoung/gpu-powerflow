# Device-Resident A0/A1 Field Solver Report

## Scope

- Cases: `case2383wp`, `case3120sp`, `case9241pegase`, `case13659pegase`, `case6468rte`
- Modes: `pure_cudss`, `bj_only`, `bj_a0_device`, `bj_a1_device`
- Block sizes for BJ/A0/A1: `8`, `16`
- Policy: no bootstrap cuDSS, no polish cuDSS, no fallback cuDSS in the primary BJ/A0/A1 paths.
- A0/A1 now keep full residual, P/Q RHS extraction, J11/J22 value update, A1 cross residual, and dx accumulation on device.

## Correction Timing

Compared with the previous host-glue implementation, the field-correction glue overhead is much smaller.

| mode | previous wall ms | previous non-cuDSS overhead ms | device wall ms mean | device non-cuDSS overhead ms mean |
|---|---:|---:|---:|---:|
| A0 | 1.596 | 1.440 | 0.685 | 0.078 |
| A1 | 2.217 | 1.969 | 0.984 | 0.156 |

The remaining wall time is mostly not full-vector D2H/H2D plumbing anymore. It is now the actual device-side work: full residual SpMV, J11/J22 value scatter, J11/J22 factor/solve, and A1 J12/J21 SpMV.

| case | A0 wall ms mean | A0 overhead ms mean | A1 wall ms mean | A1 overhead ms mean |
|---|---:|---:|---:|---:|
| case2383wp | 0.456 | 0.072 | 0.683 | 0.147 |
| case3120sp | 0.526 | 0.074 | 0.780 | 0.149 |
| case9241pegase | 0.781 | 0.081 | 1.111 | 0.162 |
| case13659pegase | 1.034 | 0.085 | 1.407 | 0.167 |
| case6468rte | 0.752 | 0.079 | 1.079 | 0.159 |

## NR Outcomes

Pure cuDSS converged on all five cases in 3 to 6 NR iterations.

| mode | block | converged cases | notes |
|---|---:|---:|---|
| BJ only | 8 | 0 / 5 | no full-J fallback; several cases diverged or hit max iterations |
| BJ only | 16 | 0 / 5 | no full-J fallback; several cases diverged |
| A0 device | 8 | 0 / 5 | case2383wp and case6468rte got close but did not meet `1e-8` in 20 iterations |
| A0 device | 16 | 0 / 5 | case2383wp got close; hard cases still unstable |
| A1 device | 8 | 1 / 5 | converged only `case6468rte`; total speedup vs pure cuDSS was 1.185x |
| A1 device | 16 | 2 / 5 | converged `case2383wp` and `case6468rte`; speedups were 0.695x and 1.073x |

Best converged A0/A1 rows:

| case | mode | block | NR iters | pure NR iters | total ms | pure ms | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| case2383wp | A1 device | 16 | 16 | 6 | 31.935 | 22.183 | 0.695 |
| case6468rte | A1 device | 8 | 9 | 3 | 28.904 | 34.257 | 1.185 |
| case6468rte | A1 device | 16 | 10 | 3 | 31.923 | 34.257 | 1.073 |

## Answers

1. Did device-resident implementation remove the 80-90% non-cuDSS glue overhead?
   Yes. A0 non-cuDSS overhead dropped from about `1.440 ms` to `0.078 ms`; A1 dropped from about `1.969 ms` to `0.156 ms`.

2. Is A0 wall time below 0.5 ms on most cases?
   No. A0 averaged `0.685 ms`; only `case2383wp` was below 0.5 ms on average.

3. Is A1 wall time below 0.8 ms on most cases?
   No. A1 averaged `0.984 ms`; `case2383wp` and `case3120sp` were near or below 0.8 ms, but larger cases were above it.

4. Does A0 or A1 converge without full J cuDSS fallback?
   A0 did not converge on any selected case within 20 NR iterations. A1 converged on `case6468rte` for block 8/16 and on `case2383wp` for block 16.

5. Are NR iterations close to pure cuDSS?
   No. Pure cuDSS needed 3 to 6 iterations. The converged A1 runs needed 9, 10, and 16 iterations.

6. Is total linear solve time lower than pure cuDSS?
   Only narrowly for `case6468rte` A1 block 8. `case2383wp` A1 block 16 converged but used more linear time than pure cuDSS.

7. Which mode should be kept?
   Keep A1 only as a diagnostic or case-specific experimental path. Do not keep A0/A1 as a general replacement for full J cuDSS. The device rewrite fixed the plumbing bottleneck, but the field-corrected BJ step is still not robust enough across cases.

## Output Files

- `results/device_field_solver_summary.csv`
- `results/device_field_solver_iters.csv`
- `results/device_field_solver_timing.csv`
- `results/device_field_solver_report.md`
