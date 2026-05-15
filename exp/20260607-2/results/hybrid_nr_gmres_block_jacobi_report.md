# Hybrid NR GMRES Block-Jacobi Smoke Results

This run uses the minimal cuPF NR port in `exp/20260607-2`, not the cuPF repository.

Options:

- cases: `case1197`, `case2736sp`, `case3375wp`, `case6468rte`, `case_ACTIVSg10k`
- policy: cuDSS bootstrap 1 iteration, GMRES middle, cuDSS polish at `mismatch_inf <= 1e-4`
- GMRES: block size 64, restart 16, fixed 8 iterations
- preconditioner: METIS block-Jacobi, FP32, `inverse_gemv`
- fallback: enabled, accept iterative step by mismatch decrease
- warmup: 1 unrecorded run before each measured pure/hybrid run

## Summary

| case | converged | NR iters | cuDSS calls | GMRES calls | fallbacks | hybrid total (s) | pure cuDSS total (s) | speedup | final mismatch inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case1197 | true | 3 | 3 | 0 | 0 | 1.406326e-02 | 1.425557e-02 | 1.013675 | 4.645335e-12 |
| case2736sp | true | 13 | 4 | 11 | 2 | 4.988669e-02 | 2.161284e-02 | 0.433239 | 5.921139e-12 |
| case3375wp | true | 2 | 2 | 0 | 0 | 2.094100e-02 | 2.107751e-02 | 1.006519 | 1.238498e-11 |
| case6468rte | true | 3 | 2 | 1 | 0 | 3.521726e-02 | 3.350106e-02 | 0.951268 | 7.636569e-10 |
| case_ACTIVSg10k | true | 8 | 4 | 6 | 2 | 6.224923e-02 | 4.717484e-02 | 0.757838 | 1.439317e-11 |

## Interpretation

- The hybrid policy does accept inaccurate GMRES corrections by NR mismatch decrease.
- `case6468rte` accepts one GMRES middle step and still converges with fewer cuDSS calls than pure cuDSS.
- `case2736sp` and `case_ACTIVSg10k` show that some GMRES steps are useful, but fallback/polish calls are still needed and total time is slower in this default setting.
- `case1197` and `case3375wp` reach polish territory immediately after bootstrap, so this policy intentionally does not use GMRES there.

Detailed per-iteration logs are in `hybrid_nr_gmres_block_jacobi_iters.csv`.
