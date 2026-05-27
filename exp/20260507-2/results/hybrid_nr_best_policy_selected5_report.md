# Hybrid NR Best Policy Selected 5

Setting: `polish=1e-4`, `block_size=64`, `restart=16`, fixed `gmres_iters=8`, `accept=0.99`, `reject=1.10`, fallback `immediate`, warmup 1.

| case | converged | pure cuDSS iters | hybrid NR iters | cuDSS calls | GMRES calls | accepted | rejected | fallback | polish | hybrid s | pure cuDSS s | speedup | final inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | true | 6 | 35 | 5 | 33 | 30 | 3 | 3 | 1 | 1.040184e-01 | 2.211722e-02 | 0.213 | 5.453e-12 |
| case3120sp | false | 6 | 50 | 2 | 49 | 48 | 1 | 1 | 0 | 1.323422e-01 | 2.249406e-02 | 0.170 | 4.633e-01 |
| case9241pegase | true | 6 | 47 | 3 | 45 | 44 | 1 | 1 | 1 | 1.645417e-01 | 4.893690e-02 | 0.297 | 1.650e-09 |
| case13659pegase | true | 5 | 17 | 5 | 15 | 12 | 3 | 3 | 1 | 1.014115e-01 | 5.916913e-02 | 0.583 | 6.808e-12 |
| case6468rte | true | 3 | 3 | 2 | 1 | 1 | 0 | 0 | 1 | 3.356949e-02 | 3.254035e-02 | 0.969 | 7.637e-10 |

## Notes

- The setting is the previous aggregate/accepted-GMRES best policy, but it is too permissive for the longer 5-6 iteration cases.
- `case3120sp` failed by `max_nr_iters=50`; GMRES kept making small accepted mismatch decreases but did not reach the polish region.
- `case2383wp` and `case9241pegase` converged, but only after many GMRES middle steps, so they are much slower than pure cuDSS.
- `case6468rte` still behaves like the earlier run: one useful GMRES step, then cuDSS polish, roughly break-even but slightly slower here.
