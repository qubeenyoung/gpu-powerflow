# Block ILU(0) Numeric Pilot Report

This is a first numeric pilot, not a production GPU implementation. The block ILU(0) factorization and apply path used here is CPU dense FP32, then copied the middle correction back to the device for the NR update. Timing is therefore a quality-gate diagnostic, not a final performance result.

## Standalone J1/F1 Gate

| block size | gate pass cases | verdict |
|---:|---:|---|
| 16 | 5/5 | pass |
| 32 | 3/5 | pass |

Gate rule: block ILU(0) must improve dx norm ratio, dx cosine, and true linear relative residual versus block-Jacobi on at least 3 of 5 cases. bs16 passed 5/5; bs32 passed 3/5.

## Hybrid Fallback Result

| case | bs | BJ fallback | ILU0 fallback | BJ NR | ILU0 NR | BJ time ms | ILU0 time ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 16 | 4 | 2 | 8 | 12 | 30.3 | 650.5 |
| case3120sp | 16 | 4 | 3 | 8 | 10 | 32.3 | 706.1 |
| case9241pegase | 16 | 3 | 2 | 7 | 7 | 62.9 | 1558.8 |
| case13659pegase | 16 | 3 | 3 | 8 | 11 | 81.7 | 4111.9 |
| case6468rte | 16 | 1 | 1 | 4 | 3 | 38.8 | 458.4 |
| case2383wp | 32 | 4 | 2 | 7 | 12 | 28.6 | 724.5 |
| case3120sp | 32 | 4 | 2 | 10 | 17 | 34.8 | 1462.7 |
| case9241pegase | 32 | 2 | 1 | 11 | 9 | 71.5 | 2491.9 |
| case13659pegase | 32 | 3 | 3 | 7 | 9 | 78.0 | 3605.2 |
| case6468rte | 32 | 1 | 1 | 4 | 3 | 38.8 | 496.6 |

- bs16: fallback decreased on 3/5 cases and was unchanged on 2/5 cases.
- bs32: fallback decreased on 3/5 cases and was unchanged on 2/5 cases.

Fallback-first criterion is partially satisfied: both candidates reduce fallback on 3 of 5 cases. However, NR iteration count often increases, so the better linear correction does not automatically translate into fewer Newton iterations.

## Timing

| block size | preconditioner | avg setup ms | avg solve ms | avg middle total ms | avg ILU factor ms | avg ILU apply ms |
|---:|---|---:|---:|---:|---:|---:|
| 16 | metis_block_jacobi | 1.548 | 0.710 | 0.311 | 0.000 | 0.000 |
| 16 | block_ilu0 | 26.843 | 4.045 | 30.563 | 23.406 | 3.338 |
| 32 | metis_block_jacobi | 1.436 | 0.677 | 0.309 | 0.000 | 0.000 |
| 32 | block_ilu0 | 80.788 | 6.490 | 87.058 | 77.119 | 5.852 |

The CPU pilot is far slower than the GPU block-Jacobi middle solver. This does not reject block ILU(0) as a preconditioner quality idea, but it does mean a GPU level-scheduled implementation is required before any speed claim.

## Candidate Judgment

- Standalone dx quality: bs16 is better and more consistent; bs32 is mixed but still passes the 3/5 gate.
- Hybrid fallback: bs16 and bs32 both reduce fallback in 3/5 cases. bs32 gives stronger fallback reduction on case3120sp and case9241pegase, but increases NR iterations more severely on case3120sp.
- Best pilot candidate to optimize next: bs16 block_coloring, because it passed standalone on all cases and has lower factor/apply cost. bs32 remains a secondary candidate if the only target is fallback reduction.
- Numeric block ILU(0) is worth a GPU pilot only as a fallback-reduction candidate. It is not yet a speed candidate in this CPU pilot form.

## Files

- `results/block_ilu0_standalone_quality.csv`: J1/F1 standalone quality gate.
- `results/block_ilu0_hybrid_summary.csv`: combined hybrid summaries for BJ/ILU0 and bs16/bs32.
- `results/block_ilu0_hybrid_iters.csv`: combined per-iteration hybrid logs.
- `results/block_ilu0_timing.csv`: combined hybrid timing logs.
- `results/block_ilu0_shadow_dx.csv`: standalone J1 dx-quality projection; hybrid shadow-dx was not run for the CPU pilot path.
