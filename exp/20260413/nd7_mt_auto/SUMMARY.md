# ND7 MT Auto Results

- profile: `cuda_edge`
- `CUDSS_CONFIG_ND_NLEVELS`: `7`
- cuDSS MT: enabled
- `CUDSS_CONFIG_HOST_NTHREADS`: `AUTO`
- `end2end`: clean timing, repeats 10, residual dump off
- `operator`: clean timing, repeats 10, residual dump off
- `dump`: residual dump only, repeats 1, do not use for timing comparison

## Timing By Case

| case | end2end ms | operator ms | end2end iters | operator iters |
|---|---:|---:|---:|---:|
| case30_ieee | 12.367 | 11.472 | 5.0 | 5.0 |
| case118_ieee | 11.615 | 11.828 | 5.0 | 5.0 |
| case793_goc | 15.098 | 16.207 | 5.7 | 5.8 |
| case1354_pegase | 19.753 | 16.268 | 6.0 | 6.0 |
| case2746wop_k | 19.143 | 22.307 | 5.0 | 5.0 |
| case4601_goc | 27.148 | 28.538 | 6.0 | 6.0 |
| case8387_pegase | 42.801 | 43.304 | 7.0 | 7.0 |
| case9241_pegase | 42.344 | 44.755 | 8.0 | 8.0 |

## Residual Dump Files

| case | files | first | last |
|---|---:|---|---|
| case30_ieee | 5 | residual_iter0.txt | residual_iter4.txt |
| case118_ieee | 5 | residual_iter0.txt | residual_iter4.txt |
| case793_goc | 6 | residual_iter0.txt | residual_iter5.txt |
| case1354_pegase | 6 | residual_iter0.txt | residual_iter5.txt |
| case2746wop_k | 5 | residual_iter0.txt | residual_iter4.txt |
| case4601_goc | 6 | residual_iter0.txt | residual_iter5.txt |
| case8387_pegase | 7 | residual_iter0.txt | residual_iter6.txt |
| case9241_pegase | 8 | residual_iter0.txt | residual_iter7.txt |

## Files

- `results/end2end/`: clean end-to-end timing
- `results/operator/`: clean operator timing
- `results/dump/`: one-repeat residual dump run
- `timing_by_case.csv`: case-level timing table
- `residual_dump_counts.csv`: residual file counts
