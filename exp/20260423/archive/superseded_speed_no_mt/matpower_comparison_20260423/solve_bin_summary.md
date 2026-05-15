# Solve-Only Bin Summary

- Binning: by `n_bus`.
- Time unit: mean solve milliseconds per case.
- `gpu cuPF` is batch 1.
- `gpu cuPF best multibatch` chooses the fastest per-case solve time among batches 4, 16, 64, and 256 for each case, then averages within the bin.
- `case16am` is excluded from averages because both CPU baselines did not converge at tolerance 1e-8.

| case size | n used / total | CPU ref | CPU cuPF | GPU cuPF b1 | GPU cuPF best multibatch | best batch counts | speedup vs CPU ref | speedup vs CPU cuPF |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| <100 | 40/41 | 0.46788 | 0.058447 | 0.48518 | 0.041513 | b256:40 | 11.27 | 1.408 |
| 100-999 | 10/10 | 2.6148 | 0.4753 | 0.70169 | 0.098228 | b256:10 | 26.62 | 4.839 |
| 1k-9,999 | 22/22 | 38.91 | 10.434 | 2.2286 | 0.86724 | b256:22 | 44.87 | 12.03 |
| 10k-49,999 | 3/3 | 231.43 | 56.647 | 6.6237 | 3.8335 | b256:3 | 60.37 | 14.78 |
| >=50k | 2/2 | 2081.9 | 596.94 | 29.601 | 22.452 | b256:2 | 92.72 | 26.59 |
