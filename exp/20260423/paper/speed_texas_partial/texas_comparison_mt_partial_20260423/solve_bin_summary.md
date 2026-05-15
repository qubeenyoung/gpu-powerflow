# Solve-Only Bin Summary

- PARTIAL: GPU cuDSS MT auto completed only batch 1 and 4 before CUDA/NVML became unavailable. Best multibatch currently chooses only batch 4.

| case size | n used / total | CPU ref ms | CPU cuPF ms | GPU cuPF b1 ms | GPU cuPF best multibatch ms | best batch counts | speedup vs CPU ref | speedup vs CPU cuPF |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| <1k | 3/3 | 4.5446 | 1.1942 | 0.72363 | 2.7478 | b4:3 | 1.654 | 0.4346 |
| 1k-9,999 | 4/4 | 66.969 | 22.909 | 3.1193 | 6.226 | b4:4 | 10.76 | 3.68 |
| 10k-49,999 | 3/3 | 301.43 | 86.959 | 7.1484 | 14.762 | b4:3 | 20.42 | 5.891 |
| >=50k | 2/2 | 2228.5 | 641.68 | 28.367 | 50.217 | b4:2 | 44.38 | 12.78 |
