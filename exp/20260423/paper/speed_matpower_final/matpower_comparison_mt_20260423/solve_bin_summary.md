# Solve-Only Bin Summary

- GPU cuPF uses cuDSS MT auto. Best multibatch chooses among batch 4/16/64/256.

| case size | n used / total | CPU ref ms | CPU cuPF ms | GPU cuPF b1 ms | GPU cuPF best multibatch ms | best batch counts | speedup vs CPU ref | speedup vs CPU cuPF |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| <100 | 40/41 | 0.46788 | 0.058447 | 0.48574 | 0.043429 | b256:40 | 10.77 | 1.346 |
| 100-999 | 10/10 | 2.6148 | 0.4753 | 0.69964 | 0.098921 | b256:10 | 26.43 | 4.805 |
| 1k-9,999 | 22/22 | 38.91 | 10.434 | 2.2222 | 0.85017 | b256:22 | 45.77 | 12.27 |
| 10k-49,999 | 3/3 | 231.43 | 56.647 | 6.632 | 3.8255 | b256:3 | 60.5 | 14.81 |
| >=50k | 2/2 | 2081.9 | 596.94 | 29.18 | 21.648 | b256:2 | 96.17 | 27.57 |
