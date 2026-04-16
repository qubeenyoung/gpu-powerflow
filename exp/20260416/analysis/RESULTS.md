# cuDSS Reorder Dump Results

Date: 2026-04-16.

Dataset root:

```text
/workspace/datasets/matpower8.1/cuPF_datasets
```

Dump root:

```text
/workspace/exp/20260416/analysis/dumps
```

Selection policy: smallest MATPOWER case with `n_bus >= target_bus`.

The dumped matrix is the edge-based Newton Jacobian at `V0`, so the permutation
length is the Jacobian dimension, not the original bus count.

| target bus | case | n bus | n pv | n pq | Jacobian dim | Jacobian nnz | perm row len | perm col len | etree len | etree sum | nonzero etree nodes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | case118 | 118 | 53 | 64 | 181 | 1051 | 181 | 181 | 1023 | 181 | 46 |
| 500 | case_ACTIVSg500 | 500 | 55 | 444 | 943 | 6275 | 943 | 943 | 1023 | 943 | 193 |
| 1000 | case1197 | 1197 | 0 | 1196 | 2392 | 14344 | 2392 | 2392 | 1023 | 2392 | 518 |
| 5000 | case6468rte | 6468 | 291 | 6176 | 12643 | 87845 | 12643 | 12643 | 1023 | 12643 | 892 |

Files per case:

- `perm_reorder_row.txt`
- `perm_reorder_col.txt`
- `elimination_tree.txt`
- `metadata.json`

Top-level files:

- `manifest.json`
- `summary.csv`
