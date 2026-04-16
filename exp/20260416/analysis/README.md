# cuDSS Reorder Data Dump

This experiment dumps cuDSS reordering data for representative MATPOWER 8.1
cuPF inputs.

Default target bus sizes:

- 100
- 500
- 1000
- 5000

Case selection uses the smallest available `n_bus` greater than or equal to
the target. The dumped matrix is the edge-based Newton Jacobian at `V0`, matching
the cuDSS matrix used by the power-flow direct-solve benchmark path.

Outputs per case:

- `perm_reorder_row.txt`
- `perm_reorder_col.txt`
- `elimination_tree.txt`
- `metadata.json`

Build:

```bash
cmake -S /workspace/exp/20260416/analysis -B /workspace/exp/20260416/analysis/build -DCMAKE_BUILD_TYPE=Release
cmake --build /workspace/exp/20260416/analysis/build -j
```

Run:

```bash
python3 /workspace/exp/20260416/analysis/run_matpower_reorder_dump.py
```

Visualize:

```bash
python3 /workspace/exp/20260416/analysis/visualize_reorder_data.py
```

Figures are written to:

```text
/workspace/exp/20260416/analysis/figures
```
