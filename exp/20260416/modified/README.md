# Modified Newton Solver Benchmark

Compare the standard Newton solver against the modified solver on the Texas
University dataset.

Defaults:

- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`
- Cases: `cases.txt`
- Profiles: `cuda_edge`, `cuda_edge_modified`
- CUDA visibility: physical GPU `3`
- cuDSS MT: enabled
- cuDSS host threads: `AUTO`
- Measurement modes: `end2end`, `operators`

Run:

```bash
python3 /workspace/exp/20260416/modified/run_modified_benchmark.py
```

Results are written to:

```text
/workspace/exp/20260416/modified/results/<run-name>/
```

Important outputs:

- `summary.csv`: one row per measured repeat.
- `aggregates.csv`: grouped means/stdevs.
- `SUMMARY.md`: readable end-to-end and operator timing tables.
- `manifest.json`: commands and environment snapshot.
