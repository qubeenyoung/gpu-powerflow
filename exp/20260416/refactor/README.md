# Refactorization Reorder Check

Operator-timing check for standard `cuda_edge` Newton runs on the Texas
University cuPF dump dataset.

Defaults:

- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`
- Cases: `cases.txt`
- Profile: `cuda_edge` (`cuda_mixed_edge`, standard algorithm)
- Measurement mode: `operators`
- cuDSS reordering algorithms: `DEFAULT`, `ALG_1`, `ALG_2`, `ALG_3`
- cuDSS MT: enabled, host threads `AUTO`
- CUDA visibility: physical GPU `0`
- Warmup/repeats: `1 / 10`

`DEFAULT` reuses the existing run at:

```text
/workspace/exp/20260416/modified/results/texas_gpu3_mt_auto_r10
```

Run:

```bash
python3 /workspace/exp/20260416/refactor/run_refactor_reorder_benchmark.py
python3 /workspace/exp/20260416/refactor/summarize_refactor_reorder.py
```

Results are written to:

```text
/workspace/exp/20260416/refactor/results/<run-name>/
```

Top-level comparison files are written directly under:

```text
/workspace/exp/20260416/refactor/
```
