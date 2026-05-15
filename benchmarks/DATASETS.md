# Benchmark Datasets

Document date: 2026-04-10.

The benchmark runner consumes cuPF dump directories with this layout:

```text
case_name/
  dump_Ybus.mtx
  dump_Sbus.txt
  dump_V.txt
  dump_pv.txt
  dump_pq.txt
  metadata.json        # optional
```

The default dataset root is the external workspace dump set:

```text
/workspace/datasets/cuPF_benchmark_dumps
```

`/workspace/datasets/cuPF_datasets` is also present in the workspace, but the
benchmark harness expects the `dump_*.txt` and `dump_Ybus.mtx` layout shown
above.

No benchmark datasets are stored under the source tree. If you want reusable
case groups, create a local text file and pass it with `--case-list`; each line
should be a case directory name under `--dataset-root`.

## Recommended Progression

1. Start with `--cases case30_ieee case118_ieee` and `pypower cpp_naive cpp`.
2. Add `cuda_edge` after CUDA runtime correctness is stable.
3. Use the default `--mode both --warmup 1 --repeats 10` for reportable runs.
4. Add `cuda_fp64_edge` when FP64 cuDSS performance is under investigation.
5. Use large cases such as `case4601_goc`, `case8387_pegase`, and `case9241_pegase` only for dedicated timing or Nsight runs.

## Notes

- Benchmark data is stored outside the source tree to avoid committing MatrixMarket dumps.
- `cpp_naive` maps to the benchmark-only `cpp_pypowerlike` reference baseline. It is intentionally benchmark-only and should not be reintroduced into the core `PlanBuilder`.
- CUDA profiles use the same public FP64 inputs and outputs. Mixed keeps mismatch in FP64 and casts to FP32 only at the cuDSS32 RHS boundary.
