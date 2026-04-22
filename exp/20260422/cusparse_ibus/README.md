# cuSPARSE Ibus Baseline Experiment

This experiment compares the current cuPF mixed-precision `Ibus` kernel against
split-real cuSPARSE `SpMM` baselines.

The existing solver stores voltages as batch-major arrays:

```text
V[batch * n_bus + bus]
```

cuSPARSE can read that memory directly as a column-major dense matrix with
shape `n_bus x batch_size` and leading dimension `n_bus`, so this experiment
does not include any voltage transpose cost.

Measured variants:

- `custom`: copy of the production warp-per-row mixed kernel
  (`Ybus FP32`, `V FP64`, `Ibus FP64`, `J_Ibus FP32`).
- `cusparse_fp64`: four real-valued `SpMM` calls with `Ybus FP64`
  converted from the FP32 matrix used by `custom`, `V FP64`, and `Ibus FP64`.
- `cusparse_fp32`: four real-valued `SpMM` calls with `Ybus FP32`, `V FP32`,
  and `Ibus FP32`, followed by a pack kernel to produce `Ibus FP64` and
  `J_Ibus FP32`.
- `cusparse_mixed_probe`: attempts the direct `Ybus FP32 * V FP64 -> Ibus FP64`
  cuSPARSE contract and records whether the local cuSPARSE runtime supports it.

Default output:

```text
exp/20260422/cusparse_ibus/results/ibus_cusparse_spmm.csv
```

Run the default experiment with:

```bash
./exp/20260422/cusparse_ibus/run_default.sh
```
