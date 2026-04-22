# cuSPARSE Ibus Experiment Run Status

Date: 2026-04-22 UTC

Build status: success

Run status: success

GPU:

```text
NVIDIA GeForce RTX 3090, sm_86
```

Default command:

```bash
./exp/20260422/cusparse_ibus/run_default.sh
```

Outputs:

```text
exp/20260422/cusparse_ibus/results/ibus_cusparse_spmm.csv
exp/20260422/cusparse_ibus/results/SUMMARY.md
```

The first runtime attempt failed while the CUDA device was not visible in the
container. After GPU visibility recovered, the full run completed successfully.
