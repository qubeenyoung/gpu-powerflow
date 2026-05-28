# Dataset Overview

Datasets are downloaded during the Docker build and stored in the image under:

```text
/datasets
```

The repository does not keep downloaded matrix or power-grid data in the source
tree. It keeps documentation, Docker build scripts, and dataset-preparation
source code under `prepare_datasets/`.

## Build-Time Generation Pipeline

The Docker build creates datasets in this order:

1. Build and install the dataset companion tool: `prepare_dataset_vectors`.
2. Install power-system Python packages and copy only MATPOWER `data/*.m` case
   files into `/datasets/power_system/matpower`.
3. Generate MATPOWER Newton-Raphson sparse systems under
   `/datasets/power_system/nr_linear_systems`.
4. Download and extract the configured SuiteSparse Matrix Collection archives
   under `/datasets/benchmark_matrices`.
5. Generate common companion files, `rhs.mtx` and `x_true.mtx`, for both
   SuiteSparse and MATPOWER systems.

The final benchmark input contract is always:

```text
A * x = rhs
```

| Dataset family | Matrix | `rhs.mtx` | `x_true.mtx` |
|---|---|---|---|
| SuiteSparse Matrix Collection | Extracted SuiteSparse matrix | `A * x_true` | Fixed-seed random vector from `U(-1, 1)` |
| MATPOWER Newton-Raphson | `J.mtx` | Copy of `F.mtx` | SuiteSparse KLU solution of `J * x_true = F` |

The companion-file manifest is written to
`/datasets/LINEAR_SYSTEM_COMPANIONS.txt`.

## Image Paths

| Dataset type | Image path |
|---|---|
| All project datasets | `/datasets` |
| Power-system datasets | `/datasets/power_system` |
| MATPOWER case files | `/datasets/power_system/matpower` |
| MATPOWER `.mat` conversions | `/datasets/power_system/matpower_mat` |
| Newton-Raphson linear systems | `/datasets/power_system/nr_linear_systems` |
| Benchmark matrices | `/datasets/benchmark_matrices` |
| Benchmark matrix downloads | `/datasets/benchmark_matrices/downloads` |
| Benchmark matrix extracts | `/datasets/benchmark_matrices/matrices` |
| Linear-system companion manifest | `/datasets/LINEAR_SYSTEM_COMPANIONS.txt` |

## Details

- [Benchmark matrices](benchmark-matrices.md) describes selected SuiteSparse
  Matrix Collection downloads and matrix properties.
- [Power-system datasets](power-system.md) describes MATPOWER case extraction,
  Python power-system package installation, and Newton-Raphson linear-system
  generation.

Both SuiteSparse benchmark matrices and MATPOWER NR systems are normalized with
`rhs.mtx` and `x_true.mtx` companion files during the Docker build.

## Regeneration Inside A Container

The Docker build runs these scripts automatically. Inside an existing container,
the same sequence can be rerun manually:

```bash
/opt/docker-scripts/generate_power_nr_datasets.sh
/opt/docker-scripts/download_suitesparse_matrix.sh
/opt/docker-scripts/generate_linear_system_companions.sh
```

Use build-time environment variables such as `POWER_NR_CASES`,
`SUITESPARSE_MATRIX_URLS`, and `LINEAR_SYSTEM_RANDOM_SEED` to intentionally
change the generated dataset.
