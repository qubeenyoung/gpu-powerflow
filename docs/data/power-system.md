# Power-System Datasets

Power-system datasets are stored under:

```text
/datasets/power_system
```

MATPOWER-related Python tooling is installed through `matpower` and
`matpowercaseframes`. The official MATPOWER git repository is cloned only
during Docker build to extract case data from the `data` directory. The clone
is deleted before the build layer completes, so MATPOWER source code, examples,
and documentation are not kept in the image.

## MATPOWER Cases

| Item | Path |
|---|---|
| Dataset root | `/datasets/power_system/matpower` |
| Dataset manifest | `/datasets/power_system/matpower/MANIFEST.txt` |
| Copied files | `*.m` files from upstream MATPOWER `data/` |
| Removed after copy | MATPOWER source code, examples, tests, docs, git metadata |

The copied case files include common systems such as `case9.m`, `case14.m`,
`case30.m`, larger PEGASE cases, and ACTIVSg synthetic grid cases when present
in the selected MATPOWER ref.

## Build Arguments

| Argument | Default |
|---|---|
| `POWER_PYTHON_PACKAGES` | `pypower,pandapower,matpower,matpowercaseframes` |
| `MATPOWER_REF` | `master` |
| `POWER_NR_CASES` | `case30,case118,case1197,case_ACTIVSg2000,case3012wp,case6468rte,case8387pegase,case_ACTIVSg25k,case_SyntheticUSA` |
| `POWER_NR_DUMP_ITERATION` | `2` |

Example:

```bash
docker build \
  --build-arg MATPOWER_REF='8.0-release' \
  -t sparse-direct-solver:3090 .
```

## Runtime Check

Inside the container:

```bash
python3 -c 'import pypower, pandapower, matpower, matpowercaseframes'
find "$POWER_SYSTEM_DATASET_ROOT/matpower" -maxdepth 1 -name 'case*.m' | sort | head
```

## Newton-Raphson Linear Systems

The Docker build derives sparse direct-solver benchmark systems from MATPOWER
power-flow cases. The generation pipeline is:

1. Convert MATPOWER `.m` case files to PYPOWER-readable `.mat` files with
   `prepare_datasets/python/convert_m2mat.py`.
2. Load each `.mat` case with PYPOWER, build `Ybus` and `Sbus`, and reproduce
   the Newton-Raphson Jacobian assembly used by `pypower.newtonpf`.
3. Run Newton-Raphson updates until the requested dump iteration. The default
   dataset dumps the Jacobian and mismatch at iteration 2.
4. Write the base NR files in Matrix Market format:
   - `J.mtx`: sparse Jacobian, `matrix coordinate real general`
   - `F.mtx`: mismatch vector, `matrix array real general`
   - `metadata.json`: source case, dimensions, bus-type counts, and norm data
5. Run `prepare_dataset_vectors` through
   `scripts/docker/generate_linear_system_companions.sh` to write the common
   solver-benchmark companion files:
   - `rhs.mtx`: copy of `F.mtx`
   - `x_true.mtx`: solution of `J * x_true = rhs`, computed with SuiteSparse KLU

The standardized generated linear system is:

```text
J * x_true = rhs
rhs = F
```

The mismatch vector follows the PYPOWER ordering:

```text
[ real(mis[pv]); real(mis[pq]); imag(mis[pq]) ]
```

The unknown update vector follows the matching ordering:

```text
[ Va[pv]; Va[pq]; Vm[pq] ]
```

For the standardized benchmark companion files, the solved vector is named
`x_true.mtx` even though it corresponds to the NR update ordering above.

The default Docker build runs this pipeline automatically through
`scripts/docker/generate_power_nr_datasets.sh`. To regenerate the same dataset
inside an existing container, run the conversion and NR dump:

```bash
python3 prepare_datasets/python/convert_m2mat.py \
  --input-root /datasets/power_system/matpower \
  --output-root /datasets/power_system/matpower_mat \
  --cases case30 case118 case1197 case_ACTIVSg2000 case3012wp case6468rte case8387pegase case_ACTIVSg25k case_SyntheticUSA

python3 prepare_datasets/python/prepare_nr_linear_system.py \
  --mat-root /datasets/power_system/matpower_mat \
  --output-root /datasets/power_system/nr_linear_systems
```

Then generate the standardized `rhs.mtx` and `x_true.mtx` companions:

```bash
/opt/docker-scripts/generate_linear_system_companions.sh
```

Generated files are stored per case:

```text
/datasets/power_system/nr_linear_systems/<case>/
  J.mtx
  F.mtx
  rhs.mtx
  x_true.mtx
  metadata.json
```

## Selected NR Cases

The case set spans small IEEE systems, a radial 1000-bus-scale case, ACTIVSg
synthetic grids, PEGASE systems, RTE systems, and the 82,000-bus SyntheticUSA
case. This gives a broader sweep of Jacobian sizes for sparse direct-solver
experiments.

| Target scale | MATPOWER case | Buses | Jacobian size | Jacobian nnz | Ref | PV | PQ | Iteration-2 `||F||_inf` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30 | `case30` | 30 | 53 x 53 | 361 | 1 | 5 | 24 | 1.635e-02 |
| 118 | `case118` | 118 | 181 x 181 | 1,051 | 1 | 53 | 64 | 2.418e-02 |
| 1000s | `case1197` | 1,197 | 2,392 x 2,392 | 14,344 | 1 | 0 | 1,196 | 2.164e-05 |
| 2000s | `case_ACTIVSg2000` | 2,000 | 3,607 x 3,607 | 26,345 | 1 | 391 | 1,608 | 2.692e-01 |
| 3000s | `case3012wp` | 3,012 | 5,725 x 5,725 | 36,263 | 1 | 297 | 2,714 | 1.565e-03 |
| 6000s | `case6468rte` | 6,468 | 12,643 x 12,643 | 87,845 | 1 | 291 | 6,176 | 7.410e-04 |
| 8000s | `case8387pegase` | 8,387 | 14,908 x 14,908 | 110,572 | 1 | 1,864 | 6,522 | 4.039e-02 |
| 25K | `case_ACTIVSg25k` | 25,000 | recorded in `metadata.json` | recorded in `metadata.json` | recorded in `metadata.json` | recorded in `metadata.json` | recorded in `metadata.json` | recorded in `metadata.json` |
| 82K | `case_SyntheticUSA` | 82,000 | recorded in `metadata.json` | recorded in `metadata.json` | recorded in `metadata.json` | recorded in `metadata.json` | recorded in `metadata.json` | recorded in `metadata.json` |
