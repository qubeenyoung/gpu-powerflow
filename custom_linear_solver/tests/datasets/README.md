# datasets

Input matrices for the benchmark runners. Two groups:

| dir | what | git | how to (re)create |
|---|---|---|---|
| `power/` | Newton–Raphson power-flow Jacobians — the primary target domain | **tracked** (small) | `prepare_datasets/convert_linear_system.py` (see below) |
| `suitesparse/` | out-of-domain SuiteSparse matrices (circuit, 2D/3D FEM) | **git-ignored** (large) | `./fetch_suitesparse.sh` (see below) |

Each case is a directory holding two MatrixMarket files:
`J.mtx` (`coordinate real general`, the matrix) and `F.mtx` (`array real general`,
dense `n×1` RHS). The custom runner also accepts `--matrix J.mtx --rhs F.mtx`.

## power/ — power-grid Jacobians

These are committed (case118 … case_SyntheticUSA). To regenerate or add cases,
build them from MATPOWER `.m` cases with the prepare CLI (pure pandapower + scipy,
no MATLAB/Julia) — see [`prepare_datasets/`](../../../prepare_datasets/):

```sh
# from the repo root
python3 -m prepare_datasets.convert_linear_system \
    --dataset-root <MATPOWER .m root> \
    --output-root  custom_linear_solver/tests/datasets/power \
    --cases case118 case1354pegase case_ACTIVSg25k case_SyntheticUSA
```

Each `--cases` entry writes `power/<case>/{J.mtx,F.mtx,metadata.json}`.

## suitesparse/ — out-of-domain stress matrices

Large, so kept out of git and fetched on demand from the SuiteSparse Matrix
Collection (`https://sparse.tamu.edu/MM/<group>/<name>.tar.gz`):

```sh
./fetch_suitesparse.sh                 # default set (cant parabolic_fem G3_circuit bmwcra_1)
./fetch_suitesparse.sh Serena G3_circuit   # specific matrices by name
./fetch_suitesparse.sh --all           # every matrix in the script's name→group table
```

Matrices land at `suitesparse/<name>/<name>.mtx`. Add new names to the `GROUP`
table in `fetch_suitesparse.sh`.
