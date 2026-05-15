# GPU Sparse Linear Solver Benchmark for Power-Flow Jacobians

This experiment benchmarks sparse linear solver libraries on independently generated Newton-Raphson power-flow Jacobian systems. The goal is annual-report evidence for selecting the most suitable sparse linear solver library for cuPF, with cuDSS treated as the main GPU sparse direct-solver baseline.

All code, build files, logs, datasets, results, and reports for this experiment are contained under:

```bash
exp/20260510/lin_sol/
```

The experiment does not modify production cuPF source code. Repository files are read only for locating MATPOWER/PGLIB case data and optional reference helper patterns.

## Solver Candidates

The benchmark tracks these candidates:

- cuDSS
- cuSolverSP / cuSolverRF, depending on local CUDA Toolkit availability
- AMGx
- Ginkgo
- SuperLU_DIST
- STRUMPACK

SuperLU_DIST and STRUMPACK are treated as GPU-capable sparse direct solver candidates. If they cannot be built in this environment, their install scripts and build logs record the exact blocker and the report marks them `build_failed` or `unavailable`.

## Reproduction Commands

From the repository root:

```bash
python3 exp/20260510/lin_sol/scripts/inspect_environment.py

python3 exp/20260510/lin_sol/scripts/dump_matpower_systems.py \
    --cases case14,case118,case300,case1354pegase \
    --iterations 0,1,last \
    --include-synthetic-validation

cmake -S exp/20260510/lin_sol -B exp/20260510/lin_sol/build \
    -DCMAKE_BUILD_TYPE=Release
cmake --build exp/20260510/lin_sol/build -j

python3 exp/20260510/lin_sol/scripts/run_all.py \
    --warmup 3 \
    --repeats 10

python3 exp/20260510/lin_sol/scripts/make_report.py
```

For a quick smoke run:

```bash
python3 exp/20260510/lin_sol/scripts/dump_matpower_systems.py \
    --cases case14 \
    --iterations 0,1 \
    --include-synthetic-validation

python3 exp/20260510/lin_sol/scripts/run_all.py \
    --warmup 1 \
    --repeats 2 \
    --systems synthetic_validation,case14

python3 exp/20260510/lin_sol/scripts/make_report.py
```

## Outputs

- `datasets/dumped_systems/`: Matrix Market Jacobians, RHS vectors, CPU reference solutions, and metadata.
- `results/environment.json`: Hardware, CUDA, compiler, Python, and solver-library environment.
- `results/raw_json/`: One JSON result per solver/system/dtype.
- `results/summary_csv/summary.csv`: Flattened benchmark summary.
- `report/linear_solver_benchmark_report.md`: Final evidence report.

## Notes

The benchmark does not implement a custom cuSPARSE GMRES/BiCGSTAB solver. Custom cuSPARSE iterative methods belong to the separate cuITER experiment.

No result is fabricated. Missing libraries, failed builds, API limitations, and runtime failures are recorded explicitly in logs and result JSON.
