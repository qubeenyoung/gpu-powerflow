# Python Tutorial Notebooks

This directory is a narrative tutorial, not just a benchmark launcher. The
notebooks move from the power-flow equation to Newton-Raphson internals, then
to baseline implementations and cuPF acceleration choices.

Fresh benchmark outputs are written under `python/tutorial/_runs/`, which is
ignored by git.

## Reading Order

1. `01_power_system_basics_case9.ipynb`
   Turns the physical grid into `Ybus`, `Sbus`, `V`, and the nonlinear equation
   `S_calc(V) - S_spec = 0`.
2. `02_newton_raphson_and_jacobian.ipynb`
   Opens one Newton step on `case9`: mismatch, Jacobian blocks, sparse solve,
   and voltage update.
3. `03_matpower_pandapower_baseline.ipynb`
   Runs pandapower's PYPOWER-derived path and MATLAB/MATPOWER on `case6468rte`,
   with MATLAB environment checks.
4. `04_cupf_cpu_acceleration.ipynb`
   Separates UMFPACK, KLU, and fixed-pattern Jacobian effects in the cuPF CPU
   path.
5. `05_cupf_gpu_acceleration.ipynb`
   Connects cuDSS, custom solver, and Edge/EdgeAtomic/VertexWarp Jacobian fill
   choices to the same Newton bottlenecks.
6. `06_batch_torch_and_research_direction.ipynb`
   Shifts from one solve to batched/differentiable solves and closes with the
   remaining research directions.

## Prerequisites

```bash
python3 -m pip install -r requirements.txt
test -f /datasets/matpower/raw/case9.m
test -f /datasets/matpower/raw/case6468rte.m
```

MATLAB settings live in the repository root `.env` file when needed:

```bash
MATLAB_BIN=matlab
MATPOWER_HOME=/opt/matpower
MATLAB_LICMODE=onlinelicensing
MATLAB_USER_ID=your-account@example.edu
MATLAB_PASSWORD=your-password
# or: MATLAB_LICENSE_FILE=/path/to/license.lic
```

The notebooks report whether these keys are set, but never print secret values.

## Execute

```bash
jupyter nbconvert --execute python/tutorial/01_power_system_basics_case9.ipynb --to notebook --inplace
jupyter nbconvert --execute python/tutorial/02_newton_raphson_and_jacobian.ipynb --to notebook --inplace
jupyter nbconvert --execute python/tutorial/03_matpower_pandapower_baseline.ipynb --to notebook --inplace --ExecutePreprocessor.timeout=9000
jupyter nbconvert --execute python/tutorial/04_cupf_cpu_acceleration.ipynb --to notebook --inplace --ExecutePreprocessor.timeout=9000
jupyter nbconvert --execute python/tutorial/05_cupf_gpu_acceleration.ipynb --to notebook --inplace --ExecutePreprocessor.timeout=12000
jupyter nbconvert --execute python/tutorial/06_batch_torch_and_research_direction.ipynb --to notebook --inplace --ExecutePreprocessor.timeout=12000
```

The benchmark notebooks default to `case6468rte`, `warmup=0`, and `repeats=1`.
For report-quality timing, increase the notebook `REPEATS` value to 5 or more.
