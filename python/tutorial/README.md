# Python Tutorial Notebooks

This folder contains live, story-style notebooks for first-time readers of
power flow, MATPOWER/pandapower baselines, and cuPF acceleration paths.

The notebooks run small reproducible benchmarks themselves. Fresh outputs are
written under `python/tutorial/_runs/`, which is ignored by git.

## Order

1. `01_power_system_basics_case9.ipynb` - power-system vocabulary, the power
   flow problem, `case9` topology, bus types, voltage phasors, and Ybus.
2. `02_matpower_pandapower_baseline.ipynb` - live `case6468rte` pandapower
   PYPOWER-derived and MATLAB/MATPOWER baseline runs, plus Python stage
   bottleneck analysis.
3. `03_cupf_cpu_path.ipynb` - live cuPF CPU build/run and comparison of
   UMFPACK, KLU, pandapower-like Jacobian, and native fixed-pattern Jacobian.
4. `04_cupf_gpu_path.ipynb` - live cuPF GPU build/run, cuDSS, Edge,
   EdgeAtomic, VertexWarp, and optional custom solver notes.
5. `05_multibatch_python_torch_interface.ipynb` - pybind `solve_batch`,
   `NewtonOptions`, and Torch autograd usage.
6. `06_bottlenecks_and_research.ipynb` - remaining bottlenecks and future work:
   cuGraph, custom linear solver, multi-GPU, tensor core, and mixed precision.

## Prerequisites

```bash
python3 -m pip install -r requirements.txt
test -f /datasets/matpower/raw/case9.m
test -f /datasets/matpower/raw/case6468rte.m
```

MATLAB/MATPOWER, CUDA, cuDSS, and Torch sections are live but optional. If a
component is unavailable, the relevant notebook cell records a skip/error table
instead of hiding the condition.

## Execute

```bash
jupyter nbconvert --execute python/tutorial/01_power_system_basics_case9.ipynb --to notebook --inplace
jupyter nbconvert --execute python/tutorial/02_matpower_pandapower_baseline.ipynb --to notebook --inplace
jupyter nbconvert --execute python/tutorial/03_cupf_cpu_path.ipynb --to notebook --inplace
jupyter nbconvert --execute python/tutorial/04_cupf_gpu_path.ipynb --to notebook --inplace
jupyter nbconvert --execute python/tutorial/05_multibatch_python_torch_interface.ipynb --to notebook --inplace
jupyter nbconvert --execute python/tutorial/06_bottlenecks_and_research.ipynb --to notebook --inplace
```

Default live benchmark settings are intentionally short: `case6468rte`,
`warmup=0`, and `repeats=1`. Increase the notebook parameters when you want a
more stable timing report.
