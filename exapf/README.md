# ExaPF runner

This folder contains a small Julia runner for ExaPF power-flow measurements.

## Setup

Julia 1.10 is installed at `/opt/julia-1.10.10` and available as `julia`.
The project environment is in this folder:

```bash
julia --project=exapf -e 'using Pkg; Pkg.instantiate()'
```

## Run

Default cases target the 3000, 6000, and 8000 bus ranges from
`/datasets/matpower/raw`:

- `case3120sp`
- `case6470rte`
- `case8387pegase`

`case8387pegase` contains a trailing MATLAB conditional block that ExaPF's
parser cannot read. The runner leaves the source file untouched and writes a
parser-friendly copy to `exapf/generated/` automatically.

```bash
julia --project=exapf exapf/run_powerflow.jl --repeats 5 --warmups 1
```

Results are written to `exapf/results/` as CSV. You can override cases with
case names or full `.m` paths:

```bash
julia --project=exapf exapf/run_powerflow.jl \
  --cases case3012wp,case6515rte,/datasets/matpower/raw/case8387pegase.m
```

CPU mode is also available:

```bash
julia --project=exapf exapf/run_powerflow.jl --backend cpu
```
