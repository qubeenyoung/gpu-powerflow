# cuDSS Linear Solve Benchmark

This experiment is isolated from `cuPF/tests` and benchmarks a single sparse
linear system solve with cuDSS.

## Purpose

The main goal of this experiment set is to evaluate `cuDSS` as the linear
solver underneath a power-flow Newton step, and then use that baseline for
future Schur complement experiments.

The comparison target is not full end-to-end power flow. Instead, the
experiment fixes one Newton linear system from a shared power-flow case and
measures the sparse linear algebra path itself.

This separation is intentional:

- it keeps the input cases shared with `cuPF`
- it removes unrelated costs such as repeated case preprocessing
- it gives a clean baseline for comparing full `J x = b` solve with cuDSS
- it gives a clean baseline for comparing a future Schur complement workflow on
  the same `J` and `b`

In other words, this directory is meant to answer:

- how expensive is the direct cuDSS solve on a power-flow Jacobian?
- when Schur complement is introduced later, does it actually help once the
  reduced-system solve cost is included?

## Inputs

Shared cases live in `/workspace/datasets/cuPF_datasets`.

Each case directory contains:

- `dump_Ybus.mtx`
- `dump_Sbus.txt`
- `dump_V.txt`
- `dump_pv.txt`
- `dump_pq.txt`

The benchmark builds one Newton-step linear system from the initial state `V0`:

- `J = dF/dx` evaluated at `V0`
- `b = -F(V0)`

After that it runs cuDSS on the fixed `J x = b` system.

## Build

```bash
cmake -S /workspace/exp/20260409 -B /workspace/exp/20260409/build -GNinja
cmake --build /workspace/exp/20260409/build --target cudss_benchmark
```

## Run

List available cases:

```bash
/workspace/exp/20260409/build/cudss_benchmark --list-cases
```

Run one case:

```bash
/workspace/exp/20260409/build/cudss_benchmark \
  --case case118_ieee \
  --warmup 1 \
  --repeats 5 \
  --output-json /workspace/exp/20260409/results/case118_ieee.json
```

What is timed:

- one-time linear-system build on CPU
- per-run cuDSS `ANALYSIS`
- per-run cuDSS `FACTORIZATION`
- per-run cuDSS `SOLVE`

You can also point directly at a specific dataset directory with `--case-dir`.
