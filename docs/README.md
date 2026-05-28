# Documentation

This directory is organized as topic-based project documentation. The root
`README.md` stays short and focuses on quick start commands; details live here.

## Guide

| Document | Purpose |
|---|---|
| [**Results and methodology**](results-and-methodology.md) | What we built, the per-metric comparison vs cuDSS (warm + cold), key techniques (partitioned-inverse, adaptive cap, plan reuse, onetone2 fix), and how to reproduce |
| [Build and run](build-and-run.md) | Docker build arguments, runtime commands, and smoke checks |
| [Dataset overview](data/README.md) | Dataset roots and build-time dataset policy |
| [Benchmark matrices](data/benchmark-matrices.md) | Selected SuiteSparse Matrix Collection downloads and matrix properties |
| [Power-system datasets](data/power-system.md) | MATPOWER case-file policy, Python packages, and Newton-Raphson linear-system generation |
| [Benchmarking](benchmarking.md) | Standard linear-system input contract and error metrics |
| [Solver stack reference](reference/solver-stack.md) | Solver versions, source trees, build trees, and install prefixes |
| [Toolchain and dependencies](reference/dependencies.md) | Non-solver tools, CUDA/profiling tools, CLIs, and build script policy |

## Layout Convention

| Directory | Contents |
|---|---|
| `data/` | Dataset policy, paths, and matrix/case selection notes |
| `reference/` | Stable reference material: versions, paths, build configuration |

File names use kebab-case so links are predictable and shell-friendly.
