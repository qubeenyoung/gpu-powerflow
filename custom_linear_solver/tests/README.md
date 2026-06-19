# custom_linear_solver / tests

Head-to-head benchmark harness: the **custom** solver vs two baselines (**cuDSS**,
**STRUMPACK**) on the same matrices. Everything needed to build a runner, fetch a
dataset, and run a comparison lives under this directory.

```
tests/
├── runners/        # benchmark source (one self-contained file per solver)
│   ├── run_custom_solver.cu   custom solver (analyze→factorize→solve, B=1 & batch)
│   ├── io.{cpp,hpp}           MatrixMarket I/O for the custom runner
│   ├── cudss_bench.cpp        NVIDIA cuDSS (GPU sparse direct LU)
│   ├── cudss_bench2.cpp       cuDSS refactorization variant
│   └── strumpack_bench.cpp    STRUMPACK (GPU multifrontal)
├── scripts/          # one build script per solver
│   ├── build_custom.sh        → custom_linear_solver/build/custom_linear_solver_run
│   ├── build_cudss.sh         → scripts/cudss_bench
│   └── build_strumpack.sh     → scripts/strumpack_bench_{magma,nomagma}  (both GPU)
└── datasets/        # see datasets/README.md
    ├── power/                 power-grid Jacobians (tracked; the primary domain)
    ├── suitesparse/           SuiteSparse matrices (git-ignored, fetched on demand)
    └── fetch_suitesparse.sh   download SuiteSparse matrices
```

## Build

Each solver has its own script (the baselines link external GPU libraries with
machine-specific paths, so they are plain shell, not CMake). All paths are env
overridable — see the header comment of each script.

```sh
tests/scripts/build_custom.sh         # uses the project CMake
tests/scripts/build_cudss.sh          # needs cuDSS  (CUDSS_INC / CUDSS_LIB)
tests/scripts/build_strumpack.sh      # needs STRUMPACK ×2 + MAGMA (STRUMPACK_BUILD_{MAGMA,NOMAGMA}, MAGMA_PREFIX)
```

**STRUMPACK has two GPU variants** from one source:
- `strumpack_bench_magma` — STRUMPACK + MAGMA vbatched path (`TPL_ENABLE_MAGMA=ON` build).
- `strumpack_bench_nomagma` — STRUMPACK native CUDA path (MAGMA off, still GPU).

### External dependencies (outside the repo; not vendored)
| Solver | Needs | Default location (this machine) |
|---|---|---|
| custom | CUDA, METIS | toolchain |
| cuDSS | cuDSS 0.7 | `/usr/{include,lib/x86_64-linux-gnu}/libcudss/12` |
| STRUMPACK | STRUMPACK (CUDA) ×2 builds + MAGMA 2.8 | `/root/baselines/STRUMPACK/build{,_nomagma}`, `/opt/magma` |

## Datasets

- **`datasets/power/`** — Newton–Raphson power-flow Jacobians (`<case>/J.mtx` real
  diagonally-dominant `J`, `<case>/F.mtx` RHS). Small and **tracked in git**; the
  primary target domain (case118 … case_SyntheticUSA).
- **`datasets/suitesparse/`** — out-of-domain SuiteSparse matrices (circuit, 2D/3D
  FEM). Large, so **git-ignored**; fetch on demand:
  ```sh
  tests/datasets/fetch_suitesparse.sh                 # default set
  tests/datasets/fetch_suitesparse.sh Serena G3_circuit
  tests/datasets/fetch_suitesparse.sh --all
  ```

A case directory holds two MatrixMarket files: `J.mtx` (`coordinate real general`)
and `F.mtx` (`array real general`, dense `n×1`). The custom runner also accepts
`--matrix J.mtx --rhs F.mtx` directly.

## Run (head-to-head example)

```sh
CASE=datasets/power/case_ACTIVSg25k

# custom (FP64 reference, then TF32 batch)
../build/custom_linear_solver_run --matrix $CASE/J.mtx --rhs $CASE/F.mtx \
    --repeat 10 --warmup 3

# cuDSS
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcudss/12 \
    scripts/cudss_bench $CASE/J.mtx 10

# STRUMPACK (MAGMA path; depth-matched reordering)
LD_LIBRARY_PATH=/opt/magma/lib \
    scripts/strumpack_bench_magma $CASE/J.mtx 1 10 --sp_reordering_method metis
```

## Custom runner flags (`run_custom_solver`)

```
<case-dir>                 directory holding J.mtx + F.mtx (or use --matrix/--rhs)
--matrix PATH --rhs PATH   matrix + RHS as MatrixMarket files
--precision fp64|fp32|tf32 batched/factor precision (tf32 = TF32 Ozaki mma). Default fp64.
--single-precision fp64|fp32   single-system input dtype (the non-batch run)
--batch B                  also run a uniform-batch experiment with B systems
--batch-only               skip the single-system run
--repeat N / --warmup N    timed trials (median) / untimed warmup
--max-panel-width N        analyze: max supernode amalgamation width (1..64, default 8)
--serial-nd                deterministic serial METIS NodeND instead of parallel ND
--matching                 structural row/column matching before factorization
--pivot-strategy none|shift   static-diagonal-shift pivoting (default shift)
--pivot-epsilon X          shift threshold/magnitude
--dump-fronts PATH         write per-front CSV after analyze
--solution-out PATH        write recovered x as MatrixMarket
```

Output is `key=value` lines (stable; parsed by scripts): `n, nnz, analyze_ms,
factorize_ms, solve_ms, relative_residual_l2, setup_ms, batch_factor_per_sys_ms,
batch_solve_per_sys_ms, batch_relres`.

## Notes / gotchas

- **B=1 vs batch are different code paths.** The default (no `--batch`) run is the
  single-system path; `--batch B` runs the batched path. Check both
  `relative_residual_l2` (B=1) and `batch_relres` (batch).
- Accuracy (conditioning-dependent): fp64 ≈ 1e-13..1e-16; fp32 ≈ 1e-4; tf32 ≈ 1e-4
  on well-conditioned cases up to ≈ 1e-2 on large stiff grids — the TF32/Ozaki floor,
  not a bug.
- The cuDSS / STRUMPACK baselines link **external GPU libraries** whose paths are
  machine-specific; override them via the env vars documented in each build script.
