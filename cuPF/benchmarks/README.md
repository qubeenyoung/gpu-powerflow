# cuPF Benchmarks

Document date: 2026-04-10.

This benchmark harness measures the current simplified cuPF solver profiles.
It separates clean end-to-end timing from operator-level timing.

## Measurement Modes

- `end2end`: clean timing path. The cuPF benchmark build disables logging,
  scoped timing collection, and NVTX. The runner still measures top-level
  elapsed/analyze/solve wall time around each run.
- `operators`: breakdown path. Logging stays disabled, but scoped timing and
  NVTX are enabled so `METRIC` rows include operator-level timing.

## Profiles

- `pypower`: Python PYPOWER baseline.
- `cpp_naive`: C++ PyPower-like Jacobian + SuperLU reference path.
- `cpp`: core CPU FP64 path with edge-based Jacobian maps and KLU.
- `cuda_edge`: CUDA Mixed profile with FP32 edge Jacobian and cuDSS32 solve.
- `cuda_vertex`: CUDA Mixed profile with FP32 vertex Jacobian and cuDSS32 solve.

The user-facing profile names above map to these internal cuPF profiles:

- `cpp_naive` -> `cpp_pypowerlike`
- `cpp` -> `cpu_fp64_edge`
- `cuda_edge` -> `cuda_mixed_edge`
- `cuda_vertex` -> `cuda_mixed_vertex`

Additional internal profiles remain available when needed:

- `cpp_pypowerlike`: reference CPU baseline using the PyPower-like Jacobian and SuperLU solve from `cpp/src/newton_solver/reference`.
- `cpu_fp64_edge`: core CPU FP64 path with edge-based Jacobian maps and KLU.
- `cuda_mixed_edge`: CUDA Mixed profile with FP32 edge Jacobian and cuDSS32 solve.
- `cuda_mixed_vertex`: CUDA Mixed profile with FP32 vertex Jacobian and cuDSS32 solve.
- `cuda_fp64_edge`: CUDA FP64 profile with FP64 edge Jacobian and cuDSS64 solve.
- `cuda_fp64_vertex`: CUDA FP64 profile with FP64 vertex Jacobian and cuDSS64 solve.

## Quick Start

The default dataset root is the external workspace dump set:

```text
/workspace/datasets/cuPF_benchmark_dumps
```

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --cases case30_ieee case118_ieee \
  --profiles pypower cpp_naive cpp cuda_edge cuda_vertex \
  --warmup 1 \
  --repeats 10
```

The default mode is `both`, which writes both clean end-to-end and operator
breakdown results. To run only one side:

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --mode end2end \
  --cases case30_ieee case118_ieee \
  --profiles pypower cpp_naive cpp cuda_edge cuda_vertex

python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --mode operators \
  --cases case30_ieee case118_ieee \
  --profiles pypower cpp_naive cpp cuda_edge cuda_vertex
```

For a broader CPU/CUDA sweep, list the cases explicitly or pass your own `--case-list` file:

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --cases case30_ieee case118_ieee case793_goc case1354_pegase case2746wop_k case4601_goc case8387_pegase case9241_pegase \
  --profiles pypower cpp_naive cpp cuda_edge cuda_vertex \
  --warmup 1 \
  --repeats 10
```

CUDA profiles automatically enable a CUDA benchmark build:

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --cases case30_ieee case118_ieee \
  --profiles cpp cuda_edge cuda_vertex \
  --warmup 1 \
  --repeats 10
```

CUDA Mixed batch runs can be measured with `--batch-size`. `B>1` is currently
supported by cuPF CUDA Mixed profiles; CPU/PYPOWER profiles remain single-case
benchmark paths.

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --cases case30_ieee case118_ieee \
  --profiles cuda_edge cuda_vertex \
  --batch-size 8 \
  --warmup 1 \
  --repeats 10
```

To compare cuDSS reordering algorithms, choose one of `DEFAULT`, `ALG_1`, or `ALG_2`.
The selected value is passed to CMake and recorded in `manifest.json`:

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --cases case30_ieee case118_ieee \
  --profiles cuda_edge cuda_vertex cuda_fp64_edge cuda_fp64_vertex \
  --cudss-reordering-alg ALG_1 \
  --warmup 1 \
  --repeats 10
```

To test cuDSS matching and pivot epsilon, use the matching flag plus an optional
matching algorithm and epsilon value. Matching is only supported with the default
reordering path:

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --cases case30_ieee case118_ieee \
  --profiles cuda_edge cuda_vertex \
  --cudss-use-matching \
  --cudss-matching-alg ALG_5 \
  --cudss-pivot-epsilon 1e-12 \
  --warmup 1 \
  --repeats 10
```

Results are written to `benchmarks/results/<run-name>/`:

- `manifest.json`: configuration, command log, and environment snapshot.
- `summary.csv`: one row per measured run across all measurement modes.
- `aggregates.csv`: grouped statistics by mode/case/profile.
- `summary_end2end.csv` / `aggregates_end2end.csv`: clean end-to-end results.
- `summary_operators.csv` / `aggregates_operators.csv`: operator timing results.
- `SUMMARY.md`: human-readable result table.
- `raw/<mode>/`: per-run timing payloads.

When `--dump-residuals` or `--dump-newton-diagnostics` is enabled, the Python
runner builds with `ENABLE_DUMP=ON` and writes dump files under
`residuals/<mode>/<profile>/<case>/repeat_XX/`. CUDA cuDSS runs now include:

- `residual_before_update_iterN.txt`: nonlinear residual at the start of iteration `N`.
- `residual_solve_iterN.txt`: the `F_k` vector used for CUDA `J dx = F_k`.
- `jacobian_row_ptr_iter0.txt` / `jacobian_col_idx_iter0.txt`: CSR structure.
- `jacobian_values_used_iterN.txt`: CSR values actually passed to cuDSS at iteration `N`.
- `dx_iterN.txt`: cuDSS solve result.
- `linear_residual_iterN.txt`: CUDA `J_used dx - F_k`.
- `linear_diagnostics.csv`: per-iteration norms, refactorization phase, Jacobian age, and cuDSS pivot counts when available.

For CUDA Mixed `--batch-size > 1`, linear residual diagnostics currently record
the batch 0 slice.

## Direct C++ Runner

```bash
cmake -S /workspace/cuPF -B /workspace/cuPF/build/bench-end2end \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_BENCHMARKS=ON \
  -DENABLE_LOG=OFF \
  -DENABLE_TIMING=OFF \
  -DENABLE_NVTX=OFF \
  -DCUPF_CUDSS_REORDERING_ALG=DEFAULT \
  -DWITH_CUDA=ON

cmake --build /workspace/cuPF/build/bench-end2end --target cupf_case_benchmark -j

/workspace/cuPF/build/bench-end2end/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case30_ieee \
  --profile cuda_mixed_edge \
  --cudss-use-matching \
  --cudss-matching-alg ALG_5 \
  --cudss-pivot-epsilon 1e-12 \
  --batch-size 1 \
  --warmup 1 \
  --repeats 10
```

For operator timing, use the operators build:

```bash
cmake -S /workspace/cuPF -B /workspace/cuPF/build/bench-operators \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_BENCHMARKS=ON \
  -DENABLE_LOG=OFF \
  -DENABLE_TIMING=ON \
  -DENABLE_NVTX=ON \
  -DCUPF_CUDSS_REORDERING_ALG=ALG_2 \
  -DWITH_CUDA=ON

cmake --build /workspace/cuPF/build/bench-operators --target cupf_case_benchmark -j

/workspace/cuPF/build/bench-operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case30_ieee \
  --profile cuda_mixed_edge \
  --warmup 1 \
  --repeats 10
```

## Nsight

For Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx -o cupf_mixed_vertex \
  /workspace/cuPF/build/bench-operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_vertex \
  --warmup 1 \
  --repeats 1
```

For Nsight Compute, start with the Jacobian kernels:

```bash
ncu --set full \
  /workspace/cuPF/build/bench-operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_vertex \
  --warmup 1 \
  --repeats 1
```
