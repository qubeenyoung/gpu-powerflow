# 20260607-2 GMRES + METIS Block-Jacobi

This experiment adds a standalone implementation of right-preconditioned restarted GMRES for general nonsymmetric CSR systems:

- `GmresSolver`
- `MetisBlockJacobiPreconditioner`
- METIS partition/permutation utilities
- device-first `solve_device(const double* d_values, const double* d_rhs, double* d_x)` API
- host wrapper `solve(const std::vector<double>& values, const std::vector<double>& rhs)`

The setup phase builds the METIS permutation, permuted CSR pattern, dense diagonal blocks, and batched block LU/inverse. The solve phase keeps vectors on device, stores both `V` and `Z` bases for right preconditioning, and reports separate timings for preconditioner apply, SpMV, orthogonalization, dot/reduction, solution update, final residual, and permutation.

## Build

```bash
cmake -S exp/20260607-2 -B exp/20260607-2/build
cmake --build exp/20260607-2/build -j
```

## Smoke Run

```bash
exp/20260607-2/build/gmres_block_jacobi_bench \
  --synthetic-n 512 \
  --solver gmres \
  --gmres-restart 16 \
  --gmres-max-iters 32 \
  --gmres-rtol 1e-3 \
  --preconditioner metis_block_jacobi \
  --block-size 32 \
  --block-jacobi-precision fp32 \
  --block-jacobi-apply inverse_gemv
```

For dumped Newton matrices, pass MatrixMarket CSR data or cuPF text dumps as:

```bash
exp/20260607-2/build/gmres_block_jacobi_bench \
  --matrix exp/20260607-2/raw/cupf_jf_dumps/case1197/J1.txt \
  --rhs exp/20260607-2/raw/cupf_jf_dumps/case1197/F1.txt \
  --gmres-restart 16 \
  --gmres-max-iters 32 \
  --gmres-rtol 1e-3 \
  --preconditioner metis_block_jacobi \
  --block-size 64 \
  --block-jacobi-precision fp32 \
  --block-jacobi-apply lu_solve
```

## cuPF J/F Dumps

The cuPF Newton loop uses zero-based iteration labels. `jacobian_iter1.txt` and
`residual_iter1.txt` are therefore the second Newton iteration, exposed as
`J1.txt` and `F1.txt` aliases by the dump script.

```bash
exp/20260607-2/scripts/dump_cupf_jf_linear_systems.py \
  --max-iter 3 \
  --tolerance 0 \
  --min-buses 1000 \
  --max-buses 10000
```

The default dump root is `exp/20260607-2/raw/cupf_jf_dumps`, and the script
writes `linear_system_dump_summary.csv` with `J1/F1` and `J2/F2` paths.

## cuDSS Timing

`cudss_jf_bench` measures cuDSS `ANALYSIS`, `FACTORIZATION`, and `SOLVE`
separately on a dumped `Jk/Fk` pair. The phase timings exclude file parsing and
host-to-device copies.

```bash
exp/20260607-2/scripts/run_cudss_representative_bench.py \
  --precision fp64 \
  --repeats 3 \
  --iteration 1
```

The representative-case output is written to
`exp/20260607-2/results/cudss_representative_j1.csv` and `.md`.

## FLOP Reporting

cuITER BiCGSTAB + block-Jacobi FLOPs can be reported from the existing
`hybrid_nr_bench` summary and iteration CSVs. The output separates per
Newton-Raphson iteration estimates from cumulative per-case totals:

```bash
exp/20260607-2/tools/estimate_cuiter_flops.py \
  --summary exp/20260607-2/results/bicgstab_iter2_bs8_all78_summary.csv \
  --iters exp/20260607-2/results/bicgstab_iter2_bs8_all78_iters.csv \
  --out-prefix exp/20260607-2/results/bicgstab_iter2_bs8_all78_flops
```

Outputs:

- `*_flops_iters.csv`: one row per NR iteration, with
  `cuiter_total_flops_est` and `cuiter_cumulative_flops_est`
- `*_flops_summary.csv`: accumulated per-case FLOPs and average FLOPs per
  cuITER NR step / per NR iteration / per BiCGSTAB linear iteration
- `*_flops.md`: compact report and formulas

The estimator counts CSR SpMV, block-Jacobi apply, dot reductions, vector
updates, and norm checks. Add `--count-bj-setup` to include a separate dense
block LU/inverse setup estimate when setup timing columns indicate setup ran.

cuDSS does not expose sparse LU FLOP counts through the benchmark. For cuDSS,
use Nsight Compute dynamic floating-point instruction counters:

```bash
exp/20260607-2/tools/measure_gpu_flops_ncu.py \
  --output exp/20260607-2/results/cudss_case1197_j1_ncu_flops.csv \
  -- exp/20260607-2/build/cudss_jf_bench \
    --case case1197 \
    --matrix exp/20260607-2/raw/cupf_jf_dumps/case1197/J1.txt \
    --rhs exp/20260607-2/raw/cupf_jf_dumps/case1197/F1.txt \
    --precision fp64 \
    --repeats 1 \
    --csv
```

This reports GPU FP32/FP64 dynamic FLOPs from SASS instruction metrics
(`FMA = 2 FLOPs`). It does not include CPU-side cuDSS multi-threaded work.

## Iterative Timing

Representative GMRES sweeps use the same `J1/F1` dumps:

```bash
exp/20260607-2/scripts/run_gmres_representative_bench.py
```

The default sweep runs block sizes `32,64`, restarts `8,16`, max iterations
`16,32`, FP32 block-Jacobi, and `inverse_gemv` apply. Results are written to
`exp/20260607-2/results/gmres_representative_j1_sweep.csv` and `.md`.

The comparison rows for cuDSS and BiCGSTAB are printed as `n/a` in this isolated experiment until the existing solver timings are wired into the same benchmark harness.

## Hybrid NR Bench

`hybrid_nr_bench` is an experimental minimal cuPF NR port inside this
experiment tree. It does not include or patch the cuPF repository. It loads the
cuPF dump case format (`dump_Ybus.mtx`, `dump_Sbus.txt`, `dump_V.txt`,
`dump_pv.txt`, `dump_pq.txt`), runs the CUDA NR operations, and switches the
linear solver by policy:

- first NR step: cuDSS bootstrap
- middle steps: GMRES + METIS block-Jacobi
- small mismatch: cuDSS polish
- rejected GMRES correction: optional cuDSS fallback

```bash
exp/20260607-2/build/hybrid_nr_bench \
  --case case1197,case2736sp,case3375wp,case6468rte,case_ACTIVSg10k \
  --solver hybrid \
  --warmup 1 \
  --cudss-bootstrap-iters 1 \
  --cudss-polish-threshold 1e-4 \
  --block-size 64 \
  --gmres-restart 16 \
  --gmres-max-iters 8 \
  --gmres-fixed-iter-mode true \
  --enable-cudss-fallback true \
  --accept-iterative-by-mismatch true
```

The default outputs are:

- `results/hybrid_nr_gmres_block_jacobi.csv`
- `results/hybrid_nr_gmres_block_jacobi_iters.csv`

For the requested sweeps:

```bash
exp/20260607-2/scripts/run_hybrid_nr_sweep.py --mode both
```
