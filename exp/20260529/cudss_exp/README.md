# cuDSS Flat-Time Investigation Plan

Date directory note: this folder is intentionally under `exp/20290529` because
the request used `20290529`. The previous benchmark artifacts are under
`exp/20260529`.

## Question

The 78-case linear-solver benchmark shows cuDSS factorization and solve times
are almost flat across a wide matrix-size range:

| phase | small cases | largest cases |
|---|---:|---:|
| analyze | about 12 ms | about 249-309 ms |
| factorize | about 12 ms | about 15 ms |
| solve | about 2.9 ms | about 4.4 ms |

`analyze` scales with case size, but `factorize` and `solve` have a large fixed
floor. The experiment goal is to determine whether that floor comes from host
API/synchronization overhead, fixed cuDSS internal GPU work, memory allocation or
library setup, or an artifact of the benchmark structure.

## Executed Result

The executed experiment used representative cases, NVTX, nsys, and ncu directly.
The earlier baseline-repeat step was not used. See `analysis.md` for the final
diagnosis.

Short version: the flat cuDSS factorize/solve time came from first-use CUDA
module/kernel launch overhead. With default lazy module loading, the first
`factorize` pays about 15 ms in `cudaLaunchKernel`, and the first `solve` pays
about 3.5 ms. With `CUDA_MODULE_LOADING=EAGER`, that cost moves into setup and
factorize/solve scale with case size.

## Hypotheses

1. `cudssExecute + cudaDeviceSynchronize` host-side overhead dominates
   factorize/solve for small and medium cases.
2. cuDSS launches a fixed internal kernel sequence for factorize/solve, so the
   kernel-launch count and GPU active time are nearly size-independent until
   large matrices.
3. cuDSS factorize/solve use internal workspaces or allocator paths whose
   allocation/synchronization cost hides matrix-size scaling.
4. The current benchmark measures cold-ish behavior: every raw row is a separate
   process, and each process performs one analysis followed by repeated
   factorize/solve loops. We need warm in-process traces with higher repeat
   counts.
5. The wall time is flat, but GPU kernel time may not be; nsys will separate CPU
   API time, CUDA synchronization time, kernel time, and memory operations.

## Case Set

Use a size ladder rather than all 78 cases for profiling. Full profiling all 78
would be slow and visually noisy.

| tier | case | n | nnz | reason |
|---|---|---:|---:|---|
| tiny | case5 | 5 | 17 | exposes fixed overhead floor |
| small | case118 | 181 | 1051 | still flat in cuDSS timings |
| medium | case1197 | 2392 | 14344 | first non-trivial sparse case |
| medium-large | case6468rte | 12643 | 87845 | cuDSS analyze grows, factor mostly flat |
| large | case_ACTIVSg25k | 47246 | 318672 | large but still manageable for ncu |
| very-large | case_ACTIVSg70k | 134104 | 900558 | near top-end power-grid case |
| largest | case_SyntheticUSA | 156255 | 1052085 | largest current Jacobian dump |

## Experiment Phases

### 0. Baseline Recheck

Run `cudss_run` with higher in-process repeat counts to separate per-process
startup from steady-state phase time.

Command template:

```bash
/tmp/custom_linear_solver_lin_solver_bench/cudss_run \
  /datasets/matpower_linear_systems/<case> \
  --repeat 50
```

Expected decision:

- If `factorize_ms` and `solve_ms` stay flat even with repeat 50, the fixed floor
  is inside repeated cuDSS phase execution, not process startup.
- If repeat 50 drops median time significantly, the previous 10x independent
  process benchmark includes warm-up effects.

### 1. Nsight Systems: CPU/GPU Timeline

Use nsys first. It answers whether wall time is CPU API/sync bound or actual GPU
work bound.

Profile collection:

```bash
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --stats=true \
  --output exp/20290529/cudss_exp/results/nsys/<case>_repeat50 \
  /tmp/custom_linear_solver_lin_solver_bench/cudss_run \
    /datasets/matpower_linear_systems/<case> \
    --repeat 50
```

Metrics to extract:

- CUDA API time per `cudssExecute`
- `cudaDeviceSynchronize` duration after analysis/factorization/solve
- GPU kernel count per factorize and solve phase
- GPU kernel total duration per phase
- `cudaMalloc/cudaFree/cudaMemset/cudaMemcpy` count and duration inside phases
- CPU idle gaps between kernels
- Kernel launch overhead versus kernel active time

Decision table:

| observation | interpretation |
|---|---|
| CPU API/sync dominates; GPU kernels are short | fixed host/API overhead explains flat time |
| GPU kernel sequence count is fixed across sizes | cuDSS has fixed internal schedule floor |
| GPU active time grows but wall time barely changes | synchronization/API floor hides GPU scaling |
| memory allocation appears in factor/solve | workspace allocation path is part of the floor |
| only analysis has heavy memory allocation | factor/solve flatness is not allocator-driven |

### 2. Nsight Compute: Kernel-Level Cause

Use ncu after nsys identifies the important cuDSS kernels. Do not profile every
kernel for every case with full metrics; that will be too slow.

Two passes:

1. Lightweight kernel list:

```bash
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --print-summary per-kernel \
  --set basic \
  --csv \
  --log-file exp/20290529/cudss_exp/results/ncu/<case>_basic.csv \
  /tmp/custom_linear_solver_lin_solver_bench/cudss_run \
    /datasets/matpower_linear_systems/<case> \
    --repeat 3
```

2. Detailed metrics only for top kernels from the nsys/ncu basic pass:

```bash
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --kernel-name '<selected_kernel_regex>' \
  --log-file exp/20290529/cudss_exp/results/ncu/<case>_top_kernel.txt \
  /tmp/custom_linear_solver_lin_solver_bench/cudss_run \
    /datasets/matpower_linear_systems/<case> \
    --repeat 3
```

Metrics to compare across case sizes:

- kernel launch count
- kernel duration distribution
- achieved occupancy
- SM active %
- DRAM throughput and L2 throughput
- eligible warps / scheduler utilization
- grid size and block size
- launch overhead and waves per SM

Expected interpretations:

- Low occupancy and small grids on tiny/small cases means cuDSS is latency/fixed
  overhead bound.
- Similar kernel durations and grids across `case5` to `case1197` means cuDSS
  performs a minimum-size internal path.
- Large cases with higher SM active and longer kernels explain why only the tail
  of the curve grows.

### 3. Optional NVTX Instrumentation

The current runner does not emit NVTX ranges. nsys can still show CUDA API calls,
but phase attribution is cleaner with NVTX around:

- setup
- analysis
- each factorize iteration
- each solve iteration
- residual download/check

If nsys output is hard to read, add an instrumented copy or guarded NVTX calls to
`custom_linear_solver/scripts/run_cudss_solver.cpp`.

Guarding approach:

```cpp
#ifdef CLS_ENABLE_NVTX
nvtxRangePushA("cudss_factorize");
...
nvtxRangePop();
#endif
```

Then rebuild with NVTX linked:

```bash
cmake -S custom_linear_solver -B /tmp/custom_linear_solver_cudss_profile \
  -DCLS_BUILD_SCRIPTS=ON \
  -DCLS_BUILD_CUDSS_SCRIPT=ON \
  -DCLS_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/custom_linear_solver_cudss_profile -j
```

## Deliverables

1. `baseline/cudss_repeat50.csv`
   - per case wall-time recheck from `cudss_run --repeat 50`
2. `results/nsys/*.nsys-rep`, `results/nsys/*.sqlite`
   - timeline profiles
3. `results/nsys/nsys_summary.csv`
   - CUDA API time, sync time, kernel time, memcpy/malloc time by case
4. `results/ncu/*_basic.csv`
   - kernel list and coarse metrics by case
5. `results/ncu/*_top_kernel.txt`
   - detailed metrics for dominant cuDSS kernels
6. `analysis.md`
   - final diagnosis: host overhead vs fixed GPU schedule vs allocator vs true
     numerical work

## Stop Criteria

We can stop once these are answered:

1. For factorize/solve, what fraction of wall time is CUDA API/synchronization
   versus GPU kernel execution?
2. Does cuDSS launch roughly the same number of kernels for small and large
   matrices?
3. Do the dominant kernels have low occupancy or tiny grids on small cases?
4. Are allocations or device-host copies happening inside factorize/solve?
5. Does increasing in-process repeat count change the apparent flatness?
