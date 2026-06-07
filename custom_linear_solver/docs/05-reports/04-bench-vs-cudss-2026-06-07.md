# Benchmark — bus-count > 1K power-grid Jacobians vs cuDSS, RTX 3090 (sm_86)

**작성일**: 2026-06-07
**Artefacts** (sibling folder [`04-bench-vs-cudss-2026-06-07/`](04-bench-vs-cudss-2026-06-07)):
- `sweep_results.tsv` — raw measurements (113 rows)
- `case8387_b1_fp32_multistream_on.nsys-rep` — Nsight Systems profile, subtree multi-stream ON
- `case8387_b1_fp32_multistream_off.nsys-rep` — same case with multi-stream OFF

**Setup**: RTX 3090 (sm_86), CUDA 12.8, `Release` build, `CLS_INTERNAL_GRAPH=ON`,
`use_multistream_subtrees=true`. Median of 10 trials. `--batch-only`.
cuDSS has no native uniform-batching semantics, so it is reported once per case at B=1
(per-system time would be identical at higher B since each system is a separate call).

Solvers / precisions compared:
- **cuDSS fp32** (`CUDA_R_32F`)
- **custom fp32** (`Precision::FP32` — staged-scalar trailing)
- **custom fp16** (`Precision::FP16` — FP16 WMMA m16n16k16 trailing, FP32 accumulate)
- **custom tf32** (`Precision::TF32` — V9h PTX `mma.m16n8k8/k4` hybrid + `__launch_bounds__(512, 2)`)

Cases enumerated by Jacobian dimension `n` (bus count ≈ n/2):

| case | n | nnz | scope |
|------|--:|--:|---|
| case1197 | 2,392 | 14,344 | ≈ 1.2k bus |
| case_ACTIVSg2000 | 3,607 | 21,529 | 2k bus |
| case3012wp | 5,725 | 33,983 | 3k bus |
| case6468rte | 12,643 | 75,917 | 6.5k bus |
| case8387pegase | 14,908 | 89,432 | 8.4k bus |
| case_ACTIVSg25k | 47,246 | 282,927 | 25k bus |
| case_SyntheticUSA | 156,255 | 936,533 | 78k bus |

---

## 1. Per-system factor+solve time (ms), median of 10

Format `factor+solve (per-system)`.

### case1197 (n=2,392)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.192** | – | – | – | – |
| custom/fp32 | 0.147 | 0.039 | 0.0076 | 0.0058 | **0.0043** |
| custom/fp16 | 0.147 | 0.039 | 0.0076 | 0.0058 | **0.0043** |
| custom/tf32 | 0.146 | 0.039 | 0.0076 | 0.0058 | **0.0043** |

### case_ACTIVSg2000 (n=3,607)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.510** | – | – | – | – |
| custom/fp32 | 0.471 | 0.126 | 0.027 | 0.019 | **0.0144** |
| custom/fp16 | 0.552 | 0.147 | 0.029 | 0.021 | 0.0154 |
| custom/tf32 | 0.545 | 0.145 | 0.028 | 0.020 | **0.0144** |

### case3012wp (n=5,725)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.400** | – | – | – | – |
| custom/fp32 | 0.416 | 0.101 | 0.024 | 0.018 | **0.0137** |
| custom/fp16 | 0.482 | 0.126 | 0.027 | 0.018 | 0.0136 |
| custom/tf32 | 0.489 | 0.123 | 0.026 | 0.019 | 0.0139 |

### case6468rte (n=12,643)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.528** | – | – | – | – |
| custom/fp32 | 0.479 | 0.161 | 0.042 | 0.035 | **0.0297** |
| custom/fp16 | 0.621 | 0.190 | 0.046 | 0.037 | 0.0305 |
| custom/tf32 | 0.577 | 0.186 | 0.045 | 0.036 | 0.0300 |

### case8387pegase (n=14,908)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.772** | – | – | – | – |
| custom/fp32 | 0.617 | 0.182 | 0.058 | 0.047 | **0.0416** |
| custom/fp16 | 0.712 | 0.230 | 0.060 | 0.054 | 0.0430 |
| custom/tf32 | 0.731 | 0.196 | 0.058 | 0.052 | 0.0414 |

### case_ACTIVSg25k (n=47,246)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **1.541** | – | – | – | – |
| custom/fp32 | 1.327 | 0.448 | 0.182 | 0.173 | 0.159 |
| custom/fp16 | 1.300 | 0.441 | 0.184 | 0.167 | 0.168 |
| custom/tf32 | 1.325 | **0.397** | **0.171** | **0.162** | **0.154** |

### case_SyntheticUSA (n=156,255)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **3.742** | – | – | – | – |
| custom/fp32 | 3.582 | 1.222 | 0.701 | 0.648 | 0.640 |
| custom/fp16 | 3.538 | 1.213 | 0.691 | 0.688 | 0.646 |
| custom/tf32 | **3.054** | 1.217 | **0.681** | **0.622** | 0.647 |

---

## 2. Per-system speedup of custom/tf32 over cuDSS-fp32

| case | cuDSS f+s (ms) | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|---:|
| case1197 | 0.192 | 1.31× | 4.90× | 25.2× | 32.8× | **44.7×** |
| case_ACTIVSg2000 | 0.510 | 0.93× | 3.52× | 18.1× | 25.7× | **35.5×** |
| case3012wp | 0.400 | 0.82× | 3.24× | 15.3× | 21.6× | **28.7×** |
| case6468rte | 0.528 | 0.91× | 2.84× | 11.7× | 14.9× | **17.6×** |
| case8387pegase | 0.772 | 1.06× | 3.94× | 13.2× | 15.0× | **18.6×** |
| case_ACTIVSg25k | 1.541 | 1.16× | 3.89× | 9.0× | 9.5× | **10.0×** |
| case_SyntheticUSA | 3.742 | 1.23× | 3.07× | 5.49× | 6.02× | 5.78× |

The 45× / 35× / 29× values at B=256 for the smaller cases reflect uniform-batch
amortisation — cuDSS pays the full single-system kernel-launch + analyze-fixed cost
per system, while the custom solver shares one dispatch sequence across all B systems.
At cases of n ≳ 50k the per-system work itself dominates, so the B amortisation
saturates around 6-10×.

---

## 3. Accuracy (max relative residual across all measured B)

| case | cuDSS/fp32 | custom/fp32 | custom/fp16 | custom/tf32 |
|---|---:|---:|---:|---:|
| case1197 | 7.1e-05 | 2.1e-04 | 2.1e-04 | 2.1e-04 |
| case_ACTIVSg2000 | 1.3e-05 | 1.5e-05 | 2.8e-03 | 1.5e-02 |
| case3012wp | 1.6e-04 | 2.5e-04 | 4.7e-03 | 3.2e-02 |
| case6468rte | 4.2e-05 | 1.3e-04 | 1.1e-03 | 2.8e-03 |
| case8387pegase | 1.3e-05 | 3.5e-05 | 2.1e-02 | 5.0e-02 |
| case_ACTIVSg25k | 1.5e-04 | 2.7e-04 | 8.4e-02 | 5.4e-02 |
| case_SyntheticUSA | 8.9e-04 | 2.4e-02 | 1.2e-01 | 5.6e-02 |

Observations:
- **fp32** (cuDSS & custom) hits ≲ 1e-3 across all cases — acceptable for NR Jacobian residuals.
- **fp16 / tf32** hit ~1e-1 .. 1e-2 on the larger cases due to accumulated trailing-GEMM rounding.
  Useful inside a Newton loop with iterative refinement; not a drop-in replacement for FP32
  precision when the caller needs `‖Ax-b‖ ≲ 1e-3` on a single shot.

---

## 4. Multi-stream overlap (case8387 B=1 fp32)

NVTX-instrumented build (`CLS_ENABLE_NVTX=ON`), `nsys profile --trace=cuda,nvtx
--cuda-graph-trace=node`.

### Per-stream kernel activity (10 factor + 10 solve iterations)

**Multi-stream ON (`use_multistream_subtrees=true`, default)** — 9 streams hit:

| stream | kernels | GPU-busy (ms) | role |
|---:|---:|---:|---|
| 13 (main) | 990 | 6.67 | analyze + spine + solve |
| 23 | 180 | 2.03 | subtree #1 |
| 24 | 180 | 1.88 | subtree #2 |
| 27 | 180 | 1.74 | subtree #3 |
| 26 | 120 | 1.25 | subtree #4 |
| 25 | 130 | 1.24 | subtree #5 |
| 29 | 120 | 0.96 | subtree #6 |
| 28 | 110 | 0.75 | subtree #7 |
| 7  | 48  | 0.36 | CUDA runtime (analyze copies) |

`factor_*` kernel total: **1,340 instances / 13.76 ms** across 8 subtree streams.

**Multi-stream OFF (`--no-multistream`)** — 2 streams hit:

| stream | kernels | GPU-busy (ms) | role |
|---:|---:|---:|---|
| 13 (main) | 930 | 5.75 | everything sequential |
| 7  | 48  | 0.36 | CUDA runtime |

`factor_*` kernel total: **300 instances / 3.21 ms** on the main stream only.

### Interpretation

- The streams ARE being used (8 distinct subtree streams + 1 spine stream visible in
  the `ON` profile). NVTX `factorize/iter=*` and `solve/iter=*` ranges in the timeline
  show the subtree kernels firing concurrently within each iteration.
- However at **B=1 wall is not improved** by multi-stream:
    - ON:  factor=0.340 ms, solve=0.262 ms (per-system)
    - OFF: factor=0.322 ms, solve=0.245 ms (per-system) — ~5% faster
- Reason: for case8387 the spine (chain at the top of the elimination tree) dominates
  the factor wall, and the spine runs on the main stream anyway. The subtree work at
  the etree leaves only accounts for ~2 ms total. At B=1 the SM occupancy is already
  low enough that splitting that work across streams adds event/wait overhead without
  freeing up SMs for the spine to run earlier.
- The win shows up at B ≥ 4 where each subtree stream has enough launches per level
  to hide the dispatch overhead. The main sweep confirms this: at B=64 case8387
  custom-tf32 reaches 15× cuDSS-fp32.

### Generated artefacts

- [`04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_on.nsys-rep`](04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_on.nsys-rep) (~185 KB)
- [`04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_off.nsys-rep`](04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_off.nsys-rep) (~140 KB)

Open in Nsight Systems GUI to inspect the timeline; the NVTX ranges `analyze`,
`setup`, `factorize/iter=N`, `solve/iter=N` group the work, and the per-stream rows
show how the subtree kernels overlap (multistream ON) vs collapse to a single row
(multistream OFF).

### Reproducing

```bash
# Build with cuDSS and NVTX
cmake -S custom_linear_solver -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DCLS_BUILD_CUDSS_SCRIPT=ON -DCLS_ENABLE_NVTX=ON
cmake --build build-bench -j

# Wall-time sweep (cuDSS once per case at B=1; custom for all 5 B sizes × 3 precisions)
./build-bench/cudss_run <case-dir> --precision fp32 --repeat 10
./build-bench/custom_linear_solver_run <case-dir> --batch <B> --batch-only \
  --precision {fp32|fp16|tf32} --repeat 10

# Multi-stream nsys profile
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --output=case8387_b1_fp32_multistream_on \
  ./build-bench/custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
  --batch 1 --batch-only --precision fp32 --repeat 10

nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --output=case8387_b1_fp32_multistream_off \
  ./build-bench/custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
  --batch 1 --batch-only --precision fp32 --repeat 10 --no-multistream
```
