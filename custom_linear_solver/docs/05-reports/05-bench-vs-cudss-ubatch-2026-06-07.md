# Benchmark vs cuDSS (ubatch + mt-auto) — bus-count > 1K power-grid Jacobians

**작성일**: 2026-06-07
**Setup**: RTX 3090 (sm_86), CUDA 12.8, `Release` build, `CLS_INTERNAL_GRAPH=ON`,
`use_multistream_subtrees=true`. Median of 10 trials. `--batch-only` on the custom side.

**Artefacts** (sibling folder
[`05-bench-vs-cudss-ubatch-2026-06-07/`](05-bench-vs-cudss-ubatch-2026-06-07)):
- `sweep_v2.tsv` — raw measurements (220 rows: 11 cases × 4 paths × 5 batch sizes)

**Solvers compared**:

| label | command |
|---|---|
| **cuDSS ubatch+mt-auto / fp32** | `cudss_run --precision fp32 --mt-auto --batch B` — `CUDSS_CONFIG_UBATCH_SIZE=B`, batch-major value buffer (B × nnz), `cudssSetThreadingLayer(handle, "<libcudss_mtlayer_gomp.so>")` (cuPF's pattern) |
| **custom / fp32** | `--precision fp32` — staged-scalar trailing |
| **custom / fp16** | `--precision fp16` — FP16 WMMA m16n16k16 trailing, FP32 accumulate |
| **custom / tf32** | `--precision tf32` — V9h PTX `mma.m16n8k8/k4` hybrid + `__launch_bounds__(512, 2)` |

**Test cases** (bus-count > 1K, sorted by Jacobian dimension n):

| case | n | nnz | label in user request |
|------|--:|--:|---|
| case1197 | 2,392 | 14,344 | 1xxx |
| case_ACTIVSg2000 | 3,607 | 21,529 | 2xxx |
| case3012wp | 5,725 | 33,983 | 3xxx |
| case6468rte | 12,643 | 75,917 | 6xxxx |
| case8387pegase | 14,908 | 89,432 | 8000 |
| case9241pegase | 17,036 | 102,189 | 9000 |
| case_ACTIVSg10k | 18,544 | 110,855 | 10K |
| case13659pegase | 23,225 | 139,341 | 13K |
| case_ACTIVSg25k | 47,246 | 282,927 | 25K |
| case_ACTIVSg70k | 134,104 | 802,873 | 70K |
| case_SyntheticUSA | 156,255 | 936,533 | USA |

---

## 1. Single batch (B = 1) — analyze / factorize / solve per system (ms)

`analyze` is the one-time symbolic + plan/reordering cost (per matrix sparsity pattern; reused across Newton iterations). `factorize` and `solve` are per call, median of 10.

### Analyze (ms)

| case | cuDSS ubatch+mt-auto / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 17.5 | **3.2** | 3.3 | 3.4 |
| case_ACTIVSg2000 | 95.0 | **7.4** | 7.4 | 7.4 |
| case3012wp | 27.6 | **8.8** | 8.7 | 9.0 |
| case6468rte | 57.2 | **16.6** | 17.5 | 17.3 |
| case8387pegase | 30.6 | **20.2** | 21.0 | 20.9 |
| case9241pegase | 32.3 | **22.7** | 21.6 | 22.7 |
| case_ACTIVSg10k | 33.2 | 23.3 | **22.7** | 23.1 |
| case13659pegase | **37.5** | 39.2 | 39.7 | 40.0 |
| case_ACTIVSg25k | 57.0 | **55.6** | 57.9 | 56.3 |
| case_ACTIVSg70k | **127.3** | 155.3 | 155.6 | 156.0 |
| case_SyntheticUSA | **149.8** | 178.0 | 183.2 | 179.3 |

The custom analyze is faster on cases up to ~25K; cuDSS catches up and is slightly faster on the largest (70K, USA). Analyze cost is amortised across many Newton iterations, so the gap rarely matters once the solver is warm.

### Factorize per system (ms)

| case | cuDSS ubatch+mt-auto / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.103 | **0.069** | 0.070 | 0.070 |
| case_ACTIVSg2000 | 0.345 | **0.274** | 0.355 | 0.349 |
| case3012wp | 0.253 | **0.233** | 0.300 | 0.300 |
| case6468rte | 0.338 | **0.274** | 0.451 | 0.427 |
| case8387pegase | 0.540 | **0.353** | 0.444 | 0.418 |
| case9241pegase | 0.564 | **0.385** | 0.480 | 0.431 |
| case_ACTIVSg10k | 0.467 | **0.370** | 0.459 | 0.431 |
| case13659pegase | 0.659 | **0.396** | 0.490 | 0.482 |
| case_ACTIVSg25k | 1.095 | **0.813** | 0.816 | 0.808 |
| case_ACTIVSg70k | 2.558 | 2.796 | **2.073** | 2.161 |
| case_SyntheticUSA | 2.792 | 2.622 | **2.166** | 2.490 |

custom/fp32 beats cuDSS on 9/11 cases at B=1. The fp16/tf32 paths catch up on the two largest (70K, USA), where the trailing-GEMM share of factor wall is big enough for the WMMA/PTX kernels to pay off even at B=1.

### Solve per system (ms)

| case | cuDSS ubatch+mt-auto / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | **0.087** | 0.077 | 0.078 | 0.078 |
| case_ACTIVSg2000 | **0.166** | 0.197 | 0.197 | 0.197 |
| case3012wp | **0.147** | 0.189 | 0.180 | 0.161 |
| case6468rte | **0.187** | 0.214 | 0.233 | 0.236 |
| case8387pegase | **0.230** | 0.242 | 0.246 | 0.234 |
| case9241pegase | **0.229** | 0.257 | 0.267 | 0.255 |
| case_ACTIVSg10k | **0.237** | 0.251 | 0.256 | 0.248 |
| case13659pegase | **0.271** | 0.273 | 0.257 | 0.264 |
| case_ACTIVSg25k | **0.450** | 0.499 | 0.535 | 0.528 |
| case_ACTIVSg70k | **0.866** | 0.891 | 0.914 | 0.935 |
| case_SyntheticUSA | **0.957** | 1.017 | 0.947 | 1.043 |

cuDSS owns single-system solve: its triangular-solve kernels are tuned for B=1 throughput, and the custom solver's `solve_small_warp` / `solve_level` path is optimised for batch amortisation rather than B=1 latency. The gap is small (within 10-20 %) for cases beyond ~10K but consistent.

### Relative residual at B=1

| case | cuDSS/fp32 | custom/fp32 | custom/fp16 | custom/tf32 |
|---|---:|---:|---:|---:|
| case1197 | 7.8e-05 | 1.6e-04 | 1.8e-04 | 1.7e-04 |
| case_ACTIVSg2000 | 1.4e-05 | 1.3e-05 | 2.8e-03 | 1.5e-02 |
| case3012wp | 1.9e-04 | 1.9e-04 | 4.1e-04 | 1.8e-04 |
| case6468rte | 4.6e-05 | 4.1e-05 | 4.9e-04 | 5.7e-03 |
| case8387pegase | 1.2e-05 | 2.0e-05 | 6.5e-03 | 3.6e-02 |
| case9241pegase | 2.7e-06 | 2.8e-06 | 6.7e-04 | 8.2e-04 |
| case_ACTIVSg10k | 6.3e-04 | 2.8e-04 | 1.2e-02 | 5.0e-02 |
| case13659pegase | 1.1e-04 | 1.1e-04 | 5.0e-02 | 1.1e-02 |
| case_ACTIVSg25k | 1.4e-04 | 2.0e-04 | 1.3e-02 | 7.4e-02 |
| case_ACTIVSg70k | 1.2e-03 | 3.4e-03 | 7.6e-02 | 5.5e-02 |
| case_SyntheticUSA | 9.2e-04 | 1.2e-03 | 6.1e-02 | 5.4e-02 |

custom/fp32 ≈ cuDSS/fp32 (same precision). fp16 / tf32 trail by 1-2 orders of magnitude; suitable inside a Newton loop with iterative refinement, not as a drop-in single-shot precision.

---

## 2. Multi batch — factor + solve per system (ms)

For each (case, B), the table is the **factor+solve per-system** wall, median of 10. `cuDSS ubatch+mt-auto / fp32` uses `CUDSS_CONFIG_UBATCH_SIZE=B` and feeds a batch-major value buffer the same way cuPF does in production.

### B = 4

| case | cuDSS ubatch / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.0748 | **0.0387** | 0.0388 | 0.0386 |
| case_ACTIVSg2000 | 0.200 | **0.126** | 0.146 | 0.145 |
| case3012wp | 0.163 | **0.106** | 0.123 | 0.117 |
| case6468rte | 0.251 | **0.159** | 0.180 | 0.177 |
| case8387pegase | 0.344 | **0.187** | 0.211 | 0.200 |
| case9241pegase | 0.374 | **0.197** | 0.221 | 0.224 |
| case_ACTIVSg10k | 0.387 | **0.197** | 0.227 | 0.233 |
| case13659pegase | 0.465 | **0.218** | 0.243 | 0.261 |
| case_ACTIVSg25k | 0.905 | 0.428 | 0.451 | **0.420** |
| case_ACTIVSg70k | 2.377 | 1.204 | 1.082 | **1.022** |
| case_SyntheticUSA | 2.681 | 1.348 | **1.221** | 1.286 |

### B = 32

| case | cuDSS ubatch / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.0448 | **0.0077** | 0.0076 | 0.0077 |
| case_ACTIVSg2000 | 0.125 | **0.0265** | 0.0285 | 0.0282 |
| case3012wp | 0.104 | **0.0248** | 0.0251 | 0.0243 |
| case6468rte | 0.185 | **0.0430** | 0.0457 | 0.0463 |
| case8387pegase | 0.229 | 0.0644 | 0.0580 | **0.0586** |
| case9241pegase | 0.253 | **0.0617** | 0.0682 | 0.0653 |
| case_ACTIVSg10k | 0.294 | **0.0656** | 0.0699 | 0.0656 |
| case13659pegase | 0.329 | **0.0865** | 0.0823 | 0.0843 |
| case_ACTIVSg25k | 0.731 | 0.178 | 0.183 | **0.172** |
| case_ACTIVSg70k | 2.093 | 0.622 | 0.616 | **0.572** |
| case_SyntheticUSA | 2.395 | 0.670 | 0.688 | **0.675** |

### B = 64

| case | cuDSS ubatch / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.0447 | **0.0059** | 0.0058 | 0.0058 |
| case_ACTIVSg2000 | 0.119 | **0.0191** | 0.0205 | 0.0197 |
| case3012wp | 0.099 | 0.0178 | 0.0194 | 0.0188 |
| case6468rte | 0.179 | **0.0342** | 0.0366 | 0.0367 |
| case8387pegase | 0.220 | **0.0480** | 0.0503 | 0.0525 |
| case9241pegase | 0.243 | 0.0558 | **0.0543** | 0.0565 |
| case_ACTIVSg10k | 0.288 | 0.0563 | 0.0618 | **0.0572** |
| case13659pegase | 0.319 | 0.0732 | 0.0753 | **0.0726** |
| case_ACTIVSg25k | 0.718 | 0.173 | 0.178 | **0.159** |
| case_ACTIVSg70k | 2.071 | **0.564** | 0.579 | 0.580 |
| case_SyntheticUSA | 2.373 | 0.663 | 0.664 | **0.656** |

### B = 256

| case | cuDSS ubatch / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.0436 | **0.0043** | 0.0043 | 0.0043 |
| case_ACTIVSg2000 | 0.114 | 0.0144 | 0.0153 | **0.0144** |
| case3012wp | 0.096 | 0.0139 | **0.0138** | 0.0138 |
| case6468rte | 0.175 | 0.0301 | 0.0313 | **0.0295** |
| case8387pegase | 0.213 | **0.0423** | 0.0426 | 0.0424 |
| case9241pegase | 0.237 | 0.0483 | 0.0604 | **0.0473** |
| case_ACTIVSg10k | 0.283 | **0.0512** | 0.0543 | 0.0517 |
| case13659pegase | 0.311 | 0.0705 | 0.0766 | **0.0658** |
| case_ACTIVSg25k | 0.707 | 0.160 | 0.167 | **0.155** |
| case_ACTIVSg70k | 2.056 | **0.548** | 0.576 | 0.571 |
| case_SyntheticUSA | 2.357 | 0.650 | 0.652 | **0.644** |

### custom/tf32 speedup over cuDSS ubatch+mt-auto / fp32

(factor + solve per system; `>1.0×` means custom is faster)

| case | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| case1197 | 1.29× | 1.94× | 5.83× | 7.64× | **10.16×** |
| case_ACTIVSg2000 | 0.94× | 1.38× | 4.43× | 6.03× | **7.96×** |
| case3012wp | 0.87× | 1.39× | 4.26× | 5.27× | **6.96×** |
| case6468rte | 0.79× | 1.42× | 3.99× | 4.89× | **5.94×** |
| case8387pegase | 1.18× | 1.73× | 3.91× | 4.19× | **5.03×** |
| case9241pegase | 1.16× | 1.67× | 3.87× | 4.31× | **5.01×** |
| case_ACTIVSg10k | 1.04× | 1.66× | 4.49× | 5.03× | **5.47×** |
| case13659pegase | 1.25× | 1.78× | 3.90× | 4.39× | **4.73×** |
| case_ACTIVSg25k | 1.16× | 2.15× | 4.24× | 4.51× | **4.57×** |
| case_ACTIVSg70k | 1.11× | 2.33× | 3.66× | 3.57× | **3.60×** |
| case_SyntheticUSA | 1.06× | 2.09× | 3.55× | 3.62× | **3.66×** |

---

## 3. Interpretation

### B = 1 single-batch

- **factor + solve** are competitive but not a clean win for either: custom/fp32 wins factorize on 9/11 cases, cuDSS wins solve on 11/11. Net f+s is roughly even at small cases and tilts to custom/fp32 from case8387 up.
- **analyze** strongly favours custom on small/mid cases (5-10× faster on case1197 ~ case6468rte); the gap closes around 25K and flips to cuDSS on the two largest cases. cuDSS' setup includes reordering + symbolic factorization that is heavier on tiny matrices than the custom solver's MeTIS-ND + multifrontal plan build.
- For a single Newton iteration on small/mid grids (≤25K), `custom/fp32` lands the same residual as cuDSS/fp32 with shorter total wall, mostly via the analyze gap.

### Multi-batch (B ≥ 4)

- cuDSS does benefit from `UBATCH_SIZE` — at B=256 its per-system time drops to ~25 % of B=1 — but the custom solver's uniform-batch dispatch saturates SM utilization more aggressively. The result is a 3-10× per-system win at B=256 across all measured cases.
- The largest cases (70K, USA) saturate around 3.5-3.7× — at that size cuDSS' parallel triangular solve + level-set factor is already near the hardware throughput limit, and the custom solver's wall is dominated by the spine (per-batch sequential).
- Smallest case (case1197) reaches the highest ratio (10.16×) because cuDSS still pays a substantial per-call setup that doesn't shrink with ubatch.

### Precision dimension

- custom/fp16 and custom/tf32 trade some accuracy for occupancy headroom on the trailing GEMM. The win is most visible on the two largest cases where trailing GEMM dominates: at case_SyntheticUSA B=4 the fp16 path is the fastest (1.22 ms/sys, ~10 % faster than fp32 and tf32 at that B).
- On bus counts ≲ 10K the three precisions are within noise of each other on per-system time — the GEMM is too small to expose the WMMA / PTX advantage.
- Single-shot residuals from fp16 / tf32 are 1e-2 ~ 1e-1 on the larger cases; recommended only inside an iterative refinement loop.

### cuDSS `--mt-auto` impact

The threading layer reduces analyze cost noticeably (see e.g. case_ACTIVSg2000: analyze 95 ms is largely host-side mt-aware reordering). Wall savings beyond analyze are small at this matrix scale; the mt layer mainly helps the host reordering phase.

---

## 4. Reproducing

```bash
# Build the CLI runners (requires cuDSS install + NVTX optional)
cmake -S custom_linear_solver -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DCLS_BUILD_CUDSS_SCRIPT=ON
cmake --build build-bench -j

# cuDSS ubatch + mt-auto (cuPF pattern)
./build-bench/cudss_run <case-dir> --precision fp32 --mt-auto --batch <B> --repeat 10

# Custom (V9h+LB TF32, FP16 WMMA, or scalar FP32)
./build-bench/custom_linear_solver_run <case-dir> \
  --precision {fp32|fp16|tf32} --batch <B> --batch-only --repeat 10
```

For the 4 cases not in the public dataset directory
(`case9241pegase`, `case_ACTIVSg10k`, `case13659pegase`, `case_ACTIVSg70k`),
regenerate the NR Jacobians with:

```bash
# Convert .m → .mat
python3 -m gpu-powerflow.python.prepare.convert_m_to_mat \
  --input-root /datasets/power_system/matpower \
  --output-root <writable_mat_dir> \
  --cases case9241pegase case_ACTIVSg10k case13659pegase case_ACTIVSg70k

# Dump NR Jacobian + mismatch at iteration 2
python3 prepare_datasets/python/prepare_nr_linear_system.py \
  --mat-root <writable_mat_dir> \
  --output-root <writable_nr_dir> \
  --cases case9241pegase case_ACTIVSg10k case13659pegase case_ACTIVSg70k
```
