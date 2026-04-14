# Schur Complement Experiment Results

**Date**: 2026-04-09  
**GPU**: (see environment — NVIDIA, cuDSS)  
**CPU**: AMD EPYC 7313 16-Core Processor  

---

## Setup

- Warmup: 1 run, Repeats: 3 runs (mean reported)
- Mode `full`: cuDSS full sparse direct solve (ANALYSIS → FACTORIZATION → SOLVE)
- Mode `schur_j22`: cuDSS Schur complement, J22 block = last n_pq rows/cols
  - ANALYSIS → FACTORIZATION → DataGet(SCHUR_MATRIX) → FWD_PERM|FWD → copy x→b → cuSOLVER Xgetrf+Xgetrs → BWD|BWD_PERM
- Matrix precision: FP32
- Jacobian: edge_based

---

## Timing Results (ms, mean over 3 repeats)

### Full solve breakdown

| case | n_pq | n_pvpq | analysis | factor | solve | **total** |
|------|-----:|-------:|---------:|-------:|------:|----------:|
| case118_ieee | 64 | 117 | 8.97 | 0.08 | 0.07 | **9.12** |
| case793_goc | 704 | 792 | 12.09 | 0.22 | 0.09 | **12.39** |
| case1354_pegase | 1094 | 1353 | 14.71 | 0.28 | 0.13 | **15.12** |
| case2746wop_k | 2396 | 2745 | 19.92 | 0.38 | 0.16 | **20.46** |
| case4601_goc | 4468 | 4600 | 29.25 | 0.56 | 0.22 | **30.02** |
| case8387_pegase | 6522 | 8386 | 43.25 | 0.70 | 0.25 | **44.20** |
| case9241_pegase | 7796 | 9240 | 47.37 | 0.74 | 0.25 | **48.36** |

### Schur complement breakdown

| case | n_pq | analysis | factor | extract | fwd | schur_dense | bwd | **total** | vs full |
|------|-----:|---------:|-------:|--------:|----:|------------:|----:|----------:|--------:|
| case118_ieee | 64 | 9.20 | 0.09 | 0.02 | 0.04 | 0.12 | 0.05 | **9.52** | 1.04× slower |
| case793_goc | 704 | 18.18 | 0.50 | 0.45 | 0.07 | 1.73 | 0.13 | **21.05** | 1.70× slower |
| case1354_pegase | 1094 | 23.75 | 1.21 | 1.39 | 0.09 | 3.09 | 0.19 | **29.72** | 1.97× slower |
| case2746wop_k | 2396 | 41.57 | 3.27 | 6.07 | 0.11 | 8.10 | 0.32 | **59.44** | 2.91× slower |
| case4601_goc | 4468 | 84.26 | 10.96 | 20.75 | 0.15 | 19.94 | 0.54 | **136.61** | 4.55× slower |
| case8387_pegase | 6522 | 144.36 | 14.57 | 44.70 | 0.33 | 39.22 | 0.67 | **243.84** | 5.52× slower |
| case9241_pegase | 7796 | 180.46 | 17.03 | 62.91 | 0.21 | 55.28 | 0.60 | **316.48** | 6.54× slower |

---

## Key Findings

### 1. cuDSS full solve wins by a large and growing margin

The Schur complement approach is **slower for every case**, and the gap widens with n_pq:

| case | speedup (full / schur) |
|------|----------------------:|
| case118_ieee (n_pq=64) | **0.96×** (roughly parity) |
| case793_goc (n_pq=704) | **0.59×** |
| case9241_pegase (n_pq=7796) | **0.15×** |

### 2. ANALYSIS cost dominates in Schur mode

cuDSS ANALYSIS in Schur mode takes 3–4× longer than in full mode. For large cases the analysis alone exceeds the full solve's total time:

- case9241: schur analysis = 180ms vs full total = 48ms

### 3. Dense Schur extraction + solve is the bottleneck

For case9241 (n_pq=7796):
- `schur_extract`: 62.9ms — copying a 7796×7796 dense matrix from cuDSS
- `schur_dense` (cuSOLVER LU): 55.3ms — O(n_pq³) dense factorization

These together (118ms) are 2.4× the full solve total (48ms).

### 4. n_pq ratio makes Schur fundamentally uncompetitive here

For power-flow Jacobian, n_pq / dim ≈ 46–97% across cases. The Schur block is not small — it encompasses almost half the degrees of freedom. This makes:
- The dense Schur matrix huge (n_pq²)
- The dense solve expensive (n_pq³)
- The sparse savings negligible

### 5. FWD and BWD costs are negligible

`fwd_solve_sec` and `bwd_solve_sec` remain under 0.7ms even for the largest case. The bottleneck is entirely in Schur extraction and dense solve.

---

## Correctness

Both modes produce consistent solutions:

| case | full residual_inf | schur residual_inf |
|------|------------------:|-------------------:|
| case118_ieee | 1.28e-05 | 2.30e-05 |
| case793_goc | 3.85e-04 | 3.82e-04 |
| case1354_pegase | 4.37e-04 | 4.76e-04 |
| case2746wop_k | 1.15e-03 | 1.07e-03 |
| case4601_goc | 4.34e-04 | 7.02e-04 |
| case8387_pegase | 5.46e-02 | 3.57e-02 |
| case9241_pegase | 4.37e-03 | 5.43e-03 |

Schur mode residuals are within 2× of full mode, consistent with slightly more floating-point operations at FP32.

---

## Implementation Note

For a GENERAL (non-symmetric) matrix in cuDSS Schur mode, the solve sequence differs from the symmetric sample:

```
SOLVE_FWD_PERM | SOLVE_FWD   → reads b, writes x (x = L⁻¹·P·b; x[n-k:] = Schur RHS)
cudaMemcpy(b ← x)            → transfers FWD output to b (equivalent to SOLVE_DIAG with D=I)
cuSOLVER Xgetrf+Xgetrs        → dense LU solve on b[n-k:] → b[n-k:] = y₂
SOLVE_BWD | SOLVE_BWD_PERM   → reads b (y₁ + y₂), writes x (complete solution)
```

The symmetric sample's `SOLVE_DIAG` step (with swapped matrix arguments) does the x→b transfer scaled by D⁻¹. For GENERAL LU (D = I), an explicit `cudaMemcpy` is required.

---

## Conclusion

**cuDSS full sparse direct solve is the better choice for power-flow Newton steps.**

The Schur complement approach would only be competitive if:
1. n_pq were a small fraction of dim (e.g., < 5–10%)
2. The Schur complement matrix were sparse (it is dense for power flow)
3. The sparse factorization were far more expensive than the dense solve

None of these conditions hold for standard power-flow cases.
