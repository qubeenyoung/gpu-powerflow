# mysolver vs cuDSS — full power-grid benchmark (2026-05-26, KST)

Measured: `benchmark --solver mysolver,cudss-gpu --matrix-set matpower_nr --warmup-gpu`.
Phases match the cuDSS breakdown: **Analysis** (reorder + symbolic), **Factor**
(numeric), **Solve** (fwd+bwd). All times ms (warm). mysolver = production CPU
path (own-numeric, residual-gated KLU fallback). berr = componentwise backward error.

## End-to-end (Analysis + Factor + Solve)

| matrix | n | mysolver A / F / S = tot | cuDSS A / F / S = tot | end-to-end |
|---|--:|--|--|--|
| case30 | 53 | 0.05/0.01/0.00 = **0.06** | 8.54/0.08/0.06 = 8.69 | **my 135×** |
| case118 | 181 | 0.09/0.03/0.01 = **0.14** | 8.79/0.10/0.07 = 8.96 | **my 66×** |
| case1197 | 2392 | 0.72/0.13/0.09 = **0.94** | 11.61/0.22/0.11 = 11.94 | **my 13×** |
| case_ACTIVSg2000 | 3607 | 1.82/2.20/0.19 = **4.21** | 13.87/0.65/0.27 = 14.79 | **my 3.5×** |
| case3012wp | 5725 | 2.96/1.30/0.56 = **4.83** | 15.66/0.47/0.22 = 16.36 | **my 3.4×** |
| case6468rte | 12643 | 8.35/3.40/0.60 = **12.35** | 24.11/0.61/0.29 = 25.01 | **my 2.0×** |
| case8387pegase | 14908 | 8.53/4.09/1.45 = **14.08** | 28.44/1.11/0.36 = 29.92 | **my 2.1×** |
| case_ACTIVSg25k | 47246 | 26.38/21.51/7.85 = **55.74** | 62.63/1.88/0.67 = 65.18 | **my 1.2×** |
| case_SyntheticUSA | 156255 | 100.99/97.50/33.47 = 231.96 | 190.84/5.08/1.44 = **197.36** | cuDSS 1.2× |

**End-to-end: mysolver wins 8/9** (2–135×), loses only the 156k matrix by 1.2×.
The win comes from analysis: cuDSS's analysis is 8.5–191 ms (host reorder + GPU
setup/transfer) vs mysolver's 0.05–101 ms. **mysolver has the best accuracy
everywhere** (berr ~1e-16 vs cuDSS 1e-10…1e-14 on several).

## Factor phase (where cuDSS leads) + the GPU multifrontal research result

cuDSS's GPU numeric factor beats mysolver's CPU factor, growing with size (5.6×
case6468rte → 19× SyntheticUSA). The cycle-81 **GPU multifrontal** factor closes
this (METIS, kernel ms):

| matrix | mysolver CPU | cy71 GPU | **cy81 MF GPU** | cuDSS | MF vs cuDSS |
|---|--:|--:|--:|--:|--:|
| case6468rte | 3.40 | 4.45 | **0.89** | 0.61 | 1.45× behind |
| case8387pegase | 4.09 | 6.29 | **1.40** | 1.11 | 1.26× behind |
| case_ACTIVSg25k | 21.51 | 22.0 | **6.07** | 1.88 | 3.2× behind |
| case_SyntheticUSA | 97.50 | 52.6 | **23.6** | 5.08 | 4.6× behind |

GPU MF is a real **2.2–5× over cy71**, narrowing the factor gap vs cuDSS from
5–19× (CPU) to **1.3–4.6×**. It does NOT yet beat cuDSS on the factor kernel
(cy81's "2.8× faster" used a stale 2.5 ms cuDSS ref; the real value is 0.61 ms).

**Projected end-to-end with the GPU MF factor swapped in** (CPU analysis + MF
factor + CPU solve): case6468rte 9.8 ms (**2.5×** vs cuDSS 25.0), ACTIVSg25k
40.3 (**1.6×**), **SyntheticUSA 158 ms — flips to a 1.25× WIN** vs cuDSS 197.
=> with MF integrated, mysolver projected to win end-to-end on all 9.

## Remaining gaps
1. **GPU MF factor** still 1.3–4.6× behind cuDSS's kernel (large worse) — cy82+.
2. **Large-matrix SOLVE**: mysolver CPU solve is slow (SyntheticUSA 33 ms vs cuDSS
   1.4 ms, 23×); gpu_solve exists but isn't on the production large-path yet.
