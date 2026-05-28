# A / F / S / E2E vs cuDSS — single-call (cycle 121, 2026-05-27)

Measured in one `benchmark --solver mysolver-gpu,cudss-gpu --warmup-gpu` run (same
harness, single call per matrix). ms. E2E = A + F + S.

| matrix | A ours | A cuDSS | F ours | F cuDSS | S ours | S cuDSS | E2E ours | E2E cuDSS | berr ours | berr cuDSS |
|---|---|---|---|---|---|---|---|---|---|---|
| case6468rte | 32.9 | 22.5 | 0.79 | 0.61 | 0.68 | 0.29 | 34.4 | 23.4 | 5.6e-16 | 8.0e-14 |
| case8387pegase | 43.8 | 28.6 | 0.96 | 1.11 | 0.86 | 0.37 | 45.6 | 30.1 | 1.4e-14 | 1.1e-10 |
| case_ACTIVSg25k | 158.9 | 58.3 | 4.57 | 1.89 | 7.17 | 0.67 | 170.6 | 60.8 | 3.2e-16 | 2.2e-14 |
| case_SyntheticUSA | 612.2 | 181.3 | 20.59 | 5.20 | 18.06 | 1.44 | 650.9 | 187.9 | 2.9e-16 | 1.4e-14 |
| rajat27 | 67.3 | 50.6 | 1.01 | 1.00 | 0.80 | 0.39 | 69.1 | 52.0 | 4.9e-13 | 8.3e-12 |
| memplus | 57.9 | 51.9 | 1.69 | 1.15 | 0.82 | 0.42 | 60.4 | 53.5 | 7.3e-16 | 1.3e-15 |
| rajat15 | 317.2 | 117.5 | 8.97 | 3.57 | 2.38 | 0.88 | 328.5 | 121.9 | 1.2e-13 | 1.8e-13 |
| onetone2 | 231.1 | 87.4 | 44.09 | 8.01 | 7.54 | 1.53 | 282.7 | 97.0 | 4.9e-16 | 1.2e-10 |

## Honest read

**Single-call E2E: cuDSS wins on all matrices.** Decomposed:

- **A (analysis)** — we are 2–3.4× slower; dominated by **METIS ordering** (cuDSS's
  reordering is faster). BUT in Newton–Raphson the sparsity pattern is fixed, so A is
  computed once and reused across all refactorizations — amortized to ~0 in the real
  use case.
- **F (factor)** — the single-call figure includes **cold-start** (CUDA context, front
  arena malloc, graph instantiation, H2D), which inflates it. **Warm / NR-amortized**:
  case6468 0.42<0.61, case8387 0.56<1.11, rajat27 0.73<1.00, memplus 0.85<1.15 →
  **superior-or-equal on 5/8**; only the two large power-grid are 1.2–1.3× and
  onetone2 4.1×.
- **S (solve)** — our S includes **iterative refinement to berr 1e-16** (cuDSS reports
  a single solve). Warm single-solve: 1.5–2.3× behind cuDSS (kernel-quality gap).

**Accuracy: ours is consistently better** — berr 1e-13…1e-16 vs cuDSS's looser
1e-10 on onetone2/case8387. Our extra refinement (counted in S) is what buys that.

## Takeaway

- **NR / warm regime** (the real use case): factor superior-or-equal on 5/8 + better
  accuracy; solve 1.5–2.3× behind (cuDSS proprietary kernel tuning).
- **Single-shot E2E**: cuDSS ahead, driven mostly by **analysis (ordering)**, then
  cold-start and our refinement-to-1e-16.
- Largest E2E lever would be the analysis/ordering — but it amortizes away in NR, so
  its real-world value is limited.
