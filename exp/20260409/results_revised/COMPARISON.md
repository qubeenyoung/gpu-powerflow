# cuDSS Full vs Schur Complement 벤치마크

**Date**: 2026-04-09  
**GPU**: NVIDIA (cuDSS) | **CPU**: AMD EPYC 7313  
**Setup**: warmup=3, repeats=10, CUDA events 타이밍, FP32, edge_based Jacobian

---

## Full mode 단계별 (ms, mean)

| case | n_pq | analysis | factor | solve | **no_analysis** |
|------|-----:|---------:|-------:|------:|----------------:|
| case118_ieee | 64 | 15.59 | 0.07 | 0.05 | **0.12** |
| case793_goc | 704 | 19.01 | 0.12 | 0.08 | **0.19** |
| case1354_pegase | 1094 | 21.04 | 0.17 | 0.12 | **0.29** |
| case2746wop_k | 2396 | 26.46 | 0.29 | 0.15 | **0.43** |
| case4601_goc | 4468 | 36.00 | 0.45 | 0.21 | **0.65** |
| case8387_pegase | 6522 | 51.32 | 0.55 | 0.23 | **0.78** |
| case9241_pegase | 7796 | 59.48 | 0.57 | 0.23 | **0.81** |

---

## Schur mode 단계별 (ms, mean)

| case | n_pq | analysis | factor | extract | fwd | getrf | getrs | bwd | **no_analysis** |
|------|-----:|---------:|-------:|--------:|----:|------:|------:|----:|----------------:|
| case118_ieee | 64 | 15.32 | 0.08 | 0.01 | 0.03 | 0.09 | 0.03 | 0.04 | **0.28** |
| case793_goc | 704 | 24.53 | 0.41 | 0.44 | 0.06 | 1.52 | 0.21 | 0.12 | **2.76** |
| case1354_pegase | 1094 | 29.77 | 0.72 | 1.39 | 0.08 | 2.77 | 0.31 | 0.18 | **5.45** |
| case2746wop_k | 2396 | 48.22 | 2.98 | 6.07 | 0.10 | 7.30 | 0.78 | 0.31 | **17.55** |
| case4601_goc | 4468 | 90.88 | 9.70 | 20.75 | 0.14 | 18.55 | 1.43 | 0.53 | **51.09** |
| case8387_pegase | 6522 | 147.44 | 14.39 | 45.13 | 0.18 | 37.08 | 2.25 | 0.61 | **99.65** |
| case9241_pegase | 7796 | 183.02 | 16.87 | 63.96 | 0.19 | 53.62 | 2.84 | 0.59 | **138.07** |

> `diag` (cudaMemcpyAsync x→b, 0.004ms 이하) 생략.  
> `factor` in Schur = partial LU(J11) + dense Schur complement 계산 포함.

---

## Full vs Schur 비교

| case | n_pq | full no_ana (ms) | schur no_ana (ms) | schur/full |
|------|-----:|-----------------:|------------------:|-----------:|
| case118_ieee | 64 | 0.12 | 0.28 | **2.3×** |
| case793_goc | 704 | 0.19 | 2.76 | **14.3×** |
| case1354_pegase | 1094 | 0.29 | 5.45 | **19.0×** |
| case2746wop_k | 2396 | 0.43 | 17.55 | **40.5×** |
| case4601_goc | 4468 | 0.65 | 51.09 | **78.4×** |
| case8387_pegase | 6522 | 0.78 | 99.65 | **127.8×** |
| case9241_pegase | 7796 | 0.81 | 138.07 | **171.5×** |

Analysis 포함 시: Schur analysis가 full의 2.5–3.1× → 격차는 다소 줄지만 Schur가 여전히 느림.

---

## Schur 병목 분석 (case9241, n_pq=7796)

```
factorize  ████████████████████  16.87ms  (partial LU + Schur 행렬 계산)
extract    ████████████████████████████████████████  63.96ms  (7796×7796 dense 복사)
getrf      █████████████████████████████████████  53.62ms  (dense LU O(n_pq³))
getrs      ██  2.84ms
fwd+bwd    █  0.78ms
─────────────────────────────────────────────────────────
total      138.07ms  vs  full 0.81ms  →  171.5× slower
```

extract(O(n_pq²) data) + getrf(O(n_pq³) compute) 가 전체의 85%를 차지.  
n_pq가 커질수록 이 두 항이 지배하므로 격차는 n_pq와 함께 계속 커진다.

---

## 결론

Schur complement는 n_pq가 전체 dim의 일부일 때 의미가 있다.  
전력조류 Jacobian에서 n_pq/dim = 35–97%이므로 Schur block이 작지 않고,  
dense 행렬 형성·인수분해 비용이 sparse 전체 풀이보다 압도적으로 크다.  
**cuDSS full sparse direct solve가 유일한 실용적 선택이다.**
