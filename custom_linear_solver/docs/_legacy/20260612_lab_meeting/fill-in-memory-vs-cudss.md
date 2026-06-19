# Fill-in · 메모리 — custom vs cuDSS

> **상태**: reference   **갱신**: 2026-06-11
> **한 줄**: custom(멀티프론탈)과 cuDSS 의 factor fill-in(L+U nnz)은 거의 동일(둘 다 METIS-ND)하지만, **custom 의 device 메모리가 2–10× 적다** — cuDSS 의 워크스페이스 오버헤드가 크기 때문.

## 측정 방법

- **custom**: analyze 가 만든 front 구조(`data/fronts_*.csv`: fsz, nc)에서 직접 계산.
  - factor fill (L+U nnz) ≈ `Σ nc·(2·fsz − nc)` (front 별 dense L/U 패널).
  - dense-front 저장 = `front_total = Σ fsz²` (`analyze/plan/lower.cu:152`), 메모리 = `front_total × 4 B`(fp32).
- **cuDSS**: `tests/run_cudss_solver.cpp` 에 **`cudssDataGet(CUDSS_DATA_LU_NNZ)` · `cudssDataGet(CUDSS_DATA_MEMORY_ESTIMATES)` 쿼리 추가** → factorization 후 출력(빌드 타깃 `cudss_run`, `-DCLS_BUILD_CUDSS_SCRIPT=ON`). device 메모리 = MEMORY_ESTIMATES[0](permanent), [1]=peak.

## Fill-in (factor L+U nnz)

| case | n | nnz(J) | custom fill | cuDSS LU_NNZ | custom/cuDSS | fill/nnz |
|---|---:|---:|---:|---:|---:|---:|
| case3012wp | 5,725 | 36,263 | 70,885 | 69,789 | 1.02× | 1.95× |
| case6468rte | 12,643 | 87,845 | 168,105 | 160,245 | 1.05× | 1.91× |
| case8387pegase | 14,908 | 110,572 | 216,346 | 208,734 | 1.04× | 1.96× |
| case_ACTIVSg25k | 47,246 | 318,672 | 876,800 | 850,278 | 1.03× | 2.75× |
| case_SyntheticUSA | 156,255 | 1,052,085 | 3,310,313 | 3,020,747 | 1.10× | 3.15× |

→ **fill-in 거의 동일**(custom 이 2–10% 더 많음). 둘 다 METIS-ND 라 ordering 이 비슷. **fill/nnz 2–3×** — 전력망 그래프가 planar-ish 라 fill 이 낮다(좋음).

## 메모리 (B=1, fp32, device)

| case | custom front arena | cuDSS device (peak) | custom/cuDSS |
|---|---:|---:|---:|
| case3012wp | 0.6 MB | 5.9 MB | **0.10×** (10× 적음) |
| case6468rte | 1.5 MB | 9.3 MB | 0.17× |
| case8387pegase | 2.1 MB | 11.5 MB | **0.18×** (5.5× 적음) |
| case_ACTIVSg25k | 8.8 MB | 29.6 MB | 0.30× |
| case_SyntheticUSA | 39.2 MB | 77.6 MB (93.6 peak) | 0.51× (2× 적음) |

→ **custom 이 2–10× 적은 device 메모리**. 작은 케이스일수록 차이 큼(cuDSS 고정 워크스페이스 오버헤드) → 규모 커지면 factor 가 지배해 ~2× 로 수렴.

## 분석

- **fill-in 은 동급** — custom 의 dense-front 저장(`Σfsz²`)은 구조적 nnz 의 ~2.4×(명시적 0 포함)지만, **실제 factor nnz 는 cuDSS 와 거의 같다**(METIS-ND).
- **메모리는 custom 이 훨씬 tight**: 예) 8387 에서 cuDSS device 11.5 MB 는 자기 factor nnz×4(0.83 MB)의 **~14×**(워크스페이스), custom front arena 2.1 MB 는 nnz×4 의 **~2.4×**. → cuDSS 는 reordering/solve 워크스페이스가 큰 반면 custom 은 dense-front 만 들고 있다.
- custom 보조버퍼(CSR matrix, perm, asm 맵)는 O(nnz) 로 작아 front arena 가 지배 — 합쳐도 cuDSS 미만.
- **배치**: 둘 다 값 저장이 ×B(custom front arena×B, cuDSS factor 값×B); 패턴/심볼릭은 공유. 위 비율은 B=1 기준.

## 배치별 메모리 스케일링 (device, fp32)

| B | 8387 cuDSS | 8387 custom | 25K cuDSS | 25K custom |
|---|---:|---:|---:|---:|
| 1 | 11.5 MB | 2.1 MB | 29.6 MB | 8.8 MB |
| 16 | 136 MB | 33.5 MB | 327 MB | 140 MB |
| 64 | 535 MB | 134 MB | 1,281 MB | 562 MB |
| 256 | **2,129 MB** | **536 MB** | **5,094 MB** | **2,248 MB** |

- **둘 다 ~선형 증가** — per-system 메모리가 배치서 일정해진다:
  - custom per-sys = `front_total×4` = **2.1 MB(8387) / 8.8 MB(25K)**, 정확히 일정(front arena 가 batch-major 로 ×B).
  - cuDSS per-sys → **~8.4 MB(8387) / ~20 MB(25K)** 로 수렴 (B=1 의 고정 워크스페이스가 배치서 희석).
- **배치서도 custom 2.3–4× 적음**(8387 4.0×, 25K 2.3×). B=1 의 10× 는 cuDSS 고정 오버헤드 탓이고, 배치선 factor 값이 지배해 비율이 안정화.

**GPU 용량 한계** (RTX 3090 = 24 GB) — 들어가는 최대 배치:

| case | custom | cuDSS | custom 우위 |
|---|---:|---:|---|
| 8387 | ~11,000 | ~2,800 | **~4× 큰 배치** |
| 25K | ~2,700 | ~1,200 | ~2.3× |

→ custom 의 leaner per-system 메모리가 배치서 그대로 유지돼 **같은 GPU 에 2–4× 큰 배치**를 올린다.

## 한 줄

> **같은 ordering(METIS-ND)으로 fill-in 은 동급인데, custom 은 dense-front 만 들어 cuDSS 대비 device 메모리를 B=1 에서 2–10×, 배치에서 2–4× 적게 쓴다 → 같은 GPU 에 2–4× 큰 배치.**
