# 2026-06-12 Lab Meeting

오늘(2026-06-11) 재측정한 **결정적**(Ozaki TF32 + serial-ND seed 1588 + GPU 클럭 고정) 데이터 기준.
이전 2026-06-10 스냅샷은 새 자료로 대체했고, 별도 old 사본은 두지 않는다.

## 보고서 (오늘 측정)

| 보고서 | 내용 |
|---|---|
| [**06 — 선형계 단독 스윕**](../05-reports/06-cudss-vs-custom-sweep-2026-06-10.md) | J·x=b 단독. cuDSS(fp64/fp32) vs custom(fp64/fp32/tf32), 6 case × B=1/4/16/64/256, **analyze/factorize/solve** (solve 최적화 반영 재측정). 원자료 [`sweep.tsv`](../05-reports/06-cudss-vs-custom-sweep-2026-06-10/sweep.tsv) |
| [**07 — cuPF 전체 조류계산**](../05-reports/07-cupf-backend-comparison-2026-06-11.md) | NR 조류계산(graph off). fp64/mixed 에서 cuDSS vs custom(mixed factor fp32/tf32), 6 case × B=1/16/64/256. **수렴 반복수·init·solve·배치·연산자별**. 원자료 [`b1_init_solve_perop.tsv`](../05-reports/07-cupf-backend-comparison-2026-06-11/b1_init_solve_perop.tsv) · [`batch_scaling.tsv`](../05-reports/07-cupf-backend-comparison-2026-06-11/batch_scaling.tsv) · [`operator_ms_full.tsv`](../05-reports/07-cupf-backend-comparison-2026-06-11/operator_ms_full.tsv) |

## 핵심 (발표용)

- **배치(B≥16)에서 custom(mixed)이 cuDSS 대비 ~4–6×** (25K B=64: custom ~800μs vs cuDSS-fp64 4885μs).
- **mixed(FP32 step + FP64 state)는 fp64와 동일 수렴**(4–8 iters); **tf32(Ozaki)≈fp32**(factorize 1.05–1.2× TC 이득이 NR 전체에선 희석).
- **custom analyze ~1.4–2× 빠름** (METIS-ND + 멀티프론탈 심볼릭).
- 연산자별: **factorize ~45–55% + triangular-solve ~12–15%** 가 선형해(custom이 cuDSS보다 작음), 나머지(jacobian/ibus/mismatch/upload/download)는 백엔드 공통.

## 발표 문서

- [**결과 요약 — 케이스별 (공정 비교)**](results-summary.md) — 같은 정밀도 cuDSS 와 비교: 선형계 custom factorize **4.4–8.4×** vs cuDSS-fp32, 전체 조류계산 custom-mixed **B=64 3.6–4.5×** vs cuDSS-mixed. tf32≈fp32. (fp64 대비 수치는 cuDSS 의 fp32/mixed 이득 1.4–1.8× 가 섞여 부풀려짐.)
- [**우리 문제의 elimination-tree 특성**](etree-characteristics.md) — ① front 99%가 fsz≤32(CDF+통계표), ② 레벨별 front tier·max_uc 표(8387·70K, [`etree_level_8387`](figures/etree_level_8387.png)/[`70K`](figures/etree_level_70K.png)), ③ **레벨별 factorize 시간**(EXP_260611, **fp32 기준**+tf32 대조) — 개수는 작은 게 많고 시간은 상위 큰 front 가 지배. **tf32 한정 L25 spike(10%)는 uc>256 TF32 경로 이탈**(fp32 5.6%로 검증).
- [**왜 small tier 는 텐서코어 가속 불가**](small-tier-no-tensorcore.md) — trailing GEMM 4×4×2급·수십 FLOP, Ampere 텐서코어 최소 타일 16×8×8 대비 ~3% 채움(Hopper M=64·Blackwell 256은 더 불리).
- [**Fill-in · 메모리 — custom vs cuDSS**](fill-in-memory-vs-cudss.md) — factor L+U nnz 는 동급(둘 다 METIS-ND, fill/nnz 2–3×), device 메모리는 **custom 이 2–10× 적음**(cuDSS 워크스페이스 오버헤드). cuDSS 는 `cudssDataGet(LU_NNZ/MEMORY_ESTIMATES)` 쿼리로 측정.
- [**factorize 병목 — ncu 레벨·티어별 (B=1/16/64)**](factorize-bottleneck-ncu.md) — small/mid/big 커널을 레벨별로 occupancy·TC·DRAM·L1·L2 프로파일. **B=1=latency/under-fill, B≥16=메모리 대역폭 bound**, occupancy 23–33%(1 block/SM)·starved TC 는 공통. ncu metric 정식 이름·의미 포함.

## 연산자별 파이차트 (cuPF mixed, B=1)

`figures/operator_pies/<config>/<case>_B1_operator_pie.png` — solve 1회의 연산자 분해.
범위: **mixed 프로파일만**(cuDSS vs custom fp32/tf32), B=1. 생성: `scripts/plot_operator_pies.py` (입력 = 07의 `operator_ms_full.tsv`).
MATPOWER(C++) 기준 B=1 연산자 분해는 [`figures/tutorial_cpp_operator_pies/`](figures/tutorial_cpp_operator_pies/) 에
같은 타이틀 형식으로 추가했다.

| config | 케이스 (각 B=1) |
|---|---|
| [`cudss-mixed`](figures/operator_pies/cudss-mixed/) | 3xxx · 6xxx · 8xxx · 13K · 25K · usa |
| [`custom-mixed-fp32`](figures/operator_pies/custom-mixed-fp32/) | 3xxx · 6xxx · 8xxx · 13K · 25K · usa |
| [`custom-mixed-tf32`](figures/operator_pies/custom-mixed-tf32/) | 3xxx · 6xxx · 8xxx · 13K · 25K · usa |
| [`matpower(cpp)`](figures/tutorial_cpp_operator_pies/) | 3xxx · 6xxx · 8xxx · 25K · usa |

예: `figures/operator_pies/custom-mixed-tf32/case_ACTIVSg25k_B1_operator_pie.png` (25K: factorize 51% · tri-solve 32%).

## 재생성

```bash
python3 gpu-powerflow/custom_linear_solver/docs/20260612_lab_meeting/scripts/plot_operator_pies.py
python3 gpu-powerflow/custom_linear_solver/docs/20260612_lab_meeting/scripts/plot_tutorial_cpp_operator_pies.py
```
