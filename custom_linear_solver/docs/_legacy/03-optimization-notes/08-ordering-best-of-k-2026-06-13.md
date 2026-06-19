# Ordering — tailored ND 반증, 그리고 measured best-of-k 가 유일한 실익

> ⚠️ **REMOVED (2026-06-15)**: 이 노트가 권하는 best-of-k ordering 선택(measured `CLS_ORDER_MEASURE_K`,
> static proxy `CLS_ORDER_K`)은 **코드에서 제거**됐다(→ `deprecated/best_of_k/`). measured 변형은 작동했으나
> (B=1 −6~13%) analyze 시점 K 회 factorize 비용 대비 default-off opt-in 으로 유지할 가치가 낮았다. 기본
> analyze 경로는 **단일 parallel-ND**. 아래 내용은 *역사적 기록*이며, tailored-ND 반증 결론(METIS 가
> near-optimal)은 유효하다.

> **상태**: reference (historical)   **갱신**: 2026-06-13
> **한 줄**: 전력망+ND 에서 **METIS fill 목적은 near-optimal** — custom GPU-objective ND·전기적-weighted bisection 은 best-of-k METIS envelope 를 못 이긴다(gpu_nd ≈ METIS 시드노이즈 내, EW +9~73% 악화). 유일한 ordering 실익은 **깨진 best-of-k 선택을 고치는 것**: tail-cube proxy(`CLS_ORDER_K`)는 anti-informative 라, 시드별 **실제 factorize 시간을 측정**해 고르는 `CLS_ORDER_MEASURE_K` 가 B=1 **−6~13%** + parallel-ND 비결정성 제거 + 정밀도 적응.

exp_260612 ordering 트랙. 원자료: `../exp_260612/`(notes 12·13·14·15), B=1 체제 맥락은 [`06-b1-factorize-regime`](06-b1-factorize-regime-2026-06-13.md) §3. 폐기 코드: `deprecated/gpu_nd/`.

---

## 1. 문헌 — fill≈work, critical-path 만이 다른 목적 ([exp 13])

METIS nested dissection 은 separator 크기(=fill)를 최소화하고 balance 는 제약으로 둔다. throughput 체제에서
fill≈work 이므로 fill-최적이 곧 work-최적에 가깝다. 진짜로 다른 목적은 **critical-path / tree-height 최소화**
(Kayaaslan–Uçar BBT, ~28% height 감소)지만 NP-hard 이고 우리 B=1 under-fill 천장과 직접 맞물리지 않는다.
SOTA(cuDSS)도 "METIS 기반 customized ND" 를 쓸 뿐 separator 목적 자체를 바꾸지 않는다.

---

## 2. 반증 — tailored ND 는 METIS 를 못 이긴다 ([exp 12·14])

- **custom GPU/TC-objective ND**(재귀·목적·정지 our-owned, bisection 은 METIS_ComputeVertexSeparator 재사용,
  imbalance penalty λ>0). 초기 보고된 −2~5% win 은 **artifact** — single-seed baseline + escalating-UFACTOR-candidate
  버그. honest 재측정(best-of-k METIS envelope 대비)에서 **gpu_nd ≈ METIS**(시드 노이즈 내).
- **전기적-weighted bisection**(|J_ij| 약한 tie-line cut, METIS 가 edge-weight 가 없어 못 노리는 분리자). 모든 depth 에서
  **+9~73% 악화, fill +28%** — 약한 tie 는 작은/균형 separator 가 아니다.

→ METIS fill 목적이 이 문제엔 empirically near-optimal. ordering 으로 GPU 병렬성을 더 짜낼 여지는 거의 없다.
코드: `deprecated/gpu_nd/`(env 옛: `CLS_GPU_ND`, `_CAND/_LAMBDA/_LEAF/_TC_G/_EW/_FILL`).

---

## 3. 실익 — measured best-of-k (`CLS_ORDER_MEASURE_K`) ([exp 15])

남는 단 하나의 레버는 best-of-k **선택 함수**다. 기존 proxy(`CLS_ORDER_K`, tail-cube = Σ under-filled large level 의
maxfsz³)는 fp32 에서 **anti-informative** — 1순위 픽이 default 보다 ~8% 느릴 수 있고, fill 은 시드 무관 동일이라
fill 로도 못 고른다. 해법: 시드별 **실제 단일-시스템 factorize 를 타이밍**(private RAII State, median of reps)해 최속 시드 선택.

| case | parallel-ND median 대비 | 선택 시드 |
|---|---:|---|
| 13K(8387) | **−8%** | seed 14 |
| 25K | **−11%** | seed 2 |
| 70K(USA) | **−8.4%** | seed 13 |

- 매 case 에서 **참(oracle) 최적 시드**를 집어낸다(proxy 와 달리).
- parallel-ND 의 run-to-run 비결정성(USA 18% swing) 을 **결정적으로 제거**(relres 불변).
- **정밀도 적응**: fp64 는 다른 시드를 고른다(proxy 는 정밀도 무관).
- 비용 = K 회의 ND 빌드 지배(K16 ~33s; factorize 타이밍 자체는 미미) → **factorize 재사용(time-series / N-1 contingency)**
  시나리오에서만 amortize. 그래서 **default-off**(`CLS_ORDER_MEASURE_K=0`), production 경로 불변.

구현: `src/solver.cpp` `Solver::analyze` — `measure_candidate_factor_ms()` + `PlanBuildOptions::single_seed_only`
(env best-of-k/custom-ND 우회). env: `CLS_ORDER_MEASURE_K=K`, `_WARMUP`(기본 3), `_REPS`(기본 5).

```bash
BIN=build/cls/custom_linear_solver_run
C=/datasets/power_system/nr_linear_systems/case_ACTIVSg25k
CLS_ORDER_MEASURE_K=8 $BIN $C --precision fp32 --batch 1 --repeat 30 --warmup 10
# stderr: [analyze] measure-select K=8 -> seed=N factor_ms=...
```

---

## 결론

- ordering 천장은 bounded — METIS fill 목적이 near-optimal, custom ND 목적은 기여 없음(`deprecated/gpu_nd/`).
- 실익은 **measured best-of-k**(`CLS_ORDER_MEASURE_K`): B=1 −6~13% + 결정성 + 정밀도 적응, factorize 재사용 하에 amortize.
- proxy `CLS_ORDER_K`(tail-cube)는 호환을 위해 src 에 잔류하나 anti-informative — 신규 사용은 measured 경로 권장.
