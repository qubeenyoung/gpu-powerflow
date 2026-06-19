# 최적 구성 (Optimal Configuration)

> **상태**: canonical   **갱신**: 2026-06-18
> **한 줄**: 현재 최적 경로의 빌드/런타임 설정과 남아있는 토글 매핑을 한곳에.

메소드 자체의 설명은 [`main-report.md`](main-report.md), 세부 측정은
[`03-optimization-notes/`](03-optimization-notes/) 참조.

> **이전 버전과의 차이**: 예전에는 regime별로 실험 플래그를 골라 켜야 했지만, 검증된 레버는 모두
> **코드에 baked-in** 되었다 — TF32 텐서코어 trailing(mid blocked / big panel·thin-K), fused trail+extend,
> 그리고 케이스 크기별 panel width(소형 8 / 대형 16)는 이제 기본 동작이다. 따라서 최적 설정은
> **`--precision tf32` + Ozaki 보정**으로 단순해졌고, regime별 플래그 번들은 사라졌다.

---

## 1. 권장 경로

```bash
cmake -S custom_linear_solver -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DCLS_TF32_OZAKI_TC2=ON -DCLS_TF32_OZAKI_TC2_FIRST_ORDER=ON
cmake --build build -j

build/custom_linear_solver_run <case> --precision tf32 --batch 64 \
  --repeat 61 --warmup 8 --single-precision fp64
```

- **정밀도/정확도**: `--precision tf32` (front=FP32, trailing=TF32 텐서코어) **+ Ozaki first-order 보정**
  (`CLS_TF32_OZAKI_TC2[_FIRST_ORDER]`). → relres 가 FP32 band(~1e-4..1e-5)이면서 fp32 대비 빠름.
  Ozaki 미적용 raw tf32 는 더 빠르지만 relres 가 ~1e-2 로 떨어진다.
- **배치**: `--batch 64`(latency) ~ `256`(throughput).
- **panel width**: 자동 — 분석기가 `n≥16k → 16`, 그 외 `--max-panel-width`(기본 8)를 쓴다(결정적 스윕
  확정값, [`05-reports/05-tf32-reproduction-2026-06-10.md`](05-reports/05-tf32-reproduction-2026-06-10.md) §8).
  사용자 값을 그대로 강제하려면 `-DCLS_RESPECT_PANEL_CAP=ON`.
- **항상 baked-in**: no-pivot, CUDA graph, METIS-ND, **3-tier 결정적 라우팅**(small/mid/big —
  front 크기·정밀도만으로 전용 커널 결정; 경계는 워프 32(small|mid) / 점유 교차점 64(mid|big), `src/internal/types.hpp`
  `kSmallFrontMax`/`kMidFrontMax`). small tier 만 GPU under-fill 시 whole-front 커널(FactorMid)로 돌리는
  occupancy 게이트가 붙는다(좁은 레벨). **정밀도는 티어와 직교** — TF32 모드면 mid/big 가 TF32 TC trailing
  (Ozaki 시 FP32-band)을 탄다(small tier 는 sub-group 패킹 scalar). 멀티스트림 서브트리, 동형 디스패치,
  **TF32 TC trailing**, **fused trail+extend**, **register-blocked trailing**(`CLS_TRAIL_RB=4`),
  **sync-free U-solve / row-fused panel LU**. (옛 4-tier 의 panel-resident big 티어는 2026-06-18 통합에서
  제거 — [`05-reports/10-tier-consolidation-2026-06-18.md`](05-reports/10-tier-consolidation-2026-06-18.md).)
- **B=1 가속 레버**(opt-in): `--precision tf32` + Ozaki(B=1 latency-bound → TC 가 critical path 단축,
  USA −17%, [`03-optimization-notes/06-b1-factorize-regime-2026-06-13.md`](03-optimization-notes/06-b1-factorize-regime-2026-06-13.md)).
  (ordering 선택 best-of-k 는 ROI 부족으로 2026-06-15 제거 — `deprecated/best_of_k/`. 기본 단일 parallel-ND.)

fp64(`--precision fp64`, 참조 정확도 ~1e-13)·fp32(`--precision fp32`)도 동일 빌드에서 선택 가능하다.

---

## 2. 남아있는 토글

| 토글 (CMake -D / CLI) | 기본 | 역할 |
|---|---|---|
| `--precision {fp64,fp32,tf32}` | fp64 | 정밀도 regime (tf32 = TC trailing) |
| `--batch N` | 1 | 배치 (B 시스템 동시 factor/solve) |
| `--max-panel-width N` | 8 | supernode 패널 최대 열 수 (분석기가 대형은 16으로 자동 상향) |
| `--serial-nd` / `--metis-seed S` | (parallel) | ND 순서 결정성 (재현/벤치용) |
| `--no-multistream` | (multi=on) | 멀티스트림 서브트리 디스패치 비활성 |
| `CLS_TF32_OZAKI_TC2` / `..._FIRST_ORDER` | OFF | TF32 Ozaki 정확도 회복(권장 ON). first-order = tail-tail 항 생략 |
| `CLS_TRAIL_RB` | 4 | register-blocked trailing 폭 {0=scalar,2,4,8} |
| `CLS_RESPECT_PANEL_CAP` | OFF | 분석기 자동 상향 대신 `--max-panel-width` 를 그대로 사용 |
| `CLS_PAR_ND_DEPTH` / `_SMALL_BASE_THR` / `_LARGE_BASE_THR` | 4 / 4000 / 20000 | 병렬 ND 튜닝 |
| `CLS_INTERNAL_GRAPH` | ON | factor/solve 를 내부 CUDA graph 로 캡처(standalone). OFF = 외부 캡처용 |

> baked-in 되어 더 이상 토글이 아닌 것: TF32 TC trailing(mid/big), fused trail+extend, 케이스 크기별
> panel width, **register-blocked trailing · sync-free U-solve**. 음성으로 판명되어 `deprecated/` 로 옮겨진
> 실험(panel-resident big 커널, gather assembly, deep-K amalgamation, custom GPU-ND, best-of-k, tiled-trailing,
> mid-fewsync/sysblk)은 [`05-reports/10-tier-consolidation-2026-06-18.md`](05-reports/10-tier-consolidation-2026-06-18.md),
> [`03-optimization-notes/07-batch-factorize-structural-2026-06-13.md`](03-optimization-notes/07-batch-factorize-structural-2026-06-13.md)
> §negative, `deprecated/README.md` 참조.

---

## 3. 검증된 수치 (sm_86, RTX 3090)

panel width 가 케이스당 확정(소형 8 / 대형 16)되고 TC 가 기본이므로, **기본 빌드가 곧 공정 비교(best-vs-best)
설정**이다 — 예전처럼 cap 을 한쪽으로 부풀려 얻는 headline 수치는 더 이상 없다.

| case | n | tf32+Ozaki vs fp32 (B≥16, 공정 cap) | tf32 relres |
|---|---:|---|---|
| case8387pegase | 14,908 | ≈ 동률 (low-fill, TC 구조적 무이득) | FP32 band |
| case_ACTIVSg25k | 47,246 | +5~7% | ~2e-4 |
| 70k / SyntheticUSA | ≥156k | +10~16% (B=1 피크) | conditioning floor |

> **정직한 천장**: TC 자체 기여는 best-vs-best 중앙값 **~1.1×** 이며, ≤10K low-fill 에서는 per-front TC 가
> 구조적으로 이득이 없다(K=nc 가 1~2). 자세한 분석·원자료는
> [`05-reports/05-tf32-reproduction-2026-06-10.md`](05-reports/05-tf32-reproduction-2026-06-10.md) 와
> [`03-optimization-notes/03-tensor-core-investigation.md`](03-optimization-notes/03-tensor-core-investigation.md).
