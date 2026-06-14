# Ordering 으로 critical-path 단축 — 민감도·비결정성·spine-work proxy (cap 튜닝 아님)

> **상태**: investigation (promising, 조건부)   **갱신**: 2026-06-12
> **한 줄**: B=1 factorize 시간이 ND ordering 에 **15–27% 좌우**되고(같은 ND 알고리즘, seed 만 다름 — cap 튜닝 아님), production parallel-ND 는 **비결정적**이라 run 마다 ~10% 흔들린다. 어느 seed 도 robust 하게 최선은 아니지만(case 별 lottery), **fill-free symbolic proxy `spine_work = Σ 상위레벨 fsz²` 가 best/near-best ordering 을 안정적으로 예측** — 그래서 **best-of-k-by-proxy 선택**(analyze 시 k 개 ordering 중 spine_work 최소 선택)이 성립한다. 단 이득은 **B=1 한정**(batch 는 under-fill 이 사라져 washes out)이고, k× analyze 비용은 **반복-solve 워크로드**(고정 grid·다수 시나리오)에서만 amortize 된다.

## 배경

목표 3 재설정(tiling·sync 둘 다 occupancy 벽으로 반증) 후, 남은 lever 는 kernel 밖 — **ordering 으로 critical path(spine) 단축**. B=1 factorize 시간의 35–49% 가 front≤4 인 spine(직렬 critical path, 소거트리 상위 separator 체인)이고, 그 spine 의 깊이·크기는 **ordering 이 정한다**. **`reorder-lab` 브랜치는 stale**(대규모 리팩토링 이전, 실제론 TC/spine *커널* 실험이지 ordering 아님) — 코드 재사용 불가. 현행 ordering = METIS nested dissection ([`src/analyze/reorder/metis_nd.cpp`](../../src/analyze/reorder/metis_nd.cpp)), **fill 최소화** 목적(critical-path 아님). cap 튜닝(panel-width amalgamation)은 정직성 문제(notes 54/55)로 제외.

## 측정 1 — ordering 민감도 (B=1, tf32, factorize_ms)

serial-ND(결정적) seed sweep:

| case | seed1 | seed7 | seed42 | seed1588 | **spread** |
|---|---:|---:|---:|---:|---:|
| 13659 | 0.400 | **0.345** | 0.437 | 0.397 | **27%** |
| 25k | 0.622 | 0.578 | 0.689 | **0.578** | **19%** |
| 70k | 1.944 | 1.712 | **1.695** | 1.797 | **15%** |

→ **같은 ND 알고리즘에서 seed 만 바꿔도 B=1 factorize 가 15–27% 변한다.** 큰 headroom. 단 **최선 seed 가 case 마다 다름**(13659=s7, 25k=s7/s1588, 70k=s42) → "좋은 seed 하나 고정"은 불가(lottery). cap 튜닝과 무관(ordering 자체).

## 측정 2 — parallel ND 는 비결정적

production 기본은 parallel ND(multi-thread). 같은 seed42 를 2회:

| case | run1 max_fsz | run2 max_fsz |
|---|---:|---:|
| 25k | 134 | 124 |
| 70k | 248 | 270 |

→ **parallel ND 는 run 마다 다른 ordering** 을 낸다(thread 별 `std::rand` race, [metis_nd.cpp:87] 주석이 시사). 즉 **production factorize 시간이 ordering 운으로 run 마다 ~10% 흔들린다** — 재현성 문제이자, 기본 ordering 이 종종 최선이 아님을 뜻함.

## 측정 3 — symbolic proxy 가 좋은 ordering 을 예측하나? (핵심)

best-of-k 가 성립하려면 **factorize 없이**(analyze 만으로) 좋은 ordering 을 골라야 한다. 후보 proxy 와 serial-seed factorize_ms 의 상관:

- **`max_fsz`(top separator) 단독 — 불충분**: 70k seed1 은 max_fsz **229(최소)**인데 factor 1.944(**최악**) — **levels=53(최심)** 이라 separator 가 작아도 트리가 깊어 critical path 가 길다. depth↔separator trade-off 를 못 잡음.
- **`spine_work = Σ 상위 8레벨 fsz²` — 잘 예측**:

| case | min(spine_work) 선택 seed | 그 factor_ms | true best | 판정 |
|---|---|---:|---:|---|
| 13659 | s7 (19,864) | 0.345 | 0.345 | **= 최선** |
| 25k | s1588 (29,731) | 0.578 | 0.578 | **= 최선(tie)** |
| 70k | s7 (109,247) | 1.712 | 1.695 | 최선의 **1% 이내** |

→ **spine_work(fill-free, analyze 산물만으로 계산)는 best/near-best ordering 을 안정적으로 집어낸다.** depth 와 separator 크기를 함께 반영(상위 레벨 fsz² 합 = 직렬 critical-path 작업량 근사)하기 때문. max_fsz 단독의 70k 실패를 교정.

## 제안 — critical-path-aware ordering 선택 (best-of-k by spine_work)

1. analyze 시 ND 를 **k 회** 실행(parallel ND 는 빠르고 비결정적이라 k 샘플이 자연스러움).
2. 각 ordering 의 **spine_work**(symbolic, factorize 불필요) 계산.
3. **min(spine_work) ordering 선택**, 이후 모든 solve 에 재사용.

- cap 튜닝과 다른 점: **유효한 ND ordering 들 중 명시적 critical-path proxy 로 선택**(공정 비교, baseline 인플레이션 없음). multi-start 최적화에 가깝다.
- 비결정성도 부수적으로 해결(proxy 로 결정적 선택).

## 정직한 ceiling 과 조건

- **이득 = B=1 한정**: 측정 1 의 15–27% 는 B=1. batch(B≥16)는 GPU 가 front×B 로 차 under-fill 이 사라져 critical-path 우위가 **washes out** → ordering 이득 거의 0(별도 검증 필요하나 구조상 명확).
- **production default(parallel seed42) 대비 현실 이득 ~7–14%**(25k 0.673→0.578, 70k ~1.60→~1.49) — worst-to-best(15–27%)가 아니라 default-to-best.
- **경제성**: best-of-k = k× analyze(70–183 ms/회). 단발 B=1 solve 엔 **절약(0.06–0.11 ms)의 ~1000× 손해**. **반드시 반복-solve 에서만 amortize**: 고정 grid topology 를 다수 시나리오(contingency·time-series·OPF loop)로 반복하면 ordering 은 **한 번** 계산·재사용 → 매 solve 가 공짜로 더 빠른 critical path. 이게 실제 power-flow 워크로드.
- **fill 리스크**: critical-path 가 좋은 ordering 이 fill(총 작업)을 키우면 batch/throughput 손해. spine_work 선택이 fill 을 얼마나 바꾸는지 동반 측정 필요.

## 평가

목표 3 의 네 lever 중 **처음으로 구조적 벽(occupancy/critical-path)에 즉시 반증되지 않고 validated 메커니즘(spine_work proxy)을 가진 방향**. 단 (a) B=1 한정, (b) 반복-solve regime 가정 필요, (c) 이득 modest(default 대비 7–14% B=1 factorize → NR solve 의 ~3–7%). cap-free·정직.

## 다음 (프로토타입 시 — 미착수)
1. spine_work proxy 를 analyze 에 계산(이미 front 크기·레벨 산출하므로 저비용).
2. best-of-k ND + 선택 구현(analyze 내, env-gated). k=4~8.
3. **end-to-end 반복-solve 시나리오**로 측정: default vs best-of-k 의 (per-solve factor_ms, one-time analyze 비용, fill nnz). B=1 과 B=64 둘 다 — batch washes out 가설 검증.
4. spine_work↔factor_ms 상관을 6+1 케이스 전수로 굳혀 proxy 일반성 확인.

## 재현
```bash
BIN=build/custom_linear_solver_run; CASE=/path/to/case_ACTIVSg70k
for s in 1 7 42 1588; do
  $BIN $CASE --precision tf32 --repeat 30 --warmup 10 --no-multistream --serial-nd --metis-seed $s | grep factorize_ms
  $BIN $CASE --precision tf32 --repeat 1 --no-multistream --serial-nd --metis-seed $s --analyze-info | grep -E '^  L'  # maxfsz per level -> spine_work
done
```
