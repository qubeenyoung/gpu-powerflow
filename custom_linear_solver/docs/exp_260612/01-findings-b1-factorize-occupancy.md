# B=1 factorize — occupancy 진단, ordering cost model, best-of-k

> **상태**: 완료   **날짜**: 2026-06-12   **GPU**: RTX 3090 (82 SM)
> **⚠️ 정밀도 주의**: 이 문서의 모든 수치는 `--single-precision fp32`(= fp32 입력 + **fp64 factor**, ncu 가 `factor_mid_blocked<double>` 노출)로 측정됨. **진짜 fp32 factor**(`--precision fp32`, `<float>`)의 결과는 [`03-precision-correction-and-fp32.md`](03-precision-correction-and-fp32.md) 참조 — 대형 case headroom 이 더 작다. 아래의 **구조 진단·cost model·best-of-k 메커니즘은 두 정밀도에 공통**(front 구조가 정밀도-무관)이고, 정밀도는 headroom 의 *크기*만 바꾼다.
> 측정: `custom_linear_solver_run`, B=1, `factorize_ms` median(repeat 20–30/warmup 6–10).

## 0. 요약

1. **진단**: B=1 factorize 시간은 소거트리 **깊은(deep) 레벨**에 집중. 그 레벨은 front 가 물리적으로 1–66개뿐(82 SM 미달) → 각 front 1 block/SM → **under-fill(occupancy 3–30%)**. 상위 레벨(수천 small front)은 GPU 가 잘 참.
2. **cost model**: under-filled large 레벨의 `Σ maxfsz³`(= `tail_cube`)가 fp32 B=1 factor 시간을 **factorize 없이** 잘 예측(seed 간 top-1 gap 0–3%). 이것이 ordering 선택 proxy.
3. **레버**: ordering 만 듣는다(+12–15%, 천장 ~1.13–1.18×). stream 수·amalgamation·tiling·sync 는 occupancy 벽에 막혀 무효.
4. **구현**: `CLS_ORDER_K` best-of-k(결정적). 천장을 거의 전부 회수 + parallel-ND 비결정성 제거.

## 1. 구조 진단 — under-fill 은 어디서 오나

`--analyze-info` 의 레벨별 (cnt, maxfsz) 와 occupancy cost model 로 레벨별 비용 분해(seed=best):

**25K (seed7), spine=[L24..L30]** — 비용은 **mid-tier deep level L8–L23**(cnt 66→2, 각 1 wave)에 집중. spine 은 6% 뿐.

| L | cnt | maxfsz | tier | cost% |
|---:|---:|---:|---|---:|
| 8 | 66 | 99 | mid | 4.4 |
| 11 | 21 | 117 | mid | 7.3 |
| 16 | 7 | 124 | mid | 8.7 |
| 20 | 3 | 117 | mid | 7.3 |
| … | | | | |

**70K (seed13), spine=[L35..L42]** — 비용은 **big-tier deep level L14–L34**(cnt 36→2, fsz 143–226, 각 1 block/SM)에 집중. L26–34 는 front **2개뿐**(ND 2-분할) → 2/82 SM = occ 2.4%. spine 3%.

| L | cnt | maxfsz | tier | cost% |
|---:|---:|---:|---|---:|
| 19 | 12 | 206 | BIG | 5.2 |
| 21 | 7 | 219 | BIG | 6.3 |
| 26 | 2 | 226 | BIG | 6.9 |
| 28 | 2 | 207 | BIG | 5.3 |

→ **핵심**: 지배 비용은 spine(직렬 꼬리)이 아니라 그 **바로 아래 under-filled large 레벨들**. 13K(8387)은 maxfsz≤84(전부 mid 이하)라 애초에 occupancy 가 높고 headroom 작음.

## 2. ordering 민감도 — fp32 B=1 (cap 튜닝 아님, seed 만)

serial-ND seed sweep, `factorize_ms`:

| case | s1 | s7 | s42(def) | s1588 | best | worst | **spread** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8387 | 0.634 | 0.600 | 0.549 | 0.536 | 0.536 | 0.634 | **18%** |
| 25k | 2.061 | **1.614** | 2.690 | 1.866 | 1.614 | 2.690 | **67%** |
| USA | 8.398 | **7.886** | 9.570 | 9.186 | 7.886 | 9.570 | **21%** |

- 같은 ND 알고리즘, **seed 만** 바꿔 15–67% 변동. default seed42 가 25k/USA 에서 **최악**.
- production default 는 **parallel ND(비결정적)** → run 마다 8–24% 흔들림(25k 1.685–1.998). 재현성 문제이자 종종 나쁜 ordering 을 뽑음.

> 이는 occupancy 가 병목이라는 직접 증거: ordering 이 deep-level front 의 **크기(fsz³)** 를 정하고, 그게 곧 시간이다.

## 3. negative results — scheduling/occupancy 직접 레버

| 실험 | 변경 | 결과 (best seed, fp32 B=1) |
|---|---|---|
| **subtree stream 8→16** | `kMaxSubtreeStreams=16` 재빌드 | 8387 0.532→0.538, 25k 1.615→1.615, **USA 7.676→7.673** → 무효 |
| multistream on/off | `--no-multistream` | ON 이 1.10–1.12× 우세(이미 기본). 끄면 손해 |
| amalgamation | `--max-panel-width 8..32` | **25k 1.852 평탄, USA 7.47 평탄**, 8387 약간 악화 |

**왜 stream 이 안 듣나**: 지배 레벨(25k L8–23, USA L14–34)의 독립 front 수가 **전역적으로** 2–66개뿐. 8 stream 이면 이미 충분히 분산; 더 늘려도 채울 일감이 없다. USA L26–34 는 front 2개라 어떤 scheduling 도 2/82 SM 초과 불가. **intra-front 병렬(tiling)** 은 re-staging 비용으로 이전 노트에서 회귀 확인. → occupancy 천장은 구조적.

## 4. ordering cost model — `tail_cube`

best-of-k 가 성립하려면 **factorize 없이**(analyze symbolic 만으로) 좋은 ordering 을 골라야 한다. 후보 proxy 를 26 seed(독립 2세트: {1,2,3,7,11,13,42,99,1588,2718} + 42..57)에서 검증, top-1 gap(선택 ordering 의 실측 time vs 그 세트 oracle):

| proxy | 정의 | avg top-1 gap | 비고 |
|---|---|---:|---|
| spine_work (note 09) | Σ spine fsz² | ~10–32% | 약함(넓은 seed 에서 무너짐) |
| top8_work | 최심 8레벨 Σfsz² | ~6% | note 09 의 "상위레벨 fsz²" |
| cube_wave | Σ ceil(cnt/SM)·maxfsz³ (전 레벨) | ~5% | 상위 filled 레벨이 noise |
| **`tail_cube`** | **Σ maxfsz³ over (cnt<SM ∧ maxfsz>32)** | **~1.8% (0–3%)** | **채택** |

**`tail_cube` 가 두 독립 세트 모두에서 oracle 거의 일치**:

| set | 8387 | 25k | USA |
|---|---:|---:|---:|
| 42–57 (16 seed) | 0.0% | 0.0% | 0.0% |
| 10-seed (독립) | 2.6% | 0.0% | 2.8% |

**물리적 의미**: under-filled large 레벨은 1 wave 이고 그 wall-time 은 **최대 front 의 O(maxfsz³) 패널 LU**가 지배. 잘 채워진 상위 레벨은 throughput-bound 라 ordering 간 분산이 작아 **빼야** 신호가 선명해진다. 이는 문헌이 지목한 "**critical-path FLOPs along the serial tail**"의 구현(→ [02-literature-review](02-literature-review.md) Q1).

## 5. best-of-k 구현 — `CLS_ORDER_K`

`src/analyze/pipeline.cpp`:
- `CLS_ORDER_K=K` (default 1 = 기존: parallel-ND 단일 ordering, 무변경).
- K>1: **결정적 serial-ND** 를 seed `[metis_seed .. metis_seed+K-1]` 로 K 회 → 각 plan 의 `tail_cube` 계산 → 최소 선택. 비-최선 후보의 device 버퍼는 RAII 로 즉시 해제(최선만 보유).
- `ordering_cost_model(plan)` = Σ over level (cnt<SM ∧ maxfsz>kSmallFrontMax) of maxfsz³. `CLS_ORDER_SM`(default 82)로 under-fill 임계 조정.

**결과 (vs honest parallel-default median; relres 불변, 결정적)**:

| case | base median | K=8 model | sel | **speedup** | 26-seed oracle 천장 |
|---|---:|---:|---:|---:|---:|
| 13K (8387) | 0.572 | 0.509 | 49 | **1.124×** | 1.121× (도달) |
| 25K | 1.896 | 1.646 | 47 | **1.152×** | 1.175× |
| 70K (USA) | 8.476 | 7.480 | 44 | **1.133×** | 1.133× (도달) |

**caveat — K 를 키우면 오히려 회귀**: USA K=8 7.480 → K=12 7.626 → K=16 7.877. proxy 가 bulk 는 잘 순위 매기나 **극단 꼬리**(가장 낮은 proxy)에서 outlier 를 뽑을 수 있음. **K≈8 이 sweet spot.** 25K 의 1.175× 천장은 best seed 7 이 42–49 윈도 밖이라 best-of-8(base 42)로는 못 닿음 → 윈도/측정-hybrid 로만 추가 회수 가능.

## 6. honest ceiling & 다음 단계

- **ordering 단독 천장**: 26 seed oracle 기준 **1.12× / 1.18× / 1.13×**. 목표 1.2× 는 25K 만 근접, **B=1 에서 ND-seed 변동만으로는 robust 하게 도달 불가.**
- **2차 레버 부재**: stream·amalgamation·tiling·sync 전부 occupancy 벽(본 실험) → 문헌도 B=1 작은-root 전용 기법 없음을 확인.
- **남은 회수 경로(미착수, 비용 큼)**:
  1. **proxy→측정 hybrid**: 넓은 seed pool 에서 `tail_cube` 로 top-m 선별 → m 개만 실제 1-shot factorize 측정 후 최종 선택. 큰 pool 의 oracle(>1.18×)에 접근. 반복-solve 에서 amortize.
  2. **critical-path-aware separator 재배열**(Kayaaslan–Uçar BBT, tree-height −28%): ND seed lottery 대신 구조적으로 짧은 꼬리 생성. 구현 비용 큼.
- **경제성**: best-of-k = k× analyze(8387 60ms·25k 195ms·USA 715ms / 회). 단발 B=1 엔 손해, **고정 grid·다수 시나리오 반복-solve(NR·contingency·time-series)에서 ordering 1회 계산·재사용**으로 매 solve 가 공짜로 빨라짐 — 실제 power-flow 워크로드에 부합.

## 부록 — 재현 명령

```bash
BIN=build/custom_linear_solver_run
# ordering 민감도
for s in 1 7 42 1588; do $BIN <case> --single-precision fp32 --repeat 30 --warmup 10 --serial-nd --metis-seed $s | grep factorize_ms; done
# best-of-8 (결정적, tail_cube 선택) + 후보별 proxy
CLS_ORDER_K=8 $BIN <case> --single-precision fp32 --repeat 30 --warmup 10 --metis-seed 42 --analyze-info 2>&1 | grep -E 'order-cand|order-select|factorize_ms'
# stream cap 실험: src/internal/types.hpp kMaxSubtreeStreams 변경 후 재빌드
# amalgamation: --max-panel-width {8,12,16,24,32}
```
